"""
train.py — ASR (larger model + more data) on LibriSpeech.

Usage:
    mkdir -p logs
    tmux new -s train      # or: nohup python -u train.py > logs/train.log 2>&1 &
    python -u train.py 2>&1 | tee -a logs/train.log
"""

import os
import sys
import glob
import time
import math
import random
import urllib.request
import tarfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf


# ══════════════════════════════════════════════
# 0. DATA DOWNLOAD (skips if already present)
# ══════════════════════════════════════════════
def ensure_data(data_root="./data"):
    os.makedirs(data_root, exist_ok=True)
    urls = {
        "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
        "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
        "dev-clean":       "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    }
    for name, url in urls.items():
        tar_path = os.path.join(data_root, f"{name}.tar.gz")
        extract_dir = os.path.join(data_root, "LibriSpeech", name)
        if os.path.exists(extract_dir):
            print(f"✓ {name} already extracted")
            continue
        if not os.path.exists(tar_path):
            print(f"Downloading {name} (this can be large — ~23 GB for 360h)...")
            def progress(count, block_size, total_size):
                pct = count * block_size * 100 / total_size
                mb = count * block_size / 1e6
                total_mb = total_size / 1e6
                sys.stdout.write(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct:.1f}%)")
                sys.stdout.flush()
            urllib.request.urlretrieve(url, tar_path, reporthook=progress)
            print("\n  Download complete.")
        print(f"  Extracting {name}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=data_root)
        print(f"  Done.")


# ══════════════════════════════════════════════
# 1. MODEL — bigger version
# ══════════════════════════════════════════════
class MiniWav2Vec2ASR(nn.Module):
    """
    Wav2vec2-style architecture trained end-to-end with CTC.
    Scaled up: 512 dim, 12 transformer layers, ~25M params.
    """
    VOCAB = {
        0: "<blank>", 1: " ",
        2: "a", 3: "b", 4: "c", 5: "d", 6: "e", 7: "f", 8: "g", 9: "h",
        10: "i", 11: "j", 12: "k", 13: "l", 14: "m", 15: "n", 16: "o",
        17: "p", 18: "q", 19: "r", 20: "s", 21: "t", 22: "u", 23: "v",
        24: "w", 25: "x", 26: "y", 27: "z", 28: "'",
    }
    CHAR_TO_IDX = {v: k for k, v in VOCAB.items()}
    VOCAB_SIZE = len(VOCAB)

    def __init__(self, encoder_channels=512, transformer_dim=512,
                 transformer_ffn_dim=2048, transformer_heads=8,
                 transformer_layers=12, dropout=0.1,
                 freq_mask_param=40, num_freq_masks=2,
                 time_mask_param=40, num_time_masks=4):
        super().__init__()

        self.freq_mask_param = freq_mask_param
        self.num_freq_masks = num_freq_masks
        self.time_mask_param = time_mask_param
        self.num_time_masks = num_time_masks

        # 7-layer CNN feature encoder (same wav2vec2 stride config)
        cfg = [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]
        encoder_layers = []
        in_ch = 1
        for k, s in cfg:
            encoder_layers.append(nn.Conv1d(in_ch, encoder_channels, k, stride=s, bias=False))
            encoder_layers.append(nn.GroupNorm(1, encoder_channels))
            encoder_layers.append(nn.GELU())
            in_ch = encoder_channels
        self.feature_encoder = nn.Sequential(*encoder_layers)

        self.proj = nn.Linear(encoder_channels, transformer_dim)
        self.layer_norm = nn.LayerNorm(transformer_dim)
        self.feature_dropout = nn.Dropout(dropout)

        # Convolutional positional encoding
        self.pos_conv = nn.Conv1d(
            transformer_dim, transformer_dim,
            kernel_size=128, padding=64, groups=16, bias=True,
        )
        self.pos_conv = nn.utils.parametrizations.weight_norm(
            self.pos_conv, name="weight", dim=2
        )
        self.pos_norm = nn.LayerNorm(transformer_dim)

        # Transformer (12 layers, 8 heads)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=transformer_heads,
            dim_feedforward=transformer_ffn_dim, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.transformer_norm = nn.LayerNorm(transformer_dim)

        # 2-layer CTC head
        self.ctc_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(transformer_dim, transformer_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim, self.VOCAB_SIZE),
        )

        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')

        self.apply(self._init_weights)
        with torch.no_grad():
            self.ctc_head[-1].bias[0] = -0.5

        self.transformer_dim = transformer_dim

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

    def _spec_augment(self, x):
        if not self.training:
            return x
        B, T, D = x.shape
        x = x.clone()
        for b in range(B):
            for _ in range(self.num_freq_masks):
                f = random.randint(0, min(self.freq_mask_param, D - 1))
                f0 = random.randint(0, D - f)
                x[b, :, f0:f0 + f] = 0.0
            for _ in range(self.num_time_masks):
                t = random.randint(0, min(self.time_mask_param, T - 1))
                t0 = random.randint(0, T - t)
                x[b, t0:t0 + t, :] = 0.0
        return x

    def encode_text(self, text):
        text = text.lower().strip()
        return [self.CHAR_TO_IDX[ch] for ch in text if ch in self.CHAR_TO_IDX]

    def decode_tokens(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        collapsed = []
        prev = None
        for t in token_ids:
            if t != prev:
                collapsed.append(t)
            prev = t
        return "".join(self.VOCAB.get(t, "?") for t in collapsed if t != 0)

    def forward(self, waveform, targets=None, target_lengths=None):
        waveform = (waveform - waveform.mean(dim=-1, keepdim=True)) / \
                   (waveform.std(dim=-1, keepdim=True) + 1e-6)

        x = waveform.unsqueeze(1)
        x = self.feature_encoder(x)

        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.layer_norm(x)
        x = self.feature_dropout(x)

        x = self._spec_augment(x)

        residual = x
        pos = x.transpose(1, 2)
        pos = self.pos_conv(pos)
        pos = pos[:, :, :residual.shape[1]]
        pos = F.gelu(pos)
        pos = pos.transpose(1, 2)
        x = self.pos_norm(pos + residual)

        x = self.transformer(x)
        x = self.transformer_norm(x)

        logits = self.ctc_head(x)
        log_probs = logits.log_softmax(dim=-1)

        result = {"logits": logits, "log_probs": log_probs}

        if targets is not None and target_lengths is not None:
            log_probs_t = log_probs.transpose(0, 1)
            input_lengths = torch.full(
                (waveform.shape[0],), log_probs.shape[1],
                dtype=torch.long, device=waveform.device
            )
            loss = self.ctc_loss(log_probs_t, targets, input_lengths, target_lengths)
            result["loss"] = loss

        return result

    @torch.no_grad()
    def transcribe(self, waveform):
        self.eval()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        out = self.forward(waveform)
        pred_ids = out["logits"].argmax(dim=-1)
        return self.decode_tokens(pred_ids[0])


# ══════════════════════════════════════════════
# 2. DATA
# ══════════════════════════════════════════════
def load_librispeech_samples(data_dir, max_audio_len=200000):
    trans_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.trans.txt"), recursive=True))
    samples = []
    for tf in trans_files:
        base_dir = os.path.dirname(tf)
        with open(tf, "r") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    utt_id, text = parts
                    flac_path = os.path.join(base_dir, f"{utt_id}.flac")
                    if os.path.exists(flac_path):
                        info = sf.info(flac_path)
                        n_samples = int(info.duration * info.samplerate)
                        if n_samples <= max_audio_len and len(text.strip()) > 0:
                            samples.append((flac_path, text.lower().strip()))
    return samples


def load_all_train_samples(train_dirs, max_audio_len):
    all_samples = []
    for d in train_dirs:
        if os.path.exists(d):
            samples = load_librispeech_samples(d, max_audio_len)
            print(f"    {d}: {len(samples)} utterances")
            all_samples.extend(samples)
        else:
            print(f"    {d}: NOT FOUND, skipping")
    random.shuffle(all_samples)
    return all_samples


def collate_batch(batch_items, char_to_idx):
    waveforms = []
    all_tokens = []
    target_lengths = []
    for audio_path, text in batch_items:
        audio, sr = sf.read(audio_path, dtype="float32")
        waveforms.append(torch.from_numpy(audio).float())
        tokens = [char_to_idx[ch] for ch in text if ch in char_to_idx]
        all_tokens.extend(tokens)
        target_lengths.append(len(tokens))
    max_len = max(w.shape[0] for w in waveforms)
    padded = [F.pad(w, (0, max_len - w.shape[0])) for w in waveforms]
    return (
        torch.stack(padded),
        torch.tensor(all_tokens, dtype=torch.long),
        torch.tensor(target_lengths, dtype=torch.long),
    )


# ══════════════════════════════════════════════
# 3. WER
# ══════════════════════════════════════════════
def edit_distance(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]


def compute_wer(references, hypotheses):
    total_err, total_words = 0, 0
    for ref, hyp in zip(references, hypotheses):
        ref_w, hyp_w = ref.strip().split(), hyp.strip().split()
        total_err += edit_distance(ref_w, hyp_w)
        total_words += len(ref_w)
    return total_err / max(1, total_words)


# ══════════════════════════════════════════════
# 4. TRAINING
# ══════════════════════════════════════════════
def train():
    TRAIN_DIRS = [
        "./data/LibriSpeech/train-clean-100",
        "./data/LibriSpeech/train-clean-360",
    ]
    VAL_DIR  = "./data/LibriSpeech/dev-clean"
    SAVE_DIR = "./checkpoints_big"   # NEW dir so old 6.5M checkpoint isn't loaded
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Hyperparameters ──
    MAX_AUDIO_LEN   = 200000   # 12.5s at 16kHz
    BATCH_SIZE      = 4        # smaller per-step batch to fit 25M model on T4
    GRAD_ACCUM      = 8        # effective batch = 32
    NUM_EPOCHS      = 60       # 460h × 60 epochs is plenty
    VAL_EVERY_EPOCH = 2
    LOG_EVERY       = 200
    LR              = 3e-4
    WARMUP_FRACTION = 0.08
    MAX_GRAD_NORM   = 5.0
    USE_AMP         = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Model
    model = MiniWav2Vec2ASR(
        encoder_channels=512, transformer_dim=512,
        transformer_ffn_dim=2048, transformer_heads=8,
        transformer_layers=12, dropout=0.1,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,}")

    # Resume bookkeeping
    resume_path = os.path.join(SAVE_DIR, "latest.pt")
    start_epoch = 1
    global_step = 0
    best_wer = float("inf")
    ckpt = None

    if os.path.exists(resume_path):
        print(f"\nResuming from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_wer = ckpt.get("best_wer", float("inf"))
        print(f"  Resuming from epoch {start_epoch}, step {global_step}, best WER {best_wer:.2%}")

    # Data
    print(f"\nLoading data...")
    train_samples = load_all_train_samples(TRAIN_DIRS, MAX_AUDIO_LEN)
    val_samples   = load_librispeech_samples(VAL_DIR, MAX_AUDIO_LEN)
    random.shuffle(val_samples)
    print(f"  Train total: {len(train_samples)}, Val: {len(val_samples)}")

    steps_per_epoch = len(train_samples) // BATCH_SIZE
    total_steps   = (steps_per_epoch // GRAD_ACCUM) * NUM_EPOCHS
    warmup_steps  = int(WARMUP_FRACTION * total_steps)
    print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Total optimizer steps: {total_steps}, Warmup: {warmup_steps}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR,
        betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01,
    )

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    # Restore optimizer / scheduler / scaler if resuming
    if ckpt is not None:
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            for _ in range(global_step):
                scheduler.step()
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        print(f"  Optimizer/scheduler/scaler restored at step {global_step}")

    print(f"\n{'='*70}")
    print(f"Bigger model + more data: epochs {start_epoch}-{NUM_EPOCHS}")
    print(f"  Model: ~{num_params/1e6:.1f}M params  |  Eff batch: {BATCH_SIZE*GRAD_ACCUM}")
    print(f"  Data:  {len(train_samples)} utterances")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        random.shuffle(train_samples)
        epoch_loss = 0.0
        epoch_batches = 0
        accum_count = 0
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)

        for batch_start in range(0, len(train_samples) - BATCH_SIZE + 1, BATCH_SIZE):
            batch_items = train_samples[batch_start: batch_start + BATCH_SIZE]

            try:
                waveforms, targets, target_lengths = collate_batch(
                    batch_items, model.CHAR_TO_IDX
                )
            except Exception:
                continue

            waveforms      = waveforms.to(device)
            targets        = targets.to(device)
            target_lengths = target_lengths.to(device)

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                out = model(waveforms, targets=targets, target_lengths=target_lengths)
                loss = out["loss"] / GRAD_ACCUM

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * GRAD_ACCUM
            epoch_batches += 1
            accum_count += 1

            if accum_count % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % LOG_EVERY == 0:
                    avg = epoch_loss / max(1, epoch_batches)
                    lr_now = optimizer.param_groups[0]["lr"]
                    mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0

                    with torch.no_grad():
                        pred_ids = out["logits"].argmax(dim=-1)[0]
                        non_blank = (pred_ids != 0).float().mean().item()
                        unique = len(set(pred_ids.tolist()))
                        quick_pred = model.decode_tokens(pred_ids)

                    print(
                        f"  [Epoch {epoch:>3d} Step {global_step:>6d}] "
                        f"loss={avg:.4f}  "
                        f"lr={lr_now:.1e}  "
                        f"grad={grad_norm:.2f}  "
                        f"non_blank={non_blank:.1%}  "
                        f"unique={unique}  "
                        f"mem={mem:.1f}GB",
                        flush=True,
                    )
                    if len(quick_pred) > 0:
                        print(f"    → {quick_pred[:80]}", flush=True)

        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(1, epoch_batches)
        print(f"\n[Epoch {epoch}/{NUM_EPOCHS}] loss={avg_loss:.4f} time={elapsed:.0f}s", flush=True)

        # Validation
        if epoch % VAL_EVERY_EPOCH == 0 or epoch == NUM_EPOCHS:
            model.eval()
            references = []
            hypotheses = []
            n_eval = min(300, len(val_samples))

            with torch.no_grad():
                for i in range(n_eval):
                    audio_path, text = val_samples[i]
                    audio, sr = sf.read(audio_path, dtype="float32")
                    waveform = torch.from_numpy(audio).float().to(device)
                    pred = model.transcribe(waveform)
                    references.append(text)
                    hypotheses.append(pred)

            wer = compute_wer(references, hypotheses)
            non_empty = sum(1 for h in hypotheses if len(h.strip()) > 0)
            avg_pred_len = np.mean([len(h) for h in hypotheses]) if hypotheses else 0

            print(f"\n  {'─'*60}")
            print(f"  Validation at epoch {epoch}")
            print(f"    WER:             {wer:.2%}")
            print(f"    Non-empty preds: {non_empty}/{n_eval}")
            print(f"    Avg pred length: {avg_pred_len:.1f} chars")

            print(f"\n    Decoding samples:")
            for i in range(min(8, len(references))):
                print(f"      REF: {references[i][:80]}")
                hyp = hypotheses[i][:80] if hypotheses[i] else "(empty)"
                print(f"      HYP: {hyp}")
                print()

            if wer < best_wer:
                best_wer = wer
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_wer": best_wer,
                }, os.path.join(SAVE_DIR, "best_model.pt"))
                print(f"    ✓ New best WER: {best_wer:.2%}")

        # Always save latest for resume
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_wer": best_wer,
        }, os.path.join(SAVE_DIR, "latest.pt"))

        if epoch % VAL_EVERY_EPOCH == 0:
            print()

    print(f"{'='*70}")
    print(f"Training complete!")
    print(f"  Best WER: {best_wer:.2%}")
    print(f"  Checkpoints: {SAVE_DIR}")
    print(f"{'='*70}")

    print(f"\nFinal decoding:")
    model.eval()
    with torch.no_grad():
        for i in range(min(10, len(val_samples))):
            audio_path, text = val_samples[i]
            audio, sr = sf.read(audio_path, dtype="float32")
            waveform = torch.from_numpy(audio).float().to(device)
            pred = model.transcribe(waveform)
            print(f"  REF: {text[:90]}")
            print(f"  HYP: {pred[:90]}")
            print()


# ══════════════════════════════════════════════
# 5. ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == "__main__":
    ensure_data("./data")
    train()