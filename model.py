"""model.py — MiniWav2Vec2ASR architecture for inference."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniWav2Vec2ASR(nn.Module):
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
                 transformer_layers=12, dropout=0.1, **kwargs):
        super().__init__()

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

        self.pos_conv = nn.Conv1d(
            transformer_dim, transformer_dim,
            kernel_size=128, padding=64, groups=16, bias=True,
        )
        self.pos_conv = nn.utils.parametrizations.weight_norm(
            self.pos_conv, name="weight", dim=2
        )
        self.pos_norm = nn.LayerNorm(transformer_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=transformer_heads,
            dim_feedforward=transformer_ffn_dim, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.transformer_norm = nn.LayerNorm(transformer_dim)

        self.ctc_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(transformer_dim, transformer_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim, self.VOCAB_SIZE),
        )

    def decode_tokens(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        collapsed, prev = [], None
        for t in token_ids:
            if t != prev:
                collapsed.append(t)
            prev = t
        return "".join(self.VOCAB.get(t, "?") for t in collapsed if t != 0)

    def forward(self, waveform):
        waveform = (waveform - waveform.mean(dim=-1, keepdim=True)) / \
                   (waveform.std(dim=-1, keepdim=True) + 1e-6)
        x = waveform.unsqueeze(1)
        x = self.feature_encoder(x)
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.layer_norm(x)
        x = self.feature_dropout(x)

        residual = x
        pos = x.transpose(1, 2)
        pos = self.pos_conv(pos)
        pos = pos[:, :, :residual.shape[1]]
        pos = F.gelu(pos)
        pos = pos.transpose(1, 2)
        x = self.pos_norm(pos + residual)

        x = self.transformer(x)
        x = self.transformer_norm(x)
        return self.ctc_head(x)

    @torch.no_grad()
    def transcribe(self, waveform):
        self.eval()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        logits = self.forward(waveform)
        pred_ids = logits.argmax(dim=-1)
        return self.decode_tokens(pred_ids[0])