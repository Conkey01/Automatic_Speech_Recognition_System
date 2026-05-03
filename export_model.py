"""
export_model.py — Export trained ASR model for deployment.

Creates a clean, deployment-friendly file with just weights + config + vocab.
"""
import torch
from train import MiniWav2Vec2ASR   # imports your model class

CHECKPOINT_PATH = "./checkpoints_big/best_model.pt"
EXPORT_PATH     = "./asr_model_deploy.pth"

# Load the training checkpoint
ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

print(f"Source checkpoint:")
print(f"  Epoch:    {ckpt.get('epoch', '?')}")
print(f"  Step:     {ckpt.get('global_step', '?')}")
print(f"  Best WER: {ckpt.get('best_wer', float('nan')):.2%}")

# Bundle everything needed for inference
export = {
    "model_state_dict": ckpt["model_state_dict"],
    "model_config": {
        "encoder_channels":    512,
        "transformer_dim":     512,
        "transformer_ffn_dim": 2048,
        "transformer_heads":   8,
        "transformer_layers":  12,
        "dropout":             0.1,
    },
    "vocab":         MiniWav2Vec2ASR.VOCAB,
    "char_to_idx":   MiniWav2Vec2ASR.CHAR_TO_IDX,
    "vocab_size":    MiniWav2Vec2ASR.VOCAB_SIZE,
    "sample_rate":   16000,
    "best_wer":      ckpt.get("best_wer"),
    "epoch":         ckpt.get("epoch"),
}

torch.save(export, EXPORT_PATH)

import os
size_mb = os.path.getsize(EXPORT_PATH) / 1e6
print(f"\n✓ Exported to {EXPORT_PATH} ({size_mb:.1f} MB)")