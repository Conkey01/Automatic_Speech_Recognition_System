"""
transcribe.py — Load a deployed ASR model and transcribe an audio file.

Usage:
    python transcribe.py path/to/audio.flac
    python transcribe.py path/to/audio.wav
"""
import sys
import torch
import soundfile as sf
from train import MiniWav2Vec2ASR

MODEL_PATH = "./asr_model_deploy.pth"

def load_model(model_path, device):
    bundle = torch.load(model_path, map_location=device, weights_only=False)
    cfg = bundle["model_config"]
    model = MiniWav2Vec2ASR(**cfg)
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"✓ Loaded model (epoch {bundle.get('epoch')}, "
          f"WER {bundle.get('best_wer'):.2%})")
    return model, bundle["sample_rate"]

def transcribe_file(model, audio_path, expected_sr, device):
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mono
    if sr != expected_sr:
        raise ValueError(f"Audio is {sr} Hz, model expects {expected_sr} Hz. "
                         f"Resample first.")
    waveform = torch.from_numpy(audio).float().to(device)
    return model.transcribe(waveform)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_file>")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sr = load_model(MODEL_PATH, device)

    for audio_path in sys.argv[1:]:
        text = transcribe_file(model, audio_path, sr, device)
        print(f"\n{audio_path}")
        print(f"  → {text}")