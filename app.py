"""app.py — FastAPI ASR inference server with CORS for browser access."""
import io
import os
import subprocess
import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download

from model import MiniWav2Vec2ASR

HF_REPO_ID  = os.environ.get("HF_REPO_ID", "Conkey01/mini-asr")
HF_FILENAME = os.environ.get("HF_FILENAME", "asr_model.pth")
TARGET_SR   = 16000

print("Loading model…")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
bundle = torch.load(weights_path, map_location=device, weights_only=False)
model = MiniWav2Vec2ASR(**bundle["model_config"])
model.load_state_dict(bundle["model_state_dict"])
model.to(device).eval()
print(f"✓ Model loaded on {device}  (epoch {bundle.get('epoch')}, "
      f"WER {bundle.get('best_wer'):.2%})")

app = FastAPI(title="Mini-Wav2Vec2 ASR")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_audio_with_ffmpeg(raw_bytes: bytes, target_sr: int = TARGET_SR) -> np.ndarray:
    """
    Decode any audio format (webm/opus, mp3, m4a, wav, flac, ogg, etc.)
    by piping bytes through ffmpeg. Returns mono float32 at target_sr.
    """
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "wav",
        "-acodec", "pcm_s16le",
        "-ac", "1",                 # mono
        "-ar", str(target_sr),      # resample
        "pipe:1",
    ]
    proc = subprocess.run(cmd, input=raw_bytes, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed: {proc.stderr.decode('utf-8', errors='ignore')[:300]}"
        )
    audio, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    return audio


@app.get("/")
def root():
    return {"status": "ok", "device": str(device),
            "model": HF_REPO_ID, "sample_rate": TARGET_SR}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty upload")

    # Fast path: try soundfile (works for wav/flac/ogg upload-from-disk)
    audio = None
    try:
        a, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != TARGET_SR:
            # let ffmpeg handle the resample for consistency
            audio = None
        else:
            audio = a.astype(np.float32)
    except Exception:
        audio = None

    # Fallback / resample path: pipe through ffmpeg
    if audio is None:
        try:
            audio = decode_audio_with_ffmpeg(raw, TARGET_SR)
        except Exception as e:
            raise HTTPException(400, f"Could not decode audio: {e}")

    if len(audio) == 0:
        raise HTTPException(400, "Decoded audio is empty")

    waveform = torch.from_numpy(audio).float().to(device)
    text = model.transcribe(waveform)

    return JSONResponse({
        "transcription": text,
        "duration_sec": round(len(audio) / TARGET_SR, 2),
        "sample_rate": TARGET_SR,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
