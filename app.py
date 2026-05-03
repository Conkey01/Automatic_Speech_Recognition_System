"""app.py — FastAPI ASR inference server with CORS for browser access."""
import io
import os
import torch
import numpy as np
import soundfile as sf
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download

from model import MiniWav2Vec2ASR

HF_REPO_ID  = os.environ.get("HF_REPO_ID", "your-username/mini-wav2vec2-asr")
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

# ── CORS: allow any browser to call us ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # or restrict to ["https://your-username.github.io"]
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    # Try soundfile first (handles wav, flac, ogg natively).
    # Fall back to librosa (handles webm, mp3, m4a from browser recordings).
    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception:
        try:
            audio, sr = librosa.load(io.BytesIO(raw), sr=None, mono=False)
            audio = audio.T if audio.ndim > 1 else audio
        except Exception as e:
            raise HTTPException(400, f"Could not decode audio: {e}")

    if audio.ndim > 1:
        audio = audio.mean(axis=1) if audio.shape[1] < audio.shape[0] else audio.mean(axis=0)

    if sr != TARGET_SR:
        audio = librosa.resample(audio.astype(np.float32),
                                 orig_sr=sr, target_sr=TARGET_SR)

    if len(audio) == 0:
        raise HTTPException(400, "Empty audio")

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