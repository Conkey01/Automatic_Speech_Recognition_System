
const BACKEND_URL = "https://conkey01-mini-asr.hf.space";

const recordBtn = document.getElementById("recordBtn");
const recordStatus = document.getElementById("recordStatus");
const recordedAudio = document.getElementById("recordedAudio");
const fileInput = document.getElementById("fileInput");
const output = document.getElementById("output");
const meta = document.getElementById("meta");
document.getElementById("backend-url").textContent = BACKEND_URL;

// ── Recording ──
let mediaRecorder = null;
let chunks = [];
let isRecording = false;

recordBtn.addEventListener("click", async () => {
  if (!isRecording) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      chunks = [];

      mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
      mediaRecorder.onstop = async () => {
        const blob = new Blob(chunks, { type: mediaRecorder.mimeType });
        recordedAudio.src = URL.createObjectURL(blob);
        recordedAudio.hidden = false;
        stream.getTracks().forEach(t => t.stop());
        await transcribe(blob, "recording.webm");
      };

      mediaRecorder.start();
      isRecording = true;
      recordBtn.textContent = "⏹ Stop recording";
      recordBtn.classList.add("recording");
      recordStatus.textContent = "Listening…";
    } catch (err) {
      output.textContent = "Microphone access denied: " + err.message;
      output.className = "output error";
    }
  } else {
    mediaRecorder.stop();
    isRecording = false;
    recordBtn.textContent = "🔴 Start recording";
    recordBtn.classList.remove("recording");
    recordStatus.textContent = "Processing…";
  }
});

// ── File upload ──
fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (file) await transcribe(file, file.name);
});

// ── Send to backend ──
async function transcribe(blob, filename) {
  output.textContent = "Transcribing…";
  output.className = "output loading";
  meta.textContent = "";

  const formData = new FormData();
  formData.append("file", blob, filename);

  try {
    const t0 = performance.now();
    const res = await fetch(`${BACKEND_URL}/transcribe`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`HTTP ${res.status}: ${errText}`);
    }

    const data = await res.json();
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

    output.textContent = data.transcription || "(no speech detected)";
    output.className = "output";
    meta.textContent =
      `Duration: ${data.duration_sec}s · Latency: ${elapsed}s · Sample rate: ${data.sample_rate}Hz`;
    recordStatus.textContent = "";
  } catch (err) {
    output.textContent = "Error: " + err.message;
    output.className = "output error";
    recordStatus.textContent = "";
  }
}
