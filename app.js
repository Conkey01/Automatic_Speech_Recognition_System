// ═══════════════════════════════════════════════
//  CONFIG
// ═══════════════════════════════════════════════
const BACKEND_URL = "https://conkey01-mini-asr.hf.space";   // ← updated below for new name
const MAX_RECORD_SEC = 15;

// ═══════════════════════════════════════════════
//  DOM (defensive — won't crash if any are missing)
// ═══════════════════════════════════════════════
const $ = (id) => document.getElementById(id);

const recordBtn      = $("recordBtn");
const recordState    = $("recordState");
const recordedAudio  = $("recordedAudio");
const fileInput      = $("fileInput");
const dropzone       = $("dropzone");
const output         = $("output");
const meta           = $("meta");
const timer          = $("timer");
const visualizer     = $("visualizer");
const copyBtn        = $("copyBtn");
const backendUrlEl   = $("backend-url");

if (backendUrlEl) backendUrlEl.textContent = BACKEND_URL.replace(/^https?:\/\//, "");

const visCtx = visualizer ? visualizer.getContext("2d") : null;

// ═══════════════════════════════════════════════
//  RECORDING
// ═══════════════════════════════════════════════
let mediaRecorder = null;
let audioContext  = null;
let analyser      = null;
let micSource     = null;
let chunks        = [];
let isRecording   = false;
let recordStartTs = 0;
let timerInterval = null;
let autoStopTimer = null;
let visAnimFrame  = null;

if (recordBtn) {
  recordBtn.addEventListener("click", async () => {
    console.log("[ASR] Record button clicked. isRecording =", isRecording);
    if (!isRecording) await startRecording();
    else stopRecording();
  });
} else {
  console.error("[ASR] recordBtn not found in DOM!");
}

async function startRecording() {
  // Quick capability check
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError(
      "Microphone API not available. The page must be served over HTTPS " +
      "(or localhost), and your browser must support getUserMedia."
    );
    return;
  }

  try {
    console.log("[ASR] Requesting mic stream…");
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    console.log("[ASR] Got mic stream:", stream);

    // Visualizer setup (optional — won't crash if canvas missing)
    if (visualizer && visCtx) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 512;
      micSource = audioContext.createMediaStreamSource(stream);
      micSource.connect(analyser);
      drawVisualizer();
    }

    // MediaRecorder
    const mimeType = pickMimeType();
    console.log("[ASR] Using mimeType:", mimeType || "(browser default)");
    mediaRecorder = mimeType
      ? new MediaRecorder(stream, { mimeType })
      : new MediaRecorder(stream);

    chunks = [];
    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunks.push(e.data);
    };
    mediaRecorder.onstop = async () => {
      console.log("[ASR] MediaRecorder stopped. Chunks:", chunks.length);
      const type = mediaRecorder.mimeType || "audio/webm";
      const blob = new Blob(chunks, { type });
      console.log("[ASR] Blob size:", blob.size, "type:", blob.type);

      if (recordedAudio) {
        recordedAudio.src = URL.createObjectURL(blob);
        recordedAudio.hidden = false;
      }
      stream.getTracks().forEach((t) => t.stop());
      cleanupVisualizer();

      if (blob.size === 0) {
        showError("Recording was empty (no audio captured).");
        if (recordState) recordState.textContent = "Tap to record";
        return;
      }

      const ext = type.includes("mp4") ? "mp4"
                 : type.includes("ogg") ? "ogg"
                 : "webm";
      await transcribe(blob, `recording.${ext}`);
    };
    mediaRecorder.onerror = (e) => {
      console.error("[ASR] MediaRecorder error:", e);
      showError("Recording error: " + (e.error?.message || e));
    };
    mediaRecorder.start();

    // UI
    isRecording = true;
    recordBtn.classList.add("recording");
    recordBtn.setAttribute("aria-label", "Stop recording");
    if (recordState) recordState.textContent = "Recording…";
    recordStartTs = Date.now();
    startTimer();

    autoStopTimer = setTimeout(() => {
      if (isRecording) stopRecording();
    }, MAX_RECORD_SEC * 1000);
  } catch (err) {
    console.error("[ASR] startRecording failed:", err);
    let msg = err.message;
    if (err.name === "NotAllowedError") {
      msg = "Microphone permission was denied. Click the lock icon in your browser address bar and allow microphone access.";
    } else if (err.name === "NotFoundError") {
      msg = "No microphone found on this device.";
    } else if (err.name === "NotReadableError") {
      msg = "Microphone is being used by another app. Close other tabs/apps and try again.";
    }
    showError(msg);
  }
}

function stopRecording() {
  if (!isRecording || !mediaRecorder) return;
  console.log("[ASR] Stopping recording…");
  try { mediaRecorder.stop(); } catch (e) { console.error(e); }
  isRecording = false;
  recordBtn.classList.remove("recording");
  recordBtn.setAttribute("aria-label", "Start recording");
  if (recordState) recordState.textContent = "Processing…";
  stopTimer();
  if (autoStopTimer) clearTimeout(autoStopTimer);
}

// Pick the most compatible mime type the browser supports
function pickMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/mp4",
    "",
  ];
  for (const c of candidates) {
    if (c === "" || (window.MediaRecorder && MediaRecorder.isTypeSupported(c))) {
      return c;
    }
  }
  return "";
}

// ═══════════════════════════════════════════════
//  TIMER
// ═══════════════════════════════════════════════
function startTimer() {
  if (!timer) return;
  timer.classList.add("recording");
  updateTimer();
  timerInterval = setInterval(updateTimer, 100);
}

function stopTimer() {
  if (timerInterval) clearInterval(timerInterval);
  if (timer) timer.classList.remove("recording");
}

function updateTimer() {
  if (!timer) return;
  const elapsed = (Date.now() - recordStartTs) / 1000;
  const remaining = Math.max(0, MAX_RECORD_SEC - elapsed);
  const m = Math.floor(remaining / 60);
  const s = Math.floor(remaining % 60);
  timer.textContent = `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

// ═══════════════════════════════════════════════
//  VISUALIZER
// ═══════════════════════════════════════════════
function drawVisualizer() {
  if (!analyser || !visCtx || !visualizer) return;
  const bufferLen = analyser.frequencyBinCount;
  const data = new Uint8Array(bufferLen);

  function render() {
    visAnimFrame = requestAnimationFrame(render);
    analyser.getByteFrequencyData(data);

    const w = visualizer.width;
    const h = visualizer.height;
    visCtx.clearRect(0, 0, w, h);

    const barCount = 64;
    const barWidth = w / barCount;
    const step = Math.floor(bufferLen / barCount);

    for (let i = 0; i < barCount; i++) {
      const v = data[i * step] / 255;
      const barH = Math.max(2, v * h * 0.85);
      const x = i * barWidth;
      const y = (h - barH) / 2;

      const grad = visCtx.createLinearGradient(0, y, 0, y + barH);
      grad.addColorStop(0, "#ec4899");
      grad.addColorStop(0.5, "#7c3aed");
      grad.addColorStop(1, "#06b6d4");
      visCtx.fillStyle = grad;
      visCtx.fillRect(x + 1, y, barWidth - 2, barH);
    }
  }
  render();
}

function cleanupVisualizer() {
  if (visAnimFrame) cancelAnimationFrame(visAnimFrame);
  visAnimFrame = null;
  if (audioContext) {
    try { audioContext.close(); } catch (e) {}
    audioContext = null;
  }
  analyser = null;
  micSource = null;
  if (visCtx && visualizer) visCtx.clearRect(0, 0, visualizer.width, visualizer.height);
}

// ═══════════════════════════════════════════════
//  FILE UPLOAD + DRAG & DROP
// ═══════════════════════════════════════════════
if (fileInput) {
  fileInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (file) await transcribe(file, file.name);
  });
}

if (dropzone) {
  ["dragenter", "dragover"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.add("drag-over");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag-over");
    })
  );
  dropzone.addEventListener("drop", async (e) => {
    const file = e.dataTransfer.files[0];
    if (file) await transcribe(file, file.name);
  });
}

// ═══════════════════════════════════════════════
//  TRANSCRIBE
// ═══════════════════════════════════════════════
async function transcribe(blob, filename) {
  if (!output) return;
  output.textContent = "Transcribing…";
  output.className = "output loading";
  if (meta) { meta.hidden = true; meta.innerHTML = ""; }
  if (copyBtn) copyBtn.hidden = true;

  const wakeHint = setTimeout(() => {
    output.innerHTML = "Waking up server <span style='opacity:0.6'>(free Spaces sleep after inactivity)…</span>";
  }, 4000);

  const formData = new FormData();
  formData.append("file", blob, filename);

  try {
    const t0 = performance.now();
    const res = await fetch(`${BACKEND_URL}/transcribe`, {
      method: "POST",
      body: formData,
    });
    clearTimeout(wakeHint);

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`HTTP ${res.status}: ${errText}`);
    }

    const data = await res.json();
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
    const text = (data.transcription || "").trim();

    if (text) {
      output.textContent = text;
      output.className = "output success";
      if (copyBtn) copyBtn.hidden = false;
    } else {
      output.innerHTML = '<span class="output-placeholder">No speech detected</span>';
      output.className = "output empty";
    }

    if (meta) {
      meta.innerHTML = `
        <span class="meta-item"><strong>Duration</strong>${data.duration_sec}s</span>
        <span class="meta-item"><strong>Latency</strong>${elapsed}s</span>
        <span class="meta-item"><strong>Sample rate</strong>${(data.sample_rate / 1000).toFixed(0)} kHz</span>
      `;
      meta.hidden = false;
    }

    if (recordState) recordState.textContent = "Tap to record";
  } catch (err) {
    clearTimeout(wakeHint);
    showError(err.message);
    if (recordState) recordState.textContent = "Tap to record";
  }
}

function showError(msg) {
  console.error("[ASR]", msg);
  if (!output) { alert(msg); return; }
  output.textContent = "Error: " + msg;
  output.className = "output error";
}

// ═══════════════════════════════════════════════
//  COPY TO CLIPBOARD
// ═══════════════════════════════════════════════
if (copyBtn) {
  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(output.textContent);
      copyBtn.classList.add("copied");
      setTimeout(() => copyBtn.classList.remove("copied"), 1500);
    } catch (e) { console.error(e); }
  });
}

// ═══════════════════════════════════════════════
//  RESPONSIVE CANVAS
// ═══════════════════════════════════════════════
function resizeCanvas() {
  if (!visualizer) return;
  const rect = visualizer.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  visualizer.width = rect.width * dpr;
  visualizer.height = rect.height * dpr;
  if (visCtx) visCtx.scale(dpr, dpr);
}
window.addEventListener("resize", resizeCanvas);
resizeCanvas();

console.log("[ASR] Frontend initialized. Backend:", BACKEND_URL);