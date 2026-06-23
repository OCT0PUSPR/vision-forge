/* vision-forge web GUI
 *
 * Two modes:
 *   1. Image upload (drag-drop / browse)  -> POST /infer (multipart)
 *   2. Webcam streaming                   -> WebSocket /ws/stream (base64 frames)
 *
 * The annotated image is drawn server-side and returned as a data URL; the
 * raw detections are also returned so we can draw boxes client-side on the
 * canvas, keeping the UI responsive and consistent across both modes.
 */
(function () {
  "use strict";

  const els = {
    taskSelect: document.getElementById("task-select"),
    backendSelect: document.getElementById("backend-select"),
    btnWebcam: document.getElementById("btn-webcam"),
    btnStop: document.getElementById("btn-stop"),
    dropzone: document.getElementById("dropzone"),
    fileInput: document.getElementById("file-input"),
    canvas: document.getElementById("view"),
    video: document.getElementById("webcam"),
    overlayHint: document.getElementById("overlay-hint"),
    fps: document.getElementById("fps-value"),
    count: document.getElementById("count-value"),
    ms: document.getElementById("ms-value"),
    labelList: document.getElementById("label-list"),
    connDot: document.getElementById("conn-dot"),
    connLabel: document.getElementById("conn-label"),
    versionPill: document.getElementById("version-pill"),
  };

  const ctx = els.canvas.getContext("2d");
  const PALETTE = [
    "#ff3838", "#ff9f38", "#ffd738", "#97ff38", "#38ff74",
    "#38ffeb", "#389fff", "#384cff", "#9738ff", "#eb38ff",
    "#ff389f", "#a0a0a0",
  ];

  let ws = null;
  let streaming = false;
  let stream = null;
  let frameIndex = 0;
  let sendBusy = false;
  let lastTs = 0;
  let fpsEMA = 0;
  let rafId = null;

  function colorForIndex(i) {
    if (i < 0) i = -i;
    return PALETTE[i % PALETTE.length];
  }

  function setConn(state) {
    if (state === "online") {
      els.connDot.className = "dot online";
      els.connLabel.textContent = "connected";
    } else {
      els.connDot.className = "dot offline";
      els.connLabel.textContent = "disconnected";
    }
  }

  function hideHint() {
    els.overlayHint.classList.add("hidden");
  }

  function drawDetections(result) {
    const dets = (result && result.detections) || [];
    els.count.textContent = String(dets.length);
    els.ms.textContent = String(Math.round(result.inference_ms || 0));

    ctx.lineWidth = 2;
    ctx.font = "14px Inter, sans-serif";
    ctx.textBaseline = "top";

    dets.forEach(function (d) {
      const idx = d.track_id != null ? d.track_id : d.class_id || 0;
      const color = colorForIndex(idx);
      const [x1, y1, x2, y2] = d.bbox;

      if (d.mask && d.mask.length) {
        ctx.save();
        ctx.globalAlpha = 0.3;
        ctx.fillStyle = color;
        ctx.beginPath();
        d.mask.forEach(function (p, i) {
          if (i === 0) ctx.moveTo(p[0], p[1]);
          else ctx.lineTo(p[0], p[1]);
        });
        ctx.closePath();
        ctx.fill();
        ctx.restore();
      }

      ctx.strokeStyle = color;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      if (d.keypoints && d.keypoints.length) {
        ctx.fillStyle = color;
        d.keypoints.forEach(function (kp) {
          if (kp.confidence > 0.2) {
            ctx.beginPath();
            ctx.arc(kp.x, kp.y, 3, 0, Math.PI * 2);
            ctx.fill();
          }
        });
      }

      let caption = d.label;
      if (d.track_id != null) caption = "#" + d.track_id + " " + caption;
      caption += " " + (d.confidence != null ? d.confidence.toFixed(2) : "");
      const tw = ctx.measureText(caption).width;
      ctx.fillStyle = color;
      ctx.fillRect(x1, Math.max(0, y1 - 18), tw + 8, 18);
      ctx.fillStyle = "#0b0e14";
      ctx.fillText(caption, x1 + 4, Math.max(0, y1 - 17));
    });

    renderLabels(result.counts_by_label || {});
    renderClassification(result.classification);
  }

  function renderLabels(counts) {
    els.labelList.innerHTML = "";
    Object.keys(counts).forEach(function (label) {
      const li = document.createElement("li");
      li.textContent = label + " ×" + counts[label];
      els.labelList.appendChild(li);
    });
  }

  function renderClassification(cls) {
    if (!cls || !cls.length) return;
    ctx.font = "16px Inter, sans-serif";
    ctx.textBaseline = "top";
    cls.slice(0, 5).forEach(function (c, i) {
      const text = c[0] + "  " + (c[1] * 100).toFixed(1) + "%";
      const y = 10 + i * 24;
      ctx.fillStyle = "rgba(0,0,0,0.55)";
      const tw = ctx.measureText(text).width;
      ctx.fillRect(8, y, tw + 12, 22);
      ctx.fillStyle = "#38bdf8";
      ctx.fillText(text, 14, y + 2);
    });
    els.count.textContent = cls.length ? "top-" + Math.min(5, cls.length) : "0";
  }

  function updateFps() {
    const now = performance.now();
    if (lastTs) {
      const dt = now - lastTs;
      const inst = dt > 0 ? 1000 / dt : 0;
      fpsEMA = fpsEMA ? fpsEMA * 0.8 + inst * 0.2 : inst;
      els.fps.textContent = fpsEMA.toFixed(1);
    }
    lastTs = now;
  }

  /* ---------------- image upload (POST /infer) ---------------- */
  function handleFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    const img = new Image();
    img.onload = function () {
      els.canvas.width = img.naturalWidth;
      els.canvas.height = img.naturalHeight;
      ctx.drawImage(img, 0, 0);
      hideHint();
    };
    const url = URL.createObjectURL(file);
    img.src = url;

    const form = new FormData();
    form.append("file", file);
    form.append("task", els.taskSelect.value);
    if (els.backendSelect.value) form.append("backend", els.backendSelect.value);

    fetch("/infer", { method: "POST", body: form })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data.error) { alert("Inference error: " + data.error); return; }
        // Redraw original then overlay detections.
        img.onload = null;
        ctx.drawImage(img, 0, 0);
        drawDetections(data.result);
        URL.revokeObjectURL(url);
      })
      .catch(function (e) { console.error(e); });
  }

  els.dropzone.addEventListener("click", function () { els.fileInput.click(); });
  els.fileInput.addEventListener("change", function (e) {
    if (e.target.files && e.target.files[0]) handleFile(e.target.files[0]);
  });
  ["dragenter", "dragover"].forEach(function (ev) {
    els.dropzone.addEventListener(ev, function (e) {
      e.preventDefault();
      els.dropzone.classList.add("drag");
    });
  });
  ["dragleave", "drop"].forEach(function (ev) {
    els.dropzone.addEventListener(ev, function (e) {
      e.preventDefault();
      els.dropzone.classList.remove("drag");
    });
  });
  els.dropzone.addEventListener("drop", function (e) {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  /* ---------------- webcam streaming (WebSocket) ---------------- */
  function wsUrl() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    return proto + "://" + location.host + "/ws/stream";
  }

  function openSocket() {
    return new Promise(function (resolve, reject) {
      ws = new WebSocket(wsUrl());
      ws.onopen = function () { setConn("online"); resolve(); };
      ws.onclose = function () { setConn("offline"); };
      ws.onerror = function (e) { reject(e); };
      ws.onmessage = function (ev) {
        let msg;
        try { msg = JSON.parse(ev.data); } catch (_) { return; }
        if (msg.type === "error") { console.warn("ws error:", msg.message); }
        else if (msg.type === "result") {
          // Draw the latest webcam frame then overlay returned detections.
          ctx.drawImage(els.video, 0, 0, els.canvas.width, els.canvas.height);
          drawDetections(msg.result);
          updateFps();
        }
        sendBusy = false;
      };
    });
  }

  function captureFrame() {
    // Downscale to a working buffer to keep payloads small.
    const tmp = document.createElement("canvas");
    tmp.width = els.canvas.width;
    tmp.height = els.canvas.height;
    const tctx = tmp.getContext("2d");
    tctx.drawImage(els.video, 0, 0, tmp.width, tmp.height);
    return tmp.toDataURL("image/jpeg", 0.7);
  }

  function streamLoop() {
    if (!streaming) return;
    if (ws && ws.readyState === WebSocket.OPEN && !sendBusy) {
      sendBusy = true;
      ws.send(
        JSON.stringify({
          type: "frame",
          task: els.taskSelect.value,
          backend: els.backendSelect.value || undefined,
          image: captureFrame(),
          frame_index: frameIndex++,
          annotate: false,
        })
      );
    }
    rafId = requestAnimationFrame(streamLoop);
  }

  async function startWebcam() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true });
      els.video.srcObject = stream;
      await els.video.play();
      els.canvas.width = els.video.videoWidth || 640;
      els.canvas.height = els.video.videoHeight || 480;
      hideHint();
      await openSocket();
      streaming = true;
      frameIndex = 0;
      els.btnWebcam.disabled = true;
      els.btnStop.disabled = false;
      streamLoop();
    } catch (e) {
      alert("Could not access webcam: " + e.message);
    }
  }

  function stopWebcam() {
    streaming = false;
    if (rafId) cancelAnimationFrame(rafId);
    if (stream) { stream.getTracks().forEach(function (t) { t.stop(); }); stream = null; }
    if (ws) { ws.close(); ws = null; }
    els.btnWebcam.disabled = false;
    els.btnStop.disabled = true;
    setConn("offline");
  }

  els.btnWebcam.addEventListener("click", startWebcam);
  els.btnStop.addEventListener("click", stopWebcam);

  /* ---------------- init ---------------- */
  fetch("/health")
    .then(function (r) { return r.json(); })
    .then(function (h) {
      if (h.version) els.versionPill.textContent = "v" + h.version;
    })
    .catch(function () {});

  window.addEventListener("beforeunload", stopWebcam);
})();
