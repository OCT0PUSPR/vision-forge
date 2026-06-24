# vision-forge 🔥

> Real-time, multi-task computer vision in one coherent platform — detection, segmentation, pose, tracking & classification, exposed through a Python library, a FastAPI service, a thin SDK/CLI, and a clean browser GUI.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![CI](https://img.shields.io/badge/build-CI-success.svg)](.github/workflows/ci.yml)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Backend: Ultralytics YOLO](https://img.shields.io/badge/engine-YOLOv8-00C2FF.svg)](https://github.com/ultralytics/ultralytics)

`vision-forge` ships a **from-scratch, anchor-free CenterNet object detector** (built by hand in
PyTorch — no ultralytics / huggingface / timm / detectron in the model, loss, decode or training
loop) as the **default** detection engine, behind a single normalized result schema, a drawing
layer, a streaming pipeline, and a websocket-powered web UI. The
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) family and an optional
[HuggingFace Transformers](https://huggingface.co/) DETR / ViT backend remain available as opt-in
baselines. Run it on a laptop CPU with the built-in synthetic `--demo`, point it at an image/video,
or stream your webcam straight to the browser.

> **From-scratch ML highlight.** The `visionforge/models/centernet/` package implements a ResNet-like
> backbone, a transposed-conv upsampling neck to stride-4, and center/wh/offset heads; a
> penalty-reduced Gaussian focal loss with Gaussian-splat targets; top-k + NMS decode; a procedural
> shapes dataset *and* a Pascal VOC loader; a from-scratch mAP@0.5; a resumable AdamW + cosine-LR
> training loop (MPS→CUDA→CPU); and an ONNX-export + onnxruntime inference path. A committed proof
> checkpoint (trained locally to **mAP@0.5 ≈ 0.85** on the procedural set) makes the default detector
> work out of the box. See [ARCHITECTURE.md](ARCHITECTURE.md) and the
> [From-scratch detector](#-from-scratch-centernet-detector) section.

---

## ✨ Features

- **From-scratch CenterNet detector** — a hand-written anchor-free detector (backbone + upsample
  neck + center/wh/offset heads, Gaussian-focal loss, top-k+NMS decode, from-scratch mAP@0.5,
  resumable AdamW/cosine training, ONNX export). The **default** detection engine.
- **5 vision tasks** — object detection, instance segmentation, pose/keypoints, multi-object
  tracking (ByteTrack), and image classification, behind one API.
- **Pluggable backends** — default from-scratch `centernet` (+ `centernet-onnx`); optional
  [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) `baseline`; optional HuggingFace
  `transformers` pipeline (`facebook/detr-resnet-50` detector, `google/vit-base-patch16-224`
  classifier); torch-free YOLO `onnx`.
- **Normalized schema** — every backend returns the same `Detection` / `FrameResult` shape
  (xyxy boxes, masks, keypoints, track ids), unit-tested and JSON-serializable.
- **FastAPI service** — `/health`, `/models`, `/infer` (multipart upload), and a live
  `/ws/stream` WebSocket.
- **Browser GUI** — drag-and-drop an image or stream your webcam over the websocket; dark, modern,
  zero-build vanilla JS, renders boxes/masks/keypoints + live FPS on a canvas.
- **No-hardware demo** — `--source demo` generates synthetic frames so you can try the whole
  pipeline on a CPU with no camera and no committed weights.
- **Lazy & cached** — model weights auto-download from Ultralytics/HF on first use and are cached;
  nothing heavy is imported until you actually run inference.
- **Light footprint** — no weights, datasets, or `node_modules` in the repo; a `requirements-min.txt`
  path runs the tests with zero torch.

---

## 🏗️ Architecture

```mermaid
flowchart LR
    subgraph Clients
        CLI["CLI / SDK<br/>python -m visionforge.cli"]
        WEB["Web GUI<br/>(canvas + getUserMedia)"]
    end

    subgraph Service["FastAPI service (visionforge.api.server)"]
        HTTP["POST /infer<br/>GET /models /health"]
        WS["WebSocket /ws/stream"]
        STATIC["GET / -> web GUI"]
    end

    subgraph Core["Core library (visionforge)"]
        PIPE["VisionPipeline<br/>(pipeline.py)"]
        REG["ModelRegistry<br/>(lazy + cached)"]
        SCHEMA["Detection / FrameResult<br/>(schema.py)"]
        DRAW["draw.py<br/>boxes / masks / keypoints"]
        VIDEO["video.py<br/>file | webcam | demo"]
    end

    subgraph Backends
        YOLO["YOLO backend<br/>ultralytics .predict()/.track()"]
        HF["HF backend<br/>transformers pipeline"]
    end

    CLI --> PIPE
    WEB <-->|base64 frames / JSON| WS
    WEB -->|multipart| HTTP
    WEB --> STATIC
    HTTP --> PIPE
    WS --> PIPE
    PIPE --> REG --> YOLO
    REG --> HF
    PIPE --> VIDEO
    PIPE --> DRAW
    YOLO --> SCHEMA
    HF --> SCHEMA
    SCHEMA --> PIPE
```

Text fallback:

```
Clients (CLI / Web GUI)
        |
        v
FastAPI service  --  /infer (HTTP)  |  /ws/stream (WebSocket)  |  / (GUI)
        |
        v
VisionPipeline -> ModelRegistry (lazy, cached)
        |                 |
        v                 v
   draw.py / video.py   YOLO backend  ||  HF backend
        |                 |                 |
        +------>  Normalized FrameResult / Detection schema
```

---

## 🚀 Quickstart

```bash
# 1. clone & enter
git clone https://github.com/OCT0PUSPR/vision-forge && cd vision-forge

# 2a. lightweight path (no torch) — runs tests, lint, the synthetic frame demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-min.txt
pytest -q

# 2b. full path (real inference) — pulls torch + ultralytics + transformers
pip install -r requirements.txt        # or: pip install -e ".[ml,dev]"

# 3. run the no-hardware synthetic demo (weights auto-download on first infer)
python -m visionforge.cli detect --source demo --task detection

# 4. launch the API + web GUI, then open http://localhost:8000
python -m visionforge.cli serve --port 8000
```

### Docker

```bash
docker compose up --build           # serves on http://localhost:8000
# or
docker build -t vision-forge . && docker run -p 8000:8000 vision-forge
```

---

## 🧠 From-scratch CenterNet detector

The default detector is **not** a wrapper — it is a hand-written, anchor-free
[CenterNet](https://arxiv.org/abs/1904.07850) ("Objects as Points") implemented from scratch in
PyTorch under [`visionforge/models/centernet/`](visionforge/models/centernet/). No ultralytics /
huggingface / timm / detectron is used in the model, loss, decode or training loop (only `torch`,
`torchvision.ops` NMS/box utils, and numpy). Full design in [ARCHITECTURE.md](ARCHITECTURE.md).

| Piece | File | What |
| ----- | ---- | ---- |
| Backbone + neck + heads | `model.py` | from-scratch ResNet-like residual backbone, transposed-conv upsample neck to **stride-4**, and `hm` / `wh` / `offset` heads |
| Losses + targets | `losses.py`, `targets.py` | penalty-reduced **Gaussian focal loss** + Gaussian-splat target rendering; masked **L1** on wh/offset |
| Decode | `postprocess.py` | sigmoid → 3×3 max-pool peak NMS → top-k → box reconstruct → per-class NMS |
| Data | `dataset.py` | **procedural shapes** dataset (local, fast) + **Pascal VOC** loader (torchvision auto-download) |
| Metric | `metrics.py` | **mAP@0.5 from scratch** (greedy IoU matching, all-points AP) |
| Train / eval | `engine.py` | DataLoader, AdamW, **cosine LR**, grad-clip, resumable checkpoints, MPS→CUDA→CPU |
| Inference / export | `infer.py`, `export.py` | torch + **onnxruntime** backends → normalized schema; ONNX export |

### Trained proof model (real numbers)

Trained locally on **Apple Silicon (MPS)** on the procedural shapes set — this proves the
architecture, loss and loop are correct:

```
variant=lite (½-width ResNet-18, 3.71M params)  data=3000 train / 500 val  epochs=30  ~13 min on MPS

epoch  0  train_loss 4.74  map@0.5 0.139
epoch  3  train_loss 1.74  map@0.5 0.315
epoch 10  train_loss 0.77  map@0.5 0.816
epoch 12  train_loss 0.59  map@0.5 0.846   <- best
epoch 29  train_loss 0.12  map@0.5 0.820
done: best mAP@0.5 = 0.846   per-class AP@0.5: rectangle 0.83 · circle 0.85 · triangle 0.78
```

The slim (optimizer-free, ~15 MB) checkpoint `weights/centernet_shapes.pt` is committed, so the
default detector and the API work out of the box. ONNX inference (`weights/centernet_shapes.onnx`,
exported on demand) runs in **~12 ms/frame** on CPU via onnxruntime. The complete, unedited training
log is committed at [`docs/centernet_shapes_train_log.jsonl`](docs/centernet_shapes_train_log.jsonl).
Independent re-evaluation of the committed checkpoint on a held-out 500-image val set reproduces
**mAP@0.5 = 0.847** (score-thr 0.2) / **0.871** (score-thr 0.05).

### Train it yourself

```bash
pip install -r requirements-train.txt      # torch + torchvision + onnx(runtime)

# Procedural proof run (reproduces the table above; <~45 min on MPS):
python scripts/train_centernet.py --dataset shapes --variant lite --epochs 30 \
    --batch-size 32 --train-size 3000 --val-size 500 --out runs/centernet_shapes

# Real-data demo on a small Pascal VOC subset (auto-downloads VOC):
python scripts/train_centernet.py --dataset voc --variant r18 --input-size 384 \
    --voc-subset 1500 --epochs 12 --out runs/centernet_voc

# Resume any run:
python scripts/train_centernet.py --dataset shapes --resume runs/centernet_shapes/last.pt

# Export the trained checkpoint to ONNX:
python scripts/export_centernet_onnx.py --checkpoint runs/centernet_shapes/best.pt \
    --out weights/centernet_shapes.onnx
```

Use the from-scratch detector explicitly (or fall back to the YOLO baseline):

```bash
python -m visionforge.cli detect --source img.jpg --backend centernet        # from-scratch (default)
python -m visionforge.cli detect --source img.jpg --backend centernet-onnx   # torch-free onnxruntime
python -m visionforge.cli detect --source img.jpg --backend baseline         # Ultralytics YOLO
```

```python
from visionforge import VisionPipeline
pipe = VisionPipeline(task="detection")              # default backend = centernet
result = pipe.run_image("photo.jpg")
```

### Scaling up (full VOC / COCO on a GPU / Colab)

The procedural set proves correctness; for real-world numbers, scale on a GPU runtime with the
turnkey script (auto-downloads full Pascal VOC, trains, and exports ONNX):

```bash
# In Colab / on any CUDA box, after cloning + `pip install -r requirements-train.txt`:
python scripts/scale_up_voc_colab.py --variant r18 --input-size 512 \
    --batch-size 32 --epochs 70 --out runs/centernet_voc_full
```

Recommended real-data recipe (single mid-range GPU, a few hours): `variant r18`/`r34`,
`input-size 512`, `batch-size 32`, `epochs 70–140`, cosine schedule (built in). For COCO, point a
COCO-format loader at the same `build_targets` / `engine.train` API. Full VOC/COCO training needs a
real GPU and the full dataset download — it is intentionally **not** run in CI.

---

## 🧑‍💻 Usage examples

**CLI**

```bash
# synthetic demo, 30 frames
python -m visionforge.cli detect --source demo --task detection

# a single image, save annotated output
python -m visionforge.cli detect --source photo.jpg --task segmentation --save out.jpg

# pose estimation on a video file
python -m visionforge.cli detect --source clip.mp4 --task pose

# webcam multi-object tracking (ids persist across frames)
python -m visionforge.cli detect --source 0 --task tracking

# alternate HuggingFace DETR detector
python -m visionforge.cli detect --source photo.jpg --task detection --backend hf

# write a synthetic demo image (no model needed)
python -m visionforge.cli demo-image --out demo.jpg
```

**Python library / SDK**

```python
from visionforge import VisionPipeline

pipe = VisionPipeline(task="detection")          # backend="hf" for DETR
result = pipe.run_image("photo.jpg")
print(result.count_by_label())                   # {'person': 2, 'dog': 1}
for det in result.detections:
    print(det.label, round(det.confidence, 2), det.bbox)

# stream a video / webcam / demo
for idx, res, annotated in pipe.run_stream("demo", max_frames=10, annotate=True):
    print(idx, res.to_dict()["counts_by_label"])
```

**HTTP API (curl)**

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/models | python -m json.tool
curl -s -F "file=@photo.jpg" -F "task=detection" -F "annotate=true" \
     http://localhost:8000/infer | python -m json.tool
```

---

## 📚 API reference

| Method | Path          | Body / params                                                                 | Returns |
| ------ | ------------- | ----------------------------------------------------------------------------- | ------- |
| GET    | `/`           | —                                                                             | Web GUI (HTML) |
| GET    | `/health`     | —                                                                             | `{status, version, device}` |
| GET    | `/models`     | —                                                                             | Available tasks, backends, default thresholds |
| POST   | `/infer`      | multipart: `file` (image), `task`, `backend?`, `annotate?` (bool)             | `{result: FrameResult, annotated?: dataURL}` |
| WS     | `/ws/stream`  | JSON frames (see protocol below)                                              | JSON `result` messages |

### WebSocket protocol (`/ws/stream`)

Client → server (text JSON):

```json
{
  "type": "frame",
  "task": "detection",
  "backend": "yolo",
  "image": "data:image/jpeg;base64,<...>",
  "frame_index": 12,
  "annotate": false
}
```

Server → client (text JSON):

```json
{
  "type": "result",
  "frame_index": 12,
  "result": { "task": "detection", "count": 3, "detections": [ ... ] },
  "annotated": "data:image/jpeg;base64,<...>"  // only if annotate=true
}
```

Also supported: `{"type":"ping"}` → `{"type":"pong"}`; any error returns
`{"type":"error","message":"..."}`. The browser GUI sends raw frames and draws the boxes
client-side from `result.detections`, so it stays responsive even without `annotate`.

### `FrameResult` JSON shape

```json
{
  "task": "detection",
  "width": 640, "height": 480,
  "frame_index": 0, "inference_ms": 12.4,
  "model": "yolov8n.pt",
  "count": 2,
  "counts_by_label": {"person": 2},
  "detections": [
    {"label": "person", "confidence": 0.91, "bbox": [x1,y1,x2,y2],
     "class_id": 0, "track_id": 3, "mask": [[x,y],...], "keypoints": [...]}
  ],
  "classification": null
}
```

---

## ⚙️ Configuration

All settings load from environment variables (prefix `VF_`), optionally via a `.env` file. See
[`.env.example`](.env.example). Secrets (`HF_TOKEN`) are read from the environment only — never
hardcoded.

| Variable                  | Default                        | Description |
| ------------------------- | ------------------------------ | ----------- |
| `VF_DEVICE`               | `auto`                         | `auto` picks cuda → mps → cpu (training picks mps → cuda → cpu) |
| `VF_DEFAULT_DETECTOR`     | `centernet`                    | Default detection backend: `centernet` (from-scratch), `baseline` (YOLO), `hf`, `onnx` |
| `VF_CENTERNET_CHECKPOINT` | `weights/centernet_shapes.pt`  | From-scratch CenterNet `.pt` checkpoint |
| `VF_CENTERNET_ONNX_PATH`  | `weights/centernet_shapes.onnx`| From-scratch CenterNet ONNX graph (for `centernet-onnx`) |
| `VF_CONF_THRESHOLD`       | `0.25`                         | Detection confidence threshold |
| `VF_IOU_THRESHOLD`        | `0.45`                         | NMS IoU threshold |
| `VF_IMAGE_SIZE`           | `640`                          | Inference image size |
| `VF_DETECTION_MODEL`      | `yolov8n.pt`                   | YOLO detection weights id |
| `VF_SEGMENTATION_MODEL`   | `yolov8n-seg.pt`               | YOLO segmentation weights id |
| `VF_POSE_MODEL`           | `yolov8n-pose.pt`              | YOLO pose weights id |
| `VF_TRACKING_MODEL`       | `yolov8n.pt`                   | YOLO model used with ByteTrack |
| `VF_CLASSIFICATION_MODEL` | `google/vit-base-patch16-224`  | HF classifier id |
| `VF_HF_DETECTION_MODEL`   | `facebook/detr-resnet-50`      | HF alternate detector id |
| `VF_TRACKER`              | `bytetrack.yaml`               | Ultralytics tracker config |
| `VF_HOST` / `VF_PORT`     | `0.0.0.0` / `8000`             | Server bind address |
| `VF_MAX_UPLOAD_MB`        | `25`                           | Max `/infer` upload size |
| `HF_TOKEN`                | _(unset)_                      | HuggingFace token for gated models |

---

## 🗂️ Project structure

```
vision-forge/
├── visionforge/
│   ├── __init__.py            # public API, lazy VisionPipeline export
│   ├── config.py              # pydantic-settings, device auto-detect
│   ├── pipeline.py            # VisionPipeline orchestration
│   ├── cli.py                 # detect / serve / demo-image commands
│   ├── core/
│   │   ├── schema.py          # Detection / FrameResult / Keypoint (dep-free)
│   │   ├── draw.py            # boxes/masks/keypoints + pure color helpers
│   │   └── video.py           # file | webcam | synthetic demo iterator, FPS
│   ├── models/
│   │   ├── registry.py        # lazy, cached task->backend registry
│   │   ├── centernet/         # FROM-SCRATCH CenterNet detector (default)
│   │   │   ├── model.py       #   backbone + upsample neck + hm/wh/offset heads
│   │   │   ├── losses.py      #   Gaussian focal loss + L1 (torch)
│   │   │   ├── targets.py     #   Gaussian-splat target rendering (torch-free)
│   │   │   ├── postprocess.py #   top-k + NMS decode
│   │   │   ├── dataset.py     #   procedural shapes + Pascal VOC loaders
│   │   │   ├── metrics.py     #   from-scratch mAP@0.5 (torch-free)
│   │   │   ├── engine.py      #   AdamW + cosine LR train/eval, resumable
│   │   │   ├── infer.py       #   torch + onnxruntime backends -> schema
│   │   │   └── export.py      #   ONNX export
│   │   ├── yolo_backend.py    # ultralytics wrapper (baseline) -> schema
│   │   ├── onnx_backend.py    # YOLOv8 ONNX (torch-free baseline)
│   │   └── hf_backend.py      # transformers detector + classifier
│   └── api/
│       ├── server.py          # FastAPI app (HTTP + WebSocket + static)
│       ├── encoding.py        # base64 / data-URL / image (de)serialization
│       └── web/               # index.html + app.js + style.css (zero-build)
├── scripts/                   # train_centernet.py, export_centernet_onnx.py, scale_up_voc_colab.py
├── weights/centernet_shapes.pt# committed slim PROOF checkpoint (~15MB, <25MB cap)
├── tests/                     # pytest — light path needs no torch; ML tests skip without torch
├── examples/                  # run_demo.py, infer_image.py
├── requirements.txt           # full runtime (incl. torch/ultralytics/transformers)
├── requirements-train.txt     # from-scratch ML training/eval/export deps (torch/torchvision/onnx)
├── requirements-min.txt       # lightweight test/CI path (no torch)
├── pyproject.toml             # packaging, ruff, pytest config
├── Dockerfile / docker-compose.yml
├── .env.example
└── .github/workflows/ci.yml   # compileall + ruff + pytest + docker build
```

---

## 📈 Benchmarks

Indicative throughput on CPU (`yolov8n.pt`, 640px). Numbers vary by hardware and are not measured
in CI — run `python -m visionforge.cli detect --source demo` to benchmark locally via the printed
per-frame milliseconds and the GUI's live FPS meter.

| Task          | Model            | Device     | Typical |
| ------------- | ---------------- | ---------- | ------- |
| Detection     | yolov8n          | laptop CPU | ~6–15 FPS |
| Segmentation  | yolov8n-seg      | laptop CPU | ~4–10 FPS |
| Pose          | yolov8n-pose     | laptop CPU | ~5–12 FPS |
| Tracking      | yolov8n+ByteTrack| laptop CPU | ~5–12 FPS |
| Any (YOLOv8n) | —                | CUDA GPU   | 60+ FPS |

---

## 🗺️ Roadmap

- [ ] Batched / async inference workers for higher websocket throughput
- [ ] ONNX / TensorRT export path for edge deployment (`*.onnx`, `*.engine`)
- [ ] Server-side annotated video recording + MP4 export
- [ ] Region-of-interest & line-crossing counting on top of tracking
- [ ] Open-vocabulary detection backend (YOLO-World / Grounding DINO)
- [ ] Prometheus metrics endpoint + structured logging
- [ ] Auth (API keys) and per-client rate limiting on the API
- [ ] Model warmup on startup + readiness gating

---

## 🤝 Contributing

Issues and PRs welcome. Run `ruff check .` and `pytest -q` before submitting. Heavy ML deps are
optional for the test suite (everything heavy is import-guarded).

## 📄 License

Released under the **MIT License** © 2026 **OCT0PUSPR**. See [LICENSE](LICENSE).
