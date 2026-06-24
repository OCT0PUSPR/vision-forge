# vision-forge — Architecture

This document describes how vision-forge is put together: the core library, the
**from-scratch CenterNet detector** (the default detection engine), the optional
baseline backends, the FastAPI service, and the production-hardening layer.

---

## 1. High-level shape

```
Clients (CLI / SDK / Web GUI)
        │
        ▼
FastAPI service  ──  /infer (HTTP)  │  /ws/stream (WebSocket)  │  / (GUI)  │  /metrics
        │            auth · rate-limit · validation · security headers · structlog
        ▼
VisionPipeline ─▶ ModelManager (LRU cache · circuit breaker · retry · metrics)
        │                 │
        │                 ▼
        │           ModelRegistry (lazy, cached, task→backend)
        │                 │
        │     ┌───────────┼─────────────┬───────────┬──────────────┐
        │     ▼           ▼             ▼           ▼              ▼
        │  centernet  centernet-onnx  baseline    hf            onnx
        │  (FROM       (onnxruntime)   (YOLO)     (DETR/ViT)    (YOLO ONNX)
        │   SCRATCH)
        ▼     └───────────┴─────────────┴───────────┴──────────────┘
   draw.py / video.py        │
        │                    ▼
        └──────▶  Normalized FrameResult / Detection schema ──▶ JSON / DB / GUI
```

Every backend — including the from-scratch CenterNet — returns the same
`FrameResult` / `Detection` schema (`visionforge/core/schema.py`), so the
pipeline, drawing layer, API and GUI are backend-agnostic.

---

## 2. The from-scratch CenterNet detector (default)

Package: `visionforge/models/centernet/`. Implemented by hand in PyTorch — **no
ultralytics / huggingface / timm / detectron** in the model, loss, decode or
training loop (only `torch`, `torchvision.ops` NMS/box utils, and numpy).

It follows the anchor-free "Objects as Points" (CenterNet) formulation: detect
objects as **center-point peaks** in a per-class heatmap, then regress box
size and a sub-pixel center offset at each peak. No anchors, no region
proposals, NMS only as a light post-filter.

### 2.1 Network (`model.py`)

| Stage | What | Output stride |
| ----- | ---- | ------------- |
| **Backbone** | ResNet-like stack of `BasicBlock` residual units (conv→BN→ReLU ×2 + identity/projection skip), written from scratch. `width` scales channels; `lite`=½-width ResNet-18, `r18`=full ResNet-18, `r34`=`[3,4,6,3]`. | 4 → 8 → 16 → 32 |
| **Neck** | 3× transposed-conv (`ConvTranspose2d`, ×2 each) upsampling tower lifting stride-32 features back to **stride-4**. | 32 → 16 → 8 → 4 |
| **Heads** | three `3×3 conv → ReLU → 1×1 conv` heads on the stride-4 map: `hm` (per-class center logits), `wh` (box w/h), `offset` (sub-pixel center). | 4 |

The heatmap head's final bias is initialised to `-log((1-π)/π)` with π=0.01 so
the initial sigmoid output is ~0.01 — this prevents the focal loss from being
swamped by background early in training.

### 2.2 Losses & target rendering (`losses.py`, `targets.py`)

* **Penalty-reduced Gaussian focal loss** on the heatmap (`neg_loss`): positives
  use `(1-p)^α log p`; negatives are down-weighted by `(1-y)^β` so pixels near a
  true center are penalised less. α=2, β=4.
* **Gaussian-splat targets** (`draw_gaussian` / `gaussian_radius`, torch-free in
  `targets.py`): each GT center is rendered as a 2D Gaussian whose radius is the
  CenterNet IoU-overlap heuristic, blended with `np.maximum`.
* **L1 regression** (`reg_l1_loss`) on `wh` and `offset`, gathered only at GT
  center indices and masked over real objects.
* Combined: `L = L_hm + 0.1·L_wh + L_off`.

### 2.3 Decode (`postprocess.py`)

`sigmoid(hm)` → 3×3 max-pool peak "NMS" → top-k peaks → reconstruct boxes from
`wh` + `offset` (×stride-4) → score threshold → per-class `torchvision.ops`
NMS → `[x1,y1,x2,y2,score,class]`.

### 2.4 Data (`dataset.py`)

* **`ShapesDetectionDataset`** — fully procedural, local, fast: random
  backgrounds with rectangles / circles / triangles, rendered with a hand-written
  numpy scanline polygon fill and circle rasteriser; tight GT boxes derived
  analytically. Deterministic per `(seed, index)` → reproducible & resumable.
  Augmentation: horizontal flip + brightness jitter. This is the **primary
  training target** and proves the architecture/loss/loop are correct.
* **`VOCDetectionDataset`** — thin wrapper over `torchvision.datasets.VOCDetection`
  (auto-downloads Pascal VOC) mapping the 20 VOC classes into the same dense
  target format for a real-data demo.

### 2.5 Metric (`metrics.py`)

**mAP@0.5 implemented from scratch** (numpy only): greedy score-sorted TP/FP
matching by IoU, per-class precision/recall accumulation, all-points
(continuous) AP integration, averaged over classes. Torch-free, so it is unit
tested on the light CI path.

### 2.6 Training engine (`engine.py`)

DataLoader + AdamW + **linear-warmup → cosine-annealing** LR, gradient clipping,
per-epoch mAP eval, best-checkpoint saving, and fully **resumable** checkpoints
(model + optimizer + epoch). Device auto-select order: **MPS → CUDA → CPU**.
A slim (optimizer-free) checkpoint is also written for committable deployment.

### 2.7 Inference & export (`infer.py`, `export.py`)

* **`CenterNetBackend`** — PyTorch inference from a trained `.pt` checkpoint →
  normalized `FrameResult`. This is the vision-forge **default detector**.
* **`CenterNetOnnxBackend`** — onnxruntime inference (torch-free) with the decode
  (sigmoid, max-pool peak NMS, top-k, per-class NMS) reimplemented in numpy.
* **`export.py`** — `torch.onnx.export` to a 3-output (`hm`/`wh`/`offset`) graph.

---

## 3. Trained proof model & metrics

Trained locally on **Apple Silicon MPS** on the procedural shapes dataset:

| Run | Variant | Params | Data | Epochs | Wall time (MPS) | **mAP@0.5** |
| --- | ------- | ------ | ---- | ------ | --------------- | ----------- |
| Proof (committed) | `lite` (½-width R18) | 3.71 M | 3 000 train / 500 val | 30 | ~13 min | **0.846** (best) |

Per-class AP@0.5 at the final epoch: rectangle 0.83, circle 0.85, triangle 0.78.
Loss fell 7.02 → 0.115. The slim checkpoint (`weights/centernet_shapes.pt`,
~15 MB) is committed so the default detector works out of the box; it also ships
as ONNX via `scripts/export_centernet_onnx.py`.

Scale to real data with `scripts/scale_up_voc_colab.py` (full Pascal VOC / COCO
on a GPU runtime) — see the README "Scaling up" section.

---

## 4. Backends & registry

`ModelRegistry` (`models/registry.py`) maps `(task, backend)` → a lazily-built,
cached backend. Detection defaults to `centernet`; `VF_DEFAULT_DETECTOR` can
switch it (e.g. to `baseline`).

| Backend | Engine | Tasks | Notes |
| ------- | ------ | ----- | ----- |
| `centernet` | **from-scratch PyTorch** | detection | **default** |
| `centernet-onnx` | from-scratch via onnxruntime | detection | torch-free deploy |
| `baseline` (alias `yolo`) | Ultralytics YOLOv8 | detection/seg/pose/tracking | optional baseline |
| `hf` | HF Transformers (DETR / ViT) | detection / classification | optional |
| `onnx` | YOLOv8 ONNX | detection | torch-free baseline |

`ModelManager` (`models/manager.py`) wraps the registry with an LRU cache, a
per-backend circuit breaker, retry-with-backoff on load, and Prometheus metrics.

---

## 5. Production-hardening layer

* **API** (`api/`) — FastAPI app: `/infer`, `/ws/stream`, `/health`, `/ready`,
  `/models`, `/metrics`, static GUI. Middleware: API-key auth, per-client rate
  limiting, request-size & image validation, security headers, structlog request
  logging, request IDs.
* **Persistence** (`db/`) — SQLAlchemy 2.0 models + repository + session; Alembic
  migrations.
* **Workers** (`worker/`) — in-process job manager (+ optional arq/redis).
* **Observability** (`observability/`) — structlog JSON logging + Prometheus
  metrics.
* **Reliability** (`reliability.py`) — circuit breaker + retry helpers.

---

## 6. The torch-free boundary (why CI stays light)

The light test/CI path installs **no torch**. This is preserved by:

* keeping the schema, drawing, encoding, config, registry routing, and the
  CenterNet **numpy** pieces (`targets.py`, `metrics.py`) torch-free;
* importing torch **lazily** inside backend `load()`/`infer()` methods and at the
  top of the torch-only CenterNet modules (`model`/`losses`/`postprocess`/
  `engine`/`infer`/`export`), which the light path never imports;
* gating the torch CenterNet tests with `pytest.importorskip("torch")` so they
  skip cleanly without torch and run fully on the `ml-tests` CI job;
* omitting the torch-only modules from the **light-path coverage gate** (they are
  covered by the torch `ml-tests` job instead).

Training / ML deps live in `requirements-train.txt`; the green gate uses
`requirements-min.txt`.
