"""Inference backends for the from-scratch CenterNet detector.

Two interchangeable backends, both producing the normalized ``FrameResult`` /
``Detection`` schema so they drop straight into the existing pipeline / API:

* :class:`CenterNetBackend` — PyTorch inference from a trained ``.pt``
  checkpoint. This is the **default** detector for vision-forge.
* :class:`CenterNetOnnxBackend` — onnxruntime inference from an exported
  ``.onnx`` graph (torch-free at inference time). The numpy pre/post-processing
  (letterbox-free square resize, sigmoid, top-k, NMS) is implemented here.

Heavy imports (torch / onnxruntime) happen lazily inside ``load`` so importing
this module is cheap and torch-free until a backend is actually used.
"""

from __future__ import annotations

import os
import time
from typing import Any, List, Optional, Sequence

import numpy as np

from visionforge.core.schema import Detection, FrameResult

# Default class names for the shipped proof checkpoint (procedural shapes).
SHAPE_CLASSES = ["rectangle", "circle", "triangle"]
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resize_rgb(image: np.ndarray, size: int):
    """Resize an RGB uint8 array to ``(size, size)``; return (img, sx, sy)."""
    h, w = image.shape[:2]
    try:
        import cv2

        resized = np.asarray(cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR))
    except Exception:  # pragma: no cover - PIL fallback
        from PIL import Image

        resized = np.asarray(Image.fromarray(image).resize((size, size)))
    return resized, size / w, size / h


def _preprocess(image: np.ndarray, size: int):
    """RGB uint8 -> normalized NCHW float32 tensor + scale factors."""
    resized, sx, sy = _resize_rgb(image, size)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    arr = arr.transpose(2, 0, 1)[None]  # NCHW
    return np.ascontiguousarray(arr, dtype=np.float32), sx, sy


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Pure-numpy NMS returning kept indices (used by the ONNX path)."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]
    return keep


def _decode_numpy(
    hm: np.ndarray,
    wh: np.ndarray,
    offset: np.ndarray,
    k: int,
    down_ratio: int,
) -> np.ndarray:
    """Decode raw head arrays (1,C,H,W) -> (M,6) [x1,y1,x2,y2,score,class].

    Coordinates are in stride-4 output units scaled back by ``down_ratio``.
    Implements the 3x3 max-pool peak NMS and top-k in numpy (ONNX path).
    """
    hm = _sigmoid(hm[0])  # (C, H, W)
    c, h, w = hm.shape
    # 3x3 max-pool peak suppression.
    pooled = np.zeros_like(hm)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = np.roll(np.roll(hm, dy, axis=1), dx, axis=2)
            pooled = np.maximum(pooled, shifted)
    keep_mask = hm == pooled
    hm = hm * keep_mask

    flat = hm.reshape(-1)
    if flat.size == 0:
        return np.zeros((0, 6), dtype=np.float32)
    topk = min(k, flat.size)
    inds = np.argpartition(-flat, topk - 1)[:topk]
    inds = inds[np.argsort(-flat[inds])]
    scores = flat[inds]
    classes = inds // (h * w)
    pix = inds % (h * w)
    ys = (pix // w).astype(np.float32)
    xs = (pix % w).astype(np.float32)

    off = offset[0]  # (2, H, W)
    wh_a = wh[0]
    xs = xs + off[0].reshape(-1)[pix]
    ys = ys + off[1].reshape(-1)[pix]
    ws = wh_a[0].reshape(-1)[pix]
    hs = wh_a[1].reshape(-1)[pix]

    x1 = (xs - ws / 2) * down_ratio
    y1 = (ys - hs / 2) * down_ratio
    x2 = (xs + ws / 2) * down_ratio
    y2 = (ys + hs / 2) * down_ratio
    return np.stack([x1, y1, x2, y2, scores, classes.astype(np.float32)], axis=1)


def _to_frame_result(
    dets: np.ndarray,
    class_names: Sequence[str],
    orig_w: int,
    orig_h: int,
    sx: float,
    sy: float,
    frame_index: int,
    elapsed_ms: float,
    model_name: str,
    score_threshold: float,
    task: str = "detection",
) -> FrameResult:
    """Convert (M,6) decoded detections (in input-size px) into a FrameResult."""
    out: List[Detection] = []
    for row in dets:
        x1, y1, x2, y2, score, cls = row
        if score < score_threshold:
            continue
        # Undo the square-resize scaling back to original pixels.
        bx1 = float(np.clip(x1 / sx, 0, orig_w))
        by1 = float(np.clip(y1 / sy, 0, orig_h))
        bx2 = float(np.clip(x2 / sx, 0, orig_w))
        by2 = float(np.clip(y2 / sy, 0, orig_h))
        if bx2 <= bx1 or by2 <= by1:
            continue
        cid = int(cls)
        label = class_names[cid] if cid < len(class_names) else str(cid)
        out.append(
            Detection(label=label, confidence=float(score), bbox=(bx1, by1, bx2, by2), class_id=cid)
        )
    return FrameResult(
        detections=out,
        task=task,
        width=orig_w,
        height=orig_h,
        frame_index=frame_index,
        inference_ms=elapsed_ms,
        model=model_name,
    )


class CenterNetBackend:
    """PyTorch CenterNet detector -> normalized schema (the DEFAULT detector)."""

    SUPPORTED_TASKS = ("detection",)

    def __init__(
        self,
        checkpoint: str,
        device: str = "cpu",
        conf: float = 0.3,
        iou: float = 0.5,
        image_size: int = 256,
        topk: int = 100,
        class_names: Optional[Sequence[str]] = None,
        variant: str = "lite",
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.conf = conf
        self.iou = iou
        self.image_size = image_size
        self.topk = topk
        self.class_names = list(class_names) if class_names else None
        self.variant = variant
        self.task = "detection"
        self._model: Any = None
        self._torch_device: Any = None
        self._down_ratio = 4

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "The CenterNet backend requires torch. Install training deps: pip install -r requirements-train.txt"
            ) from exc
        from visionforge.models.centernet.engine import select_device
        from visionforge.models.centernet.model import build_centernet

        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(
                f"CenterNet checkpoint not found: {self.checkpoint}. "
                "Train one with scripts/train_centernet.py (see README)."
            )
        ckpt = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        num_classes = int(ckpt.get("num_classes", len(self.class_names or SHAPE_CLASSES)))
        cfg = ckpt.get("config", {})
        variant = cfg.get("variant", self.variant)
        self.image_size = cfg.get("input_size", self.image_size)
        if self.class_names is None:
            self.class_names = SHAPE_CLASSES if num_classes == len(SHAPE_CLASSES) else [str(i) for i in range(num_classes)]
        self._torch_device = select_device(self.device)
        model = build_centernet(num_classes, variant=variant)
        model.load_state_dict(ckpt["model"])
        model.eval().to(self._torch_device)
        self._down_ratio = model.down_ratio
        self._model = model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def infer(self, image: np.ndarray, frame_index: int = 0) -> FrameResult:
        import torch

        from visionforge.models.centernet.postprocess import decode_detections

        self.load()
        orig_h, orig_w = image.shape[:2]
        tensor_np, sx, sy = _preprocess(image, self.image_size)
        start = time.perf_counter()
        with torch.no_grad():
            t = torch.from_numpy(tensor_np).to(self._torch_device)
            outputs = self._model(t)
            dets = decode_detections(
                outputs, k=self.topk, score_threshold=self.conf,
                nms_iou=self.iou, down_ratio=self._down_ratio,
            )[0]
        elapsed = (time.perf_counter() - start) * 1000.0
        return _to_frame_result(
            dets, self.class_names or SHAPE_CLASSES, orig_w, orig_h, sx, sy,
            frame_index, elapsed, os.path.basename(self.checkpoint), self.conf,
        )

    def predict(self, image: np.ndarray, frame_index: int = 0) -> FrameResult:
        return self.infer(image, frame_index=frame_index)


class CenterNetOnnxBackend:
    """CenterNet detector via onnxruntime (torch-free inference path)."""

    SUPPORTED_TASKS = ("detection",)

    def __init__(
        self,
        onnx_path: str,
        device: str = "cpu",
        conf: float = 0.3,
        iou: float = 0.5,
        image_size: int = 256,
        topk: int = 100,
        class_names: Optional[Sequence[str]] = None,
        down_ratio: int = 4,
    ) -> None:
        self.onnx_path = onnx_path
        self.device = device
        self.conf = conf
        self.iou = iou
        self.image_size = image_size
        self.topk = topk
        self.class_names = list(class_names) if class_names else SHAPE_CLASSES
        self.down_ratio = down_ratio
        self.task = "detection"
        self._session: Any = None
        self._input_name: Optional[str] = None

    def load(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("The ONNX backend requires onnxruntime. Install with: pip install onnxruntime") from exc
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(
                f"ONNX model not found: {self.onnx_path}. Export with scripts/export_centernet_onnx.py."
            )
        providers = ["CPUExecutionProvider"]
        if self.device.startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(self.onnx_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def infer(self, image: np.ndarray, frame_index: int = 0) -> FrameResult:
        self.load()
        assert self._session is not None
        orig_h, orig_w = image.shape[:2]
        tensor_np, sx, sy = _preprocess(image, self.image_size)
        start = time.perf_counter()
        hm, wh, offset = self._session.run(None, {self._input_name: tensor_np})
        elapsed = (time.perf_counter() - start) * 1000.0
        dets = _decode_numpy(hm, wh, offset, self.topk, self.down_ratio)
        # Per-class NMS in numpy.
        kept_rows: List[np.ndarray] = []
        if len(dets):
            for c in np.unique(dets[:, 5]):
                cls_rows = dets[dets[:, 5] == c]
                keep = _nms_numpy(cls_rows[:, :4], cls_rows[:, 4], self.iou)
                kept_rows.append(cls_rows[keep])
        dets = np.concatenate(kept_rows, axis=0) if kept_rows else np.zeros((0, 6), dtype=np.float32)
        return _to_frame_result(
            dets, self.class_names, orig_w, orig_h, sx, sy,
            frame_index, elapsed, os.path.basename(self.onnx_path), self.conf,
        )

    def predict(self, image: np.ndarray, frame_index: int = 0) -> FrameResult:
        return self.infer(image, frame_index=frame_index)
