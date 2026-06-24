"""HuggingFace Transformers backends (alternate detector + classifier).

Uses ``transformers.pipeline`` for object detection (default DETR
``facebook/detr-resnet-50``) and image classification. All heavy imports are
guarded so the module loads fine without transformers/torch present.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional, Tuple

from visionforge.core.schema import Detection, FrameResult


def _to_pil(image):
    """Coerce a numpy RGB array (or pass-through PIL image) to PIL.Image."""
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("The HF backend requires Pillow.") from exc

    if hasattr(image, "save"):  # already a PIL image
        return image
    import numpy as np

    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = arr.astype("uint8")
    return Image.fromarray(arr)


class HFDetectionBackend:
    """Transformers object-detection pipeline wrapped to our schema."""

    def __init__(
        self,
        model_id: str = "facebook/detr-resnet-50",
        device: str = "cpu",
        conf: float = 0.5,
        hf_token: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.conf = conf
        self.hf_token = hf_token
        self.task = "detection"
        self._pipe: Optional[Any] = None

    def load(self) -> None:
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "The HF backend requires 'transformers' (and torch). Install with: pip install transformers torch"
            ) from exc

        device_index = self._device_index()
        kwargs = {"model": self.model_id, "device": device_index}
        if self.hf_token:
            kwargs["token"] = self.hf_token
        self._pipe = pipeline("object-detection", **kwargs)

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def _device_index(self) -> int:
        # transformers pipelines accept an int device: -1 cpu, >=0 cuda idx.
        if self.device.startswith("cuda"):
            parts = self.device.split(":")
            return int(parts[1]) if len(parts) > 1 else 0
        return -1

    def predict(self, image, frame_index: int = 0) -> FrameResult:
        self.load()
        assert self._pipe is not None
        pil = _to_pil(image)
        width, height = pil.size
        start = time.perf_counter()
        raw = self._pipe(pil, threshold=self.conf)
        elapsed = (time.perf_counter() - start) * 1000.0

        detections: List[Detection] = []
        for item in raw:
            box = item["box"]
            detections.append(
                Detection(
                    label=item["label"],
                    confidence=float(item["score"]),
                    bbox=(
                        float(box["xmin"]),
                        float(box["ymin"]),
                        float(box["xmax"]),
                        float(box["ymax"]),
                    ),
                )
            )
        return FrameResult(
            detections=detections,
            task="detection",
            width=width,
            height=height,
            frame_index=frame_index,
            inference_ms=elapsed,
            model=self.model_id,
        )

    def infer(self, image, frame_index: int = 0) -> FrameResult:
        return self.predict(image, frame_index=frame_index)


class HFClassificationBackend:
    """Transformers image-classification pipeline -> top-k labels."""

    def __init__(
        self,
        model_id: str = "google/vit-base-patch16-224",
        device: str = "cpu",
        top_k: int = 5,
        hf_token: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.top_k = top_k
        self.hf_token = hf_token
        self.task = "classification"
        self._pipe: Optional[Any] = None

    def load(self) -> None:
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("The HF classification backend requires 'transformers' (and torch).") from exc
        device_index = 0 if self.device.startswith("cuda") else -1
        kwargs = {"model": self.model_id, "device": device_index}
        if self.hf_token:
            kwargs["token"] = self.hf_token
        self._pipe = pipeline("image-classification", **kwargs)

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def predict(self, image, frame_index: int = 0) -> FrameResult:
        self.load()
        assert self._pipe is not None
        pil = _to_pil(image)
        width, height = pil.size
        start = time.perf_counter()
        raw = self._pipe(pil, top_k=self.top_k)
        elapsed = (time.perf_counter() - start) * 1000.0
        topk: List[Tuple[str, float]] = [(item["label"], float(item["score"])) for item in raw]
        return FrameResult(
            detections=[],
            task="classification",
            width=width,
            height=height,
            frame_index=frame_index,
            inference_ms=elapsed,
            model=self.model_id,
            classification=topk,
        )

    def infer(self, image, frame_index: int = 0) -> FrameResult:
        return self.predict(image, frame_index=frame_index)
