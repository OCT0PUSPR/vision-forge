"""Model registry: maps ``(task, backend)`` to a lazily-loaded backend object.

Backends are cached by a composite key so repeated requests for the same task
reuse the already-loaded weights. Nothing is loaded until ``get`` is called.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from visionforge.config import Settings, get_settings

# Tasks routed to each backend family.
YOLO_TASKS = ("detection", "segmentation", "pose", "tracking")
HF_TASKS = ("detection", "classification")
ONNX_TASKS = ("detection",)
CENTERNET_TASKS = ("detection",)

VALID_TASKS = ("detection", "segmentation", "pose", "tracking", "classification")
# 'centernet' is the from-scratch detector (DEFAULT for detection).
# 'baseline' is an explicit alias for the Ultralytics YOLO detector.
# 'centernet-onnx' runs the from-scratch model through onnxruntime (torch-free).
VALID_BACKENDS = ("centernet", "centernet-onnx", "yolo", "baseline", "hf", "onnx")


class ModelRegistry:
    """Thread-safe, lazy-loading registry of inference backends."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #
    # Map user-facing backend aliases to the canonical builder name.
    _ALIASES = {"baseline": "yolo"}

    @staticmethod
    def available() -> List[Dict[str, Any]]:
        """List the task/backend combinations the registry knows about."""
        entries: List[Dict[str, Any]] = []
        # From-scratch CenterNet is the default detector.
        entries.append({"task": "detection", "backend": "centernet", "default": True})
        entries.append({"task": "detection", "backend": "centernet-onnx"})
        for task in YOLO_TASKS:
            entries.append({"task": task, "backend": "yolo"})
        entries.append({"task": "detection", "backend": "baseline"})  # alias of yolo
        entries.append({"task": "detection", "backend": "hf"})
        entries.append({"task": "classification", "backend": "hf"})
        entries.append({"task": "detection", "backend": "onnx"})
        return entries

    def default_backend(self, task: str) -> str:
        """Pick a sensible default backend for ``task``.

        Detection defaults to the from-scratch ``centernet`` unless overridden by
        ``VF_DEFAULT_DETECTOR`` (e.g. ``baseline`` to use the YOLO path).
        """
        if task == "classification":
            return "hf"
        if task == "detection":
            return getattr(self.settings, "default_detector", "centernet")
        return "yolo"

    def resolve(self, task: str, backend: Optional[str] = None) -> str:
        if task not in VALID_TASKS:
            raise ValueError(f"Unknown task {task!r}. Valid: {', '.join(VALID_TASKS)}")
        backend = backend or self.default_backend(task)
        backend = self._ALIASES.get(backend, backend)
        if backend not in VALID_BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}. Valid: {', '.join(VALID_BACKENDS)}")
        if backend == "yolo" and task not in YOLO_TASKS:
            raise ValueError(f"YOLO backend does not support task {task!r}")
        if backend == "hf" and task not in HF_TASKS:
            raise ValueError(f"HF backend does not support task {task!r}")
        if backend == "onnx" and task not in ONNX_TASKS:
            raise ValueError(f"ONNX backend does not support task {task!r}")
        if backend in ("centernet", "centernet-onnx") and task not in CENTERNET_TASKS:
            raise ValueError(f"CenterNet backend does not support task {task!r}")
        return backend

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #
    def get(self, task: str, backend: Optional[str] = None) -> Any:
        """Return a (cached) backend instance for ``task``/``backend``."""
        backend = self.resolve(task, backend)
        key = f"{backend}:{task}"
        with self._lock:
            if key not in self._cache:
                self._cache[key] = self._build(task, backend)
            return self._cache[key]

    def _build(self, task: str, backend: str) -> Any:
        s = self.settings
        device = s.resolved_device
        if backend == "centernet":
            from visionforge.models.centernet.infer import CenterNetBackend

            return CenterNetBackend(
                checkpoint=s.centernet_checkpoint,
                device=device,
                conf=s.centernet_conf,
                iou=s.centernet_iou,
                image_size=s.centernet_image_size,
                topk=s.centernet_topk,
            )
        if backend == "centernet-onnx":
            from visionforge.models.centernet.infer import CenterNetOnnxBackend

            return CenterNetOnnxBackend(
                onnx_path=s.centernet_onnx_path,
                device=device,
                conf=s.centernet_conf,
                iou=s.centernet_iou,
                image_size=s.centernet_image_size,
                topk=s.centernet_topk,
            )
        if backend == "yolo":
            from visionforge.models.yolo_backend import YoloBackend

            return YoloBackend(
                model_id=s.model_for(task),
                task=task,
                device=device,
                conf=s.conf_threshold,
                iou=s.iou_threshold,
                image_size=s.image_size,
                tracker=s.tracker,
            )
        if backend == "onnx":
            from visionforge.models.onnx_backend import OnnxDetectionBackend

            return OnnxDetectionBackend(
                onnx_path=s.onnx_model_path,
                device=device,
                conf=s.conf_threshold,
                iou=s.iou_threshold,
                image_size=s.image_size,
            )
        # backend == "hf"
        if task == "classification":
            from visionforge.models.hf_backend import HFClassificationBackend

            return HFClassificationBackend(
                model_id=s.classification_model,
                device=device,
                hf_token=s.hf_token,
            )
        from visionforge.models.hf_backend import HFDetectionBackend

        return HFDetectionBackend(
            model_id=s.hf_detection_model,
            device=device,
            conf=s.conf_threshold,
            hf_token=s.hf_token,
        )

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


_GLOBAL_REGISTRY: Optional[ModelRegistry] = None
_GLOBAL_LOCK = threading.Lock()


def get_registry() -> ModelRegistry:
    """Return a process-wide shared registry."""
    global _GLOBAL_REGISTRY
    with _GLOBAL_LOCK:
        if _GLOBAL_REGISTRY is None:
            _GLOBAL_REGISTRY = ModelRegistry()
        return _GLOBAL_REGISTRY
