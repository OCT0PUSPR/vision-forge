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

VALID_TASKS = ("detection", "segmentation", "pose", "tracking", "classification")
VALID_BACKENDS = ("yolo", "hf")


class ModelRegistry:
    """Thread-safe, lazy-loading registry of inference backends."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #
    @staticmethod
    def available() -> List[Dict[str, Any]]:
        """List the task/backend combinations the registry knows about."""
        entries: List[Dict[str, Any]] = []
        for task in YOLO_TASKS:
            entries.append({"task": task, "backend": "yolo"})
        entries.append({"task": "detection", "backend": "hf"})
        entries.append({"task": "classification", "backend": "hf"})
        return entries

    def default_backend(self, task: str) -> str:
        """Pick a sensible default backend for ``task``."""
        if task == "classification":
            return "hf"
        return "yolo"

    def resolve(self, task: str, backend: Optional[str] = None) -> str:
        if task not in VALID_TASKS:
            raise ValueError(
                f"Unknown task {task!r}. Valid: {', '.join(VALID_TASKS)}"
            )
        backend = backend or self.default_backend(task)
        if backend not in VALID_BACKENDS:
            raise ValueError(
                f"Unknown backend {backend!r}. Valid: {', '.join(VALID_BACKENDS)}"
            )
        if backend == "yolo" and task not in YOLO_TASKS:
            raise ValueError(f"YOLO backend does not support task {task!r}")
        if backend == "hf" and task not in HF_TASKS:
            raise ValueError(f"HF backend does not support task {task!r}")
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
