"""Singleton ModelManager: warmup, LRU cache, thread-safe inference.

Wraps the :class:`ModelRegistry` with:
    * an LRU eviction policy over loaded backends (bounded memory),
    * a per-backend circuit breaker around inference,
    * retry+backoff on model loading (downloads),
    * Prometheus latency/outcome metrics,
    * batched multi-image inference.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import List, Optional

from visionforge.config import Settings, get_settings
from visionforge.core.schema import FrameResult
from visionforge.models.registry import ModelRegistry, get_registry
from visionforge.observability.logging import get_logger
from visionforge.observability.metrics import get_metrics
from visionforge.reliability import CircuitBreaker, retry_call

log = get_logger("visionforge.models.manager")


class ModelManager:
    """Process-wide manager for inference backends."""

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        settings: Optional[Settings] = None,
        max_cached: int = 4,
    ) -> None:
        self.registry = registry or get_registry()
        self.settings = settings or get_settings()
        self.max_cached = max_cached
        self._cache: "OrderedDict[str, object]" = OrderedDict()
        self._breakers: dict = {}
        self._lock = threading.RLock()
        self._metrics = get_metrics()
        self._warmed: set = set()

    # ------------------------------------------------------------------ #
    # loading / caching
    # ------------------------------------------------------------------ #
    def _key(self, task: str, backend: Optional[str]) -> str:
        resolved = self.registry.resolve(task, backend)
        return f"{resolved}:{task}"

    def get_backend(self, task: str, backend: Optional[str] = None):
        """Return a loaded backend, loading (with retry) + caching on demand."""
        key = self._key(task, backend)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        # Build outside the lock-held load to avoid blocking other tasks.
        backend_obj = self.registry.get(task, backend)

        def _load():
            if hasattr(backend_obj, "load"):
                backend_obj.load()
            return backend_obj

        retry_call(
            _load,
            attempts=3,
            base_delay=1.0,
            max_delay=10.0,
        )
        with self._lock:
            self._cache[key] = backend_obj
            self._cache.move_to_end(key)
            self._breakers.setdefault(key, CircuitBreaker())
            self._evict_if_needed()
        log.info("model_loaded", key=key)
        return backend_obj

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self.max_cached:
            old_key, _ = self._cache.popitem(last=False)
            self._breakers.pop(old_key, None)
            log.info("model_evicted", key=old_key)

    # ------------------------------------------------------------------ #
    # warmup
    # ------------------------------------------------------------------ #
    def warmup(self, tasks: Optional[List[str]] = None) -> List[str]:
        """Eagerly load (and run one tiny inference for) the given tasks.

        Returns the list of tasks that warmed up successfully. Failures are
        logged but do not raise, so startup degrades gracefully.
        """
        tasks = tasks or ["detection"]
        warmed: List[str] = []
        for task in tasks:
            try:
                backend = self.get_backend(task)
                # One tiny synthetic forward pass to JIT/allocate.
                try:
                    import numpy as np

                    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
                    backend.infer(dummy, frame_index=0)
                except Exception as exc:  # noqa: BLE001
                    log.warning("warmup_infer_skipped", task=task, error=str(exc))
                self._warmed.add(task)
                warmed.append(task)
                log.info("warmup_ok", task=task)
            except Exception as exc:  # noqa: BLE001
                log.warning("warmup_failed", task=task, error=str(exc))
        return warmed

    @property
    def warmed_tasks(self) -> set:
        return set(self._warmed)

    def is_ready(self) -> bool:
        """Readiness: at least one backend is loaded and circuit not open."""
        with self._lock:
            if not self._cache:
                return False
            return any(b.allow() for b in self._breakers.values())

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    def infer(
        self,
        image,
        task: str,
        backend: Optional[str] = None,
        frame_index: int = 0,
    ) -> FrameResult:
        """Run a single inference through the circuit breaker, with metrics."""
        key = self._key(task, backend)
        backend_obj = self.get_backend(task, backend)
        with self._lock:
            breaker = self._breakers.setdefault(key, CircuitBreaker())
        backend_name = key.split(":", 1)[0]

        start = time.perf_counter()
        try:
            result = breaker.call(backend_obj.infer, image, frame_index=frame_index)
        except Exception:
            self._metrics.inference_total.labels(task=task, backend=backend_name, outcome="error").inc()
            raise
        elapsed = time.perf_counter() - start
        self._metrics.inference_latency_seconds.labels(task=task, backend=backend_name).observe(elapsed)
        self._metrics.inference_total.labels(task=task, backend=backend_name, outcome="ok").inc()
        return result

    def infer_batch(
        self,
        images: List,
        task: str,
        backend: Optional[str] = None,
    ) -> List[FrameResult]:
        """Run inference over a list of images (sequential; backend may batch)."""
        return [self.infer(img, task=task, backend=backend, frame_index=i) for i, img in enumerate(images)]

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def shutdown(self) -> None:
        """Release all cached models (graceful shutdown)."""
        with self._lock:
            self._cache.clear()
            self._breakers.clear()
            self._warmed.clear()
        log.info("model_manager_shutdown")


_MANAGER: Optional[ModelManager] = None
_MANAGER_LOCK = threading.Lock()


def get_model_manager() -> ModelManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = ModelManager()
        return _MANAGER


def reset_model_manager() -> None:
    """Test helper to clear the singleton."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is not None:
            _MANAGER.shutdown()
        _MANAGER = None
