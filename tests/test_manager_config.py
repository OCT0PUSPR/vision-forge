"""Tests for ModelManager caching/LRU/circuit-breaker and config validation."""

import pytest

from visionforge.config import Settings, detect_device
from visionforge.models.manager import ModelManager
from visionforge.models.registry import ModelRegistry


class _StubBackend:
    """A fake backend that records calls and can be made to fail."""

    def __init__(self, name="stub", fail=False):
        self.name = name
        self.fail = fail
        self.loaded = False
        self.calls = 0

    def load(self):
        self.loaded = True

    def infer(self, image, frame_index=0):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        from visionforge.core.schema import FrameResult

        return FrameResult(task="detection", frame_index=frame_index)


class _StubRegistry(ModelRegistry):
    def __init__(self, backend):
        # Bypass parent __init__ heavy bits.
        self._backend = backend

    def resolve(self, task, backend=None):
        return backend or "yolo"

    def get(self, task, backend=None):
        return self._backend


def test_manager_caches_backend():
    backend = _StubBackend()
    mgr = ModelManager(registry=_StubRegistry(backend))
    b1 = mgr.get_backend("detection")
    b2 = mgr.get_backend("detection")
    assert b1 is b2
    assert backend.loaded is True


def test_manager_infer_records_and_returns():
    backend = _StubBackend()
    mgr = ModelManager(registry=_StubRegistry(backend))
    result = mgr.infer(object(), task="detection")
    assert result.task == "detection"
    assert backend.calls == 1


def test_manager_circuit_breaker_opens_on_failures():
    from visionforge.reliability import CircuitBreakerError

    backend = _StubBackend(fail=True)
    mgr = ModelManager(registry=_StubRegistry(backend))
    # default breaker threshold is 5 consecutive failures
    for _ in range(5):
        with pytest.raises(RuntimeError):
            mgr.infer(object(), task="detection")
    # next call short-circuits via the open breaker
    with pytest.raises(CircuitBreakerError):
        mgr.infer(object(), task="detection")


def test_manager_is_ready():
    backend = _StubBackend()
    mgr = ModelManager(registry=_StubRegistry(backend))
    assert mgr.is_ready() is False  # nothing loaded yet
    mgr.get_backend("detection")
    assert mgr.is_ready() is True


def test_manager_lru_eviction():
    backend = _StubBackend()

    class MultiRegistry(_StubRegistry):
        def resolve(self, task, backend=None):
            return "yolo"

        def get(self, task, backend=None):
            return _StubBackend(name=task)

    mgr = ModelManager(registry=MultiRegistry(backend), max_cached=2)
    # Distinct keys via task name in the key
    mgr.get_backend("detection")
    mgr.get_backend("segmentation")
    mgr.get_backend("pose")
    # cache bounded to 2
    assert len(mgr._cache) <= 2


def test_detect_device_explicit():
    assert detect_device("cpu") == "cpu"
    assert detect_device("cuda") == "cuda"


def test_settings_validate_startup_rejects_bad_threshold():
    s = Settings()
    s.conf_threshold = 2.0
    with pytest.raises(ValueError):
        s.validate_startup()


def test_settings_production_requires_auth_and_cors():
    s = Settings()
    s.env = "production"
    s.require_auth = True
    s.cors_origins = "*"
    with pytest.raises(ValueError):
        s.validate_startup()
    s.cors_origins = "https://example.com"
    s.require_auth = False
    with pytest.raises(ValueError):
        s.validate_startup()
