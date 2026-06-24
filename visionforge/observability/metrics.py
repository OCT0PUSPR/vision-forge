"""Prometheus metrics.

All metric objects are created against a module-local registry so importing this
module never touches the global default registry twice (which would raise on
reload). When ``prometheus_client`` is absent, no-op stand-ins are used so the
rest of the app keeps working.
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST as _CONTENT_TYPE_LATEST,
    )
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _HAS_PROM = True
except Exception:  # pragma: no cover - exercised only without the dep
    _HAS_PROM = False
    _CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


CONTENT_TYPE_LATEST = _CONTENT_TYPE_LATEST


class _NoopMetric:
    """A metric that silently swallows all calls (no prometheus installed)."""

    def labels(self, *args, **kwargs) -> "_NoopMetric":
        return self

    def observe(self, *_a, **_k) -> None:  # Histogram
        pass

    def inc(self, *_a, **_k) -> None:  # Counter
        pass

    def set(self, *_a, **_k) -> None:  # Gauge
        pass


# Latency buckets tuned for CPU inference (10ms .. 30s).
_LATENCY_BUCKETS = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)


class Metrics:
    """Container holding the application's metric instruments.

    Attributes are annotated ``Any`` because each instrument is either a real
    prometheus collector or a :class:`_NoopMetric` depending on whether the
    optional ``prometheus_client`` dependency is installed.
    """

    registry: Any
    requests_total: Any
    errors_total: Any
    inference_latency_seconds: Any
    request_latency_seconds: Any
    frames_per_second: Any
    inference_total: Any
    active_jobs: Any

    def __init__(self, registry: Optional[Any] = None) -> None:
        if _HAS_PROM:
            self.registry = registry or CollectorRegistry()
            self.requests_total = Counter(
                "vf_requests_total",
                "Total HTTP requests.",
                ["method", "path", "status"],
                registry=self.registry,
            )
            self.errors_total = Counter(
                "vf_errors_total",
                "Total handled errors.",
                ["type"],
                registry=self.registry,
            )
            self.inference_latency_seconds = Histogram(
                "inference_latency_seconds",
                "Model inference latency in seconds.",
                ["task", "backend"],
                buckets=_LATENCY_BUCKETS,
                registry=self.registry,
            )
            self.request_latency_seconds = Histogram(
                "vf_request_latency_seconds",
                "End-to-end HTTP request latency.",
                ["method", "path"],
                buckets=_LATENCY_BUCKETS,
                registry=self.registry,
            )
            self.frames_per_second = Gauge(
                "frames_per_second",
                "Most recent measured frames-per-second.",
                ["task"],
                registry=self.registry,
            )
            self.inference_total = Counter(
                "vf_inference_total",
                "Total inference calls.",
                ["task", "backend", "outcome"],
                registry=self.registry,
            )
            self.active_jobs = Gauge(
                "vf_active_jobs",
                "Number of in-flight background jobs.",
                registry=self.registry,
            )
        else:  # pragma: no cover - fallback path
            self.registry = None
            noop = _NoopMetric()
            self.requests_total = noop
            self.errors_total = noop
            self.inference_latency_seconds = noop
            self.request_latency_seconds = noop
            self.frames_per_second = noop
            self.inference_total = noop
            self.active_jobs = noop

    def render(self) -> bytes:
        """Return the metrics exposition text (bytes) for the /metrics route."""
        if _HAS_PROM and self.registry is not None:
            return generate_latest(self.registry)
        return b"# prometheus_client not installed\n"


_METRICS: Optional[Metrics] = None


def get_metrics() -> Metrics:
    """Return the process-wide metrics singleton."""
    global _METRICS
    if _METRICS is None:
        _METRICS = Metrics()
    return _METRICS
