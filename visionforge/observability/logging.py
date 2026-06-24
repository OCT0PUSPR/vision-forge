"""Structured JSON logging built on structlog (with a stdlib fallback).

Every log line carries a ``request_id`` when emitted inside a request scope (set
by the API middleware via :func:`bind_request_id`). If structlog is not
installed the module degrades to the standard library ``logging`` so imports
never fail in the lightweight environment.
"""

from __future__ import annotations

import contextvars
import logging
import sys
from typing import Any, MutableMapping, Optional

# Holds the current request id for the active async/sync context.
_request_id_ctx: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar("request_id", default=None)

try:
    import structlog

    _HAS_STRUCTLOG = True
except Exception:  # pragma: no cover - exercised only without the dep
    structlog = None  # type: ignore[assignment]
    _HAS_STRUCTLOG = False

_configured = False


def bind_request_id(request_id: Optional[str]) -> "contextvars.Token":
    """Bind ``request_id`` to the current context; returns a reset token."""
    return _request_id_ctx.set(request_id)


def reset_request_id(token: "contextvars.Token") -> None:
    _request_id_ctx.reset(token)


def get_request_id() -> Optional[str]:
    return _request_id_ctx.get()


def _add_request_id(_logger: Any, _method: str, event_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """structlog processor: inject the current request id into every event."""
    rid = _request_id_ctx.get()
    if rid is not None:
        event_dict["request_id"] = rid
    return event_dict


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    """Configure process-wide structured logging. Idempotent."""
    global _configured
    if _configured:
        return
    log_level = getattr(logging, level.upper(), logging.INFO)

    if _HAS_STRUCTLOG:
        renderer = structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                _add_request_id,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                renderer,
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
            cache_logger_on_first_use=True,
        )
    else:  # pragma: no cover - fallback path
        logging.basicConfig(
            level=log_level,
            stream=sys.stdout,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    _configured = True


def get_logger(name: str = "visionforge") -> Any:
    """Return a bound logger (structlog if available, else stdlib)."""
    if not _configured:
        configure_logging()
    if _HAS_STRUCTLOG:
        return structlog.get_logger(name)
    return _StdlibAdapter(logging.getLogger(name))


class _StdlibAdapter:  # pragma: no cover - only used without structlog
    """Tiny shim so call sites can use kwargs like structlog regardless."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _fmt(self, event: str, **kw: Any) -> str:
        rid = _request_id_ctx.get()
        if rid:
            kw["request_id"] = rid
        extras = " ".join(f"{k}={v}" for k, v in kw.items())
        return f"{event} {extras}".strip()

    def info(self, event: str, **kw: Any) -> None:
        self._logger.info(self._fmt(event, **kw))

    def warning(self, event: str, **kw: Any) -> None:
        self._logger.warning(self._fmt(event, **kw))

    def error(self, event: str, **kw: Any) -> None:
        self._logger.error(self._fmt(event, **kw))

    def debug(self, event: str, **kw: Any) -> None:
        self._logger.debug(self._fmt(event, **kw))

    def exception(self, event: str, **kw: Any) -> None:
        self._logger.exception(self._fmt(event, **kw))

    def bind(self, **_kw: Any) -> "_StdlibAdapter":
        return self
