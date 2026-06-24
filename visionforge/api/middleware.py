"""ASGI/HTTP middleware: request IDs, structured access logs, security headers.

Implemented as Starlette ``BaseHTTPMiddleware`` subclasses. Kept separate from
``server.py`` so they can be unit-tested and reused.
"""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from visionforge.observability.logging import (
    bind_request_id,
    get_logger,
    reset_request_id,
)
from visionforge.observability.metrics import get_metrics

log = get_logger("visionforge.api")

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign/propagate a request id, emit structured access logs + metrics."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        token = bind_request_id(request_id)
        request.state.request_id = request_id
        metrics = get_metrics()
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            log.exception(
                "request_unhandled_error",
                method=request.method,
                path=request.url.path,
            )
            raise
        finally:
            elapsed = time.perf_counter() - start
            # Use the route template when available to bound label cardinality.
            route = request.scope.get("route")
            path_label = getattr(route, "path", request.url.path)
            try:
                metrics.requests_total.labels(method=request.method, path=path_label, status=str(status_code)).inc()
                metrics.request_latency_seconds.labels(method=request.method, path=path_label).observe(elapsed)
            except Exception:  # noqa: BLE001  # nosec B110 - metrics must never break a request
                pass
            log.info(
                "request",
                method=request.method,
                path=request.url.path,
                status=status_code,
                duration_ms=round(elapsed * 1000, 2),
            )
            reset_request_id(token)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach standard security headers to every response."""

    def __init__(self, app, enable_hsts: bool = False, csp: str = "") -> None:
        super().__init__(app)
        self.enable_hsts = enable_hsts
        self.csp = csp or (
            "default-src 'self'; "
            "img-src 'self' data: blob:; "
            "media-src 'self' blob:; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "connect-src 'self' ws: wss:; "
            "frame-ancestors 'none'; "
            "base-uri 'self'"
        )

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Permissions-Policy", "camera=(self), microphone=(), geolocation=()")
        response.headers.setdefault("Content-Security-Policy", self.csp)
        if self.enable_hsts:
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=31536000; includeSubDomains",
            )
        # Propagate request id back to the client for correlation.
        rid = getattr(request.state, "request_id", None)
        if rid:
            response.headers.setdefault(REQUEST_ID_HEADER, rid)
        return response
