"""Structured application errors and their JSON serialization.

Every error carries a stable machine ``code``, an HTTP ``status`` and an
optional ``details`` dict. The API layer installs exception handlers that turn
these (and unexpected exceptions) into a consistent JSON envelope.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class VisionForgeError(Exception):
    """Base class for all application errors."""

    status: int = 500
    code: str = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        status: Optional[int] = None,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if status is not None:
            self.status = status
        if code is not None:
            self.code = code
        self.details = details or {}

    def to_dict(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }
        if request_id:
            body["error"]["request_id"] = request_id
        return body


class ValidationError(VisionForgeError):
    status = 422
    code = "validation_error"


class AuthError(VisionForgeError):
    status = 401
    code = "unauthorized"


class RateLimitError(VisionForgeError):
    status = 429
    code = "rate_limited"


class PayloadTooLargeError(VisionForgeError):
    status = 413
    code = "payload_too_large"


class UnsupportedMediaTypeError(VisionForgeError):
    status = 415
    code = "unsupported_media_type"


class NotFoundError(VisionForgeError):
    status = 404
    code = "not_found"


class InferenceError(VisionForgeError):
    status = 500
    code = "inference_failed"


class ServiceUnavailableError(VisionForgeError):
    status = 503
    code = "service_unavailable"
