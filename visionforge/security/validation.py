"""Pure, dependency-light validation helpers for untrusted input.

These are deliberately stdlib-only so they are fast to import and exhaustively
unit-testable without FastAPI/torch. The API layer wraps the raised
``ValidationError`` / ``PayloadTooLargeError`` / ``UnsupportedMediaTypeError``
into structured JSON responses.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

from visionforge.errors import (
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
    ValidationError,
)

# Allowed upload content types.
ALLOWED_IMAGE_MIME = frozenset({"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"})
ALLOWED_VIDEO_MIME = frozenset({"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"})

# Magic-number signatures for common image formats (defense in depth: do not
# trust the client-declared content type alone).
_MAGIC = {
    "image/jpeg": [b"\xff\xd8\xff"],
    "image/png": [b"\x89PNG\r\n\x1a\n"],
    "image/webp": [b"RIFF"],  # followed by WEBP at offset 8
    "image/bmp": [b"BM"],
}


def validate_content_type(
    content_type: Optional[str],
    allowed: Iterable[str] = ALLOWED_IMAGE_MIME,
) -> str:
    """Validate and normalize a declared content type."""
    if not content_type:
        raise UnsupportedMediaTypeError("Missing content type")
    ct = content_type.split(";")[0].strip().lower()
    allowed_set = set(allowed)
    if ct not in allowed_set:
        raise UnsupportedMediaTypeError(
            f"Unsupported media type {ct!r}",
            details={"allowed": sorted(allowed_set)},
        )
    return ct


def validate_size(num_bytes: int, max_mb: float) -> None:
    """Reject payloads larger than ``max_mb`` megabytes."""
    if num_bytes <= 0:
        raise ValidationError("Empty payload")
    max_bytes = int(max_mb * 1024 * 1024)
    if num_bytes > max_bytes:
        raise PayloadTooLargeError(
            f"Payload {num_bytes} bytes exceeds {max_mb} MB limit",
            details={"max_bytes": max_bytes, "got_bytes": num_bytes},
        )


def sniff_image_mime(data: bytes) -> Optional[str]:
    """Best-effort detection of an image MIME type from magic bytes."""
    if not data:
        return None
    for mime, sigs in _MAGIC.items():
        for sig in sigs:
            if data.startswith(sig):
                if mime == "image/webp":
                    # RIFF....WEBP
                    if len(data) >= 12 and data[8:12] == b"WEBP":
                        return mime
                    continue
                return mime
    return None


def validate_image_bytes(
    data: bytes,
    *,
    max_mb: float,
    declared_type: Optional[str] = None,
) -> str:
    """Run the full image upload validation gauntlet, returning the MIME type.

    Checks size, optional declared content type, and the actual magic bytes.
    """
    validate_size(len(data), max_mb)
    if declared_type:
        validate_content_type(declared_type, ALLOWED_IMAGE_MIME)
    sniffed = sniff_image_mime(data)
    if sniffed is None:
        raise UnsupportedMediaTypeError(
            "Payload does not look like a supported image",
            details={"allowed": sorted(ALLOWED_IMAGE_MIME)},
        )
    return sniffed


def validate_dimensions(
    width: int,
    height: int,
    *,
    max_pixels: int = 8000 * 8000,
    max_side: int = 8000,
    min_side: int = 1,
) -> Tuple[int, int]:
    """Reject absurd frame dimensions (decompression-bomb guard)."""
    if width < min_side or height < min_side:
        raise ValidationError(
            "Frame too small",
            details={"width": width, "height": height},
        )
    if width > max_side or height > max_side:
        raise ValidationError(
            f"Frame side exceeds {max_side}px",
            details={"width": width, "height": height, "max_side": max_side},
        )
    if width * height > max_pixels:
        raise ValidationError(
            "Frame has too many pixels",
            details={"pixels": width * height, "max_pixels": max_pixels},
        )
    return width, height


def validate_task(task: Optional[str], valid: Iterable[str]) -> str:
    """Normalize/validate a task string from an untrusted client."""
    valid_set = set(valid)
    normalized = (task or "detection").strip().lower()
    if normalized not in valid_set:
        raise ValidationError(
            f"Unsupported task {normalized!r}",
            details={"valid": sorted(valid_set)},
        )
    return normalized


def validate_threshold(value: Optional[float], name: str) -> Optional[float]:
    """Validate an optional confidence/iou threshold is within [0, 1]."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a number") from exc
    if not 0.0 <= v <= 1.0:
        raise ValidationError(
            f"{name} must be between 0 and 1",
            details={name: v},
        )
    return v
