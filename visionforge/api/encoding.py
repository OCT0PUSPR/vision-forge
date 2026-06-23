"""Base64 / image (de)serialization helpers used by the API layer.

Kept in their own module (and import-light at top level) so the parsing helpers
can be reasoned about and tested independently of FastAPI.
"""

from __future__ import annotations

import base64
import binascii
import re
from typing import Optional

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[\w/.+-]+)?;base64,(?P<data>.*)$", re.DOTALL)


def strip_data_url(payload: str) -> str:
    """Return the raw base64 portion of a possible ``data:`` URL.

    Accepts either a bare base64 string or a full ``data:image/...;base64,...``
    URL (as produced by ``canvas.toDataURL`` in the browser).
    """
    payload = payload.strip()
    match = _DATA_URL_RE.match(payload)
    if match:
        return match.group("data")
    return payload


def decode_base64(payload: str) -> bytes:
    """Decode a (possibly data-URL-wrapped) base64 string to bytes."""
    raw = strip_data_url(payload)
    try:
        return base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image payload") from exc


def encode_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    """Wrap raw image bytes into a browser-ready ``data:`` URL."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def bytes_to_rgb_array(image_bytes: bytes):
    """Decode encoded image bytes (jpeg/png/...) into an RGB numpy array."""
    import numpy as np

    try:
        import cv2  # type: ignore

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image bytes")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except ImportError:  # pragma: no cover - fallback path
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.asarray(img)


def rgb_array_to_bytes(image, mime: str = "image/jpeg", quality: int = 85) -> bytes:
    """Encode an RGB numpy array back into compressed image bytes."""
    try:
        import cv2  # type: ignore

        bgr = cv2_from_rgb(image)
        ext = ".png" if mime.endswith("png") else ".jpg"
        params = (
            [cv2.IMWRITE_JPEG_QUALITY, quality] if ext == ".jpg" else []
        )
        ok, buf = cv2.imencode(ext, bgr, params)
        if not ok:
            raise ValueError("Failed to encode image")
        return buf.tobytes()
    except ImportError:  # pragma: no cover - fallback path
        import io

        from PIL import Image

        fmt = "PNG" if mime.endswith("png") else "JPEG"
        out = io.BytesIO()
        Image.fromarray(image).save(out, format=fmt, quality=quality)
        return out.getvalue()


def cv2_from_rgb(image):
    """Convert an RGB array to BGR for OpenCV writing."""
    import cv2  # type: ignore

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def validate_task(task: Optional[str], valid) -> str:
    """Normalize/validate a task string from an untrusted client."""
    task = (task or "detection").strip().lower()
    if task not in valid:
        raise ValueError(f"Unsupported task {task!r}. Valid: {', '.join(valid)}")
    return task
