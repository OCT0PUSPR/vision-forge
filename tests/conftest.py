"""Shared pytest fixtures."""

import io
import os

import pytest

# Force a known config BEFORE importing the app/settings anywhere.
os.environ.setdefault("VF_DEVICE", "cpu")
os.environ.setdefault("VF_ENV", "development")
os.environ.setdefault("VF_JSON_LOGS", "false")


def _make_png_bytes(width: int = 32, height: int = 32) -> bytes:
    """Create a tiny valid PNG using Pillow (always available in CI)."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(120, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def png_bytes():
    return _make_png_bytes()


@pytest.fixture()
def reset_state():
    """Reset API singletons + settings cache so env overrides take effect."""
    from visionforge.config import get_settings

    get_settings.cache_clear()
    try:
        from visionforge.api.deps import reset_singletons

        reset_singletons()
    except Exception:
        pass
    yield
    get_settings.cache_clear()
    try:
        from visionforge.api.deps import reset_singletons

        reset_singletons()
    except Exception:
        pass
