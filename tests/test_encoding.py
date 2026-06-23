"""Unit tests for the base64/data-URL parsing helpers (no heavy deps)."""

import base64

import pytest

from visionforge.api.encoding import (
    decode_base64,
    encode_data_url,
    strip_data_url,
    validate_task,
)


def test_strip_data_url_passthrough():
    assert strip_data_url("aGVsbG8=") == "aGVsbG8="


def test_strip_data_url_with_prefix():
    payload = "data:image/jpeg;base64,aGVsbG8="
    assert strip_data_url(payload) == "aGVsbG8="


def test_strip_data_url_png_prefix():
    payload = "data:image/png;base64,QUJD"
    assert strip_data_url(payload) == "QUJD"


def test_decode_base64_roundtrip():
    original = b"vision-forge bytes \x00\x01\x02"
    encoded = base64.b64encode(original).decode("ascii")
    assert decode_base64(encoded) == original


def test_decode_base64_from_data_url():
    original = b"\xff\xd8\xff\xe0jpegish"
    url = encode_data_url(original, mime="image/jpeg")
    assert url.startswith("data:image/jpeg;base64,")
    assert decode_base64(url) == original


def test_decode_base64_invalid_raises():
    with pytest.raises(ValueError):
        decode_base64("not valid base64 !!!")


def test_encode_data_url_shape():
    url = encode_data_url(b"abc", mime="image/png")
    assert url.startswith("data:image/png;base64,")


def test_validate_task_default():
    assert validate_task(None, ("detection", "pose")) == "detection"


def test_validate_task_normalizes_case():
    assert validate_task("  POSE ", ("detection", "pose")) == "pose"


def test_validate_task_rejects_unknown():
    with pytest.raises(ValueError):
        validate_task("bogus", ("detection", "pose"))
