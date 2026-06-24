"""Additional coverage for ONNX predict path, arq import, and server helpers."""

import numpy as np
import pytest


def test_arq_worker_importable():
    # Module must import without arq/redis installed.
    from visionforge.worker import arq_worker

    assert hasattr(arq_worker, "process_video_job")
    assert hasattr(arq_worker, "WorkerSettings")


def test_onnx_predict_with_fake_session(monkeypatch):
    from visionforge.models.onnx_backend import OnnxDetectionBackend

    backend = OnnxDetectionBackend(onnx_path="unused.onnx", conf=0.3, iou=0.5, image_size=640)

    # Fake an onnxruntime session: returns a single high-confidence person.
    class _FakeSession:
        def __init__(self):
            self._n = 100

        def run(self, _outputs, _feed):
            out = np.zeros((1, 84, self._n), dtype=np.float32)
            out[0, :4, 0] = [320, 240, 100, 80]
            out[0, 4, 0] = 0.95
            return [out]

    backend._session = _FakeSession()
    backend._input_name = "images"
    # Bypass load() (which would look for the file).
    monkeypatch.setattr(backend, "load", lambda: None)

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = backend.predict(img)
    assert result.task == "detection"
    assert result.width == 640
    assert any(d.label == "person" for d in result.detections)


def test_onnx_missing_file_raises():
    from visionforge.models.onnx_backend import OnnxDetectionBackend

    backend = OnnxDetectionBackend(onnx_path="/no/such/model.onnx")
    with pytest.raises((FileNotFoundError, RuntimeError)):
        backend.load()


def test_server_decode_image_validates():
    import io

    from PIL import Image

    from visionforge.api.server import _decode_image
    from visionforge.errors import UnsupportedMediaTypeError

    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format="PNG")
    arr = _decode_image(buf.getvalue(), "image/png")
    assert arr.shape[2] == 3

    with pytest.raises(UnsupportedMediaTypeError):
        _decode_image(b"not an image at all", "text/plain")


def test_registry_resolve_onnx_only_detection():
    from visionforge.models.registry import ModelRegistry

    r = ModelRegistry()
    assert r.resolve("detection", "onnx") == "onnx"
    with pytest.raises(ValueError):
        r.resolve("pose", "onnx")
    with pytest.raises(ValueError):
        r.resolve("detection", "bogus")
