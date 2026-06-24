"""Tests for the HuggingFace backend normalization (with a fake pipeline).

No transformers/torch needed: we inject a fake callable in place of the loaded
``transformers.pipeline`` to exercise the result-mapping code paths.
"""

import numpy as np

from visionforge.models.hf_backend import (
    HFClassificationBackend,
    HFDetectionBackend,
    _to_pil,
)


def test_to_pil_from_numpy():
    arr = np.zeros((10, 12, 3), dtype=np.uint8)
    img = _to_pil(arr)
    assert img.size == (12, 10)  # PIL is (w, h)


def test_to_pil_passthrough():
    from PIL import Image

    img = Image.new("RGB", (4, 4))
    assert _to_pil(img) is img


def test_hf_detection_normalizes():
    backend = HFDetectionBackend(model_id="facebook/detr-resnet-50", conf=0.5)

    # Inject a fake pipeline (skip real model load).
    def fake_pipe(pil, threshold=0.5):
        return [
            {
                "label": "cat",
                "score": 0.93,
                "box": {"xmin": 10, "ymin": 20, "xmax": 110, "ymax": 220},
            },
            {
                "label": "dog",
                "score": 0.71,
                "box": {"xmin": 0, "ymin": 0, "xmax": 50, "ymax": 50},
            },
        ]

    backend._pipe = fake_pipe
    result = backend.predict(np.zeros((240, 320, 3), dtype=np.uint8))
    assert result.task == "detection"
    assert len(result) == 2
    assert result.detections[0].label == "cat"
    assert result.detections[0].bbox == (10.0, 20.0, 110.0, 220.0)
    assert result.width == 320 and result.height == 240


def test_hf_detection_device_index():
    cpu = HFDetectionBackend(device="cpu")
    assert cpu._device_index() == -1
    cuda = HFDetectionBackend(device="cuda:1")
    assert cuda._device_index() == 1
    cuda0 = HFDetectionBackend(device="cuda")
    assert cuda0._device_index() == 0


def test_hf_classification_normalizes():
    backend = HFClassificationBackend(model_id="google/vit-base-patch16-224", top_k=3)

    def fake_pipe(pil, top_k=5):
        return [
            {"label": "tabby cat", "score": 0.8},
            {"label": "tiger cat", "score": 0.15},
        ]

    backend._pipe = fake_pipe
    result = backend.predict(np.zeros((50, 50, 3), dtype=np.uint8))
    assert result.task == "classification"
    assert result.classification[0][0] == "tabby cat"
    assert result.classification[0][1] == 0.8
    assert len(result.detections) == 0
