"""Tests for the opencv-backed draw_detections rendering (needs numpy+cv2)."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from visionforge.core.draw import draw_detections  # noqa: E402
from visionforge.core.schema import Detection, FrameResult, Keypoint  # noqa: E402
from visionforge.models.onnx_backend import OnnxDetectionBackend  # noqa: E402


def test_draw_boxes_masks_keypoints():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    fr = FrameResult(
        detections=[
            Detection(
                "person",
                0.9,
                (10, 10, 90, 180),
                class_id=0,
                track_id=2,
                keypoints=[Keypoint(40, 30, 0.9, "nose"), Keypoint(35, 50, 0.8)],
            ),
            Detection(
                "dog",
                0.7,
                (100, 100, 180, 180),
                class_id=16,
                mask=[[100, 100], [180, 100], [180, 180], [100, 180]],
            ),
        ],
        task="detection",
        width=200,
        height=200,
        inference_ms=12.3,
    )
    out = draw_detections(img, fr)
    assert out.shape == (200, 200, 3)
    # Something was drawn (image no longer all-zero).
    assert out.sum() > 0


def test_draw_empty_result():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    out = draw_detections(img, FrameResult(task="detection"))
    assert out.shape == (50, 50, 3)


def test_onnx_postprocess_decodes_synthetic_output():
    backend = OnnxDetectionBackend(onnx_path="unused.onnx", conf=0.3, iou=0.5)
    # Build a fake YOLOv8 output (1, 84, N): 4 bbox + 80 class scores. N must be
    # > 84 so the (channels, anchors) orientation heuristic transposes correctly.
    n = 100
    out = np.zeros((1, 84, n), dtype=np.float32)
    # anchor 0: centered box, high score on class 0 (person)
    out[0, :4, 0] = [320, 240, 100, 80]  # cx, cy, w, h (in 640 space)
    out[0, 4, 0] = 0.95
    # anchor 1: low score -> filtered out
    out[0, :4, 1] = [100, 100, 20, 20]
    out[0, 5, 1] = 0.1
    # anchor 2: class 16 (dog)
    out[0, :4, 2] = [500, 400, 60, 60]
    out[0, 20, 2] = 0.8

    dets = backend._postprocess([out], scale=1.0, pad_left=0, pad_top=0, orig_shape=(480, 640))
    labels = {d.label for d in dets}
    assert "person" in labels
    # the 0.1-score detection should be gone
    assert all(d.confidence >= 0.3 for d in dets)
