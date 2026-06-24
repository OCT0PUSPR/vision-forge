"""Unit tests for the ONNX backend's pure numpy pre/post-processing."""

import numpy as np
import pytest

from visionforge.models.onnx_backend import COCO_NAMES, _letterbox, _nms


def test_letterbox_square_output():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out, scale, left, top = _letterbox(img, new_shape=640)
    assert out.shape == (640, 640, 3)
    # wider than tall -> scaled by width, padded top/bottom
    assert scale == pytest.approx(640 / 200)
    assert left == 0
    assert top > 0


def test_letterbox_preserves_aspect():
    img = np.zeros((640, 320, 3), dtype=np.uint8)
    out, scale, left, top = _letterbox(img, new_shape=640)
    assert out.shape == (640, 640, 3)
    assert scale == pytest.approx(1.0)
    assert left > 0
    assert top == 0


def test_nms_removes_overlaps():
    boxes = np.array(
        [
            [0, 0, 10, 10],
            [1, 1, 11, 11],  # heavily overlaps box 0
            [100, 100, 110, 110],  # disjoint
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = _nms(boxes, scores, iou_threshold=0.5)
    assert 0 in keep  # highest score kept
    assert 2 in keep  # disjoint kept
    assert 1 not in keep  # suppressed


def test_nms_empty():
    assert _nms(np.empty((0, 4)), np.empty((0,)), 0.5) == []


def test_nms_keeps_all_when_disjoint():
    boxes = np.array([[0, 0, 5, 5], [50, 50, 55, 55]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    keep = _nms(boxes, scores, iou_threshold=0.5)
    assert sorted(keep) == [0, 1]


def test_coco_names_count():
    assert len(COCO_NAMES) == 80
    assert "person" in COCO_NAMES
