"""Unit tests for the normalized result schema.

These deliberately avoid torch / ultralytics / numpy so they run in the
lightweight CI path.
"""

import math

import pytest

from visionforge.core.schema import (
    Detection,
    FrameResult,
    Keypoint,
    bbox_area,
    bbox_iou,
    clamp,
    normalize_bbox,
)


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(11, 0, 10) == 10
    # tolerant of swapped bounds
    assert clamp(5, 10, 0) == 5


def test_normalize_bbox_orders_corners():
    assert normalize_bbox([10, 20, 5, 8]) == (5.0, 8.0, 10.0, 20.0)


def test_normalize_bbox_clamps_to_bounds():
    out = normalize_bbox([-5, -5, 200, 200], width=100, height=80)
    assert out == (0.0, 0.0, 100.0, 80.0)


def test_normalize_bbox_bad_length():
    with pytest.raises(ValueError):
        normalize_bbox([1, 2, 3])


def test_bbox_area_and_iou():
    a = (0, 0, 10, 10)
    b = (5, 5, 15, 15)
    assert bbox_area(a) == 100
    # intersection 5x5=25, union=175
    assert math.isclose(bbox_iou(a, b), 25 / 175, rel_tol=1e-6)
    # disjoint boxes
    assert bbox_iou((0, 0, 1, 1), (5, 5, 6, 6)) == 0.0
    # identical boxes
    assert bbox_iou(a, a) == 1.0


def test_detection_post_init_normalizes():
    d = Detection(label="cat", confidence=0.9, bbox=(30, 40, 10, 20))
    assert d.bbox == (10.0, 20.0, 30.0, 40.0)
    assert d.area == 400.0
    assert d.center == (20.0, 30.0)


def test_detection_roundtrip_dict():
    d = Detection(
        label="person",
        confidence=0.873,
        bbox=(1, 2, 3, 4),
        class_id=0,
        track_id=7,
        keypoints=[Keypoint(1.0, 2.0, 0.9, "nose")],
    )
    data = d.to_dict()
    assert data["label"] == "person"
    assert data["confidence"] == 0.873
    assert data["track_id"] == 7
    assert data["keypoints"][0]["name"] == "nose"

    restored = Detection.from_dict(data)
    assert restored.label == d.label
    assert restored.track_id == 7
    assert restored.keypoints[0].name == "nose"


def test_frame_result_counts_and_filter():
    dets = [
        Detection("person", 0.9, (0, 0, 10, 10)),
        Detection("person", 0.4, (0, 0, 5, 5)),
        Detection("dog", 0.8, (0, 0, 8, 8)),
    ]
    fr = FrameResult(detections=dets, task="detection", width=100, height=100)
    assert len(fr) == 3
    assert fr.count_by_label() == {"person": 2, "dog": 1}
    assert sorted(fr.labels) == ["dog", "person", "person"]

    filtered = fr.filter_by_confidence(0.5)
    assert len(filtered) == 2
    assert filtered.count_by_label() == {"person": 1, "dog": 1}
    # original unchanged
    assert len(fr) == 3


def test_frame_result_roundtrip_dict():
    fr = FrameResult(
        detections=[Detection("car", 0.5, (0, 0, 4, 4), class_id=2)],
        task="detection",
        width=64,
        height=48,
        frame_index=12,
        inference_ms=8.345,
        model="yolov8n.pt",
    )
    data = fr.to_dict()
    assert data["count"] == 1
    assert data["frame_index"] == 12
    assert data["inference_ms"] == 8.34 or data["inference_ms"] == 8.35

    restored = FrameResult.from_dict(data)
    assert restored.task == "detection"
    assert restored.width == 64
    assert len(restored) == 1
    assert restored.detections[0].class_id == 2


def test_classification_serialization():
    fr = FrameResult(
        task="classification",
        classification=[("cat", 0.91234), ("dog", 0.05)],
    )
    data = fr.to_dict()
    assert data["classification"][0] == ["cat", 0.9123]
    restored = FrameResult.from_dict(data)
    assert restored.classification[0][0] == "cat"
