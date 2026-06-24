"""Tests for the from-scratch mAP@0.5 metric (numpy only, no torch needed).

These run in the light CI path because the metric module is pure numpy.
"""

import numpy as np
import pytest

from visionforge.models.centernet.metrics import (
    MeanAveragePrecision,
    _ap_from_pr,
    iou_matrix,
)


def test_iou_matrix_identity():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    iou = iou_matrix(a, a)
    assert iou.shape == (1, 1)
    assert iou[0, 0] == pytest.approx(1.0)


def test_iou_matrix_disjoint_and_partial():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[100, 100, 110, 110], [0, 0, 10, 5]], dtype=np.float32)
    iou = iou_matrix(a, b)
    assert iou[0, 0] == pytest.approx(0.0)
    # half-overlap box: intersection 50, union 100 -> 0.5
    assert iou[0, 1] == pytest.approx(0.5)


def test_iou_matrix_empty():
    assert iou_matrix(np.zeros((0, 4)), np.zeros((0, 4))).shape == (0, 0)


def test_ap_from_pr_perfect():
    recall = np.array([0.5, 1.0])
    precision = np.array([1.0, 1.0])
    assert _ap_from_pr(recall, precision) == pytest.approx(1.0)


def test_map_perfect_predictions():
    metric = MeanAveragePrecision(num_classes=2, iou_threshold=0.5)
    gt_boxes = np.array([[0, 0, 10, 10], [20, 20, 40, 40]], dtype=np.float32)
    gt_labels = [0, 1]
    preds = np.array(
        [[0, 0, 10, 10, 0.9, 0], [20, 20, 40, 40, 0.95, 1]], dtype=np.float32
    )
    metric.update(preds, gt_boxes, gt_labels)
    out = metric.compute()
    assert out["map"] == pytest.approx(1.0, abs=1e-6)


def test_map_all_false_positives():
    metric = MeanAveragePrecision(num_classes=1, iou_threshold=0.5)
    gt_boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    # prediction far from GT -> false positive, AP should be 0
    preds = np.array([[100, 100, 110, 110, 0.9, 0]], dtype=np.float32)
    metric.update(preds, gt_boxes, [0])
    assert metric.compute()["map"] == pytest.approx(0.0)


def test_map_no_predictions():
    metric = MeanAveragePrecision(num_classes=1, iou_threshold=0.5)
    metric.update(np.zeros((0, 6), dtype=np.float32), np.array([[0, 0, 5, 5]], dtype=np.float32), [0])
    assert metric.compute()["map"] == pytest.approx(0.0)


def test_map_duplicate_predictions_one_tp():
    """Two overlapping preds for one GT: first is TP, second is FP."""
    metric = MeanAveragePrecision(num_classes=1, iou_threshold=0.5)
    gt = np.array([[0, 0, 10, 10]], dtype=np.float32)
    preds = np.array(
        [[0, 0, 10, 10, 0.9, 0], [0, 0, 10, 10, 0.8, 0]], dtype=np.float32
    )
    metric.update(preds, gt, [0])
    out = metric.compute()
    # recall maxes at 1.0 with precision dropping to 0.5; AP stays 1.0 here
    # because the first (highest-score) prediction already achieves full recall.
    assert 0.5 <= out["map"] <= 1.0
