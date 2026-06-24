"""Tests for Gaussian-splat rendering + target encoding (numpy only).

The ``draw_gaussian`` / ``gaussian_radius`` helpers and ``build_targets`` are
pure numpy, so these run in the light CI path too.
"""

import numpy as np
import pytest

# Import from the torch-free ``targets`` module so these tests run in the
# lightweight (no-torch) CI path. ``losses``/``postprocess`` re-export them.
from visionforge.models.centernet.targets import (
    _gaussian2d,
    build_targets,
    draw_gaussian,
    gaussian_radius,
)


def test_gaussian_radius_positive_and_scales():
    small = gaussian_radius((10, 10), min_overlap=0.7)
    big = gaussian_radius((40, 40), min_overlap=0.7)
    assert small >= 0
    assert big > small  # bigger objects -> bigger splat radius


def test_gaussian2d_peaks_at_center():
    g = _gaussian2d((7, 7), sigma=1.0)
    assert g.shape == (7, 7)
    assert g[3, 3] == pytest.approx(1.0)
    assert g[0, 0] < g[3, 3]


def test_draw_gaussian_peak_is_one():
    hm = np.zeros((32, 32), dtype=np.float32)
    draw_gaussian(hm, (16, 16), radius=4)
    assert hm[16, 16] == pytest.approx(1.0)
    # falls off away from the center
    assert hm[16, 16] > hm[16, 22]


def test_draw_gaussian_clips_at_border():
    hm = np.zeros((20, 20), dtype=np.float32)
    # center near the corner; must not raise and must place a peak
    draw_gaussian(hm, (1, 1), radius=5)
    assert hm[1, 1] == pytest.approx(1.0)


def test_draw_gaussian_outside_is_noop():
    hm = np.zeros((10, 10), dtype=np.float32)
    out = draw_gaussian(hm, (100, 100), radius=3)
    assert out.sum() == 0.0


def test_build_targets_shapes_and_indices():
    boxes = [[10.0, 10.0, 30.0, 30.0], [40.0, 40.0, 60.0, 60.0]]
    labels = [0, 2]
    out = build_targets(boxes, labels, num_classes=3, output_h=64, output_w=64, max_objects=16)
    assert out["hm"].shape == (3, 64, 64)
    assert out["wh"].shape == (16, 2)
    assert out["offset"].shape == (16, 2)
    assert out["reg_mask"][:2].sum() == 2
    assert out["reg_mask"][2:].sum() == 0
    # center of box 0 = (20, 20); wh = (20, 20)
    assert out["wh"][0].tolist() == [20.0, 20.0]
    cx_int, cy_int = 20, 20
    assert out["ind"][0] == cy_int * 64 + cx_int
    # the heatmap peak for class 0 sits at the center
    assert out["hm"][0, cy_int, cx_int] == pytest.approx(1.0)
    assert out["hm"][2].max() == pytest.approx(1.0)  # class 2 splat exists


def test_build_targets_skips_degenerate_box():
    out = build_targets([[10, 10, 10, 10]], [0], num_classes=1, output_h=32, output_w=32)
    assert out["reg_mask"].sum() == 0
