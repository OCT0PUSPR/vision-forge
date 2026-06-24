"""Torch-free target rendering for CenterNet (numpy only).

This module deliberately avoids importing ``torch`` so it can be imported (and
unit-tested) in the lightweight, no-torch CI environment. It holds:

    * ``gaussian_radius`` / ``_gaussian2d`` / ``draw_gaussian`` — the
      penalty-reduced focal-loss target rendering (Gaussian "splats").
    * ``build_targets`` — encode one image's GT boxes/labels into the dense
      CenterNet target arrays (heatmap, wh, offset, center index, reg mask).

The differentiable losses (``losses.py``) and the decode (``postprocess.py``)
re-export these names, so existing imports keep working while the numpy core
stays torch-free.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

DOWN_RATIO = 4


# --------------------------------------------------------------------------- #
# Gaussian-splat target rendering
# --------------------------------------------------------------------------- #
def gaussian_radius(det_size: Tuple[float, float], min_overlap: float = 0.7) -> float:
    """Radius such that a box within ``min_overlap`` IoU stays inside the splat.

    Solves the three quadratics from the CenterNet reference implementation
    (corner / inside / outside displacement cases) and returns the min positive
    root. ``det_size`` is ``(height, width)`` in heatmap units.
    """
    height, width = det_size

    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = max(0.0, b1 * b1 - 4 * a1 * c1) ** 0.5
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4.0
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = max(0.0, b2 * b2 - 4 * a2 * c2) ** 0.5
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = max(0.0, b3 * b3 - 4 * a3 * c3) ** 0.5
    r3 = (b3 + sq3) / (2 * a3)

    return max(0.0, min(r1, r2, r3))


def _gaussian2d(shape: Tuple[int, int], sigma: float = 1.0) -> np.ndarray:
    """A 2D Gaussian kernel normalised to peak 1.0 at the center."""
    m = (shape[0] - 1.0) / 2.0
    n = (shape[1] - 1.0) / 2.0
    y = np.arange(-m, m + 1).reshape(-1, 1)
    x = np.arange(-n, n + 1).reshape(1, -1)
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap: np.ndarray, center: Tuple[int, int], radius: int, k: float = 1.0) -> np.ndarray:
    """Splat a Gaussian peak into ``heatmap`` (in place, ``np.maximum`` blend).

    ``heatmap`` is a single-class ``(H, W)`` array; ``center`` is ``(x, y)`` in
    heatmap coordinates. Overlapping splats keep the elementwise maximum so two
    nearby centers do not wash each other out.
    """
    diameter = 2 * radius + 1
    gaussian = _gaussian2d((diameter, diameter), sigma=diameter / 6.0)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    if right <= -left or bottom <= -top:  # entirely outside
        return heatmap

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


# --------------------------------------------------------------------------- #
# Dense target encoding
# --------------------------------------------------------------------------- #
def build_targets(
    boxes: Sequence[Sequence[float]],
    labels: Sequence[int],
    num_classes: int,
    output_h: int,
    output_w: int,
    max_objects: int = 64,
    min_overlap: float = 0.7,
) -> dict:
    """Encode one image's GT into dense CenterNet target tensors.

    ``boxes`` are xyxy in *stride-4 (output) pixel* coordinates. Returns numpy
    arrays for ``hm`` (C,H,W), ``wh`` (K,2), ``offset`` (K,2), ``ind`` (K,) flat
    center index, ``reg_mask`` (K,).
    """
    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wh = np.zeros((max_objects, 2), dtype=np.float32)
    offset = np.zeros((max_objects, 2), dtype=np.float32)
    ind = np.zeros((max_objects,), dtype=np.int64)
    reg_mask = np.zeros((max_objects,), dtype=np.float32)

    for k, (box, cls) in enumerate(zip(boxes, labels)):
        if k >= max_objects:
            break
        x1, y1, x2, y2 = box
        x1 = float(np.clip(x1, 0, output_w - 1))
        y1 = float(np.clip(y1, 0, output_h - 1))
        x2 = float(np.clip(x2, 0, output_w - 1))
        y2 = float(np.clip(y2, 0, output_h - 1))
        h, w = y2 - y1, x2 - x1
        if h <= 0 or w <= 0:
            continue
        radius = max(0, int(gaussian_radius((h, w), min_overlap)))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        cx_int, cy_int = int(cx), int(cy)
        cx_int = min(cx_int, output_w - 1)
        cy_int = min(cy_int, output_h - 1)

        draw_gaussian(hm[int(cls)], (cx_int, cy_int), radius)
        wh[k] = [w, h]
        offset[k] = [cx - cx_int, cy - cy_int]
        ind[k] = cy_int * output_w + cx_int
        reg_mask[k] = 1.0

    return {
        "hm": hm,
        "wh": wh,
        "offset": offset,
        "ind": ind,
        "reg_mask": reg_mask,
    }
