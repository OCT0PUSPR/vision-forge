"""Unit tests for the pure drawing helpers (no numpy/opencv needed)."""

import pytest

from visionforge.core.draw import (
    PALETTE,
    color_for_index,
    color_for_label,
    contrasting_text_color,
    format_label,
    rgb_to_bgr,
    scale_bbox,
    summarize,
)
from visionforge.core.schema import Detection, FrameResult


def test_color_for_index_wraps_palette():
    assert color_for_index(0) == PALETTE[0]
    assert color_for_index(len(PALETTE)) == PALETTE[0]
    assert color_for_index(len(PALETTE) + 1) == PALETTE[1]


def test_color_for_index_handles_negative():
    # negative ids (rare) must still map into the palette
    assert color_for_index(-1) == color_for_index(1)


def test_color_for_label_is_stable():
    assert color_for_label("person") == color_for_label("person")
    assert color_for_label("") == PALETTE[0]


def test_rgb_to_bgr():
    assert rgb_to_bgr((10, 20, 30)) == (30, 20, 10)


def test_contrasting_text_color():
    # bright background -> black text
    assert contrasting_text_color((255, 255, 255)) == (0, 0, 0)
    # dark background -> white text
    assert contrasting_text_color((0, 0, 0)) == (255, 255, 255)


def test_format_label_variants():
    assert format_label("dog") == "dog"
    assert format_label("dog", 0.876) == "dog 0.88"
    assert format_label("dog", 0.5, track_id=3) == "#3 dog 0.50"
    assert format_label("dog", track_id=9) == "#9 dog"


def test_scale_bbox():
    out = scale_bbox((10, 20, 30, 40), from_size=(100, 200), to_size=(200, 100))
    # x doubled, y halved
    assert out == (20.0, 10.0, 60.0, 20.0)


def test_scale_bbox_zero_safe():
    out = scale_bbox((1, 2, 3, 4), from_size=(0, 0), to_size=(10, 10))
    assert out == (1.0, 2.0, 3.0, 4.0)


def test_summarize_empty():
    fr = FrameResult(task="detection", inference_ms=12.4)
    assert "no detections" in summarize(fr)
    assert "12.4ms" in summarize(fr)


def test_summarize_counts():
    fr = FrameResult(
        detections=[
            Detection("person", 0.9, (0, 0, 1, 1)),
            Detection("person", 0.8, (0, 0, 1, 1)),
            Detection("dog", 0.7, (0, 0, 1, 1)),
        ],
        inference_ms=5.0,
    )
    text = summarize(fr)
    assert "2 person" in text
    assert "1 dog" in text


@pytest.mark.parametrize("idx", [0, 5, 11, 12, 99])
def test_palette_returns_valid_rgb(idx):
    color = color_for_index(idx)
    assert len(color) == 3
    assert all(0 <= c <= 255 for c in color)
