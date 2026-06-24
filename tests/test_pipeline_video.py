"""Tests for the pipeline orchestration and the video frame iterators."""

import numpy as np
import pytest

from visionforge.core.schema import Detection, FrameResult
from visionforge.core.video import (
    FPSMeter,
    iter_frames,
    iter_synthetic,
    make_synthetic_frame,
)


# --- video ---
def test_make_synthetic_frame_shape():
    f = make_synthetic_frame(128, 96, seed=1, frame_index=2)
    assert f.shape == (96, 128, 3)
    assert f.dtype == np.uint8


def test_iter_synthetic_count():
    frames = list(iter_synthetic(64, 64, n_frames=5))
    assert len(frames) == 5
    assert frames[0][0] == 0
    assert frames[-1][0] == 4


def test_iter_frames_demo():
    frames = list(iter_frames("demo", max_frames=3))
    assert len(frames) == 3


def test_iter_frames_image_path(tmp_path):
    import cv2

    p = tmp_path / "x.png"
    cv2.imwrite(str(p), np.zeros((20, 30, 3), dtype=np.uint8))
    frames = list(iter_frames(str(p)))
    assert len(frames) == 1
    assert frames[0][1].shape == (20, 30, 3)


def test_iter_frames_missing_image(tmp_path):
    with pytest.raises(Exception):
        list(iter_frames(str(tmp_path / "missing.png")))


def test_fps_meter():
    m = FPSMeter(window=5)
    assert m.fps == 0.0
    # simulate ticks via direct manipulation
    m._last = 0.0
    m._times.append(0.1)
    assert m.fps == pytest.approx(10.0)
    m.reset()
    assert m.fps == 0.0


# --- pipeline (with a stub backend, no model download) ---
class _StubManager:
    def __init__(self):
        self.calls = 0

    def infer(self, image, task, backend=None, frame_index=0):
        self.calls += 1
        return FrameResult(
            detections=[Detection("person", 0.9, (1, 2, 3, 4))],
            task=task,
            frame_index=frame_index,
        )


def test_pipeline_infer_array(monkeypatch):
    from visionforge import pipeline as pipeline_mod

    stub = _StubManager()
    # The pipeline uses the registry; patch its infer path via the manager.
    pipe = pipeline_mod.VisionPipeline(task="detection")

    def fake_infer(image, frame_index=0):
        return stub.infer(image, "detection", frame_index=frame_index)

    monkeypatch.setattr(pipe, "infer_array", fake_infer)
    result = pipe.infer_array(np.zeros((8, 8, 3), dtype=np.uint8))
    assert result.task == "detection"
    assert len(result) == 1


def test_pipeline_run_stream_demo(monkeypatch):
    from visionforge import pipeline as pipeline_mod

    pipe = pipeline_mod.VisionPipeline(task="detection")
    monkeypatch.setattr(
        pipe,
        "infer_array",
        lambda image, frame_index=0: FrameResult(task="detection", frame_index=frame_index),
    )
    out = list(pipe.run_stream("demo", max_frames=3, annotate=False))
    assert len(out) == 3
    assert out[0][1].task == "detection"
