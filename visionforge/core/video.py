"""Frame sources (file / webcam / synthetic demo) and an FPS meter.

The synthetic demo generator is dependency-light (needs numpy only) so a
laptop with no camera and no model weights can still exercise the pipeline.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Iterator, Optional, Tuple, Union

Source = Union[str, int]


class FPSMeter:
    """Rolling-average frames-per-second estimator."""

    def __init__(self, window: int = 30) -> None:
        self.window = window
        self._times: Deque[float] = deque(maxlen=window)
        self._last: Optional[float] = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            self._times.append(now - self._last)
        self._last = now
        return self.fps

    @property
    def fps(self) -> float:
        if not self._times:
            return 0.0
        avg = sum(self._times) / len(self._times)
        return 1.0 / avg if avg > 0 else 0.0

    def reset(self) -> None:
        self._times.clear()
        self._last = None


def make_synthetic_frame(
    width: int = 640,
    height: int = 480,
    seed: int = 0,
    frame_index: int = 0,
):
    """Generate a deterministic synthetic RGB frame (numpy uint8 array).

    Produces a gradient background plus a few moving colored shapes so that the
    drawing/pipeline path has something non-trivial to render even with no
    camera or model. Requires numpy.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    # Gradient background.
    ys = np.linspace(20, 90, height, dtype=np.float32).reshape(-1, 1)
    xs = np.linspace(30, 110, width, dtype=np.float32).reshape(1, -1)
    base = (ys + xs) / 2.0
    img = np.stack(
        [
            np.clip(base * 0.6, 0, 255),
            np.clip(base * 0.9, 0, 255),
            np.clip(base * 1.2, 0, 255),
        ],
        axis=-1,
    ).astype(np.uint8)

    # A few "objects" that drift with frame_index.
    n_shapes = 3
    phase = frame_index * 0.08
    for i in range(n_shapes):
        color = tuple(int(c) for c in rng.integers(80, 255, size=3))
        cx = int((0.5 + 0.35 * np.sin(phase + i * 2.1)) * width)
        cy = int((0.5 + 0.30 * np.cos(phase * 0.8 + i * 1.7)) * height)
        size = int(40 + 20 * i)
        x1 = max(0, cx - size)
        y1 = max(0, cy - size)
        x2 = min(width, cx + size)
        y2 = min(height, cy + size)
        img[y1:y2, x1:x2] = color
    return img


def iter_synthetic(
    width: int = 640,
    height: int = 480,
    n_frames: int = 60,
    seed: int = 0,
) -> Iterator[Tuple[int, "object"]]:
    """Yield ``(index, frame)`` tuples of synthetic frames."""
    for i in range(n_frames):
        yield i, make_synthetic_frame(width, height, seed=seed, frame_index=i)


def iter_frames(
    source: Source,
    *,
    max_frames: Optional[int] = None,
    demo_frames: int = 60,
    demo_size: Tuple[int, int] = (640, 480),
) -> Iterator[Tuple[int, "object"]]:
    """Unified frame iterator over file / webcam / image / synthetic-demo.

    ``source`` semantics:
        * ``"demo"``          -> synthetic frames (no hardware needed)
        * ``int`` or digit str -> webcam index (OpenCV ``VideoCapture``)
        * image path           -> a single frame
        * video path           -> decode the video

    Yields ``(frame_index, rgb_numpy_array)``. Heavy decoding needs opencv.
    """
    if isinstance(source, str) and source.lower() == "demo":
        w, h = demo_size
        n = max_frames if max_frames is not None else demo_frames
        yield from iter_synthetic(w, h, n_frames=n)
        return

    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Reading files/webcam requires opencv-python(-headless). "
            "Use source='demo' for a no-dependency synthetic stream."
        ) from exc

    # Single image path -> yield one frame and stop.
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    if isinstance(source, str) and source.lower().endswith(image_exts):
        img = cv2.imread(source)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {source}")
        yield 0, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return

    # Webcam index or video file.
    cap_arg: Source
    if isinstance(source, int):
        cap_arg = source
    elif isinstance(source, str) and source.isdigit():
        cap_arg = int(source)
    else:
        cap_arg = source

    cap = cv2.VideoCapture(cap_arg)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source!r}")
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            idx += 1
            if max_frames is not None and idx >= max_frames:
                break
    finally:
        cap.release()
