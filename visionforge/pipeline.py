"""High-level orchestration: source -> backend -> (draw) -> results.

``VisionPipeline`` is the main entry point for both the CLI and the API. It ties
together the model registry, the frame iterator and the drawing helpers.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

from visionforge.config import Settings, get_settings
from visionforge.core.schema import FrameResult
from visionforge.core.video import FPSMeter, iter_frames
from visionforge.models.registry import ModelRegistry, get_registry


class VisionPipeline:
    """Run a vision task over images, videos, webcams or the synthetic demo."""

    def __init__(
        self,
        task: str = "detection",
        backend: Optional[str] = None,
        registry: Optional[ModelRegistry] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or get_registry()
        self.task = task
        self.backend_name = self.registry.resolve(task, backend)

    # ------------------------------------------------------------------ #
    # single image
    # ------------------------------------------------------------------ #
    def infer_array(self, image, frame_index: int = 0) -> FrameResult:
        """Run inference on an in-memory RGB numpy array."""
        backend = self.registry.get(self.task, self.backend_name)
        return backend.infer(image, frame_index=frame_index)

    def run_image(self, path: str) -> FrameResult:
        """Load an image from disk and run a single inference."""
        for _, frame in iter_frames(path, max_frames=1):
            return self.infer_array(frame, frame_index=0)
        raise FileNotFoundError(f"No frame produced from: {path}")

    def annotate(self, image, result: FrameResult):
        """Draw ``result`` onto ``image`` (returns a new annotated array)."""
        from visionforge.core.draw import draw_detections

        return draw_detections(image, result)

    # ------------------------------------------------------------------ #
    # streaming
    # ------------------------------------------------------------------ #
    def run_stream(
        self,
        source,
        *,
        max_frames: Optional[int] = None,
        annotate: bool = False,
    ) -> Iterator[Tuple[int, FrameResult, Optional["object"]]]:
        """Iterate frames and yield ``(index, FrameResult, annotated_or_None)``.

        For the ``tracking`` task, ids persist across frames because the
        underlying YOLO ``.track()`` call uses ``persist=True``.
        """
        meter = FPSMeter()
        for idx, frame in iter_frames(source, max_frames=max_frames):
            result = self.infer_array(frame, frame_index=idx)
            meter.tick()
            annotated = self.annotate(frame, result) if annotate else None
            yield idx, result, annotated

    @property
    def fps_meter(self) -> FPSMeter:  # convenience for callers wanting their own
        return FPSMeter()
