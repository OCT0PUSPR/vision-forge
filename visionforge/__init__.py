"""vision-forge: a real-time, multi-task computer vision platform.

Public API surface:
    >>> from visionforge import VisionPipeline, Detection, FrameResult
    >>> pipe = VisionPipeline(task="detection")
    >>> result = pipe.run_image("path/to/image.jpg")

Heavy backends (torch / ultralytics / transformers) are imported lazily so
that the lightweight pieces (schema, drawing helpers, config) work without
them installed.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "OCT0PUSPR"
__license__ = "MIT"

from visionforge.core.schema import Detection, FrameResult, Keypoint

__all__ = [
    "Detection",
    "FrameResult",
    "Keypoint",
    "__version__",
    "__author__",
    "__license__",
]


def __getattr__(name: str):
    # Lazy export of VisionPipeline so importing the package never pulls in
    # heavy optional dependencies unless the pipeline is actually requested.
    if name == "VisionPipeline":
        from visionforge.pipeline import VisionPipeline

        return VisionPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
