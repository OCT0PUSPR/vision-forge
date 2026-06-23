"""Normalized result schema shared by every backend.

These dataclasses are deliberately dependency-free (stdlib only) so they can be
imported and unit-tested without torch / ultralytics / numpy installed.

Coordinate convention:
    * ``bbox`` is always ``[x1, y1, x2, y2]`` in absolute pixel coordinates,
      top-left origin, with ``x1 <= x2`` and ``y1 <= y2``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

BBox = Tuple[float, float, float, float]


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive range ``[low, high]``."""
    if low > high:
        low, high = high, low
    return max(low, min(high, value))


def normalize_bbox(
    bbox: Sequence[float],
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> BBox:
    """Return a well-ordered xyxy bbox, optionally clamped to image bounds.

    Accepts any 4-length sequence, guarantees ``x1<=x2`` / ``y1<=y2`` and, when
    ``width``/``height`` are provided, clamps the box inside the image.
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 elements, got {len(bbox)}")
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if width is not None:
        x1 = clamp(x1, 0.0, float(width))
        x2 = clamp(x2, 0.0, float(width))
    if height is not None:
        y1 = clamp(y1, 0.0, float(height))
        y2 = clamp(y2, 0.0, float(height))
    return (x1, y1, x2, y2)


def bbox_area(bbox: Sequence[float]) -> float:
    """Area of an xyxy box (0 for degenerate boxes)."""
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Keypoint:
    """A single 2D keypoint with optional visibility/confidence."""

    x: float
    y: float
    confidence: float = 1.0
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Detection:
    """A single normalized prediction.

    Holds whatever a given task produced: a box always; a polygon mask and/or
    keypoints when segmentation/pose ran; a ``track_id`` when tracking ran.
    """

    label: str
    confidence: float
    bbox: BBox
    class_id: Optional[int] = None
    mask: Optional[List[List[float]]] = None  # polygon as [[x, y], ...]
    keypoints: Optional[List[Keypoint]] = None
    track_id: Optional[int] = None

    def __post_init__(self) -> None:
        self.bbox = normalize_bbox(self.bbox)
        self.confidence = float(self.confidence)

    @property
    def area(self) -> float:
        return bbox_area(self.bbox)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(v), 2) for v in self.bbox],
            "class_id": self.class_id,
            "track_id": self.track_id,
        }
        if self.mask is not None:
            data["mask"] = self.mask
        if self.keypoints is not None:
            data["keypoints"] = [kp.to_dict() for kp in self.keypoints]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Detection":
        kps = data.get("keypoints")
        keypoints = (
            [Keypoint(**kp) if isinstance(kp, dict) else Keypoint(*kp) for kp in kps]
            if kps
            else None
        )
        return cls(
            label=data["label"],
            confidence=data["confidence"],
            bbox=tuple(data["bbox"]),  # type: ignore[arg-type]
            class_id=data.get("class_id"),
            mask=data.get("mask"),
            keypoints=keypoints,
            track_id=data.get("track_id"),
        )


@dataclass
class FrameResult:
    """All detections for one frame, plus light metadata."""

    detections: List[Detection] = field(default_factory=list)
    task: str = "detection"
    width: int = 0
    height: int = 0
    frame_index: int = 0
    inference_ms: float = 0.0
    model: Optional[str] = None
    # Classification convenience: top-k label/score pairs.
    classification: Optional[List[Tuple[str, float]]] = None

    def __len__(self) -> int:
        return len(self.detections)

    @property
    def labels(self) -> List[str]:
        return [d.label for d in self.detections]

    def count_by_label(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for det in self.detections:
            counts[det.label] = counts.get(det.label, 0) + 1
        return counts

    def filter_by_confidence(self, threshold: float) -> "FrameResult":
        kept = [d for d in self.detections if d.confidence >= threshold]
        return FrameResult(
            detections=kept,
            task=self.task,
            width=self.width,
            height=self.height,
            frame_index=self.frame_index,
            inference_ms=self.inference_ms,
            model=self.model,
            classification=self.classification,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "width": self.width,
            "height": self.height,
            "frame_index": self.frame_index,
            "inference_ms": round(float(self.inference_ms), 2),
            "model": self.model,
            "count": len(self.detections),
            "counts_by_label": self.count_by_label(),
            "detections": [d.to_dict() for d in self.detections],
            "classification": (
                [[c[0], round(float(c[1]), 4)] for c in self.classification]
                if self.classification
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameResult":
        return cls(
            detections=[Detection.from_dict(d) for d in data.get("detections", [])],
            task=data.get("task", "detection"),
            width=data.get("width", 0),
            height=data.get("height", 0),
            frame_index=data.get("frame_index", 0),
            inference_ms=data.get("inference_ms", 0.0),
            model=data.get("model"),
            classification=(
                [tuple(c) for c in data["classification"]]  # type: ignore[misc]
                if data.get("classification")
                else None
            ),
        )
