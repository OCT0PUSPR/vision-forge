"""Ultralytics YOLO backend.

Wraps ``ultralytics.YOLO`` and converts its native results into our normalized
``FrameResult`` / ``Detection`` schema. Supports detection, segmentation, pose
and tracking. Heavy imports (torch/ultralytics) happen lazily inside methods so
this module can be imported in the lightweight environment.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional

from visionforge.core.schema import Detection, FrameResult, Keypoint

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


class YoloBackend:
    """Lazy-loading wrapper around a single Ultralytics model."""

    SUPPORTED_TASKS = ("detection", "segmentation", "pose", "tracking")

    def __init__(
        self,
        model_id: str,
        task: str = "detection",
        device: str = "cpu",
        conf: float = 0.25,
        iou: float = 0.45,
        image_size: int = 640,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        self.model_id = model_id
        self.task = task
        self.device = device
        self.conf = conf
        self.iou = iou
        self.image_size = image_size
        self.tracker = tracker
        self._model: Optional[Any] = None

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """Instantiate the underlying model (auto-downloads weights once)."""
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "The YOLO backend requires the 'ultralytics' package. Install it with: pip install ultralytics"
            ) from exc
        self._model = YOLO(self.model_id)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    def predict(self, image, frame_index: int = 0) -> FrameResult:
        """Run a single-image forward pass and normalize the output."""
        self.load()
        assert self._model is not None
        start = time.perf_counter()
        results = self._model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        elapsed = (time.perf_counter() - start) * 1000.0
        return self._convert(results[0], frame_index, elapsed)

    def track(self, image, frame_index: int = 0, persist: bool = True) -> FrameResult:
        """Run tracking (ByteTrack) and normalize, preserving track ids."""
        self.load()
        assert self._model is not None
        start = time.perf_counter()
        results = self._model.track(
            source=image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            tracker=self.tracker,
            persist=persist,
            verbose=False,
        )
        elapsed = (time.perf_counter() - start) * 1000.0
        return self._convert(results[0], frame_index, elapsed)

    def infer(self, image, frame_index: int = 0) -> FrameResult:
        """Dispatch to ``track`` for the tracking task, else ``predict``."""
        if self.task == "tracking":
            return self.track(image, frame_index=frame_index)
        return self.predict(image, frame_index=frame_index)

    # ------------------------------------------------------------------ #
    # normalization
    # ------------------------------------------------------------------ #
    def _convert(self, result: Any, frame_index: int, elapsed_ms: float) -> FrameResult:
        """Translate one Ultralytics ``Results`` object into ``FrameResult``."""
        names = getattr(result, "names", {}) or {}
        height, width = 0, 0
        orig = getattr(result, "orig_shape", None)
        if orig is not None and len(orig) >= 2:
            height, width = int(orig[0]), int(orig[1])

        detections: List[Detection] = []
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        keypoints = getattr(result, "keypoints", None)

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            track_ids = None
            if getattr(boxes, "id", None) is not None:
                track_ids = boxes.id.cpu().numpy().astype(int)

            # Pre-extract per-detection mask polygons / keypoints.
            mask_polys = self._extract_masks(masks)
            kp_lists = self._extract_keypoints(keypoints)

            for i in range(len(xyxy)):
                cls_id = int(classes[i])
                label = names.get(cls_id, str(cls_id))
                det = Detection(
                    label=label,
                    confidence=float(confs[i]),
                    bbox=tuple(float(v) for v in xyxy[i]),  # type: ignore[arg-type]
                    class_id=cls_id,
                    mask=mask_polys[i] if i < len(mask_polys) else None,
                    keypoints=kp_lists[i] if i < len(kp_lists) else None,
                    track_id=int(track_ids[i]) if track_ids is not None else None,
                )
                detections.append(det)

        return FrameResult(
            detections=detections,
            task=self.task,
            width=width,
            height=height,
            frame_index=frame_index,
            inference_ms=elapsed_ms,
            model=self.model_id,
        )

    @staticmethod
    def _extract_masks(masks: Any) -> List[Optional[List[List[float]]]]:
        out: List[Optional[List[List[float]]]] = []
        if masks is None:
            return out
        xy = getattr(masks, "xy", None)
        if xy is None:
            return out
        for poly in xy:
            try:
                out.append([[float(p[0]), float(p[1])] for p in poly])
            except Exception:
                out.append(None)
        return out

    @staticmethod
    def _extract_keypoints(keypoints: Any) -> List[Optional[List[Keypoint]]]:
        out: List[Optional[List[Keypoint]]] = []
        if keypoints is None:
            return out
        data = getattr(keypoints, "data", None)
        if data is None:
            return out
        arr = data.cpu().numpy()
        for person in arr:
            kps: List[Keypoint] = []
            for j, kp in enumerate(person):
                x = float(kp[0])
                y = float(kp[1])
                conf = float(kp[2]) if len(kp) > 2 else 1.0
                name = COCO_KEYPOINT_NAMES[j] if j < len(COCO_KEYPOINT_NAMES) else None
                kps.append(Keypoint(x=x, y=y, confidence=conf, name=name))
            out.append(kps)
        return out
