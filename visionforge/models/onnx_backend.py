"""ONNX Runtime detection backend (torch-free deployment path).

Runs YOLOv8 detection ONNX graphs with ``onnxruntime`` only — no torch needed at
inference time. The ONNX file is produced once by ``export_yolo_onnx`` (which
does need ultralytics/torch, run offline), then deployed standalone.

Pre/post-processing (letterbox, NMS) is implemented in numpy so the runtime
footprint is just onnxruntime + numpy + opencv.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional, Tuple

from visionforge.core.schema import Detection, FrameResult

# COCO-80 class names (YOLOv8 default training set).
COCO_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def export_yolo_onnx(
    model_id: str = "yolov8n.pt",
    imgsz: int = 640,
    opset: int = 12,
    out_dir: Optional[str] = None,
) -> str:
    """Export an Ultralytics model to ONNX. Needs ultralytics/torch (offline).

    Returns the path to the generated ``.onnx`` file.
    """
    from ultralytics import YOLO  # type: ignore

    model = YOLO(model_id)
    path = model.export(format="onnx", imgsz=imgsz, opset=opset, dynamic=False)
    return str(path)


def _letterbox(image, new_shape: int = 640, color: int = 114):
    """Resize+pad an RGB array to a square ``new_shape`` keeping aspect ratio."""
    import numpy as np

    h, w = image.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    try:
        import cv2

        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    except Exception:  # pragma: no cover - cv2 fallback
        from PIL import Image

        resized = np.asarray(Image.fromarray(image).resize((nw, nh)))
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, scale, left, top


def _nms(boxes, scores, iou_threshold: float):
    """Pure-numpy non-maximum suppression. Returns kept indices."""
    import numpy as np

    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]
    return keep


class OnnxDetectionBackend:
    """YOLOv8 detection via ONNX Runtime (no torch at inference time)."""

    def __init__(
        self,
        onnx_path: str,
        device: str = "cpu",
        conf: float = 0.25,
        iou: float = 0.45,
        image_size: int = 640,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.onnx_path = onnx_path
        self.device = device
        self.conf = conf
        self.iou = iou
        self.image_size = image_size
        self.class_names = class_names or COCO_NAMES
        self.task = "detection"
        self._session: Any = None
        self._input_name: Optional[str] = None

    def load(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("The ONNX backend requires onnxruntime. Install with: pip install onnxruntime") from exc
        import os

        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(
                f"ONNX model not found: {self.onnx_path}. Export one with export_yolo_onnx() first."
            )
        providers = ["CPUExecutionProvider"]
        if self.device.startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(self.onnx_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def _preprocess(self, image):
        import numpy as np

        padded, scale, pad_left, pad_top = _letterbox(image, self.image_size)
        tensor = padded.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)[None]  # NCHW
        return np.ascontiguousarray(tensor), scale, pad_left, pad_top

    def _postprocess(
        self, output, scale: float, pad_left: int, pad_top: int, orig_shape: Tuple[int, int]
    ) -> List[Detection]:
        import numpy as np

        # YOLOv8 ONNX output: (1, 84, 8400) -> (8400, 84)
        preds = np.squeeze(output[0], axis=0)
        if preds.shape[0] < preds.shape[1]:
            preds = preds.transpose(1, 0)
        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores.max(axis=1)

        mask = confidences >= self.conf
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        if len(boxes_xywh) == 0:
            return []

        # xywh (center) -> xyxy, then undo letterbox.
        cx, cy, w, h = (boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3])
        x1 = (cx - w / 2 - pad_left) / scale
        y1 = (cy - h / 2 - pad_top) / scale
        x2 = (cx + w / 2 - pad_left) / scale
        y2 = (cy + h / 2 - pad_top) / scale
        xyxy = np.stack([x1, y1, x2, y2], axis=1)

        keep = _nms(xyxy, confidences, self.iou)
        oh, ow = orig_shape
        dets: List[Detection] = []
        for i in keep:
            cid = int(class_ids[i])
            label = self.class_names[cid] if cid < len(self.class_names) else str(cid)
            dets.append(
                Detection(
                    label=label,
                    confidence=float(confidences[i]),
                    bbox=(
                        float(max(0, x1[i])),
                        float(max(0, y1[i])),
                        float(min(ow, x2[i])),
                        float(min(oh, y2[i])),
                    ),
                    class_id=cid,
                )
            )
        return dets

    def predict(self, image, frame_index: int = 0) -> FrameResult:
        self.load()
        assert self._session is not None
        import numpy as np  # noqa: F401

        orig_h, orig_w = image.shape[:2]
        tensor, scale, pad_left, pad_top = self._preprocess(image)
        start = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: tensor})
        elapsed = (time.perf_counter() - start) * 1000.0
        dets = self._postprocess(outputs, scale, pad_left, pad_top, (orig_h, orig_w))
        return FrameResult(
            detections=dets,
            task="detection",
            width=orig_w,
            height=orig_h,
            frame_index=frame_index,
            inference_ms=elapsed,
            model=self.onnx_path,
        )

    def infer(self, image, frame_index: int = 0) -> FrameResult:
        return self.predict(image, frame_index=frame_index)
