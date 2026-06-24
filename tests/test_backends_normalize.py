"""Tests for backend result-normalization logic using fake model outputs.

These exercise the conversion from raw model results into the normalized
``FrameResult`` schema without loading any real weights.
"""

import numpy as np

from visionforge.models.yolo_backend import YoloBackend


class _FakeArray:
    """Mimics the .cpu().numpy() chain of a torch tensor."""

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeBoxes:
    def __init__(self, xyxy, conf, cls, ids=None):
        self.xyxy = _FakeArray(xyxy)
        self.conf = _FakeArray(conf)
        self.cls = _FakeArray(cls)
        self.id = _FakeArray(ids) if ids is not None else None

    def __len__(self):
        return len(self.xyxy._arr)


class _FakeResult:
    def __init__(self, boxes, names, shape=(480, 640)):
        self.boxes = boxes
        self.names = names
        self.orig_shape = shape
        self.masks = None
        self.keypoints = None


def test_yolo_convert_detection():
    backend = YoloBackend(model_id="yolov8n.pt", task="detection")
    boxes = _FakeBoxes(
        xyxy=[[10, 20, 110, 220], [0, 0, 50, 50]],
        conf=[0.9, 0.5],
        cls=[0, 16],
    )
    result = _FakeResult(boxes, names={0: "person", 16: "dog"})
    fr = backend._convert(result, frame_index=3, elapsed_ms=12.5)
    assert fr.task == "detection"
    assert fr.width == 640 and fr.height == 480
    assert fr.frame_index == 3
    assert len(fr) == 2
    assert fr.detections[0].label == "person"
    assert fr.detections[1].label == "dog"
    assert fr.detections[0].bbox == (10.0, 20.0, 110.0, 220.0)


def test_yolo_convert_with_track_ids():
    backend = YoloBackend(model_id="yolov8n.pt", task="tracking")
    boxes = _FakeBoxes(
        xyxy=[[1, 2, 3, 4]],
        conf=[0.8],
        cls=[0],
        ids=[7],
    )
    result = _FakeResult(boxes, names={0: "person"})
    fr = backend._convert(result, frame_index=0, elapsed_ms=1.0)
    assert fr.detections[0].track_id == 7


def test_yolo_convert_empty():
    backend = YoloBackend(model_id="yolov8n.pt", task="detection")

    class _Empty:
        boxes = None
        masks = None
        keypoints = None
        names = {}
        orig_shape = (100, 100)

    fr = backend._convert(_Empty(), frame_index=0, elapsed_ms=0.5)
    assert len(fr) == 0
    assert fr.width == 100


def test_yolo_extract_keypoints():
    backend = YoloBackend(model_id="yolov8n-pose.pt", task="pose")

    class _KP:
        data = _FakeArray(np.array([[[10.0, 20.0, 0.9], [30.0, 40.0, 0.1]]]))

    kps = backend._extract_keypoints(_KP())
    assert len(kps) == 1
    assert len(kps[0]) == 2
    assert kps[0][0].x == 10.0
    assert kps[0][0].name == "nose"


def test_yolo_extract_masks():
    backend = YoloBackend(model_id="yolov8n-seg.pt", task="segmentation")

    class _Masks:
        xy = [np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])]

    masks = backend._extract_masks(_Masks())
    assert len(masks) == 1
    assert masks[0] == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
