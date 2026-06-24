"""Datasets for the from-scratch CenterNet detector.

Two datasets, both yielding the dense CenterNet training tensors via
``build_targets``:

* :class:`ShapesDetectionDataset` — a **fully procedural**, local, fast
  shapes-detection dataset. Each image is a random background with a handful of
  geometric shapes (rectangle / circle / triangle); the shape class and tight
  bounding box are the labels. No downloads, deterministic per-seed, generated
  on the fly so it is essentially free to scale. This is the primary training
  target to prove the architecture + loss + loop.

* :class:`VOCDetectionDataset` — a thin wrapper over
  ``torchvision.datasets.VOCDetection`` (auto-downloads Pascal VOC) that maps
  VOC's 20 classes onto the same target format for a real-data demo.

Both produce a fixed-size square ``input_size`` RGB tensor and the encoded
targets at stride ``DOWN_RATIO``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from visionforge.models.centernet.targets import DOWN_RATIO, build_targets

SHAPE_CLASSES = ["rectangle", "circle", "triangle"]

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


# --------------------------------------------------------------------------- #
# Procedural shapes dataset
# --------------------------------------------------------------------------- #
def _draw_filled_polygon(img: np.ndarray, pts: np.ndarray, color: Sequence[int]) -> None:
    """Scanline-fill a convex polygon into ``img`` (numpy only, no cv2)."""
    pts = pts.astype(np.float32)
    ys = pts[:, 1]
    y0, y1 = int(np.floor(ys.min())), int(np.ceil(ys.max()))
    h, w = img.shape[:2]
    y0, y1 = max(0, y0), min(h - 1, y1)
    n = len(pts)
    for y in range(y0, y1 + 1):
        xs: List[float] = []
        for i in range(n):
            xa, ya = pts[i]
            xb, yb = pts[(i + 1) % n]
            if (ya <= y < yb) or (yb <= y < ya):
                t = (y - ya) / (yb - ya + 1e-9)
                xs.append(xa + t * (xb - xa))
        if len(xs) >= 2:
            xs.sort()
            xl, xr = int(np.floor(xs[0])), int(np.ceil(xs[-1]))
            xl, xr = max(0, xl), min(w - 1, xr)
            if xr >= xl:
                img[y, xl : xr + 1] = color


def _draw_circle(img: np.ndarray, cx: int, cy: int, r: int, color: Sequence[int]) -> None:
    h, w = img.shape[:2]
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    ys = np.arange(y0, y1)[:, None]
    xs = np.arange(x0, x1)[None, :]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
    region = img[y0:y1, x0:x1]
    region[mask] = color


def render_shapes_image(
    size: int,
    rng: np.random.Generator,
    max_objects: int = 6,
    min_objects: int = 1,
) -> Tuple[np.ndarray, List[List[float]], List[int]]:
    """Render one procedural image; return ``(rgb_uint8, boxes_xyxy, labels)``."""
    bg = rng.integers(20, 90, size=3)
    img = np.tile(bg.astype(np.uint8), (size, size, 1))
    # Light gradient so it is not a flat field.
    grad = np.linspace(0, 40, size, dtype=np.float32)
    img = np.clip(img.astype(np.float32) + grad[:, None, None], 0, 255).astype(np.uint8)

    n = int(rng.integers(min_objects, max_objects + 1))
    boxes: List[List[float]] = []
    labels: List[int] = []
    for _ in range(n):
        cls = int(rng.integers(0, len(SHAPE_CLASSES)))
        s = int(rng.integers(size // 10, size // 4))
        cx = int(rng.integers(s, size - s))
        cy = int(rng.integers(s, size - s))
        # Bright, saturated color distinct from background.
        color = rng.integers(120, 256, size=3).tolist()
        if cls == 0:  # rectangle
            half = s
            x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
            img[max(0, y1) : y2, max(0, x1) : x2] = color
        elif cls == 1:  # circle
            _draw_circle(img, cx, cy, s, color)
            x1, y1, x2, y2 = cx - s, cy - s, cx + s, cy + s
        else:  # triangle (equilateral-ish)
            pts = np.array([[cx, cy - s], [cx - s, cy + s], [cx + s, cy + s]], dtype=np.float32)
            _draw_filled_polygon(img, pts, color)
            x1, y1, x2, y2 = cx - s, cy - s, cx + s, cy + s
        boxes.append([float(x1), float(y1), float(x2), float(y2)])
        labels.append(cls)
    return img, boxes, labels


def _augment(
    img: np.ndarray,
    boxes: List[List[float]],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[List[float]]]:
    """Cheap detection augmentation: random horizontal flip + brightness jitter."""
    h, w = img.shape[:2]
    if rng.random() < 0.5:  # horizontal flip
        img = img[:, ::-1, :].copy()
        boxes = [[w - x2, y1, w - x1, y2] for (x1, y1, x2, y2) in boxes]
    if rng.random() < 0.5:  # brightness
        factor = float(rng.uniform(0.7, 1.3))
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return img, boxes


def _to_tensor_and_targets(
    img: np.ndarray,
    boxes: List[List[float]],
    labels: List[int],
    num_classes: int,
    input_size: int,
    down_ratio: int,
    max_objects: int,
) -> Dict[str, torch.Tensor]:
    """Normalise the image and build the dense targets at stride ``down_ratio``."""
    out_size = input_size // down_ratio
    # Boxes are in input pixels -> scale into output (stride-4) units.
    scaled = [[c / down_ratio for c in box] for box in boxes]
    targets = build_targets(
        scaled, labels, num_classes, out_size, out_size, max_objects=max_objects
    )
    tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
    # ImageNet-ish normalisation keeps activations centered.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return {
        "input": tensor,
        "hm": torch.from_numpy(targets["hm"]),
        "wh": torch.from_numpy(targets["wh"]),
        "offset": torch.from_numpy(targets["offset"]),
        "ind": torch.from_numpy(targets["ind"]),
        "reg_mask": torch.from_numpy(targets["reg_mask"]),
        # Raw GT for mAP eval (xyxy in input pixels + class), padded to max_objects.
        "gt_boxes": _pad_boxes(boxes, max_objects),
        "gt_labels": _pad_labels(labels, max_objects),
        "num_gt": torch.tensor(min(len(boxes), max_objects), dtype=torch.int64),
    }


def _pad_boxes(boxes: List[List[float]], max_objects: int) -> torch.Tensor:
    arr = np.zeros((max_objects, 4), dtype=np.float32)
    for i, b in enumerate(boxes[:max_objects]):
        arr[i] = b
    return torch.from_numpy(arr)


def _pad_labels(labels: List[int], max_objects: int) -> torch.Tensor:
    arr = np.full((max_objects,), -1, dtype=np.int64)
    for i, c in enumerate(labels[:max_objects]):
        arr[i] = c
    return torch.from_numpy(arr)


class ShapesDetectionDataset(Dataset):
    """Procedural shapes-detection dataset (rectangle / circle / triangle)."""

    classes = SHAPE_CLASSES

    def __init__(
        self,
        length: int = 2000,
        input_size: int = 256,
        max_objects: int = 6,
        seed: int = 0,
        augment: bool = True,
        down_ratio: int = DOWN_RATIO,
    ) -> None:
        self.length = length
        self.input_size = input_size
        self.max_objects = max_objects
        self.seed = seed
        self.augment = augment
        self.down_ratio = down_ratio
        self.num_classes = len(SHAPE_CLASSES)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Deterministic per (seed, idx) so runs are reproducible & resumable.
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        img, boxes, labels = render_shapes_image(
            self.input_size, rng, max_objects=self.max_objects
        )
        if self.augment:
            img, boxes = _augment(img, boxes, rng)
        return _to_tensor_and_targets(
            img, boxes, labels, self.num_classes, self.input_size, self.down_ratio, self.max_objects
        )


# --------------------------------------------------------------------------- #
# Pascal VOC dataset (real data; auto-download via torchvision)
# --------------------------------------------------------------------------- #
class VOCDetectionDataset(Dataset):
    """Pascal VOC detection via ``torchvision.datasets.VOCDetection``.

    Resizes each image to a square ``input_size`` (boxes scaled accordingly) and
    encodes the same dense CenterNet targets. Downloads VOC on first use.
    """

    classes = VOC_CLASSES

    def __init__(
        self,
        root: str = "data/voc",
        year: str = "2007",
        image_set: str = "train",
        input_size: int = 384,
        max_objects: int = 50,
        download: bool = True,
        augment: bool = True,
        down_ratio: int = DOWN_RATIO,
        subset: Optional[int] = None,
    ) -> None:
        from torchvision.datasets import VOCDetection

        self.input_size = input_size
        self.max_objects = max_objects
        self.augment = augment
        self.down_ratio = down_ratio
        self.num_classes = len(VOC_CLASSES)
        self._cls_to_idx = {c: i for i, c in enumerate(VOC_CLASSES)}
        self._voc = VOCDetection(root=root, year=year, image_set=image_set, download=download)
        self._subset = subset

    def __len__(self) -> int:
        if self._subset is not None:
            return min(self._subset, len(self._voc))
        return len(self._voc)

    def _parse(self, target: dict) -> Tuple[List[List[float]], List[int]]:
        objs = target["annotation"]["object"]
        if isinstance(objs, dict):
            objs = [objs]
        boxes: List[List[float]] = []
        labels: List[int] = []
        for o in objs:
            name = o["name"]
            if name not in self._cls_to_idx:
                continue
            b = o["bndbox"]
            boxes.append([float(b["xmin"]), float(b["ymin"]), float(b["xmax"]), float(b["ymax"])])
            labels.append(self._cls_to_idx[name])
        return boxes, labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        from PIL import Image

        img_pil, target = self._voc[idx]
        boxes, labels = self._parse(target)
        ow, oh = img_pil.size
        resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR", 2)
        img_pil = img_pil.resize((self.input_size, self.input_size), resample)
        img = np.asarray(img_pil.convert("RGB"))
        sx = self.input_size / ow
        sy = self.input_size / oh
        boxes = [[x1 * sx, y1 * sy, x2 * sx, y2 * sy] for (x1, y1, x2, y2) in boxes]

        rng = np.random.default_rng(idx)
        if self.augment:
            img, boxes = _augment(img, boxes, rng)
        return _to_tensor_and_targets(
            img, boxes, labels, self.num_classes, self.input_size, self.down_ratio, self.max_objects
        )
