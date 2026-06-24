"""From-scratch mean Average Precision (mAP@0.5) for object detection.

Implements the standard VOC-style AP:
    * match predictions to ground truth greedily by descending score,
    * a prediction is a true positive if IoU with an unmatched GT of the same
      class >= ``iou_threshold``, else a false positive,
    * accumulate precision/recall, then integrate AP via the "all-points"
      (continuous) interpolation used by the modern VOC/COCO definition,
    * average AP over classes -> mAP.

No external metric libraries — only numpy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes -> ``(len(a), len(b))``."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def _ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    """All-points AP: area under the monotonically-decreasing PR envelope."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    # Make precision monotonically decreasing from the right.
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


class MeanAveragePrecision:
    """Accumulate per-image detections + GT, then compute mAP@``iou_threshold``.

    Predictions per image: ``(M, 6)`` rows ``[x1, y1, x2, y2, score, class]``.
    Ground truth per image: ``boxes (K, 4)`` xyxy and ``labels (K,)``.
    """

    def __init__(self, num_classes: int, iou_threshold: float = 0.5) -> None:
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        # Per class: list of (score, is_tp); plus a GT count.
        self._scores: Dict[int, List[float]] = {c: [] for c in range(num_classes)}
        self._tps: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
        self._n_gt: Dict[int, int] = {c: 0 for c in range(num_classes)}

    def update(
        self,
        pred: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: Sequence[int],
    ) -> None:
        gt_labels_arr = np.asarray(gt_labels)
        for c in range(self.num_classes):
            gt_mask = gt_labels_arr == c
            gt_c = gt_boxes[gt_mask]
            self._n_gt[c] += len(gt_c)

            pred_mask = pred[:, 5].astype(int) == c if len(pred) else np.zeros(0, dtype=bool)
            pred_c = pred[pred_mask] if len(pred) else np.zeros((0, 6), dtype=np.float32)
            if len(pred_c) == 0:
                continue
            order = np.argsort(-pred_c[:, 4])
            pred_c = pred_c[order]
            matched = np.zeros(len(gt_c), dtype=bool)
            ious = iou_matrix(pred_c[:, :4], gt_c[:, :4]) if len(gt_c) else None
            for i in range(len(pred_c)):
                self._scores[c].append(float(pred_c[i, 4]))
                if ious is None or len(gt_c) == 0:
                    self._tps[c].append(0)
                    continue
                j = int(np.argmax(ious[i]))
                if ious[i, j] >= self.iou_threshold and not matched[j]:
                    matched[j] = True
                    self._tps[c].append(1)
                else:
                    self._tps[c].append(0)

    def compute(self) -> Dict[str, Any]:
        """Return ``{'map': ..., 'ap_per_class': {c: ap}}``."""
        aps: Dict[int, float] = {}
        for c in range(self.num_classes):
            n_gt = self._n_gt[c]
            if n_gt == 0:
                continue
            if not self._scores[c]:
                aps[c] = 0.0
                continue
            scores = np.array(self._scores[c])
            tps = np.array(self._tps[c])
            order = np.argsort(-scores)
            tps = tps[order]
            fps = 1 - tps
            tp_cum = np.cumsum(tps)
            fp_cum = np.cumsum(fps)
            recall = tp_cum / (n_gt + 1e-9)
            precision = tp_cum / np.clip(tp_cum + fp_cum, 1e-9, None)
            aps[c] = _ap_from_pr(recall, precision)
        mean_ap = float(np.mean(list(aps.values()))) if aps else 0.0
        return {"map": mean_ap, "ap_per_class": aps}
