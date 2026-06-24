"""CenterNet decode: heatmap peak extraction -> boxes.

``decode`` / ``ctdet_decode`` turn raw network outputs into top-k boxes via a
3x3 max-pool "NMS" on the heatmap, top-k peak selection, and box reconstruction
from ``wh`` + ``offset``. An optional torchvision NMS pass removes residual
duplicates.

The torch-free dense target encoder ``build_targets`` lives in the ``targets``
module and is re-exported here for backward compatibility.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Re-export the torch-free target encoder + stride constant.
from visionforge.models.centernet.targets import DOWN_RATIO, build_targets  # noqa: F401


def _nms_pool(heat: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Keep only local maxima via a max-pool; suppresses adjacent duplicates."""
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores: torch.Tensor, k: int = 100) -> Tuple[torch.Tensor, ...]:
    """Top-k over a (N, C, H, W) heatmap -> per-image scores/classes/coords."""
    n, c, h, w = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(n, c, -1), k)
    topk_inds = topk_inds % (h * w)
    topk_ys = (topk_inds // w).float()
    topk_xs = (topk_inds % w).float()

    topk_score, topk_ind = torch.topk(topk_scores.view(n, -1), k)
    topk_classes = (topk_ind // k).int()
    topk_inds = topk_inds.view(n, -1).gather(1, topk_ind)
    topk_ys = topk_ys.view(n, -1).gather(1, topk_ind)
    topk_xs = topk_xs.view(n, -1).gather(1, topk_ind)
    return topk_score, topk_inds, topk_classes, topk_ys, topk_xs


def _gather(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    return feat.gather(1, ind)


def ctdet_decode(
    hm: torch.Tensor,
    wh: torch.Tensor,
    offset: torch.Tensor,
    k: int = 100,
) -> torch.Tensor:
    """Decode raw heads into ``(N, K, 6)`` = ``[x1, y1, x2, y2, score, class]``.

    Coordinates are in *stride-4 output* units; multiply by ``DOWN_RATIO`` to
    recover input-image pixels. ``hm`` must already be sigmoid-activated.
    """
    n = hm.size(0)
    hm = _nms_pool(hm)
    scores, inds, classes, ys, xs = _topk(hm, k=k)

    offset = offset.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
    offset = _gather(offset, inds)
    xs = xs.view(n, k, 1) + offset[:, :, 0:1]
    ys = ys.view(n, k, 1) + offset[:, :, 1:2]

    wh = wh.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
    wh = _gather(wh, inds)

    scores = scores.view(n, k, 1)
    classes = classes.view(n, k, 1).float()
    half_w = wh[..., 0:1] / 2
    half_h = wh[..., 1:2] / 2
    bboxes = torch.cat([xs - half_w, ys - half_h, xs + half_w, ys + half_h], dim=2)
    return torch.cat([bboxes, scores, classes], dim=2)


def decode_detections(
    outputs: dict,
    k: int = 100,
    score_threshold: float = 0.3,
    nms_iou: float = 0.5,
    down_ratio: int = DOWN_RATIO,
) -> List[np.ndarray]:
    """Full decode for a batch -> list (per image) of ``(M, 6)`` numpy arrays.

    Each row is ``[x1, y1, x2, y2, score, class]`` in *input-image* pixels.
    Applies score thresholding then a per-class torchvision NMS.
    """
    from torchvision.ops import batched_nms

    hm = torch.sigmoid(outputs["hm"])
    dets = ctdet_decode(hm, outputs["wh"], outputs["offset"], k=k)
    dets = dets.detach().cpu()
    dets[..., :4] *= down_ratio

    results: List[np.ndarray] = []
    for i in range(dets.size(0)):
        d = dets[i]
        keep_score = d[:, 4] >= score_threshold
        d = d[keep_score]
        if d.numel() == 0:
            results.append(np.zeros((0, 6), dtype=np.float32))
            continue
        boxes = d[:, :4]
        scores = d[:, 4]
        classes = d[:, 5].long()
        keep = batched_nms(boxes, scores, classes, nms_iou)
        results.append(d[keep].numpy().astype(np.float32))
    return results
