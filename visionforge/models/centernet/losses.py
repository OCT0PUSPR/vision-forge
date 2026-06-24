"""CenterNet differentiable losses (from scratch, torch).

Implements, by hand:
    * ``neg_loss`` — the penalty-reduced pixelwise logistic focal loss on the
      heatmap (Eq. 1 of "Objects as Points"): positives use ``(1-p)^a log p``,
      negatives are down-weighted by ``(1-y)^b`` so pixels near a center are
      penalised less.
    * ``reg_l1_loss`` — masked L1 on the ``wh`` and ``offset`` regressions,
      gathered only at ground-truth center locations.

The Gaussian-splat target rendering (``gaussian_radius`` / ``draw_gaussian``)
lives in the torch-free ``targets`` module and is re-exported here for backward
compatibility. Everything operates on stride-4 (``DOWN_RATIO``) feature maps.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-export the torch-free target-rendering helpers.
from visionforge.models.centernet.targets import (  # noqa: F401
    _gaussian2d,
    draw_gaussian,
    gaussian_radius,
)


# --------------------------------------------------------------------------- #
# Differentiable losses (torch)
# --------------------------------------------------------------------------- #
def neg_loss(pred: torch.Tensor, gt: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """Penalty-reduced pixelwise logistic focal loss on a sigmoid heatmap.

    ``pred`` is post-sigmoid in ``[0, 1]`` with shape ``(N, C, H, W)``; ``gt`` is
    the rendered Gaussian target in the same shape. Positives are exactly the
    pixels where ``gt == 1`` (the object centers).
    """
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, beta)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss_ = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.sum()
    pos_sum = pos_loss.sum()
    neg_sum = neg_loss_.sum()

    if num_pos == 0:
        return -neg_sum
    return -(pos_sum + neg_sum) / num_pos


def _gather_feat(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """Gather rows of ``feat`` (N, H*W, C) at flat indices ``ind`` (N, K)."""
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    return feat.gather(1, ind)


def transpose_and_gather_feat(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """Reshape (N, C, H, W) -> (N, H*W, C) then gather at center indices ``ind``."""
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    return _gather_feat(feat, ind)


def reg_l1_loss(pred: torch.Tensor, mask: torch.Tensor, ind: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Masked L1 loss on a regression head, gathered at center indices.

    ``pred`` is ``(N, C, H, W)``; ``ind`` ``(N, K)`` flat center indices;
    ``target`` ``(N, K, C)`` the GT values; ``mask`` ``(N, K)`` 1 for real
    objects, 0 for padding. Averaged over the number of real objects.
    """
    pred = transpose_and_gather_feat(pred, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
    loss = loss / (mask.sum() + 1e-4)
    return loss


class CenterNetLoss(nn.Module):
    """Combined CenterNet loss: focal(hm) + ``wh_weight`` * L1(wh) + L1(offset)."""

    def __init__(self, hm_weight: float = 1.0, wh_weight: float = 0.1, off_weight: float = 1.0) -> None:
        super().__init__()
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.off_weight = off_weight

    def forward(self, outputs: dict, targets: dict) -> Tuple[torch.Tensor, dict]:
        hm = torch.sigmoid(outputs["hm"])
        hm_loss = neg_loss(hm, targets["hm"])
        wh_loss = reg_l1_loss(outputs["wh"], targets["reg_mask"], targets["ind"], targets["wh"])
        off_loss = reg_l1_loss(outputs["offset"], targets["reg_mask"], targets["ind"], targets["offset"])

        total = self.hm_weight * hm_loss + self.wh_weight * wh_loss + self.off_weight * off_loss
        stats = {
            "loss": float(total.detach().cpu()),
            "hm_loss": float(hm_loss.detach().cpu()),
            "wh_loss": float(wh_loss.detach().cpu()),
            "off_loss": float(off_loss.detach().cpu()),
        }
        return total, stats
