"""From-scratch CenterNet-style anchor-free detector (PyTorch).

This module implements the network architecture by hand — no ultralytics /
huggingface / timm / detectron. Only ``torch`` and ``torch.nn`` are used.

Architecture
------------
* **Backbone** — a ResNet-18-like stack of residual blocks built from scratch
  (``conv -> BN -> ReLU`` with identity / projection shortcuts). Produces a
  stride-32 feature map from a 3xHxW image.
* **Neck** — a transposed-convolution upsampling tower that lifts the stride-32
  features back to stride-4 (4x the spatial resolution of the input / 4).
* **Heads** — three ``3x3 conv -> ReLU -> 1x1 conv`` heads on the stride-4
  feature map:
    * ``heatmap`` (``num_classes`` channels) — per-class object-center heatmap,
      bias-initialised so the initial sigmoid output is small (~0.01), which
      stabilises the focal loss early in training.
    * ``wh`` (2 channels) — object width/height in *heatmap* (stride-4) units.
    * ``offset`` (2 channels) — sub-pixel center offset to undo integer
      quantisation of the center location.

The output stride is fixed at 4, matching the original CenterNet paper
("Objects as Points", Zhou et al. 2019). The decode step lives in
``postprocess.py``; the losses live in ``losses.py``.

Heavy ``torch`` import is deliberately top-level here because this module is
*only* imported on the training / torch path. The repo's light (no-torch) CI
path never imports it — see ``visionforge/models/centernet/__init__.py`` which
import-guards everything.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn

# Output stride of the stride-4 feature map the heads operate on.
DOWN_RATIO = 4


def _conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding (bias folded into the following BN)."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


def _conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution used for residual projection shortcuts."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """A ResNet basic residual block: two 3x3 convs + identity/projection skip."""

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_ch, out_ch, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # Projection shortcut when the shape changes (stride or channel count).
        self.downsample: nn.Module
        if stride != 1 or in_ch != out_ch * self.expansion:
            self.downsample = nn.Sequential(
                _conv1x1(in_ch, out_ch * self.expansion, stride),
                nn.BatchNorm2d(out_ch * self.expansion),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class ResNetBackbone(nn.Module):
    """ResNet-18-like backbone (4 stages of basic blocks), written from scratch.

    ``width`` scales every channel count, letting us shrink the network for a
    fast procedural-dataset proof run (``width=0.5`` => ResNet-18-lite) while
    keeping a full-width option for real datasets.
    """

    def __init__(self, layers: List[int] = [2, 2, 2, 2], width: float = 1.0) -> None:
        super().__init__()
        c = [int(round(ch * width)) for ch in (64, 64, 128, 256, 512)]
        self.out_channels = c[4]

        # Stem: stride-2 conv + stride-2 maxpool => stride-4 after the stem.
        self.stem = nn.Sequential(
            nn.Conv2d(3, c[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.in_ch = c[0]
        self.layer1 = self._make_stage(c[1], layers[0], stride=1)  # stride 4
        self.layer2 = self._make_stage(c[2], layers[1], stride=2)  # stride 8
        self.layer3 = self._make_stage(c[3], layers[2], stride=2)  # stride 16
        self.layer4 = self._make_stage(c[4], layers[3], stride=2)  # stride 32

        self._init_weights()

    def _make_stage(self, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        layers: List[nn.Module] = [BasicBlock(self.in_ch, out_ch, stride)]
        self.in_ch = out_ch * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_ch, out_ch))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class UpsampleNeck(nn.Module):
    """Transposed-conv upsampling tower: stride-32 features -> stride-4.

    Three ``ConvTranspose2d`` blocks each double the spatial resolution
    (32 -> 16 -> 8 -> 4), mirroring the CenterNet deconv head. Each block is
    ``3x3 conv (channel reduce) -> BN -> ReLU -> 4x4 deconv (x2) -> BN -> ReLU``.
    """

    def __init__(self, in_channels: int, channels: List[int] = [256, 128, 64]) -> None:
        super().__init__()
        self.out_channels = channels[-1]
        layers: List[nn.Module] = []
        ch = in_channels
        for out_ch in channels:
            layers.append(nn.Conv2d(ch, out_ch, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
            )
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            ch = out_ch
        self.deconv = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


class Head(nn.Module):
    """A CenterNet output head: 3x3 conv -> ReLU -> 1x1 conv to ``out_ch``."""

    def __init__(self, in_ch: int, out_ch: int, head_ch: int = 64, head_bias: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, head_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_ch, out_ch, kernel_size=1, bias=True),
        )
        # Initialise the final layer; heatmap heads use a strong negative bias
        # so the initial sigmoid prob is ~exp(head_bias), stabilising focal loss.
        final = self.conv[-1]
        assert isinstance(final, nn.Conv2d) and final.bias is not None
        nn.init.normal_(final.weight, std=0.001)
        nn.init.constant_(final.bias, head_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CenterNet(nn.Module):
    """The full from-scratch CenterNet detector.

    Forward returns a dict with raw (pre-activation) head tensors:
        * ``hm``     : (N, num_classes, H/4, W/4) center logits
        * ``wh``     : (N, 2, H/4, W/4) width/height in stride-4 units
        * ``offset`` : (N, 2, H/4, W/4) sub-pixel center offsets

    The heatmap is returned as logits; callers apply ``sigmoid`` (the focal loss
    in ``losses.py`` clamps the post-sigmoid values for numerical stability).
    """

    down_ratio = DOWN_RATIO

    def __init__(
        self,
        num_classes: int,
        backbone_layers: List[int] = [2, 2, 2, 2],
        width: float = 1.0,
        head_channels: int = 64,
        neck_channels: List[int] = [256, 128, 64],
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = ResNetBackbone(layers=backbone_layers, width=width)
        self.neck = UpsampleNeck(self.backbone.out_channels, channels=neck_channels)
        feat_ch = self.neck.out_channels
        # Prior-prob bias for the heatmap head (focal-loss stability): start the
        # sigmoid at ~0.01 so the network is not overwhelmed by background.
        prior = 0.01
        hm_bias = -math.log((1 - prior) / prior)
        self.hm = Head(feat_ch, num_classes, head_channels, head_bias=hm_bias)
        self.wh = Head(feat_ch, 2, head_channels, head_bias=0.0)
        self.offset = Head(feat_ch, 2, head_channels, head_bias=0.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.neck(self.backbone(x))
        return {
            "hm": self.hm(feat),
            "wh": self.wh(feat),
            "offset": self.offset(feat),
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_centernet(num_classes: int, variant: str = "lite") -> CenterNet:
    """Factory for named architecture variants.

    * ``lite``  — ResNet-18-lite (width 0.5), fast for the procedural proof run.
    * ``r18``   — full ResNet-18 width, for real datasets (VOC/COCO).
    * ``r34``   — deeper [3,4,6,3] stack for stronger real-data results.
    """
    if variant == "lite":
        return CenterNet(num_classes, backbone_layers=[2, 2, 2, 2], width=0.5, neck_channels=[128, 64, 64])
    if variant == "r18":
        return CenterNet(num_classes, backbone_layers=[2, 2, 2, 2], width=1.0, neck_channels=[256, 128, 64])
    if variant == "r34":
        return CenterNet(num_classes, backbone_layers=[3, 4, 6, 3], width=1.0, neck_channels=[256, 128, 64])
    raise ValueError(f"Unknown CenterNet variant: {variant!r}. Choose from lite|r18|r34.")
