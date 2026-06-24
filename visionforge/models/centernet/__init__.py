"""From-scratch CenterNet anchor-free detector for vision-forge.

A hand-written CenterNet ("Objects as Points") implementation: a ResNet-like
backbone, a transposed-conv upsampling neck to stride-4, and center/wh/offset
heads, trained with a penalty-reduced Gaussian focal loss + L1 regression.
No ultralytics / huggingface / timm / detectron in the core model.

Submodules
----------
* ``model``       — backbone, neck, heads, the ``CenterNet`` module.
* ``losses``      — Gaussian-splat rendering + focal/L1 losses.
* ``postprocess`` — target encoding + top-k/NMS decode.
* ``dataset``     — procedural shapes dataset + Pascal VOC loader.
* ``metrics``     — from-scratch mAP@0.5.
* ``engine``      — device select, train loop, evaluate, checkpointing.
* ``infer``       — torch + onnxruntime inference backends (normalized schema).
* ``export``      — ONNX export of a trained checkpoint.

Heavy torch imports live inside the submodules, so importing this package is
*not* free of torch — only import it on the training / ML path. The light
(no-torch) CI path never imports it. The class-name constants below are
torch-free and safe to import anywhere.
"""

from __future__ import annotations

# Torch-free constants (safe to import without torch installed).
SHAPE_CLASSES = ["rectangle", "circle", "triangle"]
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

__all__ = ["SHAPE_CLASSES", "VOC_CLASSES"]
