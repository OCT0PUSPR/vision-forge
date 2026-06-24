"""Export a trained CenterNet checkpoint to ONNX.

Produces a 3-output graph (``hm``, ``wh``, ``offset``) consumed by
:class:`~visionforge.models.centernet.infer.CenterNetOnnxBackend`. Requires
torch (run offline / on the training path); the resulting ``.onnx`` runs with
onnxruntime alone.
"""

from __future__ import annotations

import os
from typing import Optional


def export_centernet_onnx(
    checkpoint: str,
    out_path: Optional[str] = None,
    image_size: Optional[int] = None,
    opset: int = 12,
) -> str:
    """Export ``checkpoint`` to ONNX; return the written ``.onnx`` path."""
    import torch

    from visionforge.models.centernet.model import build_centernet

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    num_classes = int(ckpt["num_classes"])
    cfg = ckpt.get("config", {})
    variant = cfg.get("variant", "lite")
    size = image_size or cfg.get("input_size", 256)

    model = build_centernet(num_classes, variant=variant)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if out_path is None:
        out_path = os.path.splitext(checkpoint)[0] + ".onnx"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    dummy = torch.randn(1, 3, size, size)
    dynamic_axes = {
        "input": {0: "batch"},
        "hm": {0: "batch"},
        "wh": {0: "batch"},
        "offset": {0: "batch"},
    }
    # Force the legacy TorchScript exporter. In torch >= 2.9 the torch.export
    # (dynamo) exporter becomes the default, which pulls in ``onnxscript`` as a
    # hard dependency; pinning ``dynamo=False`` keeps export working with just
    # torch + onnx installed. The kwarg is absent on older torch, so we fall back
    # gracefully if it is not accepted.
    try:
        torch.onnx.export(
            model,
            (dummy,),
            out_path,
            input_names=["input"],
            output_names=["hm", "wh", "offset"],
            opset_version=opset,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    except TypeError:  # pragma: no cover - very old torch without the dynamo kwarg
        torch.onnx.export(
            model,
            (dummy,),
            out_path,
            input_names=["input"],
            output_names=["hm", "wh", "offset"],
            opset_version=opset,
            dynamic_axes=dynamic_axes,
        )
    return out_path
