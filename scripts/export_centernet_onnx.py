#!/usr/bin/env python3
"""Export a trained CenterNet checkpoint to ONNX.

Example
-------
    python scripts/export_centernet_onnx.py \
        --checkpoint runs/centernet_shapes/best.pt \
        --out weights/centernet_shapes.onnx
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visionforge.models.centernet.export import export_centernet_onnx  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Export CenterNet checkpoint to ONNX.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--opset", type=int, default=12)
    args = p.parse_args()

    out = export_centernet_onnx(
        checkpoint=args.checkpoint, out_path=args.out,
        image_size=args.image_size, opset=args.opset,
    )
    print(f"Exported ONNX graph to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
