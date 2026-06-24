#!/usr/bin/env python3
"""Turnkey cloud / Colab scale-up trainer for the from-scratch CenterNet.

Trains the CenterNet detector on the **full Pascal VOC** (or, with a flag, a
COCO-format directory) on a GPU runtime, then exports ONNX. Designed to be
pasted into a Colab cell or run on any CUDA box:

    # In Colab (after %cd into the cloned repo):
    !pip install -r requirements-train.txt
    !python scripts/scale_up_voc_colab.py --epochs 70 --variant r18 \
        --input-size 512 --batch-size 32 --out runs/centernet_voc_full

    # Full VOC 07+12 trainval is auto-downloaded by torchvision on first run.

Recommended recipe for real VOC numbers (single mid-range GPU, ~a few hours):
    * variant r18 or r34, input-size 512, batch-size 32, epochs 70-140,
    * lr 1.25e-4 * (batch_size / 16), cosine schedule (built in).

This script intentionally mirrors ``train_centernet.py`` but defaults to the
larger real-data recipe and always exports ONNX at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visionforge.models.centernet.dataset import VOC_CLASSES, VOCDetectionDataset  # noqa: E402
from visionforge.models.centernet.engine import TrainConfig, train  # noqa: E402
from visionforge.models.centernet.export import export_centernet_onnx  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Scale-up VOC CenterNet trainer (+ ONNX export).")
    p.add_argument("--variant", choices=["lite", "r18", "r34"], default="r18")
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--input-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--voc-root", default="data/voc")
    p.add_argument("--voc-year", default="2012", help="2007 | 2012")
    p.add_argument("--out", default="runs/centernet_voc_full")
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--no-export", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log_file = open(os.path.join(args.out, "train_log.jsonl"), "a")

    def on_log(record: dict) -> None:
        line = json.dumps(record)
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    train_ds = VOCDetectionDataset(
        root=args.voc_root, year=args.voc_year, image_set="train",
        input_size=args.input_size, download=True, augment=True,
    )
    val_ds = VOCDetectionDataset(
        root=args.voc_root, year=args.voc_year, image_set="val",
        input_size=args.input_size, download=True, augment=False,
    )
    cfg = TrainConfig(
        variant=args.variant, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, num_workers=args.num_workers, input_size=args.input_size,
    )
    on_log({"event": "start", "dataset": "voc-full", "num_classes": len(VOC_CLASSES),
            "train_size": len(train_ds), "val_size": len(val_ds), "config": cfg.__dict__})

    metrics = train(
        train_ds, val_ds, len(VOC_CLASSES), cfg,
        out_dir=args.out, device=args.device, resume=args.resume, on_log=on_log,
    )
    on_log({"event": "final", "metrics": metrics})
    log_file.close()

    best = os.path.join(args.out, "best.pt")
    if not args.no_export and os.path.exists(best):
        onnx_path = export_centernet_onnx(best, os.path.join(args.out, "best.onnx"), image_size=args.input_size)
        print(f"Exported ONNX to {onnx_path}")
    print(f"\nFinal VOC mAP@0.5: {metrics.get('map'):.4f} (best {metrics.get('best_map'):.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
