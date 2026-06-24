#!/usr/bin/env python3
"""Train the from-scratch CenterNet detector.

Two datasets:
    * ``shapes`` (default) — fully procedural, local, fast. Reaches a strong
      mAP@0.5 quickly and proves the architecture + loss + loop are correct.
    * ``voc`` — Pascal VOC (auto-download via torchvision) for real-data numbers.

Examples
--------
    # Procedural proof run (under ~45 min on MPS):
    python scripts/train_centernet.py --dataset shapes --epochs 20 \
        --out runs/centernet_shapes

    # Real-data demo on a small VOC subset:
    python scripts/train_centernet.py --dataset voc --epochs 10 \
        --voc-subset 1500 --variant r18 --input-size 384 --out runs/centernet_voc

    # Resume:
    python scripts/train_centernet.py --dataset shapes --resume runs/centernet_shapes/last.pt

The full training log is streamed to stdout AND appended to ``<out>/train_log.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running as a standalone script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visionforge.models.centernet.engine import TrainConfig, train  # noqa: E402


def build_datasets(args):
    if args.dataset == "shapes":
        from visionforge.models.centernet.dataset import SHAPE_CLASSES, ShapesDetectionDataset

        train_ds = ShapesDetectionDataset(
            length=args.train_size, input_size=args.input_size, seed=args.seed, augment=True
        )
        val_ds = ShapesDetectionDataset(
            length=args.val_size, input_size=args.input_size, seed=args.seed + 99, augment=False
        )
        return train_ds, val_ds, len(SHAPE_CLASSES)

    if args.dataset == "voc":
        from visionforge.models.centernet.dataset import VOC_CLASSES, VOCDetectionDataset

        train_ds = VOCDetectionDataset(
            root=args.voc_root, year=args.voc_year, image_set="train",
            input_size=args.input_size, download=True, augment=True, subset=args.voc_subset,
        )
        val_ds = VOCDetectionDataset(
            root=args.voc_root, year=args.voc_year, image_set="val",
            input_size=args.input_size, download=True, augment=False,
            subset=(args.voc_subset // 4 if args.voc_subset else None),
        )
        return train_ds, val_ds, len(VOC_CLASSES)

    raise ValueError(f"Unknown dataset: {args.dataset}")


def main() -> int:
    p = argparse.ArgumentParser(description="Train from-scratch CenterNet.")
    p.add_argument("--dataset", choices=["shapes", "voc"], default="shapes")
    p.add_argument("--variant", choices=["lite", "r18", "r34"], default="lite")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--train-size", type=int, default=2000)
    p.add_argument("--val-size", type=int, default=400)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="auto|cpu|mps|cuda")
    p.add_argument("--eval-score-threshold", type=float, default=0.2,
                   help="Score threshold for eval decode (lower -> fuller PR curve).")
    p.add_argument("--out", default="runs/centernet")
    p.add_argument("--resume", default=None)
    p.add_argument("--eval-max-batches", type=int, default=None)
    # VOC-specific
    p.add_argument("--voc-root", default="data/voc")
    p.add_argument("--voc-year", default="2007")
    p.add_argument("--voc-subset", type=int, default=None)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train_log.jsonl")
    log_file = open(log_path, "a")

    def on_log(record: dict) -> None:
        line = json.dumps(record)
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    train_ds, val_ds, num_classes = build_datasets(args)
    cfg = TrainConfig(
        variant=args.variant, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, num_workers=args.num_workers, input_size=args.input_size, seed=args.seed,
        eval_score_threshold=args.eval_score_threshold,
    )
    on_log({"event": "start", "dataset": args.dataset, "num_classes": num_classes,
            "train_size": len(train_ds), "val_size": len(val_ds), "config": cfg.__dict__})

    metrics = train(
        train_ds, val_ds, num_classes, cfg,
        out_dir=args.out, device=args.device, resume=args.resume,
        eval_max_batches=args.eval_max_batches, on_log=on_log,
    )
    on_log({"event": "final", "metrics": metrics})
    log_file.close()
    print(f"\nFinal mAP@0.5: {metrics.get('map'):.4f}  (best {metrics.get('best_map'):.4f})")
    print(f"Best checkpoint: {os.path.join(args.out, 'best.pt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
