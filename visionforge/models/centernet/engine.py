"""Training / evaluation engine for the from-scratch CenterNet.

Provides:
    * ``select_device`` — MPS > CUDA > CPU auto-selection.
    * ``collate`` — batch the dict samples from the datasets.
    * ``evaluate`` — run the model over a loader and compute mAP@0.5 with the
      from-scratch :class:`MeanAveragePrecision`.
    * ``train`` — AdamW + cosine-annealing LR, gradient clipping, periodic eval,
      best-checkpoint saving, and resumable checkpoints (model+optimizer+epoch).

Kept dependency-light (torch + numpy); no Lightning / accelerate.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from visionforge.models.centernet.losses import CenterNetLoss
from visionforge.models.centernet.metrics import MeanAveragePrecision
from visionforge.models.centernet.model import CenterNet, build_centernet
from visionforge.models.centernet.postprocess import decode_detections


def select_device(preferred: Optional[str] = None) -> torch.device:
    """Auto-select the best device: explicit > MPS > CUDA > CPU.

    MPS is preferred first per the task brief (Apple Silicon proof run), then
    CUDA, then CPU.
    """
    if preferred and preferred != "auto":
        return torch.device(preferred)
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack a list of sample dicts into a batch dict."""
    out: Dict[str, torch.Tensor] = {}
    for key in batch[0]:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out


@dataclass
class TrainConfig:
    """Hyper-parameters for a training run (serialised into checkpoints)."""

    variant: str = "lite"
    epochs: int = 20
    batch_size: int = 16
    lr: float = 2.5e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    grad_clip: float = 35.0
    num_workers: int = 0
    input_size: int = 256
    eval_score_threshold: float = 0.2
    eval_nms_iou: float = 0.5
    eval_topk: int = 100
    log_every: int = 20
    seed: int = 0
    history: List[dict] = field(default_factory=list)


@torch.no_grad()
def evaluate(
    model: CenterNet,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    *,
    score_threshold: float = 0.2,
    nms_iou: float = 0.5,
    topk: int = 100,
    down_ratio: int = 4,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Run inference over ``loader`` and return mAP@0.5 (+ per-class AP)."""
    model.eval()
    metric = MeanAveragePrecision(num_classes=num_classes, iou_threshold=0.5)
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        images = batch["input"].to(device)
        outputs = model(images)
        dets = decode_detections(
            outputs, k=topk, score_threshold=score_threshold, nms_iou=nms_iou, down_ratio=down_ratio
        )
        gt_boxes = batch["gt_boxes"].numpy()
        gt_labels = batch["gt_labels"].numpy()
        num_gt = batch["num_gt"].numpy()
        for i in range(len(dets)):
            ng = int(num_gt[i])
            metric.update(dets[i], gt_boxes[i, :ng], gt_labels[i, :ng])
    return metric.compute()


def _build_optimizer(model: CenterNet, cfg: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def _lr_at(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    """Linear warmup then cosine annealing to ~0."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + np.cos(np.pi * min(1.0, progress)))


def save_checkpoint(path: str, model: CenterNet, optimizer, epoch: int, cfg: TrainConfig, best_map: float) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_map": best_map,
            "num_classes": model.num_classes,
            "config": cfg.__dict__,
        },
        path,
    )


def save_slim_checkpoint(path: str, model: CenterNet, cfg: TrainConfig, best_map: float) -> None:
    """Save a deployment checkpoint WITHOUT optimizer state (smaller file).

    Used to produce a committable proof checkpoint under the repo's 25MB cap —
    dropping the AdamW state roughly halves the file size.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "num_classes": model.num_classes,
            "best_map": best_map,
            "config": cfg.__dict__,
        },
        path,
    )


def load_checkpoint(path: str, model: CenterNet, optimizer=None, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def train(
    train_ds: Dataset,
    val_ds: Dataset,
    num_classes: int,
    cfg: TrainConfig,
    *,
    out_dir: str = "runs/centernet",
    device: Optional[str] = None,
    resume: Optional[str] = None,
    eval_max_batches: Optional[int] = None,
    on_log=None,
) -> Dict[str, Any]:
    """Full training loop. Returns final/best metrics. Resumable via ``resume``.

    ``on_log`` is an optional callback ``(record: dict) -> None`` for streaming
    training logs to a file / stdout.
    """
    dev = select_device(device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = build_centernet(num_classes, variant=cfg.variant).to(dev)
    criterion = CenterNetLoss()
    optimizer = _build_optimizer(model, cfg)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate,
    )

    start_epoch = 0
    best_map = 0.0
    if resume and os.path.exists(resume):
        ckpt = load_checkpoint(resume, model, optimizer, map_location=dev)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_map = float(ckpt.get("best_map", 0.0))
        _log(on_log, {"event": "resume", "from": resume, "start_epoch": start_epoch, "best_map": best_map})

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = steps_per_epoch * cfg.warmup_epochs

    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best.pt")
    last_path = os.path.join(out_dir, "last.pt")
    final_metrics: Dict[str, Any] = {"map": best_map}

    global_step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        ep_start = time.time()
        running = {"loss": 0.0, "hm_loss": 0.0, "wh_loss": 0.0, "off_loss": 0.0}
        n_batches = 0
        for batch in train_loader:
            lr = _lr_at(global_step, total_steps, warmup_steps, cfg.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            images = batch["input"].to(dev)
            targets = {
                "hm": batch["hm"].to(dev),
                "wh": batch["wh"].to(dev),
                "offset": batch["offset"].to(dev),
                "ind": batch["ind"].to(dev),
                "reg_mask": batch["reg_mask"].to(dev),
            }
            outputs = model(images)
            loss, stats = criterion(outputs, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            for kk in running:
                running[kk] += stats[kk]
            n_batches += 1
            global_step += 1
            if n_batches % cfg.log_every == 0:
                _log(on_log, {
                    "event": "step", "epoch": epoch, "step": n_batches,
                    "lr": round(lr, 6), **{k: round(v / n_batches, 4) for k, v in running.items()},
                })

        avg = {k: v / max(1, n_batches) for k, v in running.items()}
        metrics = evaluate(
            model, val_loader, dev, num_classes,
            score_threshold=cfg.eval_score_threshold, nms_iou=cfg.eval_nms_iou,
            topk=cfg.eval_topk, down_ratio=model.down_ratio, max_batches=eval_max_batches,
        )
        ep_time = time.time() - ep_start
        record = {
            "event": "epoch", "epoch": epoch, "time_s": round(ep_time, 1),
            "train_loss": round(avg["loss"], 4), "hm_loss": round(avg["hm_loss"], 4),
            "wh_loss": round(avg["wh_loss"], 4), "off_loss": round(avg["off_loss"], 4),
            "map50": round(metrics["map"], 4),
        }
        cfg.history.append(record)
        _log(on_log, record)

        save_checkpoint(last_path, model, optimizer, epoch, cfg, best_map)
        if metrics["map"] >= best_map:
            best_map = metrics["map"]
            save_checkpoint(best_path, model, optimizer, epoch, cfg, best_map)
            # Also write a slim (optimizer-free) copy for committable deployment.
            save_slim_checkpoint(os.path.join(out_dir, "best_slim.pt"), model, cfg, best_map)
        final_metrics = {"map": metrics["map"], "best_map": best_map, **{f"ap_{k}": v for k, v in metrics["ap_per_class"].items()}}

    _log(on_log, {"event": "done", "best_map": best_map, "best_path": best_path})
    return final_metrics


def _log(cb, record: dict) -> None:
    if cb is not None:
        cb(record)
