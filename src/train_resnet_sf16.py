"""
train_resnet_sf16.py
====================
ResNet-18 trainer with Q1.15 (SF16) quantized forward pass on CIFAR-10.

Design contract
---------------
  FORWARD PASS  : every Conv2d and Linear layer operates on Q1.15-quantized
                  weights.  Activations and logits are also snapped to the
                  Q1.15 grid.
  BACKWARD PASS : STE (straight-through estimator) propagates gradients
                  through the quantization step to FP32 master weights.
  OPTIMIZER     : AdamW on FP32 master weights; after each step we snap
                  weights back to the Q1.15 grid.

Image pre-processing
--------------------
  CIFAR-10 images are normalized to ~[-1, 1] with standard mean/std, then
  explicitly snapped to the Q1.15 grid so every pixel is representable in
  Q1.15 format.

Outputs
-------
  logs/sf16_metrics.json   – per-epoch acc / loss
  checkpoints/sf16_best.pt – best validation checkpoint
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# ── local modules ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from sf16_quantizer import (
    quantize_images_q115,
    snap_weights_to_q115,
    weight_stats_q115,
    Q115_RESOLUTION,
)
from resnet_model import resnet18_sf16

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[SF16] %(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sf16")


# ── helpers ──────────────────────────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int, num_workers: int = 2,
                        data_root: str = "./data") -> tuple:
    """
    CIFAR-10 data loaders.

    Normalization uses ImageNet statistics scaled so that pixels land in
    approximately [-1, 1] – which is the natural Q1.15 input range.
    """
    MEAN = (0.4914, 0.4822, 0.4465)
    STD  = (0.2023, 0.1994, 0.2010)

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    val_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

    train_set = torchvision.datasets.CIFAR10(data_root, train=True,
                                              download=True,
                                              transform=train_tf)
    val_set   = torchvision.datasets.CIFAR10(data_root, train=False,
                                              download=True,
                                              transform=val_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size * 2,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, val_loader


def get_lr_scheduler(optimizer: optim.Optimizer, num_epochs: int,
                     warmup_epochs: int = 5) -> optim.lr_scheduler.LRScheduler:
    """Cosine annealing with linear warmup."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item())
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── training / validation loops ──────────────────────────────────────────────

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    snap_every_step: bool = True) -> dict:
    model.train()
    total_loss   = 0.0
    total_correct = 0
    total_samples = 0
    t_start = time.time()

    for step, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        # ── Q1.15 input quantization ─────────────────────────────────────
        # Images are already in ~[-1, 1] after normalization.
        # We snap them to the Q1.15 grid so the forward pass is purely SF16.
        images_q = quantize_images_q115(images)

        # ── forward ──────────────────────────────────────────────────────
        logits = model(images_q)
        loss   = criterion(logits, labels)

        # ── backward ─────────────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping (mirrors superfloat.cuda practice)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # ── snap weights back to Q1.15 grid ──────────────────────────────
        # This keeps the master weights on the Q1.15 representable lattice
        # so the next forward pass does not diverge from the stored values.
        if snap_every_step:
            snap_weights_to_q115(model)

        # ── bookkeeping ───────────────────────────────────────────────────
        batch_size    = labels.size(0)
        total_samples += batch_size
        total_loss    += loss.item() * batch_size
        preds          = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()

        if (step + 1) % 50 == 0:
            acc = total_correct / total_samples * 100
            log.info(f"Epoch {epoch}  step {step+1}/{len(loader)} | "
                     f"loss {total_loss/total_samples:.4f} | acc {acc:.2f}%")

    elapsed = time.time() - t_start
    return {
        "train_loss": total_loss  / total_samples,
        "train_acc":  total_correct / total_samples * 100,
        "epoch_time": elapsed,
    }


@torch.no_grad()
def validate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> dict:
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        images_q = quantize_images_q115(images)   # Q1.15 inputs
        logits   = model(images_q)
        loss     = criterion(logits, labels)

        batch_size     = labels.size(0)
        total_samples += batch_size
        total_loss    += loss.item() * batch_size
        preds          = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()

    return {
        "val_loss": total_loss  / total_samples,
        "val_acc":  total_correct / total_samples * 100,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SF16 (Q1.15) ResNet-18 trainer")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=0.01,
                        help="Peak learning rate")
    parser.add_argument("--weight_decay",type=float, default=1e-4)
    parser.add_argument("--warmup",      type=int,   default=5,
                        help="Warmup epochs")
    parser.add_argument("--num_classes", type=int,   default=10)
    parser.add_argument("--data_root",   type=str,   default="./data")
    parser.add_argument("--log_dir",     type=str,   default="./logs")
    parser.add_argument("--ckpt_dir",    type=str,   default="./checkpoints")
    parser.add_argument("--workers",     type=int,   default=2)
    parser.add_argument("--no_snap",     action="store_true",
                        help="Disable weight snapping every step")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    # ── reproducibility ───────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info(f"Using device: {device}")

    # ── directories ───────────────────────────────────────────────────────
    os.makedirs(args.log_dir,  exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_path  = os.path.join(args.log_dir,  "sf16_metrics.json")
    ckpt_path = os.path.join(args.ckpt_dir, "sf16_best.pt")

    # ── data ──────────────────────────────────────────────────────────────
    train_loader, val_loader = get_cifar10_loaders(
        args.batch_size, args.workers, args.data_root)
    log.info(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # ── model ─────────────────────────────────────────────────────────────
    model = resnet18_sf16(num_classes=args.num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"SF16 ResNet-18 | {n_params:,} parameters | "
             f"Q1.15 resolution = {Q115_RESOLUTION:.2e}")
    stats = weight_stats_q115(model)
    log.info(f"Initial weight stats: {stats}")

    # ── optimizer & scheduler ─────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(0.9, 0.999))
    scheduler = get_lr_scheduler(optimizer, args.epochs, args.warmup)
    criterion = nn.CrossEntropyLoss()

    # ── training loop ─────────────────────────────────────────────────────
    history = []
    best_val_acc = 0.0
    snap_every_step = not args.no_snap

    log.info(f"Starting SF16 training for {args.epochs} epochs")
    log.info(f"Weight snapping every step: {snap_every_step}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, snap_every_step=snap_every_step)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        w_stats    = weight_stats_q115(model)

        record = {
            "epoch":       epoch,
            "lr":          current_lr,
            **train_metrics,
            **val_metrics,
            "weight_stats": w_stats,
        }
        history.append(record)

        log.info(
            f"[Epoch {epoch:3d}/{args.epochs}] "
            f"train_loss={train_metrics['train_loss']:.4f}  "
            f"train_acc={train_metrics['train_acc']:.2f}%  "
            f"val_loss={val_metrics['val_loss']:.4f}  "
            f"val_acc={val_metrics['val_acc']:.2f}%  "
            f"lr={current_lr:.5f}  "
            f"sat={w_stats['saturated_frac']:.4f}"
        )

        # checkpoint
        if val_metrics["val_acc"] > best_val_acc:
            best_val_acc = val_metrics["val_acc"]
            torch.save({
                "epoch":         epoch,
                "model_state":   model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "best_val_acc":  best_val_acc,
                "args":          vars(args),
            }, ckpt_path)
            log.info(f"  ↑ New best val_acc={best_val_acc:.2f}% – saved to {ckpt_path}")

        # flush metrics
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

    log.info(f"Training complete.  Best val_acc = {best_val_acc:.2f}%")
    log.info(f"Metrics written to {log_path}")
    log.info(f"Best checkpoint  : {ckpt_path}")


if __name__ == "__main__":
    main()
