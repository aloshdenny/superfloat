"""ConvNeXt-Tiny on EuroSAT under SuperFloat quantization-aware training.

Trained from scratch (no ImageNet init) with an open-ended schedule: warmup,
then ReduceLROnPlateau, and stop when validation accuracy has not improved for
`--patience` epochs. There is no fixed epoch target -- the run goes as far as
the format can carry it.
"""

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from superfloat import apply_superfloat, clamp_all, disable_tf32

# EuroSAT RGB channel statistics.
MEAN = (0.3444, 0.3803, 0.4078)
STD = (0.2027, 0.1369, 0.1156)


def build_loaders(root, img_size, batch_size, workers, seed):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(),
        # Vertical flips and 90-degree rotations are label-preserving for
        # nadir satellite imagery, unlike natural-image datasets.
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    full = datasets.ImageFolder(root)
    targets = np.array(full.targets)

    # Stratified 80/20 split with a fixed seed so every format sees identical data.
    rng = np.random.RandomState(seed)
    tr_idx, va_idx = [], []
    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        cut = int(0.8 * len(idx))
        tr_idx += idx[:cut].tolist()
        va_idx += idx[cut:].tolist()

    train_ds = Subset(datasets.ImageFolder(root, transform=train_tf), tr_idx)
    val_ds = Subset(datasets.ImageFolder(root, transform=eval_tf), va_idx)

    common = dict(num_workers=workers, pin_memory=False,
                  persistent_workers=workers > 0)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          drop_last=True, **common)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    return train_ld, val_ld, len(full.classes)


def run_epoch(model, loader, device, opt=None, scaler_clip=1.0,
              amp=False, scaler=None):
    train = opt is not None
    model.train(train)
    tot_loss, correct, seen = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    # Mixed precision is only for the fp16 baseline row. Under any SFx format
    # autocast would re-round the quantized grid to fp16's 10-bit mantissa,
    # which sits below SF16's 15 significand bits.
    dev_type = device.type if hasattr(device, "type") else str(device)
    with ctx:
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=dev_type, dtype=torch.float16,
                                enabled=amp):
                out = model(x)
                loss = F.cross_entropy(out, y,
                                       label_smoothing=0.1 if train else 0.0)
            if train:
                opt.zero_grad(set_to_none=True)
                if scaler is not None and amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), scaler_clip)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), scaler_clip)
                    opt.step()
                # Keep shadow weights and BN affine params representable.
                clamp_all(model)
            bs = y.size(0)
            tot_loss += loss.item() * bs
            correct += out.argmax(1).eq(y).sum().item()
            seen += bs
    return tot_loss / seen, 100.0 * correct / seen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--format", required=True,
                   choices=["sf16", "sf8", "sf4", "fp32", "fp16"])
    p.add_argument("--data", default=os.path.expanduser("~/sfx_data/eurosat/EuroSAT_RGB"))
    p.add_argument("--out", default="runs/eurosat")
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=4e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    disable_tf32()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    train_ld, val_ld, ncls = build_loaders(args.data, args.img_size,
                                           args.batch_size, args.workers, args.seed)

    model = timm.create_model("convnext_tiny", pretrained=False, num_classes=ncls)

    n_q = 0
    if args.format not in ("fp32", "fp16"):
        bits = int(args.format[2:])
        # "head" is ConvNeXt's classifier stack; the paper keeps output logits fp32.
        n_q = apply_superfloat(model, bits, head_names=("head",))
    model.to(device)

    if args.compile:
        model = torch.compile(model)

    nparams = sum(p.numel() for p in model.parameters())
    tag = f"convnext_tiny_{args.format}_s{args.seed}"
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, f"{tag}.csv")
    meta_path = os.path.join(args.out, f"{tag}.json")

    print(f"[{tag}] device={device} params={nparams/1e6:.2f}M quantized_layers={n_q} "
          f"classes={ncls} train={len(train_ld.dataset)} val={len(val_ld.dataset)}",
          flush=True)

    use_amp = args.format == "fp16"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    warm = torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, args.warmup)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=12, min_lr=1e-6)

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc",
                                "val_loss", "val_acc", "lr", "secs"])

    best_acc, best_ep, since = 0.0, 0, 0
    for ep in range(1, args.max_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_ld, device, opt,
                                    amp=use_amp, scaler=scaler)
        va_loss, va_acc = run_epoch(model, val_ld, device, amp=use_amp)
        lr_now = opt.param_groups[0]["lr"]
        if ep <= args.warmup:
            warm.step()
        else:
            plateau.step(va_acc)
        dt = time.time() - t0

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{tr_loss:.4f}", f"{tr_acc:.3f}",
                                    f"{va_loss:.4f}", f"{va_acc:.3f}",
                                    f"{lr_now:.2e}", f"{dt:.1f}"])

        if va_acc > best_acc:
            best_acc, best_ep, since = va_acc, ep, 0
        else:
            since += 1

        print(f"[{tag}] ep {ep:3d} | train {tr_loss:.3f}/{tr_acc:5.2f}% | "
              f"val {va_loss:.3f}/{va_acc:5.2f}% | best {best_acc:5.2f}%@{best_ep} | "
              f"lr {lr_now:.2e} | {dt:.0f}s", flush=True)

        json.dump({"tag": tag, "format": args.format, "model": "convnext_tiny",
                   "dataset": "eurosat", "params": nparams,
                   "quantized_layers": n_q, "epochs_run": ep,
                   "best_val_acc": best_acc, "best_epoch": best_ep,
                   "img_size": args.img_size, "seed": args.seed},
                  open(meta_path, "w"), indent=2)

        if since >= args.patience:
            print(f"[{tag}] early stop: no val improvement in {args.patience} epochs",
                  flush=True)
            break

    print(f"[{tag}] DONE best_val_acc={best_acc:.2f} @ epoch {best_ep} "
          f"(ran {ep} epochs)", flush=True)


if __name__ == "__main__":
    main()
