"""COCO-scale from-scratch pretrain for unknown-object (box-prompt) segmentation.

This is the largest SAM-like pretrain that fits a 16 GB Mac or a single 24 GB
pod. It is not SA-1B / SAM-ViT-B: that corpus is 11M images. It *is* the
standard substitute — class-agnostic box→mask on COCO train2017 (~118k
images), random init, SuperFloat QAT.

Dataset is streamed (no pre-rasterized masks). One instance per image so an
epoch is 118k steps/batch, not 800k. Encoder is BoxSeg-S (~18M convs), head
stays fp32. Paper ladder on this corpus is fp32 / SF16 / SF8 / SF4; the
SF2–SF16 probe stays on coco128 (`sam_scratch.py`).
"""
from __future__ import annotations

import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superfloat import apply_superfloat, clamp_all, disable_tf32, sf_params

OUT = Path(os.environ.get("DOMAIN_OUT", Path(__file__).resolve().parent.parent / "results" / "domain"))


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def conv(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class BoxSegS(nn.Module):
    """~18M-param UNet. SAM-task (box channel + RGB → mask), not SAM weights."""

    def __init__(self):
        super().__init__()
        self.stem = conv(4, 64)
        self.d1 = nn.Sequential(nn.MaxPool2d(2), conv(64, 128), conv(128, 128))
        self.d2 = nn.Sequential(nn.MaxPool2d(2), conv(128, 256), conv(256, 256))
        self.d3 = nn.Sequential(nn.MaxPool2d(2), conv(256, 384), conv(384, 384))
        self.d4 = nn.Sequential(nn.MaxPool2d(2), conv(384, 384))
        self.u3 = conv(384 + 384, 256)
        self.u2 = conv(256 + 256, 128)
        self.u1 = conv(128 + 128, 64)
        self.u0 = conv(64 + 64, 64)
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        s4 = self.d4(s3)
        y = F.interpolate(s4, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        y = self.u3(torch.cat([y, s3], 1))
        y = F.interpolate(y, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.u2(torch.cat([y, s2], 1))
        y = F.interpolate(y, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.u1(torch.cat([y, s1], 1))
        y = F.interpolate(y, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        y = self.u0(torch.cat([y, s0], 1))
        return self.head(y)


def raster_poly(coords, w, h):
    xs = [coords[i] * w for i in range(0, len(coords), 2)]
    ys = [coords[i] * h for i in range(1, len(coords), 2)]
    if len(xs) < 3:
        return None
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).polygon(list(zip(xs, ys)), fill=1)
    return np.array(im, dtype=np.uint8)


def first_poly_line(lbl: Path):
    if not lbl.exists():
        return None
    for line in lbl.read_text().splitlines():
        p = line.split()
        if len(p) >= 7:
            return line
    return None


def index_split(img_dir: Path, lbl_dir: Path, one_per_image=True):
    rows = []
    for img in sorted(img_dir.glob("*.jpg")):
        lbl = lbl_dir / (img.stem + ".txt")
        line = first_poly_line(lbl)
        if line is None:
            continue
        rows.append((str(img), line))
        if not one_per_image:
            for extra in lbl.read_text().splitlines()[1:]:
                p = extra.split()
                if len(p) >= 7:
                    rows.append((str(img), extra))
    return rows


class StreamBoxDS(Dataset):
    def __init__(self, rows, size):
        self.rows = rows
        self.size = size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, line = self.rows[i]
        img = Image.open(path).convert("RGB")
        w, h = img.size
        coords = list(map(float, line.split()[1:]))
        mask = raster_poly(coords, w, h)
        if mask is None or mask.sum() < 16:
            mask = np.zeros((h, w), dtype=np.uint8)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            box = [0, 0, w - 1, h - 1]
        else:
            box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        prompt = np.zeros((h, w), dtype=np.float32)
        x1, y1, x2, y2 = box
        prompt[y1:y2 + 1, x1:x2 + 1] = 1.0
        arr = np.array(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(arr).permute(2, 0, 1)
        pr_t = torch.from_numpy(prompt)[None]
        mk_t = torch.from_numpy(mask.astype(np.float32))[None]
        sz = (self.size, self.size)
        img_t = F.interpolate(img_t[None], size=sz, mode="bilinear", align_corners=False)[0]
        pr_t = F.interpolate(pr_t[None], size=sz, mode="nearest")[0]
        mk_t = F.interpolate(mk_t[None], size=sz, mode="nearest")[0]
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        x = torch.cat([(img_t - mean) / std, pr_t], 0)
        return x, mk_t


def dice_bce(logit, target):
    bce = F.binary_cross_entropy_with_logits(logit, target)
    p = torch.sigmoid(logit)
    num = 2 * (p * target).sum(dim=(1, 2, 3))
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
    return bce + (1 - num / den).mean()


@torch.no_grad()
def mean_iou(model, loader, device, max_batches=40):
    model.eval()
    ious = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        pred = torch.sigmoid(model(x)) > 0.5
        yt = y > 0.5
        inter = (pred & yt).flatten(1).sum(1).float()
        union = (pred | yt).flatten(1).sum(1).float().clamp_min(1)
        ious.extend((inter / union).cpu().tolist())
    return float(np.mean(ious)) if ious else 0.0


def init_scale_for(bits):
    if not bits:
        return 1.0
    step = 1.0 / (2 ** (bits - 1))
    return max(1.0, 4.0 * step / 0.02)


def run(bits, coco_root, size, epochs, patience, batch, lr, seed, device, max_val=2000):
    root = Path(coco_root)
    tr_img, tr_lbl = root / "images" / "train2017", root / "labels" / "train2017"
    va_img, va_lbl = root / "images" / "val2017", root / "labels" / "val2017"
    if not tr_img.exists():
        raise SystemExit(f"missing {tr_img} — download COCO first")
    print("indexing train…", flush=True)
    tr_rows = index_split(tr_img, tr_lbl)
    print(f"train instances (1/image) {len(tr_rows)}", flush=True)
    if va_img.exists():
        va_rows = index_split(va_img, va_lbl)[:max_val]
    else:
        va_rows = tr_rows[:max_val]
    tr_ld = DataLoader(StreamBoxDS(tr_rows, size), batch_size=batch, shuffle=True,
                       num_workers=0, drop_last=True)
    va_ld = DataLoader(StreamBoxDS(va_rows, size), batch_size=batch, shuffle=False,
                       num_workers=0)

    torch.manual_seed(seed)
    random.seed(seed)
    model = BoxSegS()
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    scale = init_scale_for(bits)
    if scale != 1.0:
        with torch.no_grad():
            n = 0
            for m in model.modules():
                if isinstance(m, nn.Conv2d) and m is not model.head:
                    m.weight.mul_(scale)
                    n += 1
        print(f"init_scale {scale:.2f}x on {n} convs", flush=True)
    nq = 0
    if bits:
        nq = apply_superfloat(model, bits, head_names=("head",), quantize_activations=True)
        clamp_all(model)
        print(f"SF{bits} quantized {nq} layers vmax={sf_params(bits)[1]:.6f}", flush=True)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    best, stale, t0 = -1.0, 0, time.time()
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        losses, nseen, t_ep = [], 0, time.time()
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = dice_bce(model(x), y)
            loss.backward()
            opt.step()
            if bits:
                clamp_all(model)
            losses.append(float(loss.detach().cpu()))
            nseen += x.size(0)
            if nseen % (batch * 200) == 0:
                print(f"  ep {ep}  {nseen}/{len(tr_rows)}  loss {np.mean(losses[-50:]):.4f}",
                      flush=True)
        iou = mean_iou(model, va_ld, device)
        rec_ep = dict(epoch=ep, loss=float(np.mean(losses)), val_iou=iou,
                      epoch_min=(time.time() - t_ep) / 60)
        hist.append(rec_ep)
        print(f"ep {ep:03d}  loss {rec_ep['loss']:.4f}  val_iou {iou:.3f}  "
              f"{rec_ep['epoch_min']:.1f}m", flush=True)
        if iou > best + 1e-4:
            best, stale = iou, 0
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop at epoch {ep}", flush=True)
                break
    tag = "fp32" if not bits else f"sf{bits}"
    rec = dict(exp="sam_pretrain", model="BoxSegS", params_m=round(nparam, 2),
               bits=bits, quantized=nq, device=str(device),
               n_train=len(tr_rows), n_val=len(va_rows), size=size,
               epochs_ran=len(hist), best_val_iou=best,
               seconds=time.time() - t0, init_scale=scale, history=hist)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"sam_pretrain_{tag}.json"
    json.dump(rec, open(path, "w"), indent=2)
    print(json.dumps({k: rec[k] for k in rec if k != "history"}), flush=True)
    ckpt = OUT / f"sam_pretrain_{tag}.pt"
    torch.save({"model": model.cpu().state_dict(), "bits": bits, "best_val_iou": best}, ckpt)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--coco", default="/Users/aoxo/sfx_data/coco")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    disable_tf32()
    device = pick_device()
    print(f"device={device} bits={a.bits} coco={a.coco}", flush=True)
    run(a.bits, a.coco, a.size, a.epochs, a.patience, a.batch, a.lr, a.seed, device)


if __name__ == "__main__":
    main()
