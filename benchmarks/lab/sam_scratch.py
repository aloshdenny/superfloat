"""From-scratch SuperFloat QAT for unknown-object (box-prompt) segmentation.

Not SAM-ViT-B and not SA-1B. Those do not fit a 16 GB Mac or a $0.27/hr
3090. This is the capability question that PTQ never asked: can a randomly
initialized, class-agnostic masker learn to segment whatever is inside a box
when every conv/linear lives on the SFx grid?

Prompt = extra channel that is 1 inside the GT box, 0 outside. Target = that
instance's coco128-seg mask. Metric = mean IoU on a held-out 20% of images.
Head (1x1 conv to the mask logit) stays fp32, same recipe as YOLO/EuroSAT.
"""
from __future__ import annotations

import argparse, json, os, sys, time
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
        if len(line.split()) >= 7:
            return line
    return None


def index_rows(img_root: Path, lbl_root: Path):
    rows = []
    for img_p in sorted(img_root.rglob("*.jpg")):
        lbl = lbl_root / (img_p.stem + ".txt")
        if not lbl.exists():
            alt = lbl_root / img_p.parent.name / (img_p.stem + ".txt")
            lbl = alt if alt.exists() else None
        if lbl is None:
            continue
        line = first_poly_line(Path(lbl))
        if line:
            rows.append((str(img_p), line))
    return rows


class BoxMaskDS(Dataset):
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


def conv(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class TinyBoxSeg(nn.Module):
    """~1.1M conv params. Encoder+decoder on the SF grid; 1x1 head in fp32."""

    def __init__(self):
        super().__init__()
        self.stem = conv(4, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), conv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), conv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), conv(128, 128))
        self.up2 = conv(128 + 128, 64)
        self.up1 = conv(64 + 64, 32)
        self.up0 = conv(32 + 32, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        y = F.interpolate(s3, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up2(torch.cat([y, s2], 1))
        y = F.interpolate(y, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up1(torch.cat([y, s1], 1))
        y = F.interpolate(y, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up0(torch.cat([y, s0], 1))
        return self.head(y)


def dice_bce(logit, target):
    bce = F.binary_cross_entropy_with_logits(logit, target)
    p = torch.sigmoid(logit)
    num = 2 * (p * target).sum(dim=(1, 2, 3))
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
    return bce + (1 - num / den).mean()


@torch.no_grad()
def mean_iou(model, loader, device):
    model.eval()
    ious = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = (torch.sigmoid(model(x)) > 0.5)
        yt = y > 0.5
        inter = (pred & yt).flatten(1).sum(1).float()
        union = (pred | yt).flatten(1).sum(1).float().clamp_min(1)
        ious.extend((inter / union).cpu().tolist())
    return float(np.mean(ious)) if ious else 0.0


def init_scale_for(bits):
    if not bits:
        return 1.0
    step = 1.0 / (2 ** (bits - 1))
    # Kaiming |w|~0.02 sits under SF4's 0.125 step. Lift to ~4 grid steps.
    # BN after every conv renormalizes, so the forward is unchanged.
    return max(1.0, 4.0 * step / 0.02)


def run(bits, img_root, lbl_root, size, epochs, patience, batch, lr, seed, device):
    rng = np.random.RandomState(seed)
    rows = index_rows(Path(img_root), Path(lbl_root))
    if len(rows) < 20:
        raise SystemExit(f"only {len(rows)} instances under {img_root}")
    imgs = sorted({r[0] for r in rows})
    rng.shuffle(imgs)
    cut = max(1, int(0.2 * len(imgs)))
    va_set, tr_set = set(imgs[:cut]), set(imgs[cut:])
    tr = BoxMaskDS([r for r in rows if r[0] in tr_set], size)
    va = BoxMaskDS([r for r in rows if r[0] in va_set], size)
    tr_ld = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=0, drop_last=True)
    va_ld = DataLoader(va, batch_size=batch, shuffle=False, num_workers=0)

    torch.manual_seed(seed)
    model = TinyBoxSeg()
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
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, stale, t0 = -1.0, 0, time.time()
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = dice_bce(model(x), y)
            loss.backward()
            opt.step()
            if bits:
                clamp_all(model)
            losses.append(float(loss.detach().cpu()))
        iou = mean_iou(model, va_ld, device)
        rec_ep = dict(epoch=ep, loss=float(np.mean(losses)), val_iou=iou)
        hist.append(rec_ep)
        print(f"ep {ep:03d}  loss {rec_ep['loss']:.4f}  val_iou {iou:.3f}", flush=True)
        if iou > best + 1e-4:
            best, stale = iou, 0
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop at epoch {ep}", flush=True)
                break
    tag = "fp32" if not bits else f"sf{bits}"
    rec = dict(exp="sam_scratch", model="TinyBoxSeg", bits=bits, quantized=nq,
               device=str(device), n_train=len(tr), n_val=len(va),
               size=size, epochs_ran=len(hist), best_val_iou=best,
               seconds=time.time() - t0, init_scale=scale, history=hist[-8:])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"sam_scratch_{tag}.json"
    json.dump(rec, open(path, "w"), indent=2)
    print(json.dumps({k: rec[k] for k in rec if k != "history"}), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--images", default="/Users/aoxo/sfx_data/coco128-seg/images")
    ap.add_argument("--labels", default="/Users/aoxo/sfx_data/coco128-seg/labels")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="", help="cuda | mps | cpu (default: auto)")
    a = ap.parse_args()
    disable_tf32()
    device = torch.device(a.device) if a.device else pick_device()
    print(f"device={device} bits={a.bits}", flush=True)
    run(a.bits, a.images, a.labels, a.size, a.epochs, a.patience, a.batch, a.lr, a.seed, device)


if __name__ == "__main__":
    main()
