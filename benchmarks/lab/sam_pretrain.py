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

import argparse, json, os, queue, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superfloat import apply_superfloat, clamp_all, disable_tf32, sf_params

OUT = Path(os.environ.get("DOMAIN_OUT", Path(__file__).resolve().parent.parent / "results" / "domain"))
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


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
    with open(lbl, encoding="utf-8", errors="replace") as f:
        for line in f:
            if len(line.split()) >= 7:
                return line.rstrip("\n")
    return None


def index_split(img_dir: Path, lbl_dir: Path, one_per_image=True):
    cache = Path("/tmp") / f"sf-idx-{img_dir.parent.name}-{img_dir.name}.json"
    if cache.exists():
        rows = json.load(open(cache))
        print(f"index cache {cache.name} {len(rows)}", flush=True)
        return [tuple(r) for r in rows]
    rows = []
    for ent in os.scandir(img_dir):
        name = ent.name
        if name.startswith("._") or not name.endswith(".jpg"):
            continue
        line = first_poly_line(lbl_dir / (name[:-4] + ".txt"))
        if line is None:
            continue
        rows.append((ent.path, line))
        if not one_per_image:
            extra = (lbl_dir / (name[:-4] + ".txt")).read_text().splitlines()[1:]
            for ex in extra:
                if len(ex.split()) >= 7:
                    rows.append((ent.path, ex))
    tmp = cache.with_suffix(".tmp")
    json.dump(rows, open(tmp, "w"))
    tmp.replace(cache)
    return rows


def load_np(row, sz):
    """PIL/numpy only — safe to run on worker threads (no torch)."""
    path, line = row
    with Image.open(path) as im:
        try:
            im.draft("RGB", (sz, sz))
        except Exception:
            pass
        im = im.convert("RGB")
        if im.size != (sz, sz):
            im = im.resize((sz, sz), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) * np.float32(1.0 / 255.0)
    coords = list(map(float, line.split()[1:]))
    mask = raster_poly(coords, sz, sz)
    if mask is None or int(mask.sum()) < 4:
        mask = np.zeros((sz, sz), dtype=np.uint8)
    ys, xs = np.nonzero(mask)
    prompt = np.zeros((sz, sz), dtype=np.float32)
    if xs.size == 0:
        prompt[:, :] = 1.0
    else:
        prompt[int(ys.min()):int(ys.max()) + 1, int(xs.min()):int(xs.max()) + 1] = 1.0
    return arr, prompt, mask.astype(np.float32, copy=False)


class StreamBoxDS(Dataset):
    def __init__(self, rows, size):
        self.rows = rows
        self.size = size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        arr, prompt, mask = load_np(self.rows[i], self.size)
        img_t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
        x = torch.cat([(img_t - _MEAN) / _STD, torch.from_numpy(prompt)[None]], 0)
        return x, torch.from_numpy(mask)[None]


class ThreadPrefetch:
    """Parallel JPEG decode on threads (numpy only; torch collate on the train thread)."""

    def __init__(self, rows, size, batch, device, shuffle, threads, depth=4, drop_last=True):
        self.rows = rows
        self.size = size
        self.batch = batch
        self.device = device
        self.shuffle = shuffle
        self.threads = max(1, threads)
        self.depth = depth
        self.drop_last = drop_last
        self.pool = ThreadPoolExecutor(max_workers=self.threads)

    def __iter__(self):
        n = len(self.rows)
        idx = np.arange(n)
        if self.shuffle:
            np.random.shuffle(idx)
        if self.drop_last:
            idx = idx[: n - (n % self.batch)]
        q: queue.Queue = queue.Queue(maxsize=self.depth)
        sentinel = object()
        sz, rows, pool = self.size, self.rows, self.pool

        def produce():
            try:
                for start in range(0, len(idx), self.batch):
                    sl = idx[start:start + self.batch]
                    futs = [pool.submit(load_np, rows[int(i)], sz) for i in sl]
                    arrs, prompts, masks = zip(*[f.result() for f in futs])
                    q.put((np.stack(arrs), np.stack(prompts), np.stack(masks)))
            finally:
                q.put(sentinel)

        threading.Thread(target=produce, daemon=True).start()
        mean = _MEAN.to(self.device)
        std = _STD.to(self.device)
        while True:
            item = q.get()
            if item is sentinel:
                break
            arrs, prompts, masks = item
            img = torch.from_numpy(np.ascontiguousarray(arrs)).permute(0, 3, 1, 2).to(self.device)
            pr = torch.from_numpy(np.ascontiguousarray(prompts)).unsqueeze(1).to(self.device)
            y = torch.from_numpy(np.ascontiguousarray(masks)).unsqueeze(1).to(self.device)
            x = torch.cat([(img - mean) / std, pr], 1)
            yield x, y


def dice_bce(logit, target):
    bce = F.binary_cross_entropy_with_logits(logit, target)
    p = torch.sigmoid(logit)
    num = 2 * (p * target).sum(dim=(1, 2, 3))
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
    return bce + (1 - num / den).mean()


@torch.no_grad()
def mean_iou(model, loader, max_batches=40):
    model.eval()
    ious = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
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


def result_tag(bits, fp16=False):
    if fp16:
        return "fp16"
    return "fp32" if not bits else f"sf{bits}"


def _dev_clear(device):
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def fit_batch(model, device, size, want, use_amp=False):
    """Largest batch that survives a train step. Climbs so a failed large alloc
    cannot fragment CUDA and poison smaller sizes."""
    import gc
    model.train()
    cands = [b for b in (2, 4, 8, 12, 16, 20, 24, 32) if b <= want]
    ok = 2
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    for b in cands:
        try:
            x = torch.zeros(b, 4, size, size, device=device)
            y = torch.zeros(b, 1, size, size, device=device)
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    logit = model(x)
                loss = dice_bce(logit.float(), y)
            else:
                loss = dice_bce(model(x), y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            del x, y, loss
            _dev_clear(device)
            print(f"batch_ok {b}", flush=True)
            ok = b
        except Exception as e:
            print(f"batch_fit skip b={b} ({type(e).__name__}: {e})", flush=True)
            gc.collect()
            _dev_clear(device)
            break
    return ok


def run(bits, coco_root, size, epochs, patience, batch, lr, seed, device, threads,
        max_val=2000, fp16=False):
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

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
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
    use_amp = fp16 and device.type in ("mps", "cuda")
    if device.type == "cuda" and not bits:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if batch <= 0:
        # 16 GB unified: 8 keeps paging off. 4 GB 3050: climb until OOM (16–32).
        want = 16 if device.type == "cuda" else 8
        batch = fit_batch(model, device, size, want=want, use_amp=use_amp)
    print(f"train_batch={batch} threads={threads} size={size} amp={use_amp}", flush=True)
    prefetch = 1 if device.type == "cuda" else 4
    tr_ld = ThreadPrefetch(tr_rows, size, batch, device, True, threads, depth=prefetch)
    va_ld = ThreadPrefetch(va_rows, size, batch, device, False, max(2, threads // 2),
                           depth=min(2, prefetch), drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    best, stale, t0 = -1.0, 0, time.time()
    hist = []
    log_every = max(batch * 25, 400)
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = torch.zeros((), device=device)
        nbat, nseen, t_ep = 0, 0, time.time()
        for x, y in tr_ld:
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    logit = model(x)
                loss = dice_bce(logit.float(), y)
            else:
                loss = dice_bce(model(x), y)
            loss.backward()
            opt.step()
            if bits:
                clamp_all(model)
            loss_sum = loss_sum + loss.detach()
            nbat += 1
            nseen += x.size(0)
            if nseen % log_every == 0:
                dt = max(time.time() - t_ep, 1e-6)
                print(f"  ep {ep}  {nseen}/{len(tr_rows)}  loss {(loss_sum / nbat).item():.4f}  "
                      f"{nseen / dt:.1f} img/s", flush=True)
        iou = mean_iou(model, va_ld)
        rec_ep = dict(epoch=ep, loss=float((loss_sum / max(nbat, 1)).item()), val_iou=iou,
                      epoch_min=(time.time() - t_ep) / 60, batch=batch,
                      img_s=nseen / max(time.time() - t_ep, 1e-6))
        hist.append(rec_ep)
        print(f"ep {ep:03d}  loss {rec_ep['loss']:.4f}  val_iou {iou:.3f}  "
              f"{rec_ep['epoch_min']:.1f}m  {rec_ep['img_s']:.1f} img/s", flush=True)
        if iou > best + 1e-4:
            best, stale = iou, 0
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop at epoch {ep}", flush=True)
                break
        _dump(bits, nq, nparam, device, tr_rows, va_rows, size, hist, best, t0, scale,
              batch, fp16, False)
        _save_weights(model, bits, best, ep, fp16)
    rec = _dump(bits, nq, nparam, device, tr_rows, va_rows, size, hist, best, t0, scale,
                batch, fp16, True)
    print(json.dumps({k: rec[k] for k in rec if k != "history"}), flush=True)
    _save_weights(model, bits, best, hist[-1]["epoch"] if hist else 0, fp16)
    return rec


def _save_weights(model, bits, best, epoch, fp16=False):
    tag = result_tag(bits, fp16)
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = OUT / f"sam_pretrain_{tag}.pt"
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "bits": bits, "best_val_iou": best, "epoch": epoch}, ckpt)


def _dump(bits, nq, nparam, device, tr_rows, va_rows, size, hist, best, t0, scale, batch,
          fp16, complete):
    tag = result_tag(bits, fp16)
    rec = dict(exp="sam_pretrain", model="BoxSegS", params_m=round(nparam, 2),
               bits=bits, fp16=fp16, quantized=nq, device=str(device), batch=batch,
               n_train=len(tr_rows), n_val=len(va_rows), size=size,
               epochs_ran=len(hist), best_val_iou=best,
               seconds=time.time() - t0, init_scale=scale, history=hist[-8:],
               complete=complete)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(OUT / f"sam_pretrain_{tag}.json", "w"), indent=2)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--coco", default="/Users/aoxo/sfx_data/coco")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch", type=int, default=0, help="0 = auto-fit largest power of two")
    ap.add_argument("--threads", type=int, default=0, help="0 = all CPU cores")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="", help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--fp16", action="store_true", help="IEEE fp16 autocast (bits must be 0)")
    a = ap.parse_args()
    if a.fp16 and a.bits:
        raise SystemExit("--fp16 requires --bits 0")
    if a.bits:
        disable_tf32()
    if a.device:
        device = torch.device(a.device)
    else:
        device = pick_device()
    if device.type == "cpu":
        raise SystemExit("refusing CPU: pass --device mps (or cuda)")
    threads = a.threads if a.threads > 0 else (os.cpu_count() or 8)
    # Intra-op stays 1 so JPEG workers own the cores; MPS is the compute.
    torch.set_num_threads(1)
    print(f"device={device} bits={a.bits} fp16={a.fp16} coco={a.coco} threads={threads}",
          flush=True)
    run(a.bits, a.coco, a.size, a.epochs, a.patience, a.batch, a.lr, a.seed, device, threads,
        fp16=a.fp16)


if __name__ == "__main__":
    main()
