"""Tier C of the SuperFloat scaling-law study: critical precision vs width.

Tests the prediction that the precision at which a network collapses is set by
how the grid's zero-threshold compares to the initialisation scale:

    p* ~ log2(1/sigma_w) + 0.57,   sigma_w = sqrt(2/fan_in)  (Kaiming)
       ~ 0.5 * log2(fan_in) + 0.07

so p* should rise by half a bit for every doubling of width. ResNet-20 at width
multipliers 0.25 .. 4 spans 16x in fan_in, i.e. a predicted 2-bit shift in p*,
which a {2..8} precision grid resolves cleanly.

Depth, dataset, schedule and seed are held fixed; only width and precision
move. Every run also logs the fraction of conv weights sitting at exactly zero,
which is the mechanism the prediction is about -- if the law holds, collapse
coincides with that fraction going to ~1 at step 0.

GPU choice is measured, not assumed: modal_profile.py puts the L40S at 5,978
img/s for the widest variant at 0.9 GB peak, the best cost per image of the
five candidates (an A100-40GB is 1.8x worse value here, a B200 far worse).
CIFAR-100 is held on the GPU as a uint8 tensor with augmentation done on
device, so there is no dataloader in the loop and the probe's numbers hold.

Run:
    modal run modal/modal_scaling_c.py::prepare          # once
    modal run modal/modal_scaling_c.py --smoke           # one config, ~2 min
    modal deploy modal/modal_scaling_c.py                # then spawn the sweep
"""

import pathlib

import modal

BENCH_DIR = str(pathlib.Path(__file__).resolve().parent.parent)

app = modal.App("superfloat-scaling-c")
vol = modal.Volume.from_name("sfx-baselines", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0", "torchvision==0.23.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("numpy", "pandas")
    .add_local_dir(BENCH_DIR, remote_path="/root/sfx_bench")
)

GPU = "L40S"
OUT = "/vol/runs_scaling_c"
DATA = "/vol/datasets/cifar100"

WIDTHS = [0.25, 0.5, 1.0, 2.0, 4.0]
# denser at the low end, where the collapse happens; 16 anchors the ceiling
PRECISIONS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16]
SEEDS = [0, 1, 2]
EPOCHS = 60


# ------------------------------------------------------------------ model ---
def build_resnet(width_mult, num_classes=100):
    import torch.nn as nn

    w = [max(1, int(16 * width_mult)), max(1, int(32 * width_mult)),
         max(1, int(64 * width_mult))]

    class Block(nn.Module):
        def __init__(self, cin, cout, stride):
            super().__init__()
            self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
            self.b1 = nn.BatchNorm2d(cout, eps=0.125)
            self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
            self.b2 = nn.BatchNorm2d(cout, eps=0.125)
            self.sc = (nn.Sequential() if stride == 1 and cin == cout else
                       nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                     nn.BatchNorm2d(cout, eps=0.125)))
            self.r = nn.ReLU(inplace=True)

        def forward(self, x):
            o = self.r(self.b1(self.c1(x)))
            return self.r(self.b2(self.c2(o)) + self.sc(x))

    layers = [nn.Conv2d(3, w[0], 3, 1, 1, bias=False),
              nn.BatchNorm2d(w[0], eps=0.125), nn.ReLU(inplace=True)]
    cin = w[0]
    for stage, cout in enumerate(w):
        for blk in range(3):
            layers.append(Block(cin, cout, 2 if (stage and not blk) else 1))
            cin = cout

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(*layers)
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.head = nn.Linear(cin, num_classes)

        def forward(self, x):
            return self.head(self.pool(self.features(x)))

    return Net()


def dead_fraction(model):
    """Share of quantized conv/linear weights that sit at exactly zero.

    This is the quantity p* is a statement about: below p*, the grid's
    zero-threshold swallows most of the initialisation and there is no
    gradient signal to recover from.
    """
    import torch
    from superfloat import SFConv2d, SFLinear
    tot = zero = 0
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (SFConv2d, SFLinear)):
                from superfloat import sf_quantize_sv
                q = sf_quantize_sv(m.weight, m.sf_scale, m.sf_vmax)
                zero += (q == 0).sum().item()
                tot += q.numel()
    return zero / max(tot, 1)


# ------------------------------------------------------------------- data ---
@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 45)
def prepare():
    """Fetch CIFAR-100 into the shared volume once.

    torchvision's default fetch from toronto.edu is slow enough to blow a
    20-minute timeout partway through, so this pulls the tarball with curl
    (resumable, so a killed attempt continues rather than restarting) and lets
    torchvision do only the extract and integrity check.
    """
    import os
    import urllib.request
    import torchvision
    if os.path.isdir(f"{DATA}/cifar-100-python"):
        print("already present", flush=True)
        return
    os.makedirs(DATA, exist_ok=True)
    tar = f"{DATA}/cifar-100-python.tar.gz"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    TOTAL = 169001437
    for attempt in range(6):
        have = os.path.getsize(tar) if os.path.exists(tar) else 0
        if have >= TOTAL:
            break
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as r, \
                    open(tar, "ab" if have else "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception as exc:                          # noqa: BLE001
            print(f"  attempt {attempt}: {str(exc)[:90]} "
                  f"(have {os.path.getsize(tar)/1e6:.0f} MB)", flush=True)
    print(f"tarball {os.path.getsize(tar)/1e6:.0f} MB", flush=True)
    # download=True now finds a complete tarball, verifies its MD5 and extracts
    torchvision.datasets.CIFAR100(DATA, train=True, download=True)
    torchvision.datasets.CIFAR100(DATA, train=False, download=False)
    vol.commit()
    print("CIFAR-100 ready", flush=True)


def _load_gpu(train):
    """CIFAR-100 as uint8 GPU tensors; no dataloader in the training loop."""
    import numpy as np
    import torch
    import torchvision
    ds = torchvision.datasets.CIFAR100(DATA, train=train, download=False)
    x = torch.from_numpy(np.asarray(ds.data)).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(ds.targets, dtype=torch.long)
    return x.cuda(), y.cuda()


MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)


def _norm(xb, mean, std):
    return (xb.float().div_(255.0) - mean) / std


def _augment(xb):
    """Random crop (pad 4) + horizontal flip, done on device."""
    import torch
    import torch.nn.functional as F
    n = xb.shape[0]
    xb = F.pad(xb, (4, 4, 4, 4), mode="constant", value=0)
    i = torch.randint(0, 9, (2,), device=xb.device)
    xb = xb[:, :, i[0]:i[0] + 32, i[1]:i[1] + 32]
    flip = torch.rand(n, device=xb.device) < 0.5
    xb[flip] = torch.flip(xb[flip], dims=[3])
    return xb


# ------------------------------------------------------------------ train ---
@app.function(image=image, gpu=GPU, volumes={"/vol": vol},
              timeout=60 * 60 * 3, max_containers=10)
def train(width_mult: float, bits: int, seed: int, epochs: int = EPOCHS,
          lr: float = 1e-3, batch: int = 128):
    import json
    import os
    import sys
    import time
    import torch
    sys.path.insert(0, "/root/sfx_bench")
    from superfloat import disable_tf32, apply_superfloat, clamp_all

    disable_tf32()
    torch.manual_seed(seed)
    tag = f"w{width_mult}_sf{bits}_s{seed}"
    os.makedirs(OUT, exist_ok=True)

    xtr, ytr = _load_gpu(True)
    xte, yte = _load_gpu(False)
    mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)

    model = build_resnet(width_mult).cuda()
    nconv = apply_superfloat(model, bits=bits, head_names=("head",),
                             quantize_activations=False)
    nparams = sum(p.numel() for p in model.parameters())
    d0 = dead_fraction(model)

    # widest 3x3 conv in the net sets the prediction
    fan_in = max(m.weight[0].numel() for m in model.features.modules()
                 if hasattr(m, "weight") and m.weight.dim() == 4)
    import math
    sigma = math.sqrt(2.0 / fan_in)
    p_star = math.log2(1.0 / sigma) + 0.57

    print(f"[{tag}] params={nparams/1e6:.3f}M sf_layers={nconv} "
          f"fan_in={fan_in} sigma={sigma:.4f} p*={p_star:.2f} "
          f"dead@init={d0*100:.2f}%", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * (len(xtr) // batch),
        pct_start=0.1)
    lossf = torch.nn.CrossEntropyLoss()

    hist, best = [], 0.0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr), device="cuda")
        tl = n = 0
        for i in range(0, len(xtr) - batch + 1, batch):
            idx = perm[i:i + batch]
            xb = _norm(_augment(xtr[idx]), mean, std)
            yb = ytr[idx]
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            clamp_all(model)
            tl += loss.item() * len(idx)
            n += len(idx)

        model.eval()
        correct = vl = m = 0
        with torch.no_grad():
            for i in range(0, len(xte), 500):
                xb = _norm(xte[i:i + 500].clone(), mean, std)
                yb = yte[i:i + 500]
                out = model(xb)
                vl += lossf(out, yb).item() * len(yb)
                correct += (out.argmax(1) == yb).sum().item()
                m += len(yb)
        acc = 100.0 * correct / m
        best = max(best, acc)
        hist.append({"epoch": ep, "train_loss": tl / n, "val_loss": vl / m,
                     "val_acc": acc, "dead": dead_fraction(model)})
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"[{tag}] ep{ep} train={tl/n:.3f} val={vl/m:.3f} "
                  f"acc={acc:.2f} dead={hist[-1]['dead']*100:.1f}%", flush=True)

    rec = {"width_mult": width_mult, "bits": bits, "seed": seed,
           "params": nparams, "fan_in": fan_in, "sigma": sigma,
           "p_star_pred": p_star, "dead_at_init": d0,
           "best_acc": best, "final_acc": hist[-1]["val_acc"],
           "final_val_loss": hist[-1]["val_loss"],
           "minutes": (time.time() - t0) / 60, "history": hist}
    with open(f"{OUT}/{tag}.json", "w") as f:
        json.dump(rec, f)
    vol.commit()
    print(f"[{tag}] done best={best:.2f} in {rec['minutes']:.1f} min", flush=True)
    return {k: v for k, v in rec.items() if k != "history"}


@app.local_entrypoint()
def main(smoke: bool = False):
    if smoke:
        # one collapsed config and one healthy one, shortened
        for wm, b in ((4.0, 4), (4.0, 8)):
            print(train.remote(width_mult=wm, bits=b, seed=0, epochs=3))
        return
    jobs = [(w, b, s) for w in WIDTHS for b in PRECISIONS for s in SEEDS]
    print(f"spawning {len(jobs)} runs", flush=True)
    handles = [train.spawn(width_mult=w, bits=b, seed=s) for w, b, s in jobs]
    done = 0
    for h in handles:
        try:
            h.get()
        except Exception as exc:                          # noqa: BLE001
            print(f"  run failed: {str(exc)[:160]}", flush=True)
        done += 1
        if done % 10 == 0:
            print(f"  {done}/{len(handles)} complete", flush=True)
