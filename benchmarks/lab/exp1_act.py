"""Experiment 1: activation precision vs weight precision.

Everything measured so far is weights-only, but the Atreides systolic array
computes *in* SF, so activations flow through it in SF too. "SF4 weights work"
does not imply an SF4 chip works. This maps the (bits_w, bits_a) frontier.

Datapath modelled faithfully:

    array computes  (x / a) @ (w / s)

with x the activation entering the array and w the weight, both snapped to the
SF grid after their scales are divided out. Neither scale enters the array. The
product scale a*s is absorbed by the BatchNorm that follows every conv, exactly
as the weight scale alone was in the tier C result.

a is a per-tensor EMA of max|x| collected during training and frozen for
inference, so it is a constant in deployment -- not a per-batch dynamic max,
which would not be implementable cheaply in hardware.

    python exp1_act.py --bits-w 4 --bits-a 8 --seed 0
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from superfloat import disable_tf32, sf_params, sf_quantize_sv

DATA = "/workspace/cifar100"
OUT = "/workspace/results"
MEAN = (0.5071, 0.4865, 0.4409); STD = (0.2673, 0.2564, 0.2762)


class SFConv(nn.Conv2d):
    """Conv with independent weight and activation precision."""
    bits_w = 16; bits_a = 0; chan_norm = True
    def setup(self, bits_w, bits_a, chan_norm=True):
        self.bits_w, self.bits_a, self.chan_norm = bits_w, bits_a, chan_norm
        self.sw, self.vw = sf_params(bits_w) if bits_w else (0, 0)
        self.sa, self.va = sf_params(bits_a) if bits_a else (0, 0)
        self.register_buffer("a_scale", torch.ones(1))
        self.register_buffer("clip_n", torch.zeros((), dtype=torch.long))
        self.register_buffer("seen_n", torch.zeros((), dtype=torch.long))
        return self

    def forward(self, x):
        if self.bits_a:
            if self.training:
                m = x.detach().abs().amax()
                # EMA so the deployed scale is a constant, not a batch statistic
                self.a_scale.mul_(0.99).add_(0.01 * m.clamp_min(1e-8))
            xs = x / self.a_scale.clamp_min(1e-8)
            if not self.training:
                # mechanism metric: the analogue of dead_fraction for weights.
                # An activation floor is only interpretable if we know whether
                # it comes from clipping at the bound or from grid coarseness.
                with torch.no_grad():
                    self.clip_n += (xs.abs() > self.va).sum()
                    self.seen_n += xs.numel()
            x = sf_quantize_sv(xs, self.sa, self.va)
        w = self.weight
        if self.bits_w:
            if self.chan_norm:
                s = w.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
                w = sf_quantize_sv(w / s, self.sw, self.vw)
            else:
                w = sf_quantize_sv(w, self.sw, self.vw)
        return self._conv_forward(x, w, None)


def build(width_mult=1.0, classes=100, depth=20):
    n = (depth - 2) // 6
    w = [max(1, int(16 * width_mult)), max(1, int(32 * width_mult)),
         max(1, int(64 * width_mult))]

    class Block(nn.Module):
        def __init__(s, cin, cout, stride):
            super().__init__()
            s.c1 = SFConv(cin, cout, 3, stride, 1, bias=False)
            s.b1 = nn.BatchNorm2d(cout, eps=0.125)
            s.c2 = SFConv(cout, cout, 3, 1, 1, bias=False)
            s.b2 = nn.BatchNorm2d(cout, eps=0.125)
            s.sc = (nn.Sequential() if stride == 1 and cin == cout else
                    nn.Sequential(SFConv(cin, cout, 1, stride, bias=False),
                                  nn.BatchNorm2d(cout, eps=0.125)))
            s.r = nn.ReLU(inplace=True)
        def forward(s, x):
            o = s.r(s.b1(s.c1(x)))
            return s.r(s.b2(s.c2(o)) + s.sc(x))

    layers = [SFConv(3, w[0], 3, 1, 1, bias=False),
              nn.BatchNorm2d(w[0], eps=0.125), nn.ReLU(inplace=True)]
    cin = w[0]
    for stage, cout in enumerate(w):
        for blk in range(n):
            layers.append(Block(cin, cout, 2 if (stage and not blk) else 1)); cin = cout

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.features = nn.Sequential(*layers)
            s.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
            s.head = nn.Linear(cin, classes)     # head stays FP32
        def forward(s, x): return s.head(s.pool(s.features(x)))
    return Net()


def load_gpu(train):
    import torchvision
    # download=True re-verifies the 169MB tarball's MD5 on every call; with six
    # concurrent streams that is a synchronised GPU-idle window at each config
    # start. Fetch once if missing, then never re-check.
    import os as _os
    # The directory appears the moment the tarball starts extracting, so an
    # isdir() check lets a concurrently starting stream open a half-written
    # dataset and die with "Dataset not found or corrupted". Look for the
    # extracted files instead, and if the check still loses the race, wait for
    # the download that is already in flight rather than starting a second one.
    base = _os.path.join(DATA, "cifar-100-python")
    have = all(_os.path.isfile(_os.path.join(base, f))
               for f in ("train", "test", "meta"))
    try:
        ds = torchvision.datasets.CIFAR100(DATA, train=train, download=not have)
    except RuntimeError:
        time.sleep(90)
        ds = torchvision.datasets.CIFAR100(DATA, train=train, download=True)
    x = torch.from_numpy(np.asarray(ds.data)).permute(0, 3, 1, 2).contiguous()
    return x.cuda(), torch.tensor(ds.targets, dtype=torch.long).cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits-w", type=int, default=4)
    ap.add_argument("--bits-a", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag-prefix", default="exp1")
    ap.add_argument("--no-chan-norm", action="store_true",
                    help="plain SF: quantize weights as-is, no per-channel scale")
    a = ap.parse_args()

    disable_tf32(); torch.manual_seed(a.seed)
    tag = (f"{a.tag_prefix}_w{a.bits_w}_a{a.bits_a}_d{a.depth}_s{a.seed}"
           + ("_plain" if a.no_chan_norm else ""))
    if os.path.exists(f"{OUT}/{tag}.json"):
        print(f"[{tag}] already done, skipping", flush=True); return

    model = build(depth=a.depth).cuda()
    nconv = 0
    for m in model.features.modules():
        if isinstance(m, SFConv):
            m.setup(a.bits_w, a.bits_a, not a.no_chan_norm).cuda(); nconv += 1
    print(f"[{tag}] convs={nconv} bits_w={a.bits_w} bits_a={a.bits_a}", flush=True)

    xtr, ytr = load_gpu(True); xte, yte = load_gpu(False)
    mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)
    norm = lambda t: (t.float().div(255.0) - mean) / std

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * (len(xtr) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                total_steps=steps, pct_start=0.1)
    lf = nn.CrossEntropyLoss()
    hist, best, t0 = [], 0.0, time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(xtr), device="cuda")
        for i in range(0, len(xtr) - a.batch + 1, a.batch):
            idx = perm[i:i + a.batch]
            xb = xtr[idx]
            xb = F.pad(xb, (4, 4, 4, 4))
            oy, ox = np.random.randint(0, 9, 2)
            xb = xb[:, :, oy:oy + 32, ox:ox + 32]
            flip = torch.rand(len(idx), device="cuda") < 0.5
            xb[flip] = torch.flip(xb[flip], dims=[3])
            opt.zero_grad(set_to_none=True)
            lf(model(norm(xb)), ytr[idx]).backward()
            opt.step(); sched.step()
        model.eval(); c = t = 0
        with torch.no_grad():
            for i in range(0, len(xte), 500):
                out = model(norm(xte[i:i + 500]))
                c += (out.argmax(1) == yte[i:i + 500]).sum().item(); t += len(out)
        acc = 100.0 * c / t; best = max(best, acc)
        hist.append({"epoch": ep, "acc": acc})
        if ep % 15 == 0 or ep == a.epochs - 1:
            print(f"[{tag}] ep{ep} acc={acc:.2f} best={best:.2f} "
                  f"({(time.time()-t0)/60:.0f}m)", flush=True)
    clip, scales = [], []
    for m in model.features.modules():
        if isinstance(m, SFConv) and m.bits_a:
            clip.append(float(m.clip_n.item()) / max(int(m.seen_n.item()), 1))
            scales.append(float(m.a_scale.item()))
    rec = {"exp": a.tag_prefix, "chan_norm": not a.no_chan_norm,
           "bits_w": a.bits_w, "bits_a": a.bits_a,
           "clip_frac_mean": (sum(clip)/len(clip)) if clip else 0.0,
           "clip_frac_max": max(clip) if clip else 0.0,
           "a_scale_max": max(scales) if scales else 0.0,
           "a_scale_per_layer": scales, "clip_per_layer": clip,
           "depth": a.depth, "seed": a.seed, "best_acc": best,
           "final_acc": hist[-1]["acc"], "minutes": (time.time()-t0)/60,
           "history": hist, "complete": True}
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(f"{OUT}/{tag}.json", "w"))
    print(f"[{tag}] DONE best={best:.2f}", flush=True)


if __name__ == "__main__":
    main()
