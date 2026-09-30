"""QAT the decision policy at a given precision.

One arm per precision. fp32 is the control; every SF arm trains from the same
init and the same data in the same order, so an arm difference is the grid and
not the run.

The manoeuvre distribution is heavily skewed -- three quarters of an expert's
decisions are 'pursue', while 'lead' (take the shot) and 'recover' (do not fly
into the ground) are under 1% each. Unweighted training would score 76% by
predicting 'pursue' forever and would hide precisely the decisions that decide
a fight, so the choice loss is inverse-frequency weighted and evaluation
reports per-class accuracy alongside the aggregate.
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# superfloat.py lives at the benchmarks root in this repo, but the remote lab
# boxes keep it one level up from the script instead. Try both.
_here = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_here, "..", ".."), os.path.join(_here, ".."), _here):
    sys.path.insert(0, os.path.abspath(_p))
from superfloat import apply_superfloat, clamp_all, disable_tf32

from policy import Policy, N_MAN, param_count, set_residual_format, residual_format
from pilot import MANEUVERS

OUT = os.environ.get("DOGFIGHT_OUT", "runs")
HEADS = ("head_choice", "head_score", "head_noul")


def load(path):
    d = np.load(path, allow_pickle=True)
    return (torch.from_numpy(d["X"]), torch.from_numpy(d["Yc"]),
            torch.from_numpy(d["Ys"]), torch.from_numpy(d["Yn"]))


def class_weights(y, n, cap=25.0):
    cnt = torch.bincount(y, minlength=n).float().clamp_min(1.0)
    w = cnt.sum() / (n * cnt)
    return w.clamp(max=cap)


AMP = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


@torch.no_grad()
def evaluate(model, X, Yc, Ys, Yn, dev, bs=8192, amp=None):
    model.eval()
    correct = torch.zeros(N_MAN); total = torch.zeros(N_MAN)
    ok_s = 0; n = 0; probs = []; loss = 0.0
    for i in range(0, len(X), bs):
        x = X[i:i+bs].to(dev); yc = Yc[i:i+bs].to(dev)
        ys = Ys[i:i+bs].to(dev); yn = Yn[i:i+bs].to(dev)
        with torch.autocast(dev, dtype=amp, enabled=amp is not None):
            c, s, no = model(x)
        c, s, no = c.float(), s.float(), no.float()
        loss += F.cross_entropy(c, yc, reduction="sum").item()
        pred = c.argmax(-1)
        for k in range(N_MAN):
            m = yc == k
            total[k] += m.sum().item()
            correct[k] += (pred[m] == k).sum().item()
        ok_s += (s.argmax(-1) == ys).sum().item()
        probs.append(torch.sigmoid(no).cpu())
        n += len(x)
    p = torch.cat(probs)
    per = (correct / total.clamp_min(1)).tolist()
    return {
        "choice_acc": (correct.sum() / total.sum()).item(),
        "choice_balanced_acc": float(np.mean([per[k] for k in range(N_MAN) if total[k] > 0])),
        "per_class": {MANEUVERS[k]: (per[k] if total[k] > 0 else None) for k in range(N_MAN)},
        "class_support": {MANEUVERS[k]: int(total[k]) for k in range(N_MAN)},
        "score_acc": ok_s / n,
        "noul_brier": float(((p - Yn) ** 2).mean()),
        "noul_ece": ece(p, Yn),
        "choice_nll": loss / n,
    }


def ece(p, y, bins=15):
    """Expected calibration error. For a model sold on calibrated probabilities
    this is the number that matters, and nothing in this repo has measured what
    quantization does to it."""
    p = p.flatten(); y = y.flatten().float()
    edges = torch.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p <= edges[1])
        if m.sum() == 0:
            continue
        e += (m.float().mean() * (y[m].mean() - p[m].mean()).abs()).item()
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=0, help="0 = fp32 control")
    ap.add_argument("--train", default=os.path.join(OUT, "bc_s0_e600.npz"))
    ap.add_argument("--val", default=os.path.join(OUT, "bc_val.npz"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"],
                    help="reduced-precision baseline; autocast, not a pure half datapath")
    ap.add_argument("--res-int-bits", type=int, default=-1,
                    help="saturate the residual SUM at Q(n+1).(bits-1-n); "
                         "-1 leaves it unsaturated, which is what every other "
                         "study in this repo does")
    ap.add_argument("--quant-act", action="store_true",
                    help="also saturate layer outputs (the datapath question)")
    a = ap.parse_args()

    disable_tf32()
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Yc, Ys, Yn = load(a.train)
    Xv, Ycv, Ysv, Ynv = load(a.val)

    model = Policy(a.d, a.depth).to(dev)
    amp = AMP[a.dtype]
    base = f"sf{a.bits}" if a.bits else a.dtype
    tag = f"dog_{base}{'_act' if a.quant_act else ''}"
    if a.depth != 4:
        tag += f"_d{a.depth}"
    tag += f"_s{a.seed}"
    nconv = 0
    if a.bits:
        nconv = apply_superfloat(model, a.bits, head_names=HEADS,
                                 quantize_activations=a.quant_act)
    res_fmt = None
    if a.res_int_bits >= 0:
        bits = a.bits or 8
        sc, vm = set_residual_format(model, bits, a.res_int_bits)
        res_fmt = {"total_bits": bits, "int_bits": a.res_int_bits,
                   "step": 1.0 / sc, "vmax": vm}
        tag += "_r%d" % a.res_int_bits
    print(f"[{tag}] {param_count(model)/1e6:.2f}M params, {nconv} layers quantized, dev={dev}")

    w = class_weights(Yc, N_MAN).to(dev)
    pos = ((Yn == 0).sum() / (Yn == 1).sum().clamp_min(1)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * max(1, len(Xtr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)

    g = torch.Generator().manual_seed(a.seed)
    t0 = time.time(); step = 0
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr) - a.bs + 1, a.bs):
            idx = perm[i:i+a.bs]
            x = Xtr[idx].to(dev)
            with torch.autocast(dev, dtype=amp, enabled=amp is not None):
                c, s, no = model(x)
            c, s, no = c.float(), s.float(), no.float()
            loss = (F.cross_entropy(c, Yc[idx].to(dev), weight=w)
                    + 0.3 * F.cross_entropy(s, Ys[idx].to(dev))
                    + 0.3 * F.binary_cross_entropy_with_logits(
                        no, Yn[idx].to(dev), pos_weight=pos))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if a.bits:
                clamp_all(model)      # keep weights inside the SF range
            sched.step(); step += 1
        if (ep + 1) % 10 == 0 or ep == a.epochs - 1:
            m = evaluate(model, Xv, Ycv, Ysv, Ynv, dev, amp=amp)
            print(f"  ep{ep+1:3d} acc {m['choice_acc']:.4f} bal {m['choice_balanced_acc']:.4f} "
                  f"score {m['score_acc']:.4f} brier {m['noul_brier']:.5f} ece {m['noul_ece']:.5f}",
                  flush=True)

    m = evaluate(model, Xv, Ycv, Ysv, Ynv, dev, amp=amp)
    m.update(exp="dogfight_qat", bits=a.bits, dtype=a.dtype, res_fmt=res_fmt,
             res_int_bits=a.res_int_bits, quant_act=bool(a.quant_act), seed=a.seed,
             d=a.d, depth=a.depth, epochs=a.epochs, params=param_count(model),
             layers_quantized=nconv, minutes=(time.time()-t0)/60, tag=tag, complete=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(m, open(os.path.join(OUT, tag + ".json"), "w"), indent=1)
    torch.save({"model": model.state_dict(), "cfg": {"d": a.d, "depth": a.depth,
               "bits": a.bits, "quant_act": bool(a.quant_act), "dtype": a.dtype,
               "res_int_bits": a.res_int_bits}},
               os.path.join(OUT, tag + ".pt"))
    print(f"[{tag}] done in {m['minutes']:.1f}m -> acc {m['choice_acc']:.4f} "
          f"bal {m['choice_balanced_acc']:.4f} ece {m['noul_ece']:.5f}")


if __name__ == "__main__":
    main()
