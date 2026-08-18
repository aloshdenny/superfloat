"""Scaling-law analysis for the SuperFloat precision study.

Fits the critical precision p0 -- the point at which a network stops training
at all -- as a function of width, and tests it against the prediction from
initialisation scale.

The prediction. A weight quantizes to exactly zero when |w| < Delta/2, with
Delta = 2^-(p-1). Kaiming init has sigma = sqrt(2/fan_in), so the precision at
which a fixed fraction of weights survives is

    p* = log2(1/sigma) + c = 0.5 * log2(fan_in) + c'

i.e. +0.5 bits per doubling of width, under ANY fixed survival criterion --
the criterion only moves the intercept c'.

Why a logistic fit rather than a threshold. Reading p0 off an accuracy
threshold makes the answer depend on the threshold: on this data the slope
ranges +0.08 to +0.37 across four reasonable choices, which is not a law but
an artefact of the analysis. Fitting

    acc(p) = A / (1 + exp(-k (p - p0)))

per width and taking the inflection p0 removes that freedom entirely.

Caveats this script prints rather than hides: k (transition sharpness) is not
constant across widths, so the logistic is a convenient shape and not a derived
one; and the slope error is the residual scatter about the line, not the much
smaller propagated curve-fit error, because with a handful of width points the
latter is meaninglessly tight.

    python analyze_scaling.py --results-dir <dir> --out figures/
"""

import argparse
import collections
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# shared with make_figures.py so the panels can sit side by side in the paper
COLOR = {"fp32": "#1f77b4", "fp16": "#17becf", "sf16": "#ff7f0e",
         "sf8": "#2ca02c", "sf4": "#d62728"}
WIDTH_COLOR = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
MIN_EPOCHS = 60          # anything shorter is a smoke run, not a result


def logistic(p, A, k, p0):
    return A / (1.0 + np.exp(-k * (p - p0)))


def _rows(results_dir, tier):
    """Tier records from `scaling_<tier>.jsonl` or a runs_scaling_<tier>/ dir."""
    arch = os.path.join(results_dir, f"scaling_{tier}.jsonl")
    if os.path.exists(arch):
        with open(arch) as f:
            return [json.loads(l) for l in f if l.strip()]
    out = []
    for fn in glob.glob(os.path.join(results_dir, f"runs_scaling_{tier}", "*.json")):
        if "_fp16head" in fn:
            continue
        out.append(json.load(open(fn)))
    return out


def load_c(results_dir):
    """Tier C records, keyed (width_mult, bits) -> list over seeds."""
    by = collections.defaultdict(list)
    for r in _rows(results_dir, "c"):
        # smoke runs share tags with real ones; length is the only honest check
        if len(r.get("history", [])) < MIN_EPOCHS:
            continue
        # head-quantization controls are a different condition and must not be
        # pooled into the main fit
        if r.get("quantize_head"):
            continue
        # nor may the channel-normalised arm.  It was added to tier C after
        # this analysis was written, and pooling it flattens the knee it is
        # trying to locate: normalised runs train at SF2, so the logistic sees
        # no collapse and the fitted slope drops from +0.29 to +0.14.
        if r.get("channel_norm"):
            continue
        by[(r["width_mult"], r["bits"])].append(r)
    return by


def fit_widths(by):
    """One logistic per width; returns [(log2 width, p0, stderr, A, k, n)]."""
    widths = sorted({w for w, _ in by})
    precs = sorted({b for _, b in by})
    out = []
    for w in widths:
        pts = [(b, x["best_acc"]) for b in precs for x in by.get((w, b), [])]
        if len({b for b, _ in pts}) < len(precs):
            print(f"  x{w}: incomplete ({len({b for b,_ in pts})}/{len(precs)} "
                  "precisions), skipped")
            continue
        x = np.array([b for b, _ in pts], float)
        y = np.array([v for _, v in pts], float)
        popt, pcov = curve_fit(logistic, x, y, p0=[y.max(), 2.0, 4.0],
                               maxfev=40000)
        err = float(np.sqrt(np.diag(pcov))[2])
        out.append((math.log2(w), popt[2], err, popt[0], popt[1], len(pts)))
    return out


def fit_slope(fits):
    """OLS of p0 on log2(width), with a residual-based standard error."""
    lx = np.array([f[0] for f in fits])
    ly = np.array([f[1] for f in fits])
    A = np.vstack([lx, np.ones_like(lx)]).T
    beta, *_ = np.linalg.lstsq(A, ly, rcond=None)
    resid = ly - A @ beta
    dof = max(len(lx) - 2, 1)
    s2 = float(resid @ resid) / dof
    se = math.sqrt(s2 * np.linalg.inv(A.T @ A)[0, 0])
    return beta[0], beta[1], se, resid


def figure(by, fits, slope, intercept, se, outdir):
    widths = sorted({w for w, _ in by})
    precs = sorted({b for _, b in by})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.8))

    for i, w in enumerate(widths):
        xs = [b for b in precs if by.get((w, b))]
        ys = [np.mean([x["best_acc"] for x in by[(w, b)]]) for b in xs]
        es = [np.ptp([x["best_acc"] for x in by[(w, b)]]) / 2 for b in xs]
        c = WIDTH_COLOR[i % len(WIDTH_COLOR)]
        ax1.errorbar(xs, ys, yerr=es, marker="o", ms=4, lw=1.7, capsize=2,
                     color=c, label=f"width x{w}")
        f = next((f for f in fits if abs(f[0] - math.log2(w)) < 1e-9), None)
        if f:
            g = np.linspace(min(precs), max(precs), 200)
            ax1.plot(g, logistic(g, f[3], f[4], f[1]), color=c, lw=1.0,
                     ls="--", alpha=0.6)
            ax1.axvline(f[1], color=c, lw=0.8, ls=":", alpha=0.7)
    ax1.set_xlabel("SuperFloat precision p (bits)")
    ax1.set_ylabel("best top-1 accuracy (%)")
    ax1.set_title("(a) collapse is sharp, and its location moves with width")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    lx = np.array([f[0] for f in fits])
    ly = np.array([f[1] for f in fits])
    ax2.errorbar(lx, ly, yerr=[f[2] for f in fits], marker="o", ms=6, lw=0,
                 elinewidth=1.5, capsize=3, color="#1f77b4", label="fitted p0")
    g = np.linspace(lx.min() - 0.3, lx.max() + 0.3, 50)
    ax2.plot(g, slope * g + intercept, color="#1f77b4", lw=1.8,
             label=f"measured {slope:+.3f} +/- {se:.3f} bits/doubling")
    ax2.plot(g, 0.5 * g + (ly.mean() - 0.5 * lx.mean()), color="#d62728",
             lw=1.8, ls="--",
             label="Kaiming prediction +0.500")
    ax2.set_xlabel("log2(width multiplier)")
    ax2.set_ylabel("critical precision p0 (bits)")
    ax2.set_title("(b) measured exponent is below the initialisation-scale prediction")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "scaling_critical_precision.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path



# ------------------------------------------------------- tiers A and B -----
def load_ab(results_dir, tier):
    """Tier A (QAT) or B (PTQ) records. bits=0 is that row's control."""
    return _rows(results_dir, tier)   # "_fp16head" coverage controls dropped there


def _penalty_table(rows, key_size, key_loss, step=None):
    """size -> {bits: penalty vs that size's own control}."""
    grp = collections.defaultdict(dict)
    for r in rows:
        if step is not None and r.get("step", 0) != step:
            continue
        grp[r[key_size]][r["bits"]] = r[key_loss]
    out = {}
    for size, d in grp.items():
        if 0 not in d:
            continue
        out[size] = {b: v - d[0] for b, v in d.items() if b}
    return out, {s: d[0] for s, d in grp.items() if 0 in d}


# a loss far above the uniform baseline is a diverged forward pass, not a
# measurement; ln(50304) = 10.83 for this vocabulary
RANDOM_LOSS = math.log(50304)
DIVERGED = RANDOM_LOSS * 1.2


def figure_ab(results_dir, outdir):
    a = load_ab(results_dir, "a")
    b = load_ab(results_dir, "b")
    if not a and not b:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 4.8))

    # (a) QAT
    pen_a, ctrl_a = _penalty_table(a, "size", "final_val_loss")
    order_a = sorted(pen_a, key=lambda s: next(
        r["n_nonembed"] for r in a if r["size"] == s))
    for i, s in enumerate(order_a):
        n = next(r["n_nonembed"] for r in a if r["size"] == s)
        xs = sorted(pen_a[s])
        axes[0].plot(xs, [pen_a[s][x] for x in xs], marker="o", ms=4, lw=1.7,
                     color=WIDTH_COLOR[i % len(WIDTH_COLOR)],
                     label=f"{s} ({n/1e6:.0f}M non-emb)")
    axes[0].axhline(0, color="k", lw=0.8, ls=":")
    axes[0].set_yscale("symlog", linthresh=0.01)
    axes[0].set_xlabel("SuperFloat precision p (bits)")
    axes[0].set_ylabel("val loss penalty vs FP32 (nats)")
    axes[0].set_title("(a) QAT from scratch: cost grows with N below SF6")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # (b) PTQ, N axis
    pen_b, ctrl_b = _penalty_table(b, "size", "val_loss", step=0)
    order_b = [s for s in ["70m", "160m", "410m", "1b", "1.4b", "2.8b",
                           "6.9b", "12b"] if s in pen_b]
    cmap = plt.get_cmap("viridis")
    for i, s in enumerate(order_b):
        xs = sorted(pen_b[s])
        ys = [min(pen_b[s][x], 20) for x in xs]
        axes[1].plot(xs, ys, marker="o", ms=4, lw=1.7,
                     color=cmap(i / max(len(order_b) - 1, 1)), label=s)
    axes[1].axhline(0, color="k", lw=0.8, ls=":")
    axes[1].set_yscale("symlog", linthresh=0.01)
    axes[1].set_xlabel("SuperFloat precision p (bits)")
    axes[1].set_ylabel("val loss penalty vs FP16 (nats)")
    axes[1].set_title("(b) PTQ across the Pythia ladder: threshold at SF8")
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].grid(alpha=0.3)

    # (c) D axis -- the same model, more tokens
    ck = collections.defaultdict(dict)
    for r in b:
        if r.get("step"):
            ck[(r["size"], r["step"])][r["bits"]] = r["val_loss"]
    sizes = sorted({s for s, _ in ck})
    for i, size in enumerate(sizes):
        steps = sorted({st for sz, st in ck if sz == size})
        xs, ys = [], []
        for st in steps:
            v = ck[(size, st)]
            if 0 in v and 6 in v:
                xs.append(st * 2097152 / 1e9)
                ys.append(v[6] - v[0])
        if xs:
            axes[2].plot(xs, ys, marker="o", ms=5, lw=1.8,
                         color=WIDTH_COLOR[i % len(WIDTH_COLOR)], label=size)
    axes[2].set_xlabel("training tokens (B)")
    axes[2].set_ylabel("SF6 penalty vs FP16 (nats)")
    axes[2].set_title("(c) PTQ damage grows with training tokens")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "scaling_qat_vs_ptq.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)

    # report cells that diverged rather than degraded
    bad = sorted({(r["size"], r["bits"], r.get("step", 0), round(r["val_loss"], 1))
                  for r in b if r.get("bits") and r["val_loss"] > DIVERGED})
    if bad:
        print(f"\n  {len(bad)} PTQ cells exceeded {DIVERGED:.1f} nats "
              f"(uniform = {RANDOM_LOSS:.2f}); these are diverged forward "
              "passes, not losses:")
        for s_, bits, st, v in bad[:8]:
            tag = f"@step{st}" if st else ""
            print(f"     {s_:<6} SF{bits:<3}{tag:<12} loss={v:.1f}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=".")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    by = load_c(args.results_dir)
    n = sum(len(v) for v in by.values())
    print(f"tier C: {n} complete runs over {len({w for w,_ in by})} widths")

    fits = fit_widths(by)
    if len(fits) < 3:
        print("need at least 3 complete widths to fit a slope")
        return
    print(f"\n  {'width':<9}{'pts':>5}{'plateau':>9}{'k':>7}{'p0':>7}{'+/-':>7}")
    for lx, p0, err, A, k, npts in fits:
        print(f"  x{2**lx:<8.2f}{npts:>5}{A:>9.1f}{k:>7.2f}{p0:>7.2f}{err:>7.3f}")

    slope, intercept, se, resid = fit_slope(fits)
    print(f"\n  p0 = {slope:+.3f} * log2(width) + {intercept:.3f}")
    print(f"  slope = {slope:+.3f} +/- {se:.3f} bits per width doubling")
    print(f"  Kaiming prediction +0.500 is {abs(slope-0.5)/se:.1f} sigma away")
    print(f"  residuals: {np.round(resid, 3)}")

    ks = [f[4] for f in fits]
    print(f"\n  caveat: transition sharpness k spans {min(ks):.2f}-{max(ks):.2f}, "
          "so the logistic is a fitted shape, not a derived one")

    print("\n  " + figure(by, fits, slope, intercept, se, args.out))
    p = figure_ab(args.results_dir, args.out)
    if p:
        print("  " + p)


if __name__ == "__main__":
    main()
