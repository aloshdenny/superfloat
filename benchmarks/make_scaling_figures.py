"""Figure set for the SuperFloat precision scaling study.

Four tiers, one style, matching make_figures.py so the panels can sit beside
the evaluation figures in the paper.

  A  QAT language-model ladder, 5M-85M non-embedding params
  B  PTQ across Pythia 70M-12B, plus a data axis from intermediate checkpoints
  C  CNN width sweep, with and without training-time channel normalisation
  D  Transformer scale absorption into the norm that feeds each matmul

Every penalty is computed against a control trained in the same condition:
tier D's ln_full adds two norms per block and is a different architecture, so
comparing it to the plain baseline's FP32 would credit scale absorption with
an architecture gain worth 0.17-0.25 nats.

    python make_scaling_figures.py --results-dir <dir> --out figures/
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

WIDTH_COLOR = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
MODE_COLOR = {"none": "#d62728", "ln": "#ff7f0e", "ln_full": "#2ca02c"}
MODE_LABEL = {"none": "plain SF", "ln": "absorb into existing norms",
              "ln_full": "norm before every matmul"}
RANDOM_LOSS = math.log(50304)          # uniform baseline for this vocabulary
MIN_EPOCHS = 60


def load(results_dir, tier):
    """Rows for one tier, from `scaling_<tier>.jsonl` or a runs_scaling_<tier>/ dir."""
    arch = os.path.join(results_dir, f"scaling_{tier}.jsonl")
    if os.path.exists(arch):
        with open(arch) as f:
            return [json.loads(l) for l in f if l.strip()]
    out = []
    for f in glob.glob(os.path.join(results_dir, f"runs_scaling_{tier}", "*.json")):
        if "_fp16head" in f:
            continue
        out.append(json.load(open(f)))
    return out


def logistic(p, A, k, p0):
    return A / (1.0 + np.exp(-k * (p - p0)))


# ------------------------------------------------------------------ tier C --
def fig_width(rows, outdir):
    """Critical precision vs width, and what channel normalisation does to it."""
    plain, cnorm = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        if len(r.get("history", [])) < MIN_EPOCHS or r.get("quantize_head"):
            continue
        (cnorm if r.get("channel_norm") else plain)[
            (r["width_mult"], r["bits"])].append(r)
    W = sorted({w for w, _ in plain})
    P = sorted({b for _, b in plain})

    def mean(d, w, b, key="best_acc"):
        v = d.get((w, b))
        return float(np.mean([x[key] for x in v])) if v else None

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 4.8))
    for i, w in enumerate(W):
        c = WIDTH_COLOR[i % len(WIDTH_COLOR)]
        for d, ls, ax in ((plain, "-", axes[0]), (cnorm, "-", axes[1])):
            xs = [b for b in P if mean(d, w, b) is not None]
            if not xs:
                continue
            ax.plot(xs, [mean(d, w, b) for b in xs], marker="o", ms=4, lw=1.7,
                    ls=ls, color=c, label=f"x{w}")
    for ax, t in ((axes[0], "(a) plain SF: sharp collapse, moving with width"),
                  (axes[1], "(b) + channel norm: collapse gone at every width")):
        ax.set_xlabel("SuperFloat precision p (bits)")
        ax.set_ylabel("best top-1 accuracy (%)")
        ax.set_title(t, fontsize=10)
        ax.legend(fontsize=8, title="width")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 80)

    # p0 vs width, plain only -- under channel norm there is no knee to fit
    fits = []
    for w in W:
        pts = [(b, x["best_acc"]) for b in P for x in plain.get((w, b), [])]
        if len({b for b, _ in pts}) < len(P):
            continue
        x = np.array([b for b, _ in pts], float)
        y = np.array([v for _, v in pts], float)
        popt, _ = curve_fit(logistic, x, y, p0=[y.max(), 2.0, 4.0], maxfev=40000)
        fits.append((math.log2(w), popt[2]))
    if len(fits) >= 3:
        lx = np.array([f[0] for f in fits])
        ly = np.array([f[1] for f in fits])
        A = np.vstack([lx, np.ones_like(lx)]).T
        beta, *_ = np.linalg.lstsq(A, ly, rcond=None)
        axes[2].plot(lx, ly, "o", ms=7, color="#1f77b4", label="fitted p0 (plain SF)")
        g = np.linspace(lx.min() - 0.3, lx.max() + 0.3, 50)
        axes[2].plot(g, beta[0] * g + beta[1], color="#1f77b4", lw=1.8,
                     label=f"measured {beta[0]:+.2f} bits/doubling")
        axes[2].plot(g, 0.5 * g + (ly.mean() - 0.5 * lx.mean()), color="#d62728",
                     ls="--", lw=1.8, label="Kaiming prediction +0.50")
    axes[2].set_xlabel("log2(width multiplier)")
    axes[2].set_ylabel("critical precision p0 (bits)")
    axes[2].set_title("(c) the width law holds only without normalisation",
                      fontsize=10)
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    p = os.path.join(outdir, "scaling_c_width.png")
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p


# ------------------------------------------------------------ tiers A and B --
def fig_lm(a_rows, b_rows, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 4.8))

    # (a) QAT penalty vs precision, one line per size
    g = collections.defaultdict(dict)
    N = {}
    for r in a_rows:
        if r.get("seed", 0):
            continue                      # seed replicates go in their own panel
        g[r["size"]][r["bits"]] = r["final_val_loss"]
        N[r["size"]] = r["n_nonembed"]
    order = sorted(g, key=lambda s: N[s])
    for i, s in enumerate(order):
        d = g[s]
        if 0 not in d:
            continue
        xs = sorted(b for b in d if b)
        axes[0].plot(xs, [d[b] - d[0] for b in xs], marker="o", ms=4, lw=1.7,
                     color=WIDTH_COLOR[i % len(WIDTH_COLOR)],
                     label=f"{s} ({N[s]/1e6:.0f}M)")
    axes[0].axhline(0, color="k", lw=0.8, ls=":")
    axes[0].set_yscale("symlog", linthresh=0.01)
    axes[0].set_xlabel("SuperFloat precision p (bits)")
    axes[0].set_ylabel("val loss penalty vs FP32 (nats)")
    axes[0].set_title("(a) QAT: cost below SF6 grows with model size", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # (b) PTQ penalty vs precision across the Pythia ladder
    gb = collections.defaultdict(dict)
    for r in b_rows:
        if r.get("step", 0):
            continue
        gb[r["size"]][r["bits"]] = r["val_loss"]
    ladder = [s for s in ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b",
                          "12b"] if s in gb]
    cmap = plt.get_cmap("viridis")
    for i, s in enumerate(ladder):
        d = gb[s]
        if 0 not in d:
            continue
        xs = sorted(b for b in d if b)
        axes[1].plot(xs, [min(d[b] - d[0], 20) for b in xs], marker="o", ms=4,
                     lw=1.7, color=cmap(i / max(len(ladder) - 1, 1)), label=s)
    axes[1].axhline(0, color="k", lw=0.8, ls=":")
    axes[1].set_yscale("symlog", linthresh=0.01)
    axes[1].set_xlabel("SuperFloat precision p (bits)")
    axes[1].set_ylabel("val loss penalty vs FP16 (nats)")
    axes[1].set_title("(b) PTQ: threshold at SF8, two bits worse than QAT",
                      fontsize=10)
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3)

    # (c) the data axis -- same model, more tokens.  Sampling only the last
    # four checkpoints reads as monotone growth; the full sweep from 2.1B
    # tokens shows that rise is the right-hand half of a U.
    ck = collections.defaultdict(dict)
    for r in b_rows:
        if r.get("step"):
            ck[(r["size"], r["step"])][r["bits"]] = r["val_loss"]
    sizes = [s for s in ("160m", "410m", "1.4b") if any(k[0] == s for k in ck)]
    for i, size in enumerate(sizes):
        for bits, ls, mk in ((7, "-", "o"), (8, "--", "s")):
            pts = []
            for st in sorted({s for sz, s in ck if sz == size}):
                v = ck[(size, st)]
                if 0 in v and bits in v:
                    pts.append((st * 2097152 / 1e9, min(v[bits] - v[0], 20)))
            if len(pts) < 3:
                continue
            axes[2].plot(*zip(*pts), marker=mk, ms=4.5, lw=1.8, ls=ls,
                         color=WIDTH_COLOR[i % len(WIDTH_COLOR)],
                         label=f"{size} SF{bits}")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("training tokens (B)")
    axes[2].set_ylabel("PTQ penalty vs FP16 (nats)")
    axes[2].set_title("(c) PTQ damage is U-shaped in tokens:\n"
                      "both ends of a training run are fragile", fontsize=10)
    axes[2].legend(fontsize=7, ncol=2)
    axes[2].grid(alpha=0.3, which="both")

    fig.tight_layout()
    p = os.path.join(outdir, "scaling_ab_lm.png")
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ tier D --
def fig_absorb(rows, outdir):
    g = collections.defaultdict(dict)
    for r in rows:
        g[(r["size"], r["mode"])][r["bits"]] = r["final_val_loss"]
    sizes = [s for s in ("5m", "11m") if (s, "none") in g]
    fig, axes = plt.subplots(1, len(sizes) + 1, figsize=(6.5 * (len(sizes) + 1), 4.8))

    for ax, size in zip(axes, sizes):
        for mode in ("none", "ln", "ln_full"):
            d = g.get((size, mode), {})
            if 0 not in d:
                continue
            xs = sorted(b for b in d if b)
            ax.plot(xs, [d[b] - d[0] for b in xs], marker="o", ms=5, lw=1.8,
                    color=MODE_COLOR[mode], label=MODE_LABEL[mode])
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_xlabel("SuperFloat precision p (bits)")
        ax.set_ylabel("val loss penalty vs own FP32 control (nats)")
        ax.set_title(f"({'ab'[sizes.index(size)]}) transformer, {size}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    ax = axes[-1]
    P = [2, 3, 4]
    x = np.arange(len(P))
    w = 0.35
    for j, size in enumerate(sizes):
        base = g[(size, "none")]
        d = g[(size, "ln_full")]
        vals = [100 * (1 - (d[b] - d[0]) / (base[b] - base[0])) for b in P]
        ax.bar(x + (j - 0.5) * w, vals, w, label=size,
               color=WIDTH_COLOR[j % len(WIDTH_COLOR)])
    ax.set_xticks(x)
    ax.set_xticklabels([f"SF{b}" for b in P])
    ax.set_ylabel("% of plain-SF penalty removed")
    ax.set_title("(c) scale absorption, norm before every matmul", fontsize=10)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    p = os.path.join(outdir, "scaling_d_absorption.png")
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=".")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    a = load(args.results_dir, "a")
    b = load(args.results_dir, "b")
    # exp2 re-measures the same Pythia PTQ grid on finer checkpoint spacing.
    # The 54 cells the two runs share agree to 0.007 nats, so panel (c) uses
    # the union rather than tier B's four checkpoints alone.
    exp2 = os.path.join(args.results_dir, "exp2.jsonl")
    if os.path.exists(exp2):
        seen = {(r["size"], r.get("step"), r["bits"]) for r in b}
        with open(exp2) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if (r["size"], r.get("step"), r["bits"]) not in seen:
                    b.append(r)
    c = load(args.results_dir, "c")
    d = load(args.results_dir, "d")
    print(f"loaded  A={len(a)}  B={len(b)}  C={len(c)}  D={len(d)}")
    for p in (fig_width(c, args.out), fig_lm(a, b, args.out),
              fig_absorb(d, args.out)):
        print("  " + p)


if __name__ == "__main__":
    main()
