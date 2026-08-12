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


def load_c(results_dir):
    """Tier C records, keyed (width_mult, bits) -> list over seeds."""
    by = collections.defaultdict(list)
    for f in glob.glob(os.path.join(results_dir, "runs_scaling_c", "*.json")):
        r = json.load(open(f))
        # smoke runs share tags with real ones; length is the only honest check
        if len(r.get("history", [])) < MIN_EPOCHS:
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


if __name__ == "__main__":
    main()
