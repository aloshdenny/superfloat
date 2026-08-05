"""Cross-format summary figures.

`plot_curves.py` draws one panel per format (per-run convergence). These are the
comparative views: all formats overlaid, the accuracy/storage trade-off, and the
weight- and activation-distribution figures that explain the two failure modes.

    python plot_summary.py --results-dir <dir> --out figures/
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FORMATS = ["fp32", "fp16", "sf16", "sf8", "sf4"]
LABEL = {"fp32": "FP32", "fp16": "FP16", "sf16": "SF16", "sf8": "SF8",
         "sf4": "SF4"}
COLOR = {"fp32": "#1f77b4", "fp16": "#17becf", "sf16": "#ff7f0e",
         "sf8": "#2ca02c", "sf4": "#d62728"}
SAVING = {"fp32": 0.0, "fp16": 50.0, "sf16": 50.0, "sf8": 75.0, "sf4": 87.5}
PANEL = "abcdefgh"

GROUPS = [
    ("visdrone_pretrained", "VisDrone / YOLO11x, pretrained", "mAP50-95 (%)"),
    ("visdrone_random", "VisDrone / YOLO11x, from scratch", "mAP50-95 (%)"),
    ("dota_pretrained", "DOTAv1 / YOLOv8x-OBB, pretrained", "mAP50-95 (%)"),
    ("dota_random", "DOTAv1 / YOLOv8x-OBB, from scratch", "mAP50-95 (%)"),
]


def det_curve(results_dir, name):
    p = os.path.join(results_dir, "runs", name, "results.csv")
    if not (os.path.exists(p) and os.path.getsize(p)):
        return None
    d = pd.read_csv(p)
    d.columns = [c.strip() for c in d.columns]
    m = next((c for c in d.columns if "mAP50-95" in c), None)
    return (d["epoch"], d[m] * 100.0) if m else None


def cls_curve(results_dir, fmt, seed=0):
    hits = glob.glob(os.path.join(results_dir, "**",
                                  f"convnext_tiny_{fmt}_s{seed}.csv"),
                     recursive=True)
    if not hits:
        return None
    d = pd.read_csv(hits[0])
    return (d["epoch"], d["val_acc"]) if "val_acc" in d.columns else None


def fig_overlay(results_dir, outdir):
    """All formats on shared axes -- the comparison the per-panel plots can't show."""
    panels = []
    for key, title, ylab in GROUPS:
        series = []
        for f in FORMATS:
            c = det_curve(results_dir, f"{key}_{f}")
            if c:
                series.append((f, *c))
        if series:
            panels.append((title, ylab, series))

    cls = [(f, *c) for f in FORMATS if (c := cls_curve(results_dir, f))]
    if cls:
        panels.append(("EuroSAT / ConvNeXt-Tiny, from scratch",
                       "Top-1 accuracy (%)", cls))
    if not panels:
        return

    nrow = (len(panels) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(13.2, 4.3 * nrow), squeeze=False)
    axes = axes.ravel()
    for i, (title, ylab, series) in enumerate(panels):
        ax = axes[i]
        for f, ep, y in series:
            ax.plot(ep, y, color=COLOR[f], lw=1.6, label=LABEL[f])
        ax.set_xlabel("Epochs")
        ax.set_ylabel(ylab)
        ax.set_title(f"({PANEL[i]}) {title}", fontweight="bold")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
    for j in range(len(panels), len(axes)):
        axes[j].axis("off")
    fig.suptitle("SuperFloat vs full precision — validation trajectories",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(outdir, "format_overlay.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("wrote", out)


def fig_tradeoff(results_dir, outdir):
    """Final accuracy against per-weight storage saving."""
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    marks = {"visdrone_pretrained": "o", "dota_pretrained": "s",
             "visdrone_random": "^", "dota_random": "v"}

    for key, title, _ in GROUPS:
        xs, ys, fs = [], [], []
        for f in FORMATS:
            c = det_curve(results_dir, f"{key}_{f}")
            if c:
                xs.append(SAVING[f]); ys.append(float(c[1].max())); fs.append(f)
        if not xs:
            continue
        base = next((y for y, f in zip(ys, fs) if f == "fp32"), None)
        if base:
            ys = [100.0 * y / base for y in ys]
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], "--",
                marker=marks.get(key, "o"), lw=1.3, ms=7, label=title)

    xs, ys, fs = [], [], []
    for f in FORMATS:
        c = cls_curve(results_dir, f)
        if c:
            xs.append(SAVING[f]); ys.append(float(c[1].max())); fs.append(f)
    if xs:
        base = next((y for y, f in zip(ys, fs) if f == "fp32"), None)
        if base:
            ys = [100.0 * y / base for y in ys]
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], "--", marker="D",
                lw=1.3, ms=7, label="EuroSAT / ConvNeXt-Tiny")

    ax.axhline(100, color="#888888", lw=1, ls=":")
    ax.set_xlabel("Per-weight storage saving vs FP32 (%)")
    ax.set_ylabel("Performance retained vs FP32 (%)")
    ax.set_title("Accuracy retained against storage saved", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out = os.path.join(outdir, "accuracy_vs_storage.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("wrote", out)


def fig_failure_modes(outdir):
    """The two mechanisms, drawn from measurements taken during the sweep."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 4.6))

    # (a) SF4 weight annihilation at initialisation.
    models = ["YOLOv8x-OBB\nKaiming random", "YOLOv8x-OBB\nCOCO-pretrained",
              "V-JEPA 2 ViT-L\nrandom", "V-JEPA 2 ViT-L\npretrained"]
    sf16 = [0.08, 0.40, 0.06, 0.02]
    sf8 = [21.03, 56.55, 15.50, 5.79]
    sf4 = [99.98, 99.79, 99.82, 69.80]
    x = np.arange(len(models)); w = 0.26
    a1.bar(x - w, sf16, w, label="SF16", color=COLOR["sf16"])
    a1.bar(x, sf8, w, label="SF8", color=COLOR["sf8"])
    a1.bar(x + w, sf4, w, label="SF4", color=COLOR["sf4"])
    a1.set_xticks(x); a1.set_xticklabels(models, fontsize=8)
    a1.set_ylabel("Weights quantized to exactly zero (%)")
    a1.set_title("(a) SF4 annihilates standard initialisation",
                 fontweight="bold")
    a1.axhline(50, color="#888888", ls=":", lw=1)
    a1.legend(fontsize=8)
    a1.grid(alpha=0.25, axis="y")

    # (b) V-JEPA activations against the SFx bound.
    labels = ["SF16", "SF8", "SF4"]
    with_acts = [48.81, 51.01, 47.90]
    weights_only = [96.53, 98.90, 74.22]
    x = np.arange(3); w = 0.36
    a2.bar(x - w / 2, with_acts, w, label="weights + activations",
           color=COLOR["sf4"])
    a2.bar(x + w / 2, weights_only, w, label="weights only",
           color=COLOR["sf8"])
    a2.axhline(97.07, color=COLOR["fp32"], ls="--", lw=1.5, label="FP32 (97.07)")
    a2.set_xticks(x); a2.set_xticklabels(labels)
    a2.set_ylabel("UCF101 probe accuracy (%)")
    a2.set_title("(b) V-JEPA 2: activation clamping, not precision, is the cost\n"
                 "max|a| = 256.1, 26.3% of activations exceed the ±1 bound",
                 fontweight="bold", fontsize=10)
    a2.legend(fontsize=8, loc="lower right")
    a2.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    out = os.path.join(outdir, "failure_modes.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("wrote", out)


def fig_eurosat_seeds(results_dir, outdir):
    """All three seeds per format -- SFx variance is much tighter than FP."""
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for f in FORMATS:
        for s in (0, 1, 2):
            c = cls_curve(results_dir, f, s)
            if c:
                ax.plot(c[0], c[1], color=COLOR[f], lw=1.1, alpha=0.75,
                        label=LABEL[f] if s == 0 else None)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("EuroSAT / ConvNeXt-Tiny — 3 seeds per format",
                 fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = os.path.join(outdir, "eurosat_seeds.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_overlay(a.results_dir, a.out)
    fig_tradeoff(a.results_dir, a.out)
    fig_failure_modes(a.out)
    fig_eurosat_seeds(a.results_dir, a.out)
