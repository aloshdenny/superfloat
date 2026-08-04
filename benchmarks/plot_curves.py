"""Convergence-trajectory figures in the paper's style.

One panel per numeric format. Left axis is validation loss on a log scale
(solid); right axis is validation accuracy or mAP50-95 (dashed). A vertical
line marks the epoch of peak validation performance, matching the CIFAR-10 and
ImageNet convergence figures already in the paper.

    python plot_curves.py --results-dir <dir> --out figures/

Expects `<dir>/runs/<name>/results.csv` for detection (Ultralytics format) and
`<dir>/**/convnext_tiny_<fmt>_s<seed>.csv` for classification.
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FORMATS = ["fp32", "fp16", "sf16", "sf8", "sf4"]
LABEL = {"fp32": "FP32", "fp16": "FP16", "sf16": "SF16", "sf8": "SF8",
         "sf4": "SF4"}
# Matplotlib's default blue, as used by the paper's existing figures.
BLUE = "#1f77b4"
PANEL = "abcdefgh"


def _detection(path):
    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]
    m = next((c for c in d.columns if "mAP50-95" in c), None)
    loss_cols = [c for c in d.columns if c.startswith("val/") and c.endswith("loss")]
    if m is None or not loss_cols or d.empty:
        return None
    return d["epoch"], d[loss_cols].sum(axis=1), d[m] * 100.0


def _classification(path):
    d = pd.read_csv(path)
    if d.empty or "val_loss" not in d.columns:
        return None
    return d["epoch"], d["val_loss"], d["val_acc"]


def panel(ax, ep, loss, metric, title, metric_label):
    """Log-scale loss (solid) + metric (dashed) + a convergence marker."""
    ax.plot(ep, loss, "-", color=BLUE, lw=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Validation Loss (log scale)")

    ax2 = ax.twinx()
    ax2.plot(ep, metric, "--", color=BLUE, lw=1.5)
    ax2.set_ylabel(metric_label)

    # Convergence is the epoch of peak validation performance.
    conv = int(ep.iloc[int(metric.values.argmax())])
    ax.axvline(conv, color=BLUE, lw=1.5)
    ax.annotate(f"Convergence = {conv}", xy=(conv, 0.5),
                xycoords=("data", "axes fraction"), rotation=90,
                va="center", ha="right", fontsize=8, color="#333333")
    ax.set_title(title, fontweight="bold")


def figure(specs, out_path, suptitle, metric_label):
    """specs: list of (title, epochs, loss, metric), laid out two per row."""
    ncol = 2
    nrow = (len(specs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.6 * ncol, 4.3 * nrow),
                             squeeze=False)
    axes = axes.ravel()
    for i, (title, ep, loss, met) in enumerate(specs):
        panel(axes[i], ep, loss, met, f"({PANEL[i]}) {title}", metric_label)
    for j in range(len(specs), len(axes)):
        axes[j].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontweight="bold", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print("wrote", out_path, f"({len(specs)} panels)")


def build(results_dir, outdir):
    os.makedirs(outdir, exist_ok=True)

    for ds, init, title, short in [
        ("visdrone", "pretrained", "VisDrone / YOLO11x, COCO-pretrained", "VisDrone"),
        ("visdrone", "random", "VisDrone / YOLO11x, from scratch", "VisDrone"),
        ("dota", "pretrained", "DOTAv1 / YOLOv8x-OBB, COCO-pretrained", "DOTA"),
        ("dota", "random", "DOTAv1 / YOLOv8x-OBB, from scratch", "DOTA"),
    ]:
        specs = []
        for fmt in FORMATS:
            p = os.path.join(results_dir, "runs", f"{ds}_{init}_{fmt}",
                             "results.csv")
            if not (os.path.exists(p) and os.path.getsize(p)):
                continue
            got = _detection(p)
            if got:
                specs.append((f"{LABEL[fmt]} ({short})", *got))
        if specs:
            figure(specs, os.path.join(outdir, f"convergence_{ds}_{init}.png"),
                   title, "Validation mAP50-95 (%)")

    # Classification: seed 0 of each format.
    specs = []
    for fmt in FORMATS:
        hits = glob.glob(os.path.join(results_dir, "**",
                                      f"convnext_tiny_{fmt}_s0.csv"),
                         recursive=True)
        if not hits:
            continue
        got = _classification(hits[0])
        if got:
            specs.append((f"{LABEL[fmt]} (EuroSAT)", *got))
    if specs:
        figure(specs, os.path.join(outdir, "convergence_eurosat.png"),
               "EuroSAT / ConvNeXt-Tiny, from scratch",
               "Validation Accuracy (%)")

    # The SF8 learning-rate result, side by side.
    specs = []
    for name, lbl in (("visdrone_random_sf8", "SF8 @ lr 4e-3 (collapses)"),
                      ("visdrone_random_sf8_lr1e3", "SF8 @ lr 1e-3 (trains)")):
        p = os.path.join(results_dir, "runs", name, "results.csv")
        if os.path.exists(p) and os.path.getsize(p):
            got = _detection(p)
            if got:
                specs.append((lbl, *got))
    if len(specs) == 2:
        figure(specs, os.path.join(outdir, "sf8_learning_rate.png"),
               "SF8 from random init: step size vs grid resolution",
               "Validation mAP50-95 (%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    build(a.results_dir, a.out)
