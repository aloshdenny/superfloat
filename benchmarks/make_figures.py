"""Paper figure set for the SuperFloat evaluation — one uniform style.

Supersedes plot_curves.py and plot_summary.py, which grew ad hoc and disagreed
on styling. Everything here shares one palette, one axis convention and one
convergence marker, so the panels can sit next to each other in the paper.

Convention, matching the paper's existing CIFAR-10 / ImageNet figures:
  * solid line, left axis, log scale  = validation loss
  * dashed line, right axis, linear   = accuracy or mAP50-95
  * vertical rule                     = epoch of peak validation performance

Runs that never learned are excluded from the trajectory panels and named in
EXCLUDED below with the reason. They are not hidden -- they are the subject of
their own figure (failure_modes.png), where a flat line at zero is the point
rather than noise on a shared axis.

    python make_figures.py --results-dir <dir> --out figures/
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

# Runs excluded from trajectory panels, with the reason. Each is reported in
# SUPERFLOAT_RESULTS.md; none is silently dropped.
EXCLUDED = {
    "dota_random_sf4":
        "0.0000 for 263 epochs -- Kaiming init puts 99.98% of weights below "
        "SF4's 0.0625 floor, so the network is dead at step 0",
    "visdrone_random_sf8":
        "collapsed at lr 4e-3 (0.0724); the lr 1e-3 run is shown instead",
    "visdrone_random_sf8_s1": "seed 1 of the same lr 4e-3 collapse",
    "visdrone_random_sf8_s2": "seed 2 of the same lr 4e-3 collapse",
    "dota_random_sf4_lr1e3": "SF4 floor diagnostic, never left 0.0000",
    "dota_random_sf4_lr2p5e4": "SF4 floor diagnostic, never left 0.0000",
    "dota_random_sf4_scaled": "cancelled init-scaling probe",
    "dota_random_sf4_wd0": "SF4 floor diagnostic, never left 0.0000",
}

# The from-scratch VisDrone SF8 cell is represented by the lr 1e-3 run.
SUBSTITUTE = {"visdrone_random_sf8": "visdrone_random_sf8_lr1e3"}


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------

def detection(results_dir, name):
    p = os.path.join(results_dir, "runs", name, "results.csv")
    if not (os.path.exists(p) and os.path.getsize(p)):
        return None
    d = pd.read_csv(p)
    d.columns = [c.strip() for c in d.columns]
    m = next((c for c in d.columns if "mAP50-95" in c), None)
    lcols = [c for c in d.columns if c.startswith("val/") and c.endswith("loss")]
    if m is None or not lcols or d.empty:
        return None
    return d["epoch"], d[lcols].sum(axis=1), d[m] * 100.0


def classification(results_dir, fmt, seed=0):
    hits = glob.glob(os.path.join(results_dir, "**",
                                  f"convnext_tiny_{fmt}_s{seed}.csv"),
                     recursive=True)
    if not hits:
        return None
    d = pd.read_csv(hits[0])
    if d.empty or "val_loss" not in d.columns:
        return None
    return d["epoch"], d["val_loss"], d["val_acc"]


def vjepa(results_dir, sub, fmt):
    p = os.path.join(results_dir, sub, f"{fmt}.csv")
    if not (os.path.exists(p) and os.path.getsize(p)):
        return None
    d = pd.read_csv(p)
    if d.empty or "val_loss" not in d.columns:
        return None
    return d["epoch"], d["val_loss"], d["val_acc"]


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------

def dual_panel(ax, ep, loss, metric, title, metric_label, color=None):
    c = color or "#1f77b4"
    ax.plot(ep, loss, "-", color=c, lw=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Validation Loss (log scale)")
    ax2 = ax.twinx()
    ax2.plot(ep, metric, "--", color=c, lw=1.5)
    ax2.set_ylabel(metric_label)
    conv = int(ep.iloc[int(metric.values.argmax())])
    ax.axvline(conv, color=c, lw=1.4)
    ax.annotate(f"Convergence = {conv}", xy=(conv, 0.5),
                xycoords=("data", "axes fraction"), rotation=90,
                va="center", ha="right", fontsize=8, color="#333333")
    ax.set_title(title, fontweight="bold")


def grid(specs, path, suptitle, metric_label):
    ncol = 2
    nrow = (len(specs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.8 * ncol, 4.4 * nrow),
                             squeeze=False)
    axes = axes.ravel()
    for i, (title, ep, loss, met, col) in enumerate(specs):
        dual_panel(axes[i], ep, loss, met, f"({PANEL[i]}) {title}",
                   metric_label, col)
    for j in range(len(specs), len(axes)):
        axes[j].axis("off")
    fig.suptitle(suptitle, fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"wrote {os.path.basename(path)} ({len(specs)} panels)")


# --------------------------------------------------------------------------

DET_GROUPS = [
    ("visdrone", "pretrained", "VisDrone / YOLO11x, COCO-pretrained", "VisDrone"),
    ("visdrone", "random", "VisDrone / YOLO11x, from scratch", "VisDrone"),
    ("dota", "pretrained", "DOTAv1 / YOLOv8x-OBB, COCO-pretrained", "DOTA"),
    ("dota", "random", "DOTAv1 / YOLOv8x-OBB, from scratch", "DOTA"),
]


def build(results_dir, outdir):
    os.makedirs(outdir, exist_ok=True)
    kept, skipped = [], []

    # ---- per-format trajectory grids, detection ---------------------------
    for ds, init, title, short in DET_GROUPS:
        specs = []
        for fmt in FORMATS:
            name = f"{ds}_{init}_{fmt}"
            use = SUBSTITUTE.get(name, name)
            if name in EXCLUDED and use == name:
                skipped.append((name, EXCLUDED[name]))
                continue
            got = detection(results_dir, use)
            if got:
                lbl = LABEL[fmt] + (" @ lr 1e-3" if use != name else "")
                specs.append((f"{lbl} ({short})", *got, COLOR[fmt]))
                kept.append(use)
        if specs:
            grid(specs, os.path.join(outdir, f"convergence_{ds}_{init}.png"),
                 title, "Validation mAP50-95 (%)")

    # ---- classification ---------------------------------------------------
    specs = []
    for fmt in FORMATS:
        got = classification(results_dir, fmt)
        if got:
            specs.append((f"{LABEL[fmt]} (EuroSAT)", *got, COLOR[fmt]))
    if specs:
        grid(specs, os.path.join(outdir, "convergence_eurosat.png"),
             "EuroSAT / ConvNeXt-Tiny, from scratch",
             "Validation Accuracy (%)")

    # ---- V-JEPA, weights-only (the configuration that works) --------------
    for sub, tag, ttl in (("vjepa", "ptq", "V-JEPA 2 ViT-L / UCF101 — PTQ, weights-only"),
                          ("vjepa_qat", "qat", "V-JEPA 2 ViT-L / UCF101 — QAT, weights-only")):
        specs = []
        for fmt in FORMATS:
            key = fmt if fmt in ("fp32", "fp16") else f"{fmt}_wonly"
            got = vjepa(results_dir, sub, key)
            if got:
                specs.append((f"{LABEL[fmt]} (UCF101)", *got, COLOR[fmt]))
        if specs:
            grid(specs, os.path.join(outdir, f"convergence_vjepa_{tag}.png"),
                 ttl, "Validation Accuracy (%)")

    overlay(results_dir, outdir)
    tradeoff(results_dir, outdir)
    failure_modes(results_dir, outdir)

    print(f"\nincluded {len(set(kept))} runs; excluded {len(skipped)}:")
    for n, why in skipped:
        print(f"  - {n}: {why}")


def overlay(results_dir, outdir):
    """All formats on shared axes, one panel per domain."""
    panels = []
    for ds, init, title, _ in DET_GROUPS:
        series = []
        for fmt in FORMATS:
            name = f"{ds}_{init}_{fmt}"
            use = SUBSTITUTE.get(name, name)
            if name in EXCLUDED and use == name:
                continue
            got = detection(results_dir, use)
            if got:
                series.append((fmt, got[0], got[2],
                               use != name))
        if series:
            panels.append((title, "mAP50-95 (%)", series))

    cls = [(f, *classification(results_dir, f)[::2], False)
           for f in FORMATS if classification(results_dir, f)]
    if cls:
        panels.append(("EuroSAT / ConvNeXt-Tiny", "Top-1 accuracy (%)", cls))
    vj = [(f, *vjepa(results_dir, "vjepa_qat",
                     f if f in ("fp32", "fp16") else f"{f}_wonly")[::2], False)
          for f in FORMATS
          if vjepa(results_dir, "vjepa_qat",
                   f if f in ("fp32", "fp16") else f"{f}_wonly")]
    if vj:
        panels.append(("V-JEPA 2 / UCF101 (QAT, weights-only)",
                       "Top-1 accuracy (%)", vj))

    nrow = (len(panels) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(13.6, 4.4 * nrow), squeeze=False)
    axes = axes.ravel()
    for i, (title, ylab, series) in enumerate(panels):
        ax = axes[i]
        for fmt, ep, y, subbed in series:
            ax.plot(ep, y, color=COLOR[fmt], lw=1.7,
                    label=LABEL[fmt] + (" @1e-3" if subbed else ""))
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
    fig.savefig(os.path.join(outdir, "format_overlay.png"), dpi=170)
    plt.close(fig)
    print("wrote format_overlay.png")


def tradeoff(results_dir, outdir):
    """Final performance retained against per-weight storage saved."""
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    marks = {"visdrone_pretrained": "o", "dota_pretrained": "s",
             "visdrone_random": "^", "dota_random": "v"}

    def curve(points, label, marker):
        if len(points) < 2:
            return
        base = next((v for s, v, f in points if f == "fp32"), None)
        if not base:
            return
        pts = sorted(((s, 100.0 * v / base) for s, v, _ in points))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "--",
                marker=marker, lw=1.4, ms=7, label=label)

    for ds, init, title, _ in DET_GROUPS:
        pts = []
        for fmt in FORMATS:
            name = f"{ds}_{init}_{fmt}"
            use = SUBSTITUTE.get(name, name)
            if name in EXCLUDED and use == name:
                continue
            got = detection(results_dir, use)
            if got:
                pts.append((SAVING[fmt], float(got[2].max()), fmt))
        curve(pts, title, marks.get(f"{ds}_{init}", "o"))

    pts = [(SAVING[f], float(classification(results_dir, f)[2].max()), f)
           for f in FORMATS if classification(results_dir, f)]
    curve(pts, "EuroSAT / ConvNeXt-Tiny", "D")

    pts = []
    for f in FORMATS:
        key = f if f in ("fp32", "fp16") else f"{f}_wonly"
        g = vjepa(results_dir, "vjepa_qat", key)
        if g:
            pts.append((SAVING[f], float(g[2].max()), f))
    curve(pts, "V-JEPA 2 / UCF101 (weights-only)", "*")

    ax.axhline(100, color="#888888", lw=1, ls=":")
    ax.set_xlabel("Per-weight storage saving vs FP32 (%)")
    ax.set_ylabel("Performance retained vs FP32 (%)")
    ax.set_title("Accuracy retained against storage saved", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "accuracy_vs_storage.png"), dpi=170)
    plt.close(fig)
    print("wrote accuracy_vs_storage.png")


def failure_modes(results_dir, outdir):
    """The two mechanisms, from measurements taken during the sweep."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.6, 4.8))

    models = ["YOLOv8x-OBB\nKaiming random", "YOLOv8x-OBB\nCOCO-pretrained",
              "V-JEPA 2 ViT-L\nrandom", "V-JEPA 2 ViT-L\npretrained"]
    x = np.arange(len(models)); w = 0.26
    for off, key, vals in ((-w, "sf16", [0.08, 0.40, 0.06, 0.02]),
                           (0.0, "sf8", [21.03, 56.55, 15.50, 5.79]),
                           (w, "sf4", [99.98, 99.79, 99.82, 69.80])):
        a1.bar(x + off, vals, w, label=LABEL[key], color=COLOR[key])
    a1.set_xticks(x); a1.set_xticklabels(models, fontsize=8)
    a1.set_ylabel("Weights quantized to exactly zero (%)")
    a1.set_title("(a) SF4 annihilates standard initialisation",
                 fontweight="bold")
    a1.axhline(50, color="#888888", ls=":", lw=1)
    a1.legend(fontsize=8); a1.grid(alpha=0.25, axis="y")

    labels = ["SF16", "SF8", "SF4"]
    x = np.arange(3); w = 0.36
    a2.bar(x - w/2, [48.81, 51.01, 47.90], w,
           label="weights + activations", color=COLOR["sf4"])
    a2.bar(x + w/2, [96.53, 98.90, 74.22], w,
           label="weights only", color=COLOR["sf8"])
    a2.axhline(97.07, color=COLOR["fp32"], ls="--", lw=1.5,
               label="FP32 (97.07)")
    a2.set_xticks(x); a2.set_xticklabels(labels)
    a2.set_ylabel("UCF101 probe accuracy (%)")
    a2.set_title("(b) V-JEPA 2: activation clamping, not precision, is the cost\n"
                 "max|a| = 256.1, 26.3% of activations exceed the ±1 bound",
                 fontweight="bold", fontsize=10)
    a2.legend(fontsize=8, loc="lower right"); a2.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "failure_modes.png"), dpi=170)
    plt.close(fig)
    print("wrote failure_modes.png")

    # SF8 learning-rate pair keeps its own figure: the collapsed run is the
    # subject here, so it is deliberately not excluded.
    specs = []
    for name, lbl, col in (("visdrone_random_sf8", "SF8 @ lr 4e-3 (collapses)",
                            COLOR["sf4"]),
                           ("visdrone_random_sf8_lr1e3", "SF8 @ lr 1e-3 (trains)",
                            COLOR["sf8"])):
        got = detection(results_dir, name)
        if got:
            specs.append((lbl, *got, col))
    if len(specs) == 2:
        grid(specs, os.path.join(outdir, "sf8_learning_rate.png"),
             "SF8 from random init: step size vs grid resolution",
             "Validation mAP50-95 (%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    build(a.results_dir, a.out)
