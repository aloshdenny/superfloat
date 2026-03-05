"""
compare_training.py
===================
Compare loss & accuracy convergence of SF16 (Q1.15) vs Baseline (FP32).

Usage
-----
  # After running both trainers:
  python compare_training.py

  # Or point to specific log files:
  python compare_training.py \
      --sf16_log   logs/sf16_metrics.json \
      --base_log   logs/baseline_metrics.json \
      --outdir     figures/

What it produces
----------------
  figures/loss_convergence.png   – train & val loss side-by-side
  figures/accuracy_convergence.png – train & val accuracy
  figures/combined_dashboard.png – 4-panel dashboard
  figures/summary_table.txt      – best-epoch summary statistics
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ── colour scheme ─────────────────────────────────────────────────────────────
STYLE = {
    "sf16": {
        "color":  "#FF6B6B",       # coral-red  → SF16
        "marker": "o",
        "label":  "SF16  Q1.15",
    },
    "base": {
        "color":  "#4ECDC4",       # teal       → Baseline
        "marker": "s",
        "label":  "Baseline  FP32",
    },
}

# ── plot helpers ──────────────────────────────────────────────────────────────

def _load_metrics(path: str) -> Optional[list]:
    if not os.path.exists(path):
        print(f"[WARN] metrics file not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def _extract(history: list, key: str) -> tuple:
    """Return (epochs, values) arrays for a given metric key."""
    epochs = [r["epoch"] for r in history]
    values = [r.get(key, float("nan")) for r in history]
    return np.array(epochs, dtype=float), np.array(values, dtype=float)


def _smooth(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Simple moving-average smoother."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def _apply_style():
    plt.rcParams.update({
        "figure.facecolor":  "#0D1117",
        "axes.facecolor":    "#161B22",
        "axes.edgecolor":    "#30363D",
        "axes.labelcolor":   "#E6EDF3",
        "axes.titlecolor":   "#E6EDF3",
        "axes.grid":         True,
        "grid.color":        "#21262D",
        "grid.linestyle":    "--",
        "grid.alpha":        0.7,
        "xtick.color":       "#8B949E",
        "ytick.color":       "#8B949E",
        "text.color":        "#E6EDF3",
        "legend.facecolor":  "#161B22",
        "legend.edgecolor":  "#30363D",
        "legend.labelcolor": "#E6EDF3",
        "font.family":       "DejaVu Sans",
        "font.size":         11,
    })


def _plot_metric(ax, epochs_sf16, vals_sf16,
                 epochs_base, vals_base,
                 title: str, ylabel: str,
                 lower_is_better: bool = True):
    """Plot one metric panel with both runs."""

    def _plot_run(ep, v, tag):
        s = STYLE[tag]
        ax.plot(ep, v, color=s["color"], marker=s["marker"],
                markersize=3, linewidth=1.2, alpha=0.35, label=None)
        # smooth trendline
        smooth_v = _smooth(v, window=5)
        ax.plot(ep, smooth_v, color=s["color"], linewidth=2.0,
                label=s["label"])

    if epochs_sf16 is not None and vals_sf16 is not None:
        _plot_run(epochs_sf16, vals_sf16, "sf16")
    if epochs_base is not None and vals_base is not None:
        _plot_run(epochs_base, vals_base, "base")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("Epoch", labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)
    ax.legend(loc="upper right" if lower_is_better else "lower right",
              fontsize=9, framealpha=0.8)

    # mark best epoch
    for vals, eps, tag in [
        (vals_sf16, epochs_sf16, "sf16"),
        (vals_base, epochs_base, "base"),
    ]:
        if vals is None or eps is None:
            continue
        idx = int(np.nanargmin(vals)) if lower_is_better else int(np.nanargmax(vals))
        ax.axvline(eps[idx], color=STYLE[tag]["color"], linestyle=":",
                   alpha=0.5, linewidth=1)
        ax.scatter([eps[idx]], [vals[idx]],
                   s=70, color=STYLE[tag]["color"], zorder=5, edgecolors="white")


# ── summary table ─────────────────────────────────────────────────────────────

def _build_summary(sf16_history: Optional[list],
                   base_history:  Optional[list]) -> str:
    lines = []
    lines.append("=" * 64)
    lines.append(" ResNet-18  Training Summary  –  SF16 vs Baseline FP32 ")
    lines.append("=" * 64)

    def _stats(history: Optional[list], tag: str):
        if history is None:
            lines.append(f"  {tag}: no data")
            return
        best_val = max(r["val_acc"] for r in history)
        best_ep  = next(r["epoch"] for r in reversed(history)
                        if r["val_acc"] == best_val)
        final_tl = history[-1]["train_loss"]
        final_vl = history[-1]["val_loss"]
        final_ta = history[-1]["train_acc"]
        final_va = history[-1]["val_acc"]
        epochs   = len(history)
        total_t  = sum(r.get("epoch_time", 0) for r in history)
        lines.append(f"\n  ── {tag} ──────────────────────────────────────────")
        lines.append(f"  Epochs trained        : {epochs}")
        lines.append(f"  Best val_acc          : {best_val:.2f}%  (epoch {best_ep})")
        lines.append(f"  Final train_loss      : {final_tl:.4f}")
        lines.append(f"  Final val_loss        : {final_vl:.4f}")
        lines.append(f"  Final train_acc       : {final_ta:.2f}%")
        lines.append(f"  Final val_acc         : {final_va:.2f}%")
        if total_t > 0:
            lines.append(f"  Total training time   : {total_t/60:.1f} min")

    _stats(sf16_history, "SF16  (Q1.15)")
    _stats(base_history, "Baseline (FP32)")

    if sf16_history and base_history:
        best_sf16 = max(r["val_acc"] for r in sf16_history)
        best_base = max(r["val_acc"] for r in base_history)
        delta = best_sf16 - best_base
        lines.append("\n  ── Comparison ────────────────────────────────────")
        lines.append(f"  SF16 best val_acc  : {best_sf16:.2f}%")
        lines.append(f"  Base best val_acc  : {best_base:.2f}%")
        lines.append(f"  SF16 vs Base delta : {delta:+.2f}% (positive = SF16 better)")

    lines.append("\n" + "=" * 64)
    return "\n".join(lines)


# ── runner ────────────────────────────────────────────────────────────────────

def plot_all(sf16_log: str, base_log: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    _apply_style()

    sf16_h = _load_metrics(sf16_log)
    base_h = _load_metrics(base_log)

    # ── extract individual series ─────────────────────────────────────────
    def _series(h, key):
        if h is None:
            return None, None
        return _extract(h, key)

    ep_sf16_tr, tl_sf16 = _series(sf16_h, "train_loss")
    ep_sf16_v,  vl_sf16 = _series(sf16_h, "val_loss")
    ep_sf16_ta, ta_sf16 = _series(sf16_h, "train_acc")
    ep_sf16_va, va_sf16 = _series(sf16_h, "val_acc")

    ep_base_tr, tl_base = _series(base_h, "train_loss")
    ep_base_v,  vl_base = _series(base_h, "val_loss")
    ep_base_ta, ta_base = _series(base_h, "train_acc")
    ep_base_va, va_base = _series(base_h, "val_acc")

    # ── 1. Train loss convergence ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_metric(ax,
                 ep_sf16_tr, tl_sf16,
                 ep_base_tr, tl_base,
                 "Training Loss Convergence  –  SF16 vs Baseline",
                 "Cross-Entropy Loss")
    fig.tight_layout()
    p = os.path.join(outdir, "train_loss.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")

    # ── 2. Val loss convergence ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_metric(ax,
                 ep_sf16_v, vl_sf16,
                 ep_base_v, vl_base,
                 "Validation Loss Convergence  –  SF16 vs Baseline",
                 "Cross-Entropy Loss")
    fig.tight_layout()
    p = os.path.join(outdir, "val_loss.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")

    # ── 3. Accuracy convergence ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_metric(axes[0],
                 ep_sf16_ta, ta_sf16,
                 ep_base_ta, ta_base,
                 "Train Accuracy", "Accuracy (%)", lower_is_better=False)
    _plot_metric(axes[1],
                 ep_sf16_va, va_sf16,
                 ep_base_va, va_base,
                 "Val Accuracy", "Accuracy (%)", lower_is_better=False)
    fig.suptitle("Accuracy Convergence  –  SF16 vs Baseline",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    p = os.path.join(outdir, "accuracy_convergence.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")

    # ── 4. 4-panel dashboard ──────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0D1117")
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.28)

    panels = [
        (gs[0, 0], ep_sf16_tr, tl_sf16, ep_base_tr, tl_base,
         "Train Loss",      "Loss",        True),
        (gs[0, 1], ep_sf16_v,  vl_sf16, ep_base_v,  vl_base,
         "Validation Loss", "Loss",        True),
        (gs[1, 0], ep_sf16_ta, ta_sf16, ep_base_ta, ta_base,
         "Train Accuracy",  "Accuracy (%)", False),
        (gs[1, 1], ep_sf16_va, va_sf16, ep_base_va, va_base,
         "Validation Accuracy", "Accuracy (%)", False),
    ]

    for spec, es16, vs16, ebs, vbs, title, ylabel, lib in panels:
        ax = fig.add_subplot(spec)
        _plot_metric(ax, es16, vs16, ebs, vbs, title, ylabel, lib)

    # Title bar
    fig.add_subplot(gs[:]).set_visible(False)
    fig.text(0.5, 0.98,
             "ResNet-18: SF16 (Q1.15) vs Baseline (FP32)  –  CIFAR-10",
             ha="center", va="top", fontsize=15, fontweight="bold",
             color="#E6EDF3")

    p = os.path.join(outdir, "combined_dashboard.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")

    # ── 5. LR schedule overlay ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    for h, tag in [(sf16_h, "sf16"), (base_h, "base")]:
        if h is None:
            continue
        ep, lr = _extract(h, "lr")
        s = STYLE[tag]
        ax.plot(ep, lr, color=s["color"], linewidth=2, label=s["label"])
    ax.set_title("Learning Rate Schedule  –  SF16 vs Baseline",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    p = os.path.join(outdir, "lr_schedule.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")

    # ── summary table ─────────────────────────────────────────────────────
    summary = _build_summary(sf16_h, base_h)
    print(summary)
    sp = os.path.join(outdir, "summary_table.txt")
    with open(sp, "w") as f:
        f.write(summary)
    print(f"Saved: {sp}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare SF16 vs baseline training")
    parser.add_argument("--sf16_log", type=str,
                        default="./logs/sf16_metrics.json")
    parser.add_argument("--base_log", type=str,
                        default="./logs/baseline_metrics.json")
    parser.add_argument("--outdir",   type=str,
                        default="./figures")
    args = parser.parse_args()

    if not os.path.exists(args.sf16_log) and not os.path.exists(args.base_log):
        print("[ERROR] Neither log file found.  Run the trainers first:\n"
              "  python train_resnet_sf16.py\n"
              "  python train_resnet_baseline.py")
        return

    plot_all(args.sf16_log, args.base_log, args.outdir)


if __name__ == "__main__":
    main()
