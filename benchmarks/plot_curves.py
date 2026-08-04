"""Convergence-trajectory figures in the paper's style.

Left axis: validation loss, log scale, solid. Right axis: validation accuracy
(or mAP50-95 for detection), linear, dashed. A vertical line marks the epoch of
peak validation performance, as in the CIFAR-10/ImageNet figures.
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FORMATS = ["sf16", "sf8", "sf4"]
COLORS = {"sf16": "#1f77b4", "sf8": "#ff7f0e", "sf4": "#2ca02c"}


def series():
    """Yield (dataset, format, epochs, val_loss, val_metric, metric_label)."""
    for p in sorted(glob.glob(os.path.join(HERE, "runs/eurosat/*_s0.csv"))):
        # Trajectory panels show seed 0 only; runs stop at different epochs, so
        # averaging curves across seeds would misrepresent the tail. Seed
        # spread is reported as mean +/- std in the results table instead.
        fmt = next((f for f in FORMATS if f"_{f}_" in os.path.basename(p)), None)
        if not fmt:
            continue
        d = pd.read_csv(p)
        if len(d):
            yield "EuroSAT / ConvNeXt-Tiny", fmt, d.epoch, d.val_loss, d.val_acc, "Val accuracy (%)"

    for p in sorted(glob.glob(os.path.join(HERE, "runs/**/results.csv"),
                              recursive=True)):
        name = os.path.basename(os.path.dirname(p))
        fmt = next((f for f in FORMATS if name.endswith("_" + f)), None)
        if not fmt:
            continue
        d = pd.read_csv(p)
        d.columns = [c.strip() for c in d.columns]
        m95 = next((c for c in d.columns if "mAP50-95" in c), None)
        # Total val loss = box + cls + dfl components.
        lcols = [c for c in d.columns if c.startswith("val/") and c.endswith("loss")]
        if not m95 or not lcols or not len(d):
            continue
        ds = ("VisDrone / YOLO11n" if name.startswith("visdrone")
              else "DOTAv1 / YOLOv8n-OBB")
        yield ds, fmt, d.epoch, d[lcols].sum(axis=1), d[m95], "mAP50-95"


def main():
    groups = {}
    for ds, fmt, ep, loss, met, lab in series():
        groups.setdefault((ds, lab), []).append((fmt, ep, loss, met))
    if not groups:
        print("no runs to plot yet")
        return

    outdir = os.path.join(HERE, "runs/figures")
    os.makedirs(outdir, exist_ok=True)

    for (ds, lab), runs in groups.items():
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax2 = ax.twinx()
        for fmt, ep, loss, met in sorted(runs, key=lambda r: FORMATS.index(r[0])):
            c = COLORS[fmt]
            ax.plot(ep, loss, "-", color=c, lw=1.6, label=f"{fmt.upper()} loss")
            ax2.plot(ep, met, "--", color=c, lw=1.3, alpha=0.85,
                     label=f"{fmt.upper()} {lab}")
            peak = int(ep.iloc[met.values.argmax()])
            ax.axvline(peak, color=c, ls="-", lw=0.9, alpha=0.45)

        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation loss (log)")
        ax2.set_ylabel(lab)
        ax.set_title(f"{ds} — SuperFloat convergence")
        ax.grid(alpha=0.25, which="both")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, ncol=2, loc="center right")
        fig.tight_layout()

        slug = ds.split(" / ")[0].lower().replace(".", "")
        path = os.path.join(outdir, f"convergence_{slug}.png")
        fig.savefig(path, dpi=170)
        plt.close(fig)
        print("wrote", path)


if __name__ == "__main__":
    main()
