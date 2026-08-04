"""Aggregate every finished run into the paper's table layout.

Classification rows report best top-1 val accuracy; detection rows report best
mAP50-95 (and mAP50). Runs still in flight are shown with their current best so
progress is visible mid-sweep.
"""

import glob
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FORMATS = ["sf16", "sf8", "sf4"]


def classification_rows():
    """One row per format, aggregated over seeds as mean +/- std.

    The paper reports its CIFAR tables as mean +/- std over 3 seeds precisely
    because SuperFloat's failure mode at depth is seed-dependent collapse
    rather than a uniform accuracy drop -- a point estimate would hide it.
    """
    per_fmt = {}
    for meta_path in sorted(glob.glob(os.path.join(HERE, "runs/eurosat/*.json"))):
        meta = json.load(open(meta_path))
        csv_path = meta_path.replace(".json", ".csv")
        df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
        best = df.loc[df.val_acc.idxmax()] if len(df) else None
        per_fmt.setdefault(meta["format"], []).append({
            "model": meta["model"],
            "acc": meta["best_val_acc"],
            "best_epoch": meta["best_epoch"],
            "epochs_run": meta["epochs_run"],
            "train_loss": float(best.train_loss) if best is not None else float("nan"),
            "val_loss": float(best.val_loss) if best is not None else float("nan"),
            "train_acc": float(best.train_acc) if best is not None else float("nan"),
        })

    rows = []
    for fmt, runs in per_fmt.items():
        acc = pd.Series([r["acc"] for r in runs])
        rows.append({
            "domain": "Remote sensing classification",
            "dataset": "EuroSAT",
            "model": runs[0]["model"],
            "format": fmt,
            "metric": "top-1 acc (%)",
            "best": round(acc.mean(), 2),
            # population std, matching how the paper's +/- figures read
            "std": round(acc.std(ddof=0), 2),
            "seeds": len(acc),
            "per_seed": ", ".join(f"{a:.2f}" for a in acc),
            "best_epoch": int(pd.Series([r["best_epoch"] for r in runs]).median()),
            "epochs_run": int(pd.Series([r["epochs_run"] for r in runs]).median()),
            "train_loss": round(pd.Series([r["train_loss"] for r in runs]).mean(), 4),
            "val_loss": round(pd.Series([r["val_loss"] for r in runs]).mean(), 4),
            "train_acc": round(pd.Series([r["train_acc"] for r in runs]).mean(), 2),
        })
    return rows


def detection_rows():
    rows = []
    # Recursive: Ultralytics resolves --project against its own runs_dir, so
    # the tree can come out nested as runs/detect/runs/detect/<name>/.
    for csv_path in sorted(glob.glob(os.path.join(HERE, "runs/**/results.csv"),
                                     recursive=True)):
        # Run dirs are named <dataset>[_<init>]_<format>, e.g.
        # visdrone_pretrained_sf16, dota_random_sf4, or plain visdrone_sf16
        # for the earlier nano baseline.
        name = os.path.basename(os.path.dirname(csv_path))
        fmt = next((f for f in FORMATS if name.endswith("_" + f)), "?")
        if fmt == "?":
            continue
        stem = name[: -(len(fmt) + 1)]
        if stem.endswith("_pretrained"):
            ds, init = stem[: -len("_pretrained")], "pretrained"
        elif stem.endswith("_random"):
            ds, init = stem[: -len("_random")], "random"
        else:
            ds, init = stem, "random"
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        col95 = next((c for c in df.columns if "mAP50-95" in c), None)
        col50 = next((c for c in df.columns if "mAP50(" in c), None)
        if col95 is None or not len(df):
            continue
        b = df.loc[df[col95].idxmax()]
        rows.append({
            "domain": "UAV object detection" if ds == "visdrone"
                      else "Satellite object detection",
            "dataset": f"{'VisDrone' if ds == 'visdrone' else 'DOTAv1'} ({init})",
            "model": "YOLO11x" if ds == "visdrone" else "YOLOv8x-OBB",
            "format": fmt,
            "metric": "mAP50-95",
            "best": round(float(b[col95]), 4),
            "mAP50": round(float(b[col50]), 4) if col50 else None,
            # single seed: detection configs need ~18 GB each, so parallel
            # seeds do not fit and shrinking batch would break comparability
            "std": 0.0,
            "seeds": 1,
            "per_seed": f"{float(b[col95]):.4f}",
            "best_epoch": int(b["epoch"]),
            "epochs_run": int(df["epoch"].max()),
        })
    return rows


def main():
    rows = classification_rows() + detection_rows()
    if not rows:
        print("no completed runs yet")
        return
    df = pd.DataFrame(rows)
    order = {f: i for i, f in enumerate(FORMATS)}
    df = df.sort_values(["dataset", "format"], key=lambda s: s.map(order).fillna(99)
                        if s.name == "format" else s)
    out = os.path.join(HERE, "runs/summary.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {out}")

    # Paper-style table: formats as rows, datasets as columns, mean +/- std.
    print("\n=== paper table layout ===")
    cell = {}
    for _, r in df.iterrows():
        txt = (f"{r['best']:.2f} ±{r['std']:.2f}" if r["seeds"] > 1
               else f"{r['best']:.4f}" if r["metric"] == "mAP50-95"
               else f"{r['best']:.2f}")
        cell.setdefault(r["format"], {})[f"{r['dataset']} ({r['metric']})"] = txt
    tab = pd.DataFrame(cell).T.reindex([f for f in FORMATS if f in cell])
    print(tab.fillna("--").to_string())


if __name__ == "__main__":
    main()
