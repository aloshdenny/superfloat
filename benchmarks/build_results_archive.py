"""Fold the raw run directories into one JSONL per experiment.

A live run writes one JSON per configuration so an interrupted queue can skip
what it already has. That is the wrong shape for a repository: 800 loose files
that git has to track individually. This collapses them, sorted so the archive
is stable across rebuilds and a diff shows only what actually changed.

    python build_results_archive.py --tiers /tmp/tiers --lab /tmp/collect/all --out results
"""

import argparse
import glob
import json
import os


def key(r):
    return json.dumps({k: r.get(k) for k in
                       ("size", "bits", "mode", "step", "seed", "width_mult",
                        "bits_w", "bits_a", "depth", "lr", "tpp", "channel_norm")},
                      sort_keys=True)


def write(rows, path):
    rows = sorted(rows, key=key)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", help="dir holding runs_scaling_{a,b,c,d}/")
    ap.add_argument("--lab", help="dir holding exp*.json")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.tiers:
        for t in "abcd":
            fs = glob.glob(os.path.join(a.tiers, f"runs_scaling_{t}", "*.json"))
            fs = [f for f in fs if "_fp16head" not in f]
            if fs:
                n = write([json.load(open(f)) for f in fs],
                          os.path.join(a.out, f"scaling_{t}.jsonl"))
                print(f"  scaling_{t}.jsonl  {n}")

    if a.lab:
        groups = {}
        skipped = 0
        for f in glob.glob(os.path.join(a.lab, "*.json")):
            try:
                r = json.load(open(f))
            except Exception:
                skipped += 1
                continue
            # placeholder files claim a config for another host; they are not data
            if r.get("PLACEHOLDER") or not r.get("exp"):
                skipped += 1
                continue
            groups.setdefault(r["exp"], []).append(r)
        for exp, rows in sorted(groups.items()):
            n = write(rows, os.path.join(a.out, f"{exp}.jsonl"))
            done = sum(1 for r in rows if r.get("complete"))
            print(f"  {exp}.jsonl  {n}" + ("" if done == n else f"  ({done} complete)"))
        if skipped:
            print(f"  skipped {skipped} non-result files")


if __name__ == "__main__":
    main()
