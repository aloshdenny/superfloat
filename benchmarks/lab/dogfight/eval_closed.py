"""Closed-loop evaluation: does the quantized policy still win the fight?

Validation accuracy is measured on the expert's own trajectory. In closed loop
the model flies its own, so an early wrong decision changes every state that
follows and errors compound instead of averaging out. This is the metric the
whole harness exists for; accuracy is the screen, this is the capability.

Three numbers per arm, all against a fixed scripted expert flying red:

  win rate        wins - losses - draws over identical seeds
  agreement       fraction of on-policy decisions matching what the expert
                  would have chosen in the state the MODEL actually reached
  crash rate      how often it flew into the ground, i.e. failed to `recover`

Agreement is the sensitive one: it moves long before win rate does, because a
fight can be won despite many small errors. Win rate is the one that matters.
"""
from __future__ import annotations

import argparse, collections, json, os, sys, time

import torch

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_here, "..", ".."), os.path.join(_here, ".."), _here):
    sys.path.insert(0, os.path.abspath(_p))
from superfloat import apply_superfloat, disable_tf32

from env import Engagement
from policy import Policy, encode
from pilot import MANEUVERS, MAN_IDX
import expert as E

OUT = os.environ.get("DOGFIGHT_OUT", "runs")
HEADS = ("head_choice", "head_score", "head_noul")


def load_arm(path, dev="cpu"):
    blob = torch.load(path, map_location=dev, weights_only=False)
    cfg = blob["cfg"]
    m = Policy(cfg["d"], cfg["depth"]).to(dev)
    if cfg.get("bits"):
        apply_superfloat(m, cfg["bits"], head_names=HEADS,
                         quantize_activations=cfg.get("quant_act", False))
    m.load_state_dict(blob["model"])
    m.eval()
    return m, cfg


@torch.no_grad()
def play(model, n=200, seed=1234, dev="cpu", amp=None):
    eng = Engagement(seed=seed)
    res = collections.Counter()
    agree = total = 0
    times = []
    for _ in range(n):
        bo, ro = eng.reset()
        done, info = False, {}
        while not done:
            x = torch.tensor([encode(bo)], dtype=torch.float32, device=dev)
            with torch.autocast(dev, dtype=amp, enabled=amp is not None):
                c, _s, _n = model(x)
            man = MANEUVERS[int(c.float().argmax(-1))]
            # what the teacher would have done in the state the MODEL reached
            if man == E.expert(bo):
                agree += 1
            total += 1
            bo, ro, done, info = eng.step(man, E.expert(ro))
        res[info.get("outcome", "?")] += 1
        if "t" in info:
            times.append(info["t"])
    w = res["blue_win"] + res["red_crash"]
    l = res["red_win"] + res["blue_crash"]
    return {
        "wins": w, "losses": l, "draws": n - w - l, "n": n,
        "win_rate": w / n, "loss_rate": l / n,
        "crash_rate": res["blue_crash"] / n,
        "agreement": agree / max(total, 1),
        "decisions": total,
        "median_t": sorted(times)[len(times) // 2] if times else None,
        "outcomes": dict(res),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="", help="comma-separated tags; default all .pt in OUT")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=os.path.join(OUT, "closed_loop.jsonl"))
    a = ap.parse_args()
    disable_tf32()
    import glob
    paths = ([os.path.join(OUT, t + ".pt") for t in a.arms.split(",")] if a.arms
             else sorted(glob.glob(os.path.join(OUT, "dog_*.pt"))))
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["tag"] for l in open(a.out) if l.strip()}
    with open(a.out, "a") as fh:
        for p in paths:
            tag = os.path.basename(p)[:-3]
            if tag in done:
                print("skip", tag); continue
            m, cfg = load_arm(p)
            t0 = time.time()
            r = play(m, a.n, a.seed)
            r.update(tag=tag, bits=cfg.get("bits", 0), dtype=cfg.get("dtype", "fp32"),
                     quant_act=cfg.get("quant_act", False), minutes=(time.time()-t0)/60,
                     exp="dogfight_closed_loop", complete=True)
            fh.write(json.dumps(r, sort_keys=True) + "\n"); fh.flush()
            print("%-22s W%-4d L%-4d D%-4d  win %.3f  agree %.4f  crash %.3f  (%.1fm)"
                  % (tag, r["wins"], r["losses"], r["draws"], r["win_rate"],
                     r["agreement"], r["crash_rate"], r["minutes"]), flush=True)


if __name__ == "__main__":
    main()
