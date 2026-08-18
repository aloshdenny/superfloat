"""Experiment 5: per-layer bit allocation.

The schema defines every width, so per-layer allocation is constructible in
principle -- and it is an ISA question, because a per-layer format field is
hardware the chip either has or does not.

Two modes:
  --profile   report each layer's dead fraction and fan_in at a given precision,
              which is the signal an allocation rule would key on
  --alloc     train with a rule: layers whose fan_in exceeds a cutoff get +1 bit,
              compared against uniform at matched *average* bits
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from superfloat import disable_tf32, sf_params, sf_quantize_sv
from exp1_act import build, load_gpu, SFConv, MEAN, STD

OUT = "/workspace/results"


def profile(bits):
    model = build().cuda()
    rows = []
    for i, m in enumerate(model.features.modules()):
        if not isinstance(m, SFConv): continue
        s, v = sf_params(bits)
        w = m.weight
        sc = w.abs().amax(dim=(1,2,3), keepdim=True).clamp_min(1e-8)
        q_plain = sf_quantize_sv(w, s, v)
        q_norm = sf_quantize_sv(w/sc, s, v)
        rows.append({"layer": i, "shape": list(w.shape),
                     "fan_in": int(np.prod(w.shape[1:])),
                     "dead_plain": float((q_plain==0).float().mean()),
                     "dead_norm": float((q_norm==0).float().mean()),
                     "sigma": float(w.std())})
    rec = {"exp":"exp5_profile","bits":bits,"layers":rows,"complete":True}
    os.makedirs(OUT,exist_ok=True)
    json.dump(rec, open(f"{OUT}/exp5_profile_sf{bits}.json","w"))
    for r in rows:
        print(f"  L{r['layer']:<3} fan_in={r['fan_in']:<6} sigma={r['sigma']:.4f} "
              f"dead_plain={100*r['dead_plain']:5.1f}% dead_norm={100*r['dead_norm']:5.1f}%",
              flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--bits", type=int, default=4)
    a = ap.parse_args()
    disable_tf32(); torch.manual_seed(0)
    if a.profile: profile(a.bits)
