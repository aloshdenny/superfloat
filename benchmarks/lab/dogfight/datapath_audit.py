"""How far is the trained policy from running on a pure SF register datapath?

The `+act` arms saturate weights and every Linear OUTPUT, which is much closer
to the Atreides datapath than the weights-only recipe used elsewhere in this
repo. It is still not every register. Three things escape:

  - the residual addition, `x + fc_out(...)`, whose sum on real hardware lands
    in a register that must itself be Q1.(x-1)
  - RMSNorm's rsqrt and mean-of-squares, and SiLU's sigmoid, which are not SF
    operations at all
  - the three output heads, excluded from quantization so the logits stay fp32

This measures the size of that gap rather than arguing about it: the maximum
absolute value reaching every named site, against the SF bound. It is the same
census PURE_SF.md section 1 ran on SmolLM2-360M, where the residual stream
reached tens of thousands against a representable bound of 1.
"""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_here, "..", ".."), os.path.join(_here, ".."), _here):
    sys.path.insert(0, os.path.abspath(_p))
from superfloat import sf_params, apply_superfloat

from policy import Policy
from pilot import MANEUVERS

HEADS = ("head_choice", "head_score", "head_noul")


def audit(ckpt, data, bits_override=None, n=20000):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = blob["cfg"]
    m = Policy(cfg["d"], cfg["depth"])
    bits = bits_override or cfg.get("bits") or 8
    if cfg.get("bits"):
        apply_superfloat(m, cfg["bits"], head_names=HEADS,
                         quantize_activations=cfg.get("quant_act", False))
    m.load_state_dict(blob["model"]); m.eval()
    scale, vmax = sf_params(bits)

    d = np.load(data, allow_pickle=True)
    X = torch.from_numpy(d["X"][:n])

    peak = {}
    def note(name, t):
        v = float(t.abs().max())
        peak[name] = max(peak.get(name, 0.0), v)

    with torch.no_grad():
        note("input_features", X)
        h = m.stem(X); note("stem_out", h)
        for i, b in enumerate(m.blocks):
            nrm = b.norm(h);            note(f"b{i}_norm_out", nrm)
            fi = b.fc_in(nrm);          note(f"b{i}_fc_in_out", fi)
            si = torch.nn.functional.silu(fi); note(f"b{i}_silu_out", si)
            fo = b.fc_out(si);          note(f"b{i}_fc_out_out", fo)
            h = h + fo;                 note(f"b{i}_residual_after_add", h)
        hn = m.norm(h);                 note("final_norm_out", hn)
        note("head_choice_logits", m.head_choice(hn))
        note("head_score_logits", m.head_score(hn))
        note("head_noul_logit", m.head_noul(hn))

    rows = []
    for k, v in peak.items():
        rows.append({"site": k, "peak_abs": v, "sf_bound": vmax,
                     "over_bound": v > vmax, "ratio": v / vmax})
    return {"ckpt": os.path.basename(ckpt), "bits": bits,
            "quant_act": bool(cfg.get("quant_act")), "sf_vmax": vmax,
            "sites": rows, "complete": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="runs/bc_v2_val.npz")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    r = audit(a.ckpt, a.data)
    print("%s  (bits=%d, saturated outputs=%s)  SF bound +-%.4f"
          % (r["ckpt"], r["bits"], r["quant_act"], r["sf_vmax"]))
    print("%-26s %12s %10s  %s" % ("site", "peak |x|", "x bound", ""))
    for s in r["sites"]:
        flag = "OVER" if s["over_bound"] else ""
        print("%-26s %12.4f %9.1fx  %s" % (s["site"], s["peak_abs"], s["ratio"], flag))
    over = [s for s in r["sites"] if s["over_bound"]]
    print("\n%d of %d sites exceed the SF%d register bound" % (len(over), len(r["sites"]), r["bits"]))
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
