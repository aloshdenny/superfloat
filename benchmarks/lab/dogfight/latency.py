"""Decision latency: the GPU denominator for an Atreides speedup ratio.

Read this before quoting any number out of it.

SuperFloat in this repository is SIMULATED. `sf_quantize_sv` clamps and rounds
to the grid with a straight-through estimator and the tensor stays float, so
the matmul is fp32 whatever the bit width.

That does NOT make the arms equally fast, and assuming so was the first
mistake here: measured on CPU, the SF arms run ~10x SLOWER than fp32, because
the quantizer itself executes on every forward. That cost is an artefact of
simulating the format and would not exist on a datapath whose weights are
already grid points. **Timing an SFLinear model tells you the price of the
simulation, not the price of the format, and it points the wrong way.**

The honest denominator is therefore the DEPLOYED model: the same weights,
rounded to the grid once, executed by an ordinary Linear with no quantizer in
the loop. That is what hardware actually runs, and it is what `deployed_ms`
measures below. It should come out the same for every precision on a GPU,
since a GPU has no narrow datapath to exploit -- the win from fewer bits lives
in silicon, not here. The numerator comes from the hardware side -- Atreides, or any
RTL mounted under Yosys -- where the bit width genuinely changes the datapath.
The bridge between them is MACs per decision, reported here, since

    cycles  ~= MACs / (PEs x MACs-per-PE-per-cycle)
    latency  = cycles / Fmax

and the same MAC count drives both. What DOES differ per precision on any
platform is the weight footprint, also reported, because that decides whether
the policy fits in on-chip SRAM -- which for a 2.1M-parameter model at 4 bits
is the difference between a memory-bound and a compute-bound design.
"""
from __future__ import annotations

import argparse, json, os, statistics, sys, time

import torch

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_here, "..", ".."), os.path.join(_here, ".."), _here):
    sys.path.insert(0, os.path.abspath(_p))
from superfloat import apply_superfloat, disable_tf32

from policy import Policy, N_FEAT, param_count

OUT = os.environ.get("DOGFIGHT_OUT", "runs")
HEADS = ("head_choice", "head_score", "head_noul")


def macs_per_decision(d, depth, mult=4, n_feat=N_FEAT, n_man=9, n_threat=3):
    """Multiply-accumulates in one forward pass, batch 1."""
    m = n_feat * d                       # stem
    m += depth * (d * mult * d + mult * d * d)   # fc_in + fc_out per block
    m += d * (n_man + n_threat + 1)      # heads
    return m


def deployed(model):
    """Strip the quantizer, keep the quantized values.

    Rebinds SFLinear back to nn.Linear after baking the rounded weights in, so
    the forward is a plain fp32 matmul over grid-valued weights -- what a
    deployment runs, with no simulation cost in the measurement.
    """
    import torch.nn as nn
    from superfloat import SFLinear, sf_quantize_sv
    for m in model.modules():
        if isinstance(m, SFLinear):
            with torch.no_grad():
                m.weight.copy_(sf_quantize_sv(m.weight, m.sf_scale, m.sf_vmax))
            m.__class__ = nn.Linear
    return model


def bench(model, dev, iters=2000, warmup=300, amp=None):
    x = torch.randn(1, N_FEAT, device=dev)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            with torch.autocast(dev, dtype=amp, enabled=amp is not None):
                model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            with torch.autocast(dev, dtype=amp, enabled=amp is not None):
                model(x)
            if dev == "cuda":
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return {
        "mean_ms": statistics.fmean(samples),
        "p50_ms": samples[len(samples) // 2],
        "p99_ms": samples[int(len(samples) * 0.99)],
        "min_ms": samples[0],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--bits", default="0,8,6,4,3,2")
    ap.add_argument("--pes", type=int, default=256, help="Atreides PE count, for the projection")
    ap.add_argument("--fmax-mhz", type=float, default=100.0)
    ap.add_argument("--out", default=os.path.join(OUT, "latency.json"))
    a = ap.parse_args()
    disable_tf32()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    macs = macs_per_decision(a.d, a.depth)
    params = param_count(Policy(a.d, a.depth))

    rows = []
    for b in [int(x) for x in a.bits.split(",")]:
        m = Policy(a.d, a.depth).to(dev)
        if b:
            apply_superfloat(m, b, head_names=HEADS, quantize_activations=False)
        sim = bench(m, dev, a.iters) if b else None
        dep = bench(deployed(m), dev, a.iters)
        r = {"bits": b, "deployed": dep, "simulated": sim,
             "weight_kib": (params * (b or 32)) / 8 / 1024}
        rows.append(r)
        print("SF%-4s deployed p50 %.3f ms  p99 %.3f ms | simulated p50 %s | weights %6.0f KiB"
              % (b or "fp32", dep["p50_ms"], dep["p99_ms"],
                 ("%.3f ms" % sim["p50_ms"]) if sim else "   n/a  ", r["weight_kib"]))

    spread = (max(r["deployed"]["p50_ms"] for r in rows)
              / min(r["deployed"]["p50_ms"] for r in rows))
    cyc = macs / a.pes
    proj_ms = cyc / (a.fmax_mhz * 1e6) * 1e3
    out = {
        "exp": "dogfight_latency", "device": dev, "d": a.d, "depth": a.depth,
        "params": params, "macs_per_decision": macs, "arms": rows,
        "gpu_p50_spread": spread,
        "atreides_projection": {
            "pes": a.pes, "fmax_mhz": a.fmax_mhz,
            "cycles_per_decision": cyc, "latency_ms": proj_ms,
            "note": "one MAC per PE per cycle; replace with measured RTL numbers",
        },
        "complete": True,
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("\nMACs/decision %.2fM   params %.2fM" % (macs / 1e6, params / 1e6))
    print("deployed p50 spread across precisions: %.3fx  <- expect ~1.0 on a GPU" % spread)
    sims = [r for r in rows if r["simulated"]]
    if sims:
        fp = [r for r in rows if r["bits"] == 0][0]["deployed"]["p50_ms"]
        print("simulation overhead (SFLinear vs deployed): %.1fx -- an artefact, not a result"
              % (statistics.fmean(r["simulated"]["p50_ms"] for r in sims) / fp))
    print("Atreides projection @ %d PEs, %.0f MHz: %.0f cycles = %.3f ms"
          % (a.pes, a.fmax_mhz, cyc, proj_ms))


if __name__ == "__main__":
    main()
