"""Needle-in-a-haystack for a base LM, scored by loss rather than by answer.

A prompted needle test ("the magic number is X ... what is the magic number?")
assumes an instruction-following model. These checkpoints are base LMs trained
on 6.4B tokens, so a prompted test scores ~0 for every arm and measures nothing.

The loss-based version asks the same question without needing instructions:
plant a random, unpredictable token sequence (the needle) at depth d of a long
haystack, then re-present its first `prefix` tokens at the very end and measure
NLL on the remaining tokens. A model that attends to the far context copies the
rest and the NLL collapses; a model that does not is left predicting random
tokens and pays ~log(vocab) nats. The control repeats the experiment with a
DIFFERENT needle planted, so the only thing that changes is whether the answer
is actually retrievable -- this isolates retrieval from "random tokens are hard".

    retrieval score = NLL(control) - NLL(planted), in nats; 0 = no retrieval.
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_1b import CFG, Llama, quantize, clean_state_dict, complete_shards, DTYPE

OUT = os.environ.get("NIAH_OUT", "/workspace/results")


def load_model(ckpt, bits, mode, seqlen, device="cuda", dtype=torch.bfloat16):
    model = Llama(CFG, seqlen)
    blob = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
    sd = clean_state_dict(blob["model"] if "model" in blob else blob)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    src_step = blob.get("step", -1)
    del blob, sd
    # bf16 weights: halves the resident model (5.0 -> 2.5 GB) so a 128K forward
    # fits a 20 GB card. Safe for SF8 because every SF8 grid point is exactly
    # representable in bf16; do NOT do this for SF16.
    if bits and bits > 8:
        dtype = torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    nq = quantize(model, bits, mode) if bits else 0
    print(f"  loaded step={src_step} missing={len(missing)} unexpected={len(unexpected)} "
          f"quantized={nq}", flush=True)
    return model


def haystack_tokens(data, n, rng):
    lo = rng.integers(0, len(data) - n - 1)
    return np.asarray(data[lo:lo + n], dtype=np.int64)


@torch.no_grad()
def needle_nll(model, ctx_ids, score_ids, device="cuda"):
    """NLL (nats/token) on score_ids given ctx_ids as the prefix."""
    x = torch.from_numpy(np.concatenate([ctx_ids, score_ids])[None, :]).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.hidden(x[:, :-1])
        lg = model.lm_head(h[:, -len(score_ids):])
    tgt = torch.from_numpy(score_ids[None, :]).to(device)
    return F.cross_entropy(lg.reshape(-1, CFG["vocab"]).float(), tgt.reshape(-1)).item()


def run(a):
    torch.manual_seed(a.seed)
    data = np.memmap(complete_shards(a.data)[0], dtype=DTYPE, mode="r")
    rng = np.random.default_rng(a.seed)
    model = load_model(a.ckpt, a.bits, a.mode, a.ctx,
                       dtype=torch.float32 if a.fp32 else torch.bfloat16)

    rows = []
    for depth in [float(d) for d in a.depths.split(",")]:
        planted, control = [], []
        for _ in range(a.samples):
            hay = haystack_tokens(data, a.ctx, rng)
            needle = rng.integers(0, CFG["vocab"], size=a.needle).astype(np.int64)
            other = rng.integers(0, CFG["vocab"], size=a.needle).astype(np.int64)
            cue, tail = needle[:a.prefix], needle[a.prefix:]
            # the scored sequence is body + needle + cue + tail, and that must
            # come to exactly a.ctx so it fits the RoPE buffers built for it:
            # len(body) + needle + prefix + (needle - prefix) = len(body) + 2*needle
            body = hay[: a.ctx - 2 * a.needle]
            pos = int(depth * (len(body) - 1))
            def build(planted_needle):
                return np.concatenate([body[:pos], planted_needle, body[pos:], cue])
            planted.append(needle_nll(model, build(needle), tail))
            control.append(needle_nll(model, build(other), tail))
        p, c = float(np.mean(planted)), float(np.mean(control))
        rows.append(dict(depth=depth, nll_planted=p, nll_control=c, retrieval=c - p,
                         n=a.samples))
        print(f"  depth {depth:.2f}: planted {p:.3f}  control {c:.3f}  "
              f"retrieval {c - p:+.3f} nats", flush=True)

    rec = dict(exp="niah_1b", ckpt=a.ckpt, bits=a.bits, mode=a.mode, ctx=a.ctx,
               needle=a.needle, prefix=a.prefix, samples=a.samples, seed=a.seed,
               vocab_nats=float(np.log(CFG["vocab"])), rows=rows,
               mean_retrieval=float(np.mean([r["retrieval"] for r in rows])))
    os.makedirs(OUT, exist_ok=True)
    # the checkpoint identity MUST be in the tag: two models of the same arm
    # evaluated at the same ctx (e.g. the 8K base vs the 128K-extended one)
    # otherwise silently overwrite each other's results.
    src = os.path.splitext(os.path.basename(a.ckpt))[0]
    tag = f"niah_{src}_{'bf16' if not a.bits else f'sf{a.bits}'}_ctx{a.ctx}_s{a.seed}"
    json.dump(rec, open(f"{OUT}/{tag}.json", "w"), indent=2)
    print(f"[{tag}] mean retrieval {rec['mean_retrieval']:+.3f} nats "
          f"(random baseline would be 0; perfect copy ~{rec['vocab_nats']:.2f})", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="/vol/sf1b_data")
    ap.add_argument("--bits", type=int, default=0)
    ap.add_argument("--mode", default="ln_all")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--depths", default="0.1,0.25,0.5,0.75,0.9")
    ap.add_argument("--needle", type=int, default=64)
    ap.add_argument("--prefix", type=int, default=16)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp32", action="store_true", help="keep weights fp32 (needs >20 GB at 128K)")
    a = ap.parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    run(a)


if __name__ == "__main__":
    main()
