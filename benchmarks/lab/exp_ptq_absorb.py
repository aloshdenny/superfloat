"""PTQ under scale absorption — the open cell in SCALING_LAWS.md §6 / headline.

Tier B / exp2 quantize weights in place with no per-channel scale. The C/D
fix never ran on PTQ. This is that measurement: same Pythia checkpoints and
WikiText eval as exp2, with an optional per-input-channel scale absorbed into
the preceding activation (g lives outside the matmul).

Fits a 4GB card at --batch 1 for 70m/160m/410m. 1.4b needs more VRAM.

    EXP2_OUT=./results python exp_ptq_absorb.py --prepare
    python exp_ptq_absorb.py --queue
    python exp_ptq_absorb.py --size 160m --step 143000 --bits 4 --absorb
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from superfloat import disable_tf32, sf_params, sf_quantize_sv

OUT = os.environ.get("EXP2_OUT", "/workspace/results")
EVAL = os.environ.get("EXP2_EVAL", "/workspace/eval_tokens.npy")
SEQLEN = 2048
SIZES = ["70m", "160m", "410m"]
STEPS = [143000]
PRECS = [0, 4, 6, 8]
TOK_PER_STEP = 2_097_152
HEAD = ("embed_out", "lm_head")


def prepare_eval():
    if os.path.exists(EVAL):
        print("eval slice present", flush=True)
        return
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    ids = []
    for row in ds:
        if row["text"].strip():
            ids.extend(tok(row["text"]).input_ids)
        if len(ids) >= 400_000:
            break
    os.makedirs(os.path.dirname(EVAL) or ".", exist_ok=True)
    np.save(EVAL, np.array(ids[:400_000], dtype=np.uint16))
    print(f"eval slice: {len(ids[:400_000])} tokens", flush=True)


class AbsorbLinear(nn.Module):
    """w <- quant(w / g_in); y = (x * g_in) @ w_q. g never enters the SF grid."""

    def __init__(self, linear, bits):
        super().__init__()
        scale, vmax = sf_params(bits)
        w = linear.weight.data.float()
        g = w.abs().amax(dim=0).clamp_min(1e-8)
        wq = sf_quantize_sv(w / g.unsqueeze(0), scale, vmax)
        self.weight = nn.Parameter(wq.to(linear.weight.dtype), requires_grad=False)
        self.register_buffer("g", g.to(linear.weight.dtype))
        self.bias = None
        if linear.bias is not None:
            self.bias = nn.Parameter(
                sf_quantize_sv(linear.bias.data.float(), scale, vmax).to(linear.weight.dtype),
                requires_grad=False)

    def forward(self, x):
        return F.linear(x * self.g, self.weight, self.bias)


def install_absorb(model, bits):
    n = 0
    for name, child in list(model.named_modules()):
        if not isinstance(child, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in HEAD:
            continue
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], AbsorbLinear(child, bits).to(child.weight.device))
        n += 1
    return n


def plain_ptq(model, bits):
    dead = tot = 0
    scale, vmax = sf_params(bits)
    with torch.no_grad():
        for n, m in model.named_modules():
            if not isinstance(m, nn.Linear):
                continue
            if n.rsplit(".", 1)[-1] in HEAD:
                continue
            q = sf_quantize_sv(m.weight.data.float(), scale, vmax)
            dead += (q == 0).sum().item()
            tot += q.numel()
            m.weight.data = q.to(m.weight.dtype)
    return dead, tot


def evaluate(model, batch):
    ids = np.load(EVAL).astype(np.int64)
    n_seq = len(ids) // SEQLEN
    x = torch.from_numpy(ids[:n_seq * SEQLEN]).view(n_seq, SEQLEN)
    nll = ntok = 0.0
    with torch.no_grad():
        for i in range(0, n_seq, batch):
            xb = x[i:i + batch].cuda()
            out = model(xb, labels=xb)
            k = xb.numel() - xb.shape[0]
            nll += out.loss.item() * k
            ntok += k
    return nll / ntok


def run_one(a):
    mode = "absorb" if a.absorb else "plain"
    tag = f"ptqabs_{a.size}_step{a.step}_" + (
        "fp16" if a.bits == 0 else f"sf{a.bits}_{mode}")
    path = f"{OUT}/{tag}.json"
    if os.path.exists(path):
        print(f"[{tag}] done, skip", flush=True)
        return
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{a.size}", revision=f"step{a.step}",
        torch_dtype=torch.float16).cuda().eval()
    dead = tot = nwrap = 0
    if a.bits:
        heads = [n for n, m in model.named_modules()
                 if isinstance(m, nn.Linear) and n.rsplit(".", 1)[-1] in HEAD]
        if len(heads) != 1:
            raise RuntimeError(f"expected one head, found {heads}")
        if a.absorb:
            nwrap = install_absorb(model, a.bits)
        else:
            dead, tot = plain_ptq(model, a.bits)
    loss = evaluate(model, a.batch)
    rec = {"exp": "ptq_absorb", "size": a.size, "step": a.step, "bits": a.bits,
           "absorb": bool(a.absorb and a.bits), "tokens": a.step * TOK_PER_STEP,
           "val_loss": loss, "dead_frac": dead / max(tot, 1), "wrapped": nwrap,
           "minutes": (time.time() - t0) / 60, "complete": True}
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(path, "w"))
    print(f"[{tag}] loss={loss:.4f} absorb={rec['absorb']} "
          f"dead={100*rec['dead_frac']:.1f}% {rec['minutes']:.1f}m", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="160m")
    ap.add_argument("--step", type=int, default=143000)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--absorb", action="store_true")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--queue", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if a.prepare:
        prepare_eval()
        return
    if a.queue:
        prepare_eval()
        for size in SIZES:
            a.size = size
            a.bits, a.absorb = 0, False
            run_one(a)
            for bits in PRECS:
                if bits == 0:
                    continue
                for absorb in (False, True):
                    a.bits, a.absorb = bits, absorb
                    run_one(a)
        return
    run_one(a)


if __name__ == "__main__":
    main()
