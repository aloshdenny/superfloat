"""Mixed-precision layer-group allocation, stage 2: modern block shape.

Stage 1 (exp3_mixed.py) ran this on a GPT-2-style block (LayerNorm/GELU/MHA)
and found protect-C beats uniform by ~37x at a matched 4-bit average, both
seeds. That result could be an artifact of the GPT-2 block specifically --
this repeats the same group allocation (A: q/k/v, B: mlp-in, C: o + mlp-out)
on the RMSNorm/SwiGLU/GQA block used everywhere else in this study (stage0,
puresf_llm), weights-only, no activation quantization, to check the win
survives a block shape where GQA gives k/v far fewer params than q and SwiGLU
splits "mlp-in" into two parallel projections (gate, up) instead of one.

The per-group bit assignments (uniform6 6/6/6, protect-C 5/5/8, ...) are
carried over unchanged from stage 1's 33/33/33-derived arms; GQA/SwiGLU shift
each group's real parameter share, so the achieved average bit width is not
exactly the nominal target -- it is computed from real numel counts and
printed per cell (avg_bits), same discipline exp3_mixed.py already uses.

    EXP3_OUT=./results EXP3_TOKENS=./fineweb_edu_tokens.bin \\
      python exp3_mixed_modern.py --queue
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from superfloat import disable_tf32, apply_superfloat, SFLinear, sf_quantize_sv, sf_params

VOCAB, SEQLEN = 50304, 1024
OUT = os.environ.get("EXP3_OUT", "/workspace/results")
TOKENS = os.environ.get("EXP3_TOKENS", "/workspace/fineweb_edu_tokens.bin")
TARGET_TOKENS = 500_000_000
CONFIGS = {"11m": dict(d=384, n_layer=6, nh=6, nkv=2, hidden=1024)}

# arm -> (bits_a, bits_b, bits_c), carried over from stage 1 (EXPERIMENT_PLAN_mixed.md)
ARMS = {
    "uniform6":  (6, 6, 6),
    "protect-C6": (5, 5, 8),
    "starve-C6": (7, 7, 4),
    "protect-A6": (8, 5, 5),
    "uniform4":  (4, 4, 4),
    "protect-C4": (3, 3, 6),
    "starve-C4": (5, 5, 2),
}


def prepare_tokens():
    if os.path.exists(TOKENS):
        p = np.memmap(TOKENS, dtype=np.uint16, mode="r")
        if len(p) >= TARGET_TOKENS and p[-4096:].max() > 0:
            print(f"corpus present: {len(p)/1e6:.0f}M", flush=True)
            return
        del p
        os.remove(TOKENS)
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    eot = tok.eos_token_id
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    buf = np.memmap(TOKENS, dtype=np.uint16, mode="w+", shape=(TARGET_TOKENS,))
    i, batch = 0, []

    def flush(b, i):
        for ids in tok(b)["input_ids"]:
            ids = ids + [eot]
            if i + len(ids) > TARGET_TOKENS:
                ids = ids[:TARGET_TOKENS - i]
            if not ids:
                break
            buf[i:i + len(ids)] = np.array(ids, dtype=np.uint16)
            i += len(ids)
        return i

    for row in ds:
        batch.append(row["text"])
        if len(batch) >= 2000:
            i = flush(batch, i)
            batch = []
            if i >= TARGET_TOKENS:
                break
    if i < TARGET_TOKENS:
        i = flush(batch, i)
    buf.flush()
    del buf
    print(f"corpus written: {i/1e6:.0f}M", flush=True)


# --------------------------------------------------------------- model -----
def rope(q, k, cos, sin):
    def rot(t):
        a, b = t.chunk(2, dim=-1); return torch.cat((-b, a), dim=-1)
    return q * cos + rot(q) * sin, k * cos + rot(k) * sin


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return self.w * (x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


def build(d, n_layer, nh, nkv, hidden, seqlen, use_ckpt):
    class Block(nn.Module):
        def __init__(s):
            super().__init__()
            s.nh, s.nkv, s.hd = nh, nkv, d // nh
            s.n1 = RMSNorm(d)
            s.q = nn.Linear(d, nh * s.hd, bias=False)
            s.k = nn.Linear(d, nkv * s.hd, bias=False)
            s.v = nn.Linear(d, nkv * s.hd, bias=False)
            s.o = nn.Linear(nh * s.hd, d, bias=False)
            s.n2 = RMSNorm(d)
            s.gate = nn.Linear(d, hidden, bias=False)
            s.up = nn.Linear(d, hidden, bias=False)
            s.down = nn.Linear(hidden, d, bias=False)

        def forward(s, x, cos, sin):
            B, T, C = x.shape
            h = s.n1(x)
            q = s.q(h).view(B, T, s.nh, s.hd).transpose(1, 2)
            k = s.k(h).view(B, T, s.nkv, s.hd).transpose(1, 2)
            v = s.v(h).view(B, T, s.nkv, s.hd).transpose(1, 2)
            q, k = rope(q, k, cos, sin)
            if s.nkv != s.nh:
                r = s.nh // s.nkv
                k = k.repeat_interleave(r, dim=1); v = v.repeat_interleave(r, dim=1)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            a = a.transpose(1, 2).reshape(B, T, -1)
            x = x + s.o(a)
            h = s.n2(x)
            m = F.silu(s.gate(h)) * s.up(h)
            return x + s.down(m)

    class LM(nn.Module):
        def __init__(s):
            super().__init__()
            s.wte = nn.Embedding(VOCAB, d)
            s.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
            s.nf = RMSNorm(d)
            s.head = nn.Linear(d, VOCAB, bias=False)
            s.head.weight = s.wte.weight
            s.use_ckpt = use_ckpt
            hd = d // nh
            inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
            t = torch.arange(seqlen).float()
            f = torch.outer(t, inv)
            emb = torch.cat((f, f), dim=-1)
            s.register_buffer("cos", emb.cos()[None, None], persistent=False)
            s.register_buffer("sin", emb.sin()[None, None], persistent=False)
            s.apply(s._init)
            for n_, p_ in s.named_parameters():
                if n_.endswith(("o.weight", "down.weight")):
                    nn.init.normal_(p_, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

        @staticmethod
        def _init(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        def hidden(s, idx):
            x = s.wte(idx)
            T = idx.shape[1]
            cos, sin = s.cos[:, :, :T], s.sin[:, :, :T]
            for b in s.blocks:
                if s.use_ckpt and s.training:
                    x = checkpoint(b, x, cos, sin, use_reentrant=False)
                else:
                    x = b(x, cos, sin)
            return s.nf(x)

        def forward(s, idx):
            return s.head(s.hidden(idx))

    return LM()


LAYER_GROUPS = ("A", "B", "C")   # A: q/k/v   B: gate/up   C: o + down


def install_mixed(model, bits_a, bits_b, bits_c):
    """Same recipe as exp3_mixed.install_mixed, ported to q/k/v/o/gate/up/down
    names. A and B keep an absorbable per-input-channel column scale (they are
    fed by an RMSNorm whose gain can carry it); C gets none (fed by no norm,
    residual-scaled init) -- see exp3_mixed.py for the full rationale."""
    class SFLinearCol(SFLinear):
        sf_group = None

        def forward(self, x):
            w = self.weight
            src = self.sf_group if self.sf_group is not None else [w]
            g = torch.stack([t.abs().amax(dim=0) for t in src]).amax(0).clamp_min(1e-8)
            wq = sf_quantize_sv(w / g.unsqueeze(0), self.sf_scale, self.sf_vmax)
            b = None if self.bias is None else sf_quantize_sv(
                self.bias, self.sf_scale, self.sf_vmax)
            return F.linear(x * g, wq, b)

    class SFLinearPlain(SFLinear):
        def forward(self, x):
            wq = sf_quantize_sv(self.weight, self.sf_scale, self.sf_vmax)
            b = None if self.bias is None else sf_quantize_sv(
                self.bias, self.sf_scale, self.sf_vmax)
            return F.linear(x, wq, b)

    counts = {"A": 0, "B": 0, "C": 0}
    wbits = 0.0
    wtot = 0.0
    for blk in model.blocks:
        grp = [blk.q.weight, blk.k.weight, blk.v.weight]
        for m in (blk.q, blk.k, blk.v):
            wtot += m.weight.numel(); wbits += m.weight.numel() * (bits_a or 32)
            if bits_a:
                s, v = sf_params(bits_a)
                m.__class__ = SFLinearCol; m.sf_scale, m.sf_vmax = s, v
                m.sf_group = grp; counts["A"] += 1
        grp_b = [blk.gate.weight, blk.up.weight]
        for m in (blk.gate, blk.up):
            wtot += m.weight.numel(); wbits += m.weight.numel() * (bits_b or 32)
            if bits_b:
                s, v = sf_params(bits_b)
                m.__class__ = SFLinearCol; m.sf_scale, m.sf_vmax = s, v
                m.sf_group = grp_b; counts["B"] += 1
        for m in (blk.o, blk.down):
            wtot += m.weight.numel(); wbits += m.weight.numel() * (bits_c or 32)
            if bits_c:
                s, v = sf_params(bits_c)
                m.__class__ = SFLinearPlain; m.sf_scale, m.sf_vmax = s, v
                counts["C"] += 1
    return counts, (wbits / wtot if wtot else 0.0)


def chunked_ce(model, x, y, chunk):
    h = model.hidden(x)
    T = y.shape[1]
    loss = h.new_zeros(())
    for i in range(0, T, chunk):
        logits = model.head(h[:, i:i + chunk])
        loss = loss + F.cross_entropy(
            logits.reshape(-1, VOCAB), y[:, i:i + chunk].reshape(-1)) * min(chunk, T - i)
    return loss / T


def run_one(a):
    cfg = CONFIGS[a.size]
    disable_tf32()
    torch.manual_seed(a.seed)
    if a.bits_a is not None:
        tag = f"mixmod_{a.size}_tpp{a.tpp}_{a.arm or 'a%sb%sc%s' % (a.bits_a, a.bits_b, a.bits_c)}_s{a.seed}"
    else:
        tag = f"mixmod_{a.size}_tpp{a.tpp}_" + ("fp32" if a.bits == 0 else f"sf{a.bits}") + f"_s{a.seed}"
    path = f"{OUT}/{tag}.json"
    if os.path.exists(path):
        print(f"[{tag}] done, skip", flush=True)
        return
    model = build(cfg["d"], cfg["n_layer"], cfg["nh"], cfg["nkv"], cfg["hidden"],
                  SEQLEN, a.checkpoint).cuda()
    n_ne = sum(p.numel() for p in model.parameters()) - model.wte.weight.numel()
    ncol = 0
    if a.bits:
        nconv = apply_superfloat(model, bits=a.bits, head_names=("head", "wte"),
                                 quantize_activations=False)
        assert nconv == 7 * cfg["n_layer"], f"quantized {nconv}, expected {7 * cfg['n_layer']}"
        if a.bits_a is not None:
            counts, avgbits = install_mixed(model, a.bits_a, a.bits_b, a.bits_c)
            ncol = counts["A"] + counts["B"]
            print(f"[{tag}] alloc A={a.bits_a} B={a.bits_b} C={a.bits_c} "
                  f"counts={counts} avg_bits={avgbits:.2f}", flush=True)
    eff_batch = a.micro * a.accum
    total = int(n_ne * a.tpp)
    steps = total // (eff_batch * SEQLEN)
    print(f"[{tag}] N={n_ne/1e6:.2f}M tpp={a.tpp} tokens={total/1e6:.0f}M "
          f"steps={steps} micro={a.micro} accum={a.accum} col={ncol} "
          f"mem={torch.cuda.memory_allocated()/2**20:.0f}MiB", flush=True)

    data = np.memmap(TOKENS, dtype=np.uint16, mode="r")
    assert len(data) >= TARGET_TOKENS and data[-4096:].max() > 0, "corpus incomplete"
    train_end = len(data) - 2_000_000
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=max(steps, 1), pct_start=0.02)
    rng = np.random.default_rng(a.seed)

    def get(lo, hi, bs):
        ix = rng.integers(lo, hi - SEQLEN - 1, size=bs)
        x = np.stack([data[i:i + SEQLEN] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1:i + 1 + SEQLEN] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()

    ckpt_path = f"{OUT}/{tag}.ckpt"
    hist, t0 = [], time.time()
    start_step = 0
    if os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            start_step = ck["step"] + 1
            hist = ck.get("hist", [])
            rng = np.random.default_rng(a.seed + start_step)
            print(f"[{tag}] resumed from step {start_step}/{steps}", flush=True)
        except Exception as e:
            print(f"[{tag}] checkpoint unreadable ({e}), starting fresh", flush=True)
            start_step = 0

    def save_ckpt(step):
        tmp = ckpt_path + ".tmp"
        torch.save({"step": step, "model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "hist": hist}, tmp)
        os.replace(tmp, ckpt_path)

    for step in range(start_step, steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            x, y = get(0, train_end, a.micro)
            loss = chunked_ce(model, x, y, a.chunk) / a.accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            model.eval()
            vl = k = 0.0
            with torch.no_grad():
                for _ in range(20):
                    xv, yv = get(train_end, len(data), a.micro)
                    vl += chunked_ce(model, xv, yv, a.chunk).item()
                    k += 1
            model.train()
            rec_h = {"step": step, "val_loss": vl / k,
                     "mem_mb": torch.cuda.max_memory_allocated() / 2 ** 20}
            hist.append(rec_h)
            print(f"[{tag}] {step}/{steps} val={rec_h['val_loss']:.4f} "
                  f"mem={rec_h['mem_mb']:.0f}MiB", flush=True)
            save_ckpt(step)
        elif step % 500 == 0:
            save_ckpt(step)
    rec = {"exp": "exp3_11m" if a.bits_a is None else "mixed_alloc_modern",
           "size": a.size, "bits": a.bits, "tpp": a.tpp,
           "arm": a.arm or None, "bits_a": a.bits_a, "bits_b": a.bits_b, "bits_c": a.bits_c,
           "seed": a.seed, "n_nonembed": n_ne, "tokens": total, "steps": steps,
           "micro": a.micro, "accum": a.accum, "checkpoint": a.checkpoint,
           "col_norm_layers": ncol, "final_val_loss": hist[-1]["val_loss"],
           "minutes": (time.time() - t0) / 60, "history": hist, "complete": True}
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(path, "w"))
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"[{tag}] DONE val={hist[-1]['val_loss']:.4f} ({rec['minutes']:.0f}m)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--bits-a", type=int, default=None)
    ap.add_argument("--bits-b", type=int, default=None)
    ap.add_argument("--bits-c", type=int, default=None)
    ap.add_argument("--arm", default="")
    ap.add_argument("--tpp", type=int, default=10)
    ap.add_argument("--size", default="11m", choices=sorted(CONFIGS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--micro", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--checkpoint", action="store_true", default=True)
    ap.add_argument("--no-checkpoint", dest="checkpoint", action="store_false")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--queue", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if a.prepare:
        prepare_tokens()
        return
    if a.queue:
        # fp32 control + 7 arms, two seeds, matching stage 1's cell count
        for seed in (0, 1):
            a.seed = seed
            a.bits, a.bits_a, a.bits_b, a.bits_c, a.arm = 0, None, None, None, ""
            run_one(a)
            for arm, (ba, bb, bc) in ARMS.items():
                a.bits, a.bits_a, a.bits_b, a.bits_c, a.arm = 6, ba, bb, bc, arm
                run_one(a)
        return
    run_one(a)


if __name__ == "__main__":
    main()
