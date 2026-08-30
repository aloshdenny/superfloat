"""11M D/N sweep under scale absorption (the open 4.1 follow-up).

exp3 ran this grid at 5M, where the inversion does not exist. This is the same
recipe at 11M: tpp 5/10/20/40, bits 0/2/3/4/6, one seed. 4GB-safe defaults
(microbatch 1, accum 16, checkpoint, chunked CE) keep the effective batch at
the archived 16 x 1024.

    EXP3_OUT=./results EXP3_TOKENS=./fineweb_edu_tokens.bin \\
      python exp3_11m.py --prepare
    python exp3_11m.py --tpp 20 --bits 2 --seed 0
    python exp3_11m.py --queue            # remaining cells, skip-if-done
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from superfloat import disable_tf32, apply_superfloat, SFLinear, sf_quantize_sv

VOCAB, SEQLEN = 50304, 1024
OUT = os.environ.get("EXP3_OUT", "/workspace/results")
TOKENS = os.environ.get("EXP3_TOKENS", "/workspace/fineweb_edu_tokens.bin")
TARGET_TOKENS = 500_000_000
CONFIGS = {"5m": (256, 6, 4), "11m": (384, 6, 6)}
QUEUE_TPP = (5, 10, 20, 40)
QUEUE_BITS = (0, 2, 3, 4, 6)


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


def build(d_model, n_layer, n_head, use_ckpt):
    class Attn(nn.Module):
        def __init__(s):
            super().__init__()
            s.q = nn.Linear(d_model, d_model)
            s.k = nn.Linear(d_model, d_model)
            s.v = nn.Linear(d_model, d_model)
            s.proj = nn.Linear(d_model, d_model)
            s.n_head = n_head

        def forward(s, x):
            B, T, C = x.shape
            hd = C // s.n_head
            q = s.q(x).view(B, T, s.n_head, hd).transpose(1, 2)
            k = s.k(x).view(B, T, s.n_head, hd).transpose(1, 2)
            v = s.v(x).view(B, T, s.n_head, hd).transpose(1, 2)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return s.proj(o.transpose(1, 2).reshape(B, T, C))

    class Block(nn.Module):
        def __init__(s):
            super().__init__()
            s.ln1 = nn.LayerNorm(d_model)
            s.attn = Attn()
            s.ln2 = nn.LayerNorm(d_model)
            s.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model), nn.GELU(),
                nn.Linear(4 * d_model, d_model))

        def forward(s, x):
            x = x + s.attn(s.ln1(x))
            return x + s.mlp(s.ln2(x))

    class GPT(nn.Module):
        def __init__(s):
            super().__init__()
            s.wte = nn.Embedding(VOCAB, d_model)
            s.wpe = nn.Embedding(SEQLEN, d_model)
            s.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
            s.lnf = nn.LayerNorm(d_model)
            s.head = nn.Linear(d_model, VOCAB, bias=False)
            s.head.weight = s.wte.weight
            s.use_ckpt = use_ckpt

        def hidden(s, idx):
            x = s.wte(idx) + s.wpe(torch.arange(idx.shape[1], device=idx.device))
            for b in s.blocks:
                if s.use_ckpt and s.training:
                    x = checkpoint(b, x, use_reentrant=False)
                else:
                    x = b(x)
            return s.lnf(x)

        def forward(s, idx):
            return s.head(s.hidden(idx))

    return GPT()


def install_col_norm(model):
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

    n = 0
    for blk in model.blocks:
        at = blk.attn
        grp = [at.q.weight, at.k.weight, at.v.weight]
        for m in (at.q, at.k, at.v):
            m.__class__ = SFLinearCol
            m.sf_group = grp
            n += 1
        blk.mlp[0].__class__ = SFLinearCol
        blk.mlp[0].sf_group = None
        n += 1
    return n


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
    d_model, n_layer, n_head = CONFIGS[a.size]
    disable_tf32()
    torch.manual_seed(a.seed)
    tag = f"exp3_{a.size}_tpp{a.tpp}_" + ("fp32" if a.bits == 0 else f"sf{a.bits}") + f"_s{a.seed}"
    path = f"{OUT}/{tag}.json"
    if os.path.exists(path):
        print(f"[{tag}] done, skip", flush=True)
        return
    model = build(d_model, n_layer, n_head, a.checkpoint).cuda()
    n_ne = sum(p.numel() for p in model.parameters()) \
           - model.wte.weight.numel() - model.wpe.weight.numel()
    ncol = 0
    if a.bits:
        nconv = apply_superfloat(model, bits=a.bits, head_names=("head", "wte", "wpe"),
                                 quantize_activations=False)
        assert nconv == 6 * n_layer, f"quantized {nconv}, expected {6 * n_layer}"
        ncol = install_col_norm(model)
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

    hist, t0 = [], time.time()
    for step in range(steps):
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
    rec = {"exp": "exp3_11m", "size": a.size, "bits": a.bits, "tpp": a.tpp,
           "seed": a.seed, "n_nonembed": n_ne, "tokens": total, "steps": steps,
           "micro": a.micro, "accum": a.accum, "checkpoint": a.checkpoint,
           "col_norm_layers": ncol, "final_val_loss": hist[-1]["val_loss"],
           "minutes": (time.time() - t0) / 60, "history": hist, "complete": True}
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(path, "w"))
    print(f"[{tag}] DONE val={hist[-1]['val_loss']:.4f} ({rec['minutes']:.0f}m)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=3)
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
        for tpp in QUEUE_TPP:
            for bits in QUEUE_BITS:
                a.tpp, a.bits = tpp, bits
                run_one(a)
        return
    run_one(a)


if __name__ == "__main__":
    main()
