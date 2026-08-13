"""Tier A of the SuperFloat scaling-law study: from-scratch QAT LM ladder.

Trains decoder-only LMs at four non-embedding sizes under six SFx precisions
plus an FP32 control, on a fixed pre-tokenized FineWeb-Edu slice, and records
validation loss. This is the arm that supplies L(N, D, p).

Token budget is 10x non-embedding params, i.e. half Chinchilla. That is a
deliberate compromise: a full 20x budget costs ~$210 for the precision sweep
alone, against ~$120 here. Relative degradation across precision at matched
(N, D) -- which is what the law needs -- is unaffected, but the absolute
losses are not compute-optimal and the paper has to say so.

N is counted excluding embeddings, per Kaplan and Chinchilla convention;
embeddings are tied and left in FP32 along with the LM head, matching the
recipe used elsewhere in this project.

GPU is measured, not assumed: modal_profile.py puts H100 at 61.6k tok/s for a
d=768 block stack, best value of four candidates -- B200 is 1.2x faster for
1.6x the price. Note the probe omitted embeddings and data loading, so budget
roughly 20-30% above its projection.

Run:
    modal run modal/modal_scaling_a.py::prepare_tokens   # ~1B tokens, once
    modal run modal/modal_scaling_a.py --smoke
    modal run modal/modal_scaling_a.py
"""

import pathlib

import modal

BENCH_DIR = str(pathlib.Path(__file__).resolve().parent.parent)

app = modal.App("superfloat-scaling-a")
vol = modal.Volume.from_name("sfx-baselines", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("transformers", "datasets", "numpy", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/vol/hf"})
    .add_local_dir(BENCH_DIR, remote_path="/root/sfx_bench")
)

GPU = "H100"
OUT = "/vol/runs_scaling_a"
TOKENS = "/vol/fineweb_edu_tokens.bin"

VOCAB = 50304                  # GPT-2 vocab padded to a multiple of 64
SEQLEN = 1024
# (d_model, n_layer, n_head); non-embedding params ~ 12 * L * d^2
CONFIGS = {
    "5m":  (256, 6, 4),
    "11m": (384, 6, 6),
    "25m": (512, 8, 8),
    "85m": (768, 12, 12),
}
PRECISIONS = [3, 4, 5, 6, 8, 16]
TOKENS_PER_PARAM = 10          # half Chinchilla; see module docstring
TARGET_TOKENS = 1_100_000_000  # enough for the largest run's 10x budget


@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 60 * 3, cpu=16)
def prepare_tokens():
    """Pre-tokenize FineWeb-Edu into a flat uint16 memmap, once."""
    import os
    import numpy as np
    # np.memmap(mode="w+") sizes the file up front, so existence and size say
    # nothing about how much was actually written -- a killed run leaves a
    # full-size file whose tail is still zeros. Check real content instead, or
    # a half-tokenized file would silently train on zero padding.
    if os.path.exists(TOKENS):
        probe = np.memmap(TOKENS, dtype=np.uint16, mode="r")
        if len(probe) >= TARGET_TOKENS and probe[-4096:].max() > 0:
            print(f"already present: {len(probe)/1e6:.0f}M tokens", flush=True)
            return len(probe)
        print("partial token file, re-tokenizing", flush=True)
        del probe
        os.remove(TOKENS)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    eot = tok.eos_token_id
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)

    buf = np.memmap(TOKENS, dtype=np.uint16, mode="w+", shape=(TARGET_TOKENS,))
    i = 0
    for row in ds:
        ids = tok(row["text"]).input_ids
        ids.append(eot)
        if i + len(ids) > TARGET_TOKENS:
            ids = ids[: TARGET_TOKENS - i]
        buf[i:i + len(ids)] = np.array(ids, dtype=np.uint16)
        i += len(ids)
        if i % 50_000_000 < len(ids):
            print(f"  {i/1e6:.0f}M tokens", flush=True)
            buf.flush()
            vol.commit()
        if i >= TARGET_TOKENS:
            break
    buf.flush()
    del buf
    vol.commit()
    print(f"wrote {i/1e6:.0f}M tokens", flush=True)
    return i


def build_gpt(d_model, n_layer, n_head):
    import torch
    import torch.nn as nn

    class Attn(nn.Module):
        """Explicit q/k/v projections.

        nn.MultiheadAttention stores QKV as a single raw Parameter
        (in_proj_weight), not an nn.Linear, so apply_superfloat cannot see it
        and 25% of every block would silently stay FP32 -- which would mean
        "SF4" did not actually denote SF4. Separate nn.Linear layers keep the
        whole block inside the format.
        """

        def __init__(self):
            super().__init__()
            self.q = nn.Linear(d_model, d_model)
            self.k = nn.Linear(d_model, d_model)
            self.v = nn.Linear(d_model, d_model)
            self.proj = nn.Linear(d_model, d_model)

        def forward(self, x):
            import torch.nn.functional as F
            B, T, C = x.shape
            hd = C // n_head
            q = self.q(x).view(B, T, n_head, hd).transpose(1, 2)
            k = self.k(x).view(B, T, n_head, hd).transpose(1, 2)
            v = self.v(x).view(B, T, n_head, hd).transpose(1, 2)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.proj(o.transpose(1, 2).reshape(B, T, C))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = Attn()
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                     nn.Linear(4 * d_model, d_model))

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            return x + self.mlp(self.ln2(x))

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(VOCAB, d_model)
            self.wpe = nn.Embedding(SEQLEN, d_model)
            self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
            self.lnf = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, VOCAB, bias=False)
            self.head.weight = self.wte.weight        # tied

        def forward(self, idx):
            b, t = idx.shape
            pos = torch.arange(t, device=idx.device)
            x = self.wte(idx) + self.wpe(pos)
            for blk in self.blocks:
                x = blk(x)
            return self.head(self.lnf(x))

    return GPT()


def nonembed_params(model):
    tot = sum(p.numel() for p in model.parameters())
    emb = model.wte.weight.numel() + model.wpe.weight.numel()
    return tot - emb          # head is tied to wte, already counted once


@app.function(image=image, gpu=GPU, volumes={"/vol": vol},
              timeout=60 * 60 * 12, max_containers=7)
def train(size: str, bits: int, batch: int = 16, lr: float = 6e-4):
    """One (N, p) point. bits=0 is the FP32 control."""
    import json
    import math
    import os
    import sys
    import time
    import numpy as np
    import torch
    sys.path.insert(0, "/root/sfx_bench")
    from superfloat import disable_tf32, apply_superfloat, clamp_all

    disable_tf32()
    torch.manual_seed(0)
    d_model, n_layer, n_head = CONFIGS[size]
    tag = f"{size}_" + ("fp32" if bits == 0 else f"sf{bits}")
    os.makedirs(OUT, exist_ok=True)

    model = build_gpt(d_model, n_layer, n_head).cuda()
    n_ne = nonembed_params(model)
    if bits:
        # weights only; embeddings, position table and tied head stay FP32
        nconv = apply_superfloat(model, bits=bits,
                                 head_names=("head", "wte", "wpe"),
                                 quantize_activations=False)
    else:
        nconv = 0

    total_tokens = int(n_ne * TOKENS_PER_PARAM)
    steps = total_tokens // (batch * SEQLEN)
    if bits:
        expected = 6 * n_layer          # q,k,v,proj + 2 mlp per block
        if nconv != expected:
            raise RuntimeError(
                f"quantized {nconv} layers, expected {expected}; "
                "some weights would silently stay FP32")
    print(f"[{tag}] non-embed N={n_ne/1e6:.1f}M sf_layers={nconv} "
          f"D={total_tokens/1e6:.0f}M tokens steps={steps}", flush=True)

    data = np.memmap(TOKENS, dtype=np.uint16, mode="r")
    # refuse a half-written corpus rather than quietly training on zero padding
    if len(data) < TARGET_TOKENS or data[-4096:].max() == 0:
        raise RuntimeError(
            f"token file incomplete ({len(data)} tokens, tail is zeros); "
            "re-run prepare_tokens before training")
    n_val = 2_000_000
    train_end = len(data) - n_val

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.02)
    lossf = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(0)

    def batch_from(lo, hi):
        ix = rng.integers(lo, hi - SEQLEN - 1, size=batch)
        x = np.stack([data[i:i + SEQLEN] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1:i + 1 + SEQLEN] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()

    hist, t0 = [], time.time()
    for step in range(steps):
        x, y = batch_from(0, train_end)
        opt.zero_grad(set_to_none=True)
        loss = lossf(model(x).view(-1, VOCAB), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if bits:
            clamp_all(model)

        if step % max(1, steps // 20) == 0 or step == steps - 1:
            model.eval()
            vl = k = 0.0
            with torch.no_grad():
                for _ in range(20):
                    xv, yv = batch_from(train_end, len(data))
                    vl += lossf(model(xv).view(-1, VOCAB), yv.view(-1)).item()
                    k += 1
            model.train()
            hist.append({"step": step, "train_loss": loss.item(),
                         "val_loss": vl / k,
                         "tokens": step * batch * SEQLEN})
            print(f"[{tag}] step {step}/{steps} train={loss.item():.4f} "
                  f"val={vl/k:.4f} ({(time.time()-t0)/60:.0f} min)", flush=True)

    rec = {"size": size, "bits": bits, "n_nonembed": n_ne,
           "tokens": total_tokens, "steps": steps,
           "final_val_loss": hist[-1]["val_loss"],
           "minutes": (time.time() - t0) / 60, "history": hist}
    with open(f"{OUT}/{tag}.json", "w") as f:
        json.dump(rec, f)
    vol.commit()
    print(f"[{tag}] done val={rec['final_val_loss']:.4f} "
          f"in {rec['minutes']:.0f} min", flush=True)
    return {k: v for k, v in rec.items() if k != "history"}


@app.local_entrypoint()
def main(smoke: bool = False):
    if smoke:
        print(train.remote(size="5m", bits=4))
        return
    jobs = [(s, b) for s in CONFIGS for b in [0] + PRECISIONS]
    print(f"spawning {len(jobs)} runs", flush=True)
    handles = [train.spawn(size=s, bits=b) for s, b in jobs]
    done = 0
    for h in handles:
        try:
            h.get()
        except Exception as exc:                          # noqa: BLE001
            print(f"  run failed: {str(exc)[:160]}", flush=True)
        done += 1
        print(f"  {done}/{len(handles)} complete", flush=True)
