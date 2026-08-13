"""Tier D: can a transformer reach low SF precision on SF-only hardware?

Tier C showed the CNN collapse was a scale mismatch, not a precision limit:
normalising conv weights per output channel before quantization took SF2 from
chance to within 1.5 points of SF16, with the scale absorbed by the BatchNorm
that follows every conv, so inference arithmetic stayed pure SF.

That trick does not transfer directly. BatchNorm undoes a per-OUTPUT-channel
(row) scale on the preceding conv. LayerNorm normalises across features, and a
transformer's matmul outputs go into a residual stream, so a row scale is not
recoverable downstream.

What is exactly recoverable is a per-INPUT-channel (column) scale, absorbed by
the norm that feeds the matmul:

    (W/g) . (LN(z).gamma.g + beta.g)  ==  W . (LN(z).gamma + beta)

so folding gamma' = gamma.g and beta' = beta.g at deployment leaves the matmul
holding pure SF weights. g never enters the systolic array; it lives in the
normalisation unit, exactly as the BatchNorm gain does in the CNN case.

Three modes:

  none      plain SF, the tier A baseline
  ln        column-normalise only the matmuls a norm already feeds: q, k, v
            (sharing one g, since they read the same ln1) and mlp.0
  ln_full   additionally put a norm before proj and mlp.2, so every matmul in
            the block is fed by a norm and every weight is absorbable

ln_full is a real architecture change -- two extra norms per block -- so it
gets its own FP32 control rather than being compared against the baseline's.

    modal run modal/modal_scaling_d.py::main --smoke
"""

import pathlib

import modal

BENCH_DIR = str(pathlib.Path(__file__).resolve().parent.parent)

app = modal.App("superfloat-scaling-d")
vol = modal.Volume.from_name("sfx-baselines", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0",
                 extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers", "datasets", "numpy", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/vol/hf"})
    .add_local_dir(BENCH_DIR, remote_path="/root/sfx_bench")
)

GPU = "H100"
OUT = "/vol/runs_scaling_d"
TOKENS = "/vol/fineweb_edu_tokens.bin"

VOCAB, SEQLEN = 50304, 1024
CONFIGS = {"5m": (256, 6, 4), "11m": (384, 6, 6)}
TOKENS_PER_PARAM = 10
TARGET_TOKENS = 1_100_000_000
MODES = ["none", "ln", "ln_full"]


def build_gpt(d_model, n_layer, n_head, mode):
    """GPT block stack. `mode` decides which matmuls a norm feeds."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    full = (mode == "ln_full")

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.q = nn.Linear(d_model, d_model)
            self.k = nn.Linear(d_model, d_model)
            self.v = nn.Linear(d_model, d_model)
            # a norm before proj / mlp.2 exists only in ln_full; Identity keeps
            # the parameter count and graph identical otherwise
            self.ln_a = nn.LayerNorm(d_model) if full else nn.Identity()
            self.proj = nn.Linear(d_model, d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.fc1 = nn.Linear(d_model, 4 * d_model)
            self.ln_m = nn.LayerNorm(4 * d_model) if full else nn.Identity()
            self.fc2 = nn.Linear(4 * d_model, d_model)

        def forward(self, x):
            h = self.ln1(x)
            B, T, C = h.shape
            hd = C // n_head
            q = self.q(h).view(B, T, n_head, hd).transpose(1, 2)
            k = self.k(h).view(B, T, n_head, hd).transpose(1, 2)
            v = self.v(h).view(B, T, n_head, hd).transpose(1, 2)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            o = o.transpose(1, 2).reshape(B, T, C)
            x = x + self.proj(self.ln_a(o))
            m = self.fc1(self.ln2(x))
            return x + self.fc2(self.ln_m(F.gelu(m)))

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(VOCAB, d_model)
            self.wpe = nn.Embedding(SEQLEN, d_model)
            self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
            self.lnf = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, VOCAB, bias=False)
            self.head.weight = self.wte.weight

        def forward(self, idx):
            x = self.wte(idx) + self.wpe(torch.arange(idx.shape[1],
                                                      device=idx.device))
            for blk in self.blocks:
                x = blk(x)
            return self.head(self.lnf(x))

    return GPT()


def install_col_norm(model, mode):
    """Column-normalise the matmuls a norm feeds, so the scale is absorbable.

    q/k/v share one g because they all read ln1: a norm can only carry a
    single gain vector, so three independent scales could not be folded.
    """
    import torch
    import torch.nn.functional as F
    from superfloat import SFLinear, sf_quantize_sv

    class SFLinearCol(SFLinear):
        """Weights quantized as W/g; g is applied to the input instead.

        At deployment g folds into the feeding norm (gamma' = gamma*g,
        beta' = beta*g) and disappears from the matmul, which then holds
        exactly SF grid values.
        """
        sf_group = None                      # shared scale, set below

        def forward(self, x):
            w = self.weight
            src = self.sf_group if self.sf_group is not None else [w]
            g = torch.stack([t.abs().amax(dim=0) for t in src]).amax(0)
            g = g.clamp_min(1e-8)
            wq = sf_quantize_sv(w / g.unsqueeze(0), self.sf_scale, self.sf_vmax)
            b = None if self.bias is None else sf_quantize_sv(
                self.bias, self.sf_scale, self.sf_vmax)
            return F.linear(x * g, wq, b)

    n = 0
    for blk in model.blocks:
        group = [blk.q.weight, blk.k.weight, blk.v.weight]
        for m in (blk.q, blk.k, blk.v):
            m.__class__ = SFLinearCol
            m.sf_group = group               # one g for all three
            n += 1
        blk.fc1.__class__ = SFLinearCol
        blk.fc1.sf_group = None
        n += 1
        if mode == "ln_full":
            for m in (blk.proj, blk.fc2):
                m.__class__ = SFLinearCol
                m.sf_group = None
                n += 1
    return n


@app.function(image=image, gpu=GPU, volumes={"/vol": vol},
              timeout=60 * 60 * 8, max_containers=8)
def train(size: str, bits: int, mode: str = "none", seed: int = 0,
          batch: int = 16, lr: float = 6e-4):
    import json
    import os
    import sys
    import time
    import numpy as np
    import torch
    sys.path.insert(0, "/root/sfx_bench")
    from superfloat import disable_tf32, apply_superfloat, clamp_all

    disable_tf32()
    torch.manual_seed(seed)
    d_model, n_layer, n_head = CONFIGS[size]
    tag = f"{size}_{mode}_" + ("fp32" if bits == 0 else f"sf{bits}")
    if seed:
        tag += f"_s{seed}"
    os.makedirs(OUT, exist_ok=True)

    model = build_gpt(d_model, n_layer, n_head, mode).cuda()
    tot = sum(p.numel() for p in model.parameters())
    n_ne = tot - model.wte.weight.numel() - model.wpe.weight.numel()
    ncol = 0
    if bits:
        nconv = apply_superfloat(model, bits=bits,
                                 head_names=("head", "wte", "wpe"),
                                 quantize_activations=False)
        expected = 6 * n_layer
        if nconv != expected:
            raise RuntimeError(f"quantized {nconv}, expected {expected}")
        if mode != "none":
            ncol = install_col_norm(model, mode)

    total_tokens = int(n_ne * TOKENS_PER_PARAM)
    steps = total_tokens // (batch * SEQLEN)
    print(f"[{tag}] N={n_ne/1e6:.1f}M col_norm={ncol} steps={steps}", flush=True)

    data = np.memmap(TOKENS, dtype=np.uint16, mode="r")
    if len(data) < TARGET_TOKENS or data[-4096:].max() == 0:
        raise RuntimeError("token file incomplete")
    train_end = len(data) - 2_000_000

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=steps, pct_start=0.02)
    lossf = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)

    def get(lo, hi):
        ix = rng.integers(lo, hi - SEQLEN - 1, size=batch)
        x = np.stack([data[i:i + SEQLEN] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1:i + 1 + SEQLEN] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()

    hist, t0 = [], time.time()
    for step in range(steps):
        x, y = get(0, train_end)
        opt.zero_grad(set_to_none=True)
        loss = lossf(model(x).view(-1, VOCAB), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        # under column normalisation the raw weights are deliberately off-grid
        # and only normalised at use time, so clamping them would be wrong
        if bits and mode == "none":
            clamp_all(model)
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            model.eval()
            vl = k = 0.0
            with torch.no_grad():
                for _ in range(20):
                    xv, yv = get(train_end, len(data))
                    vl += lossf(model(xv).view(-1, VOCAB), yv.view(-1)).item()
                    k += 1
            model.train()
            hist.append({"step": step, "val_loss": vl / k})
            print(f"[{tag}] {step}/{steps} val={vl/k:.4f} "
                  f"({(time.time()-t0)/60:.0f}m)", flush=True)

    rec = {"size": size, "bits": bits, "mode": mode, "seed": seed,
           "n_nonembed": n_ne, "col_norm_layers": ncol, "steps": steps,
           "final_val_loss": hist[-1]["val_loss"],
           "minutes": (time.time() - t0) / 60, "history": hist}
    with open(f"{OUT}/{tag}.json", "w") as f:
        json.dump(rec, f)
    vol.commit()
    print(f"[{tag}] done val={rec['final_val_loss']:.4f}", flush=True)
    return {k: v for k, v in rec.items() if k != "history"}


@app.local_entrypoint()
def main(smoke: bool = False):
    if smoke:
        for mode in ("none", "ln_full"):
            print(train.remote(size="5m", bits=3, mode=mode))
        return
    jobs = [(s, b, m) for s in CONFIGS for m in MODES
            for b in [0, 2, 3, 4, 6, 16]]
    print(f"spawning {len(jobs)} runs")
    for h in [train.spawn(size=s, bits=b, mode=m) for s, b, m in jobs]:
        h.get()
