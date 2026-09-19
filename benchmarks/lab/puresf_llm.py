"""Does a transformer trained end-to-end in SF8 stay SF-representable everywhere?

Every prior result in this study is weights-only. A real forward pass through
a trained model was measured directly (see SCALING_LAWS.md and the tool-use
work) and the numbers are not close to SF's +/-1 bound:

    max |activation entering a matmul|   2,298
    max |matmul output|                 15,279
    max |residual stream|               16,056

Weight-scale absorption does not touch this: the residual stream in a pre-norm
transformer is read by every sub-layer through a norm (which is scale
invariant) but is never itself rescaled, so it grows without bound across
depth. That is the actual blocker for "runs on SF-only hardware" -- a chip
whose registers and MAC array only hold SFx values has nowhere to put a
16,000-magnitude activation.

Three things are added on top of stage0_toolqat's block (RMSNorm/SwiGLU/GQA):

  1. QK-norm.  Query and key are RMSNorm'd per head before the attention dot
     product (OLMo2, Qwen3). Bounds attention logits directly.

  2. Norm gains and activations are themselves SF-quantized, not just weights.
     A norm's gamma is free to grow during ordinary training -- nothing
     constrains it -- which is why even "reads a norm" activations were not
     actually bounded in the audit above (L31.gate_proj max|in|=28.31). Gamma
     is quantized to the SF grid and every activation entering a matmul is
     quantized with an EMA-tracked scale, the same STE pattern already used
     for weights.

  3. Periodic residual renormalisation. Every `renorm_every` blocks, divide
     the residual stream by its own running max. This is architecturally
     honest rather than a trick: every downstream read of x goes through a
     norm, and RMSNorm(x/s) == RMSNorm(x), so the rescale is invisible to
     everything except the raw magnitude of x itself -- which is exactly the
     thing that needs to shrink. It changes what the network computes (the
     next layer's contribution is relatively larger against a smaller x), so
     it must be present during training, not applied after the fact.

The pass/fail criterion is not the loss. It is the audit at the bottom of this
file, which repeats the exact measurement above on the trained model and
reports whether every value anywhere in the forward pass is now within the SF
bound. A model that trains fine but still produces a 15,000-magnitude
activation has not solved the problem this file exists to test.
"""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superfloat import disable_tf32, sf_params, sf_quantize_sv

OUT = os.environ.get("PSF_OUT", "/workspace/results")
TOK = os.environ.get("PSF_TOK", "/workspace/psf")
VOCAB = 50304


class SFAct(nn.Module):
    """Quantize an activation to the SF grid with a PER-TOKEN scale.

    First version of this module used a single EMA-tracked scalar per tensor
    (one scale for the whole batch x seq x feature block). Trained end-to-end
    at 11M/20M tokens that only got the audit to 2.4x/4.6x over bound, not a
    pass. PURE_SF.md independently ran the actual granularity ablation this
    module skipped, on a real model (SmolLM2-360M, PTQ): a tensor-wide scale
    with the residual left alone is dead (10.80 nats, collapsed); a tensor
    scale even WITH a per-token residual only gets to 7.16; the config that
    actually works is per-token activations AND per-token residual together
    (a8tok_rtok, +0.28 nats vs fp32). A single EMA scalar is exactly the
    "tensor scale" arm their own data shows failing. This rewrites SFAct to
    match the arm that worked: a fresh scale per (batch, token) pair, computed
    from that token's own feature vector, not tracked across steps.

    Every call site here feeds a (B, T, C) tensor with the feature dim last
    (q/k/v and gate/up inputs are norm outputs; the attention-output and
    SwiGLU-product sites are the same shape after their own norm), so one
    `amax(dim=-1, keepdim=True)` is correct everywhere -- the same reduction
    the residual renorm already uses.

    `hard=False` rescales the quantized value back up: grid-aligned at each
    token's own scale, not bounded to [-vmax, vmax] unless that per-token
    scale is <= 1. Fine where the value is about to be re-normalised anyway.

    `hard=True` divides by the per-token scale, clamps+quantizes to
    [-vmax, vmax], and does NOT rescale back up -- real per-token clipping,
    for the two sites with no downstream norm or weight to fold a scale into
    (attention output feeding o, SwiGLU product feeding down).
    """
    def __init__(self, bits, hard=False):
        super().__init__()
        self.bits = bits; self.hard = hard
        if bits:
            self.scale, self.vmax = sf_params(bits)

    def forward(self, x):
        if not self.bits:
            return x
        s = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        xs = x / s
        q = sf_quantize_sv(xs, self.scale, self.vmax)
        return q if self.hard else q * s


class RMSNorm(nn.Module):
    """gamma is itself SF-quantized when act_bits is set, so the norm's own
    gain cannot silently grow past what the format can hold -- ordinary
    RMSNorm has nothing constraining gamma, which is why gate_proj's input
    still hit |x|=28 in the audit despite reading a norm output."""
    def __init__(self, d, act_bits=0, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d)); self.eps = eps
        self.act_bits = act_bits
        if act_bits:
            self.scale, self.vmax = sf_params(act_bits)

    def forward(self, x):
        g = self.w
        if self.act_bits:
            g = sf_quantize_sv(g, self.scale, self.vmax)
        return g * (x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


def rope(q, k, cos, sin):
    def rot(t):
        a, b = t.chunk(2, dim=-1); return torch.cat((-b, a), dim=-1)
    return q * cos + rot(q) * sin, k * cos + rot(k) * sin


class Block(nn.Module):
    def __init__(self, d, nh, nkv, hidden, act_bits, qk_norm, renorm_every, layer_idx):
        super().__init__()
        self.nh, self.nkv, self.hd = nh, nkv, d // nh
        # gated on DEPTH, not training step: this must fire the same way on
        # every forward pass, or most passes never renormalise at all (a bug
        # caught in a CPU smoke test before this ever reached a GPU -- with
        # step-gating, only 20 of 80 test steps triggered any renorm)
        self.do_renorm = bool(renorm_every) and (layer_idx + 1) % renorm_every == 0
        self.n1 = RMSNorm(d, act_bits)
        self.q = nn.Linear(d, nh * self.hd, bias=False)
        self.k = nn.Linear(d, nkv * self.hd, bias=False)
        self.v = nn.Linear(d, nkv * self.hd, bias=False)
        self.qk_norm = qk_norm
        if qk_norm:
            self.qn = RMSNorm(self.hd, act_bits)
            self.kn = RMSNorm(self.hd, act_bits)
        # a norm ahead of o and down makes their input bounded AND their
        # weight scale absorbable, extending tier D's ln_full to activations
        self.n_o = RMSNorm(d, act_bits)
        self.o = nn.Linear(nh * self.hd, d, bias=False)
        self.n2 = RMSNorm(d, act_bits)
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.n_d = RMSNorm(hidden, act_bits)
        self.down = nn.Linear(hidden, d, bias=False)
        # every activation site is hard-clamped: a CPU probe found the "soft"
        # mode leaves q/gate inputs at 2-2.6x the SF bound even when fed by a
        # gamma-quantized norm, because RMSNorm bounds RMS, not max -- a single
        # outlier coordinate survives normalisation at up to sqrt(feature_dim).
        # "reads a norm" is necessary but not sufficient for boundedness; only
        # hard clamping enforces the actual requirement, no exceptions.
        self.a_in = SFAct(act_bits, hard=True)
        self.a_qkv = SFAct(act_bits, hard=True)
        self.a_o = SFAct(act_bits, hard=True)
        self.a_mlp = SFAct(act_bits, hard=True)
        self.a_down = SFAct(act_bits, hard=True)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        h = self.a_in(self.n1(x))
        q = self.q(h).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k(h).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v(h).view(B, T, self.nkv, self.hd).transpose(1, 2)
        if self.qk_norm:
            q, k = self.qn(q), self.kn(k)
        q, k = rope(q, k, cos, sin)
        if self.nkv != self.nh:
            r = self.nh // self.nkv
            k = k.repeat_interleave(r, dim=1); v = v.repeat_interleave(r, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = self.a_qkv(a.transpose(1, 2).reshape(B, T, -1))
        x = x + self.o(self.a_o(self.n_o(a)))
        h = self.a_mlp(self.n2(x))
        m = self.a_down(self.n_d(F.silu(self.gate(h)) * self.up(h)))
        x = x + self.down(m)
        if self.do_renorm:
            # RMSNorm(x/s) == RMSNorm(x): every downstream read is invariant
            # to this, so the residual stream can be pulled back toward unit
            # scale without disturbing what any norm sees. This DOES change
            # the network's function (the next block's contribution becomes
            # relatively larger), so it is trained with the rescale present.
            s = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
            x = x / s
        return x


class Model(nn.Module):
    def __init__(self, d=384, n_layer=6, nh=6, nkv=2, hidden=1024, seqlen=1024,
                 act_bits=0, qk_norm=False, renorm_every=0):
        super().__init__()
        self.wte = nn.Embedding(VOCAB, d)
        self.blocks = nn.ModuleList([
            Block(d, nh, nkv, hidden, act_bits, qk_norm, renorm_every, i)
            for i in range(n_layer)])
        self.nf = RMSNorm(d, act_bits)
        self.head = nn.Linear(d, VOCAB, bias=False)
        self.head.weight = self.wte.weight
        hd = d // nh
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        t = torch.arange(seqlen).float()
        f = torch.outer(t, inv)
        emb = torch.cat((f, f), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None], persistent=False)
        self.apply(self._init)
        for n_, p_ in self.named_parameters():
            if n_.endswith(("o.weight", "down.weight")):
                nn.init.normal_(p_, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        x = self.wte(idx)
        T = idx.shape[1]
        cos, sin = self.cos[:, :, :T], self.sin[:, :, :T]
        for b in self.blocks:
            x = b(x, cos, sin)
        return self.head(self.nf(x))


# ------------------------------------------------------------ weight side ---
class SFLinear(nn.Linear):
    """Weights on the SF grid. Same absorption rule as stage0_toolqat: layers
    fed by a norm share that norm's scale; o/down are now fed by n_o/n_d, so
    unlike stage0 they ARE absorbable here -- the extra norms bought that."""
    sf_scale = 0.0; sf_vmax = 0.0; sf_group = None

    def forward(self, x):
        w = self.weight
        if self.sf_group is None:
            return F.linear(x, sf_quantize_sv(w, self.sf_scale, self.sf_vmax))
        g = torch.stack([t.abs().amax(dim=0) for t in self.sf_group]).amax(0).clamp_min(1e-8)
        wq = sf_quantize_sv(w / g.unsqueeze(0), self.sf_scale, self.sf_vmax)
        return F.linear(x * g, wq)


def quantize_weights(model, bits):
    scale, vmax = sf_params(bits)
    n = 0
    for blk in model.blocks:
        qkv = [blk.q.weight, blk.k.weight, blk.v.weight]
        mlp = [blk.gate.weight, blk.up.weight]
        for m, grp in ((blk.q, qkv), (blk.k, qkv), (blk.v, qkv),
                       (blk.gate, mlp), (blk.up, mlp),
                       (blk.o, [blk.o.weight]), (blk.down, [blk.down.weight])):
            m.__class__ = SFLinear
            m.sf_scale, m.sf_vmax = scale, vmax
            m.sf_group = grp
            n += 1
    return n


# ------------------------------------------------------------------ data ----
def prepare(n_tokens, out_dir):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tokens.bin")
    if os.path.exists(path):
        d = np.memmap(path, dtype=np.uint16, mode="r")
        if len(d) >= n_tokens and d[-4096:].max() > 0:
            print(f"corpus present: {len(d)/1e6:.0f}M", flush=True); return
    tok = AutoTokenizer.from_pretrained("gpt2"); eot = tok.eos_token_id
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    buf = np.memmap(path, dtype=np.uint16, mode="w+", shape=(n_tokens,))
    i, batch = 0, []
    def flush(b, i):
        for ids in tok(b)["input_ids"]:
            ids = ids + [eot]
            if i + len(ids) > n_tokens: ids = ids[:n_tokens - i]
            if not ids: break
            buf[i:i+len(ids)] = np.array(ids, dtype=np.uint16); i += len(ids)
        return i
    for row in ds:
        batch.append(row["text"])
        if len(batch) >= 2000:
            i = flush(batch, i); batch = []
            if i >= n_tokens: break
    if i < n_tokens: i = flush(batch, i)
    buf.flush(); del buf
    print(f"corpus written: {i/1e6:.0f}M", flush=True)


# ------------------------------------------------------------------ audit ---
@torch.no_grad()
def audit(model, x):
    """Reproduce the SmolLM2 measurement on THIS model. Pass/fail: does every
    value anywhere in the forward pass sit within SF's +/-1 bound?"""
    stats = {}
    hooks = []
    def hook(name):
        def f(mod, inp, out):
            y = out.detach() if not isinstance(out, tuple) else out[0].detach()
            stats[name] = float(y.abs().max())
        return f
    for i, blk in enumerate(model.blocks):
        for nm, mod in (("q", blk.q), ("o", blk.o), ("gate", blk.gate), ("down", blk.down)):
            hooks.append(mod.register_forward_hook(hook(f"L{i:02d}.{nm}")))
    resid = []
    for blk in model.blocks:
        hooks.append(blk.register_forward_hook(
            lambda m, i, o, r=resid: r.append(float(o.detach().abs().max()))))
    model.eval()
    model(x)
    for h in hooks: h.remove()
    worst_layer = max(stats.values()) if stats else 0.0
    worst_resid = max(resid) if resid else 0.0
    return {"max_layer_output": worst_layer, "max_residual": worst_resid,
            "sf_bound": 1.0, "pass": worst_layer <= 1.0 and worst_resid <= 1.0,
            "detail": stats}


# ----------------------------------------------------------------- train ----
CONFIGS = {"11m": dict(d=384, n_layer=6, nh=6, nkv=2, hidden=1024)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="11m", choices=sorted(CONFIGS))
    ap.add_argument("--arm", required=False, default="bf16",
                    choices=["bf16", "sf8_weights", "sf8_full"])
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=200_000_000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--renorm-every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prepare", type=int, default=0)
    a = ap.parse_args()

    if a.prepare:
        prepare(a.prepare, TOK); return

    disable_tf32(); torch.manual_seed(a.seed)
    os.makedirs(OUT, exist_ok=True)
    tag = f"psf_{a.size}_{a.arm}_s{a.seed}"
    if os.path.exists(f"{OUT}/{tag}.json"):
        print(f"[{tag}] done, skip", flush=True); return

    act_bits = 8 if a.arm == "sf8_full" else 0
    qk_norm = a.arm != "bf16"
    renorm = a.renorm_every if a.arm == "sf8_full" else 0
    model = Model(**CONFIGS[a.size], seqlen=a.seqlen, act_bits=act_bits,
                 qk_norm=qk_norm, renorm_every=renorm).cuda()
    n_ne = sum(p.numel() for p in model.parameters()) - model.wte.weight.numel()
    nq = 0
    if a.arm in ("sf8_weights", "sf8_full"):
        nq = quantize_weights(model, 8)
    print(f"[{tag}] N={n_ne/1e6:.1f}M arm={a.arm} act_bits={act_bits} "
          f"qk_norm={qk_norm} renorm_every={renorm} quantized_w={nq}", flush=True)

    data = np.memmap(os.path.join(TOK, "tokens.bin"), dtype=np.uint16, mode="r")
    train_end = len(data) - 2_000_000
    rng = np.random.default_rng(a.seed)
    def get(lo, hi):
        ix = rng.integers(lo, hi - a.seqlen - 1, size=a.batch)
        x = np.stack([data[i:i+a.seqlen] for i in ix]).astype(np.int64)
        y = np.stack([data[i+1:i+1+a.seqlen] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()

    steps = a.tokens // (a.batch * a.seqlen)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps,
                                                pct_start=0.02)
    lf = nn.CrossEntropyLoss()
    amp = torch.autocast("cuda", dtype=torch.bfloat16)

    hist, t0 = [], time.time()
    for step in range(steps):
        x, y = get(0, train_end)
        with amp:
            loss = lf(model(x).view(-1, VOCAB), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % max(1, steps // 12) == 0 or step == steps - 1:
            model.eval()
            vl = k = 0.0
            with torch.no_grad(), amp:
                for _ in range(15):
                    xv, yv = get(train_end, len(data))
                    vl += lf(model(xv).view(-1, VOCAB), yv.view(-1)).item(); k += 1
            model.train()
            hist.append({"step": step, "val_loss": vl / k})
            print(f"[{tag}] {step}/{steps} val={vl/k:.4f} ({(time.time()-t0)/60:.0f}m)", flush=True)

    xa, _ = get(train_end, len(data))
    aud = audit(model, xa[:2])
    rec = {"exp": "puresf_llm", "size": a.size, "arm": a.arm, "seed": a.seed,
           "n_nonembed": n_ne, "act_bits": act_bits, "qk_norm": qk_norm,
           "renorm_every": renorm, "quantized_w": nq, "tokens": a.tokens,
           "final_val_loss": hist[-1]["val_loss"], "audit": aud,
           "minutes": (time.time() - t0) / 60, "history": hist, "complete": True}
    json.dump(rec, open(f"{OUT}/{tag}.json", "w"))
    print(f"[{tag}] DONE val={rec['final_val_loss']:.4f} "
          f"audit_pass={aud['pass']} max_layer={aud['max_layer_output']:.2f} "
          f"max_resid={aud['max_residual']:.2f}", flush=True)


if __name__ == "__main__":
    main()
