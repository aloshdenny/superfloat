"""Validate SF quantization-aware training starting from a trained checkpoint.

Stage 0 showed SF8 is free when a 22.6M model is trained from scratch on modern
blocks. That does not answer the question that matters for a tool-use model,
which is whether an existing checkpoint -- one that already cost trillions of
tokens to make -- can be moved onto the SF grid for a tiny fraction of that.

SmolLM2-360M is LLaMA-shaped, so the absorption boundary is identical to
stage0_toolqat.py: q/k/v all read input_layernorm and share one per-input
channel scale, gate/up share post_attention_layernorm, and o_proj/down_proj are
fed by no norm so their weights are quantized as-is. It also has 8192 native
positions, so the target context needs no RoPE surgery.

Three measurements, in increasing cost:

  ptq   quantize the checkpoint and evaluate, no training at all. Tier B put
        the PTQ threshold at SF8, so this locates the regime where training is
        actually needed rather than assuming it.
  qat   continued pretraining with SF in the loop
  bf16  the same continued pretraining without quantization, so that "SF8 hurt
        it" is separable from "our data and schedule moved it"

The unmodified checkpoint is a fourth, free reference point.
"""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superfloat import disable_tf32, sf_params, sf_quantize_sv

MODEL = "HuggingFaceTB/SmolLM2-360M"
OUT = os.environ.get("SMOL_OUT", "/workspace/results")
TOK = os.environ.get("SMOL_TOK", "/workspace/smol")


class SFLinear(nn.Linear):
    """Weights on the SF grid. `sf_group` shares one per-input-channel scale
    across the matmuls reading the same norm; where it is None the layer is fed
    by no norm and nothing downstream could undo a scale, so the weights are
    quantized as they stand."""
    sf_scale = 0.0; sf_vmax = 0.0; sf_group = None

    def forward(self, x):
        w = self.weight
        if self.sf_group is None:
            return F.linear(x, sf_quantize_sv(w, self.sf_scale, self.sf_vmax), self.bias)
        g = torch.stack([t.abs().amax(dim=0) for t in self.sf_group]).amax(0).clamp_min(1e-8)
        wq = sf_quantize_sv(w / g.unsqueeze(0), self.sf_scale, self.sf_vmax)
        return F.linear(x * g, wq, self.bias)


def install(model, bits, mode="ln"):
    """mode `ln_all` also scales o_proj/down_proj, whose trained weights run to
    |w|=7.5 and are otherwise clipped at the SF bound of 1.0."""
    scale, vmax = sf_params(bits)
    n = 0
    for layer in model.model.layers:
        a, m = layer.self_attn, layer.mlp
        qkv = [a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]
        gu = [m.gate_proj.weight, m.up_proj.weight]
        for mod, grp in ((a.q_proj, qkv), (a.k_proj, qkv), (a.v_proj, qkv),
                         (m.gate_proj, gu), (m.up_proj, gu),
                         (a.o_proj, None), (m.down_proj, None)):
            mod.__class__ = SFLinear
            mod.sf_scale, mod.sf_vmax = scale, vmax
            if mode == "plain":
                mod.sf_group = None
            elif mode == "ln_all" and grp is None:
                # o_proj and down_proj are fed by no norm, but their input-side
                # scale is still absorbable: scaling o_proj's input channel j is
                # scaling v_proj's output row j, and scaling down_proj's input
                # channel j is scaling up_proj's output row j, since
                # (gate*up)*g == gate*(up*g). So they get their own scale and it
                # folds one layer back rather than into a norm.
                mod.sf_group = [mod.weight]
            else:
                mod.sf_group = grp
            n += 1
    return n


def fold(model):
    """Fold each absorbable scale into the norm feeding it, then confirm every
    deployed weight really sits on the grid."""
    worst, n = 0.0, 0
    for layer in model.model.layers:
        a, m = layer.self_attn, layer.mlp
        for norm, group in ((layer.input_layernorm, [a.q_proj, a.k_proj, a.v_proj]),
                            (layer.post_attention_layernorm, [m.gate_proj, m.up_proj])):
            grp = [x for x in group if x.sf_group is not None]
            if grp:
                g = torch.stack([x.weight.abs().amax(dim=0) for x in grp]).amax(0).clamp_min(1e-8)
                norm.weight.data = norm.weight.data * g
                for x in grp:
                    x.weight.data = sf_quantize_sv(x.weight.data / g.unsqueeze(0),
                                                   x.sf_scale, x.sf_vmax)
                    x.sf_group = None
            for x in group:
                k = x.weight.data.double() * x.sf_scale
                worst = max(worst, (k - k.round()).abs().max().item()); n += 1
        for x in (a.o_proj, m.down_proj):
            x.weight.data = sf_quantize_sv(x.weight.data, x.sf_scale, x.sf_vmax)
            k = x.weight.data.double() * x.sf_scale
            worst = max(worst, (k - k.round()).abs().max().item()); n += 1
    return n, worst


# ------------------------------------------------------------------ data ----
def _render_glaive(r):
    return (r.get("system") or "").strip() + "\n" + (r.get("chat") or "").strip()


def _render_hermes(r):
    out = [f"SYSTEM: You have access to the following tools:\n{r.get('tools') or ''}"]
    for m in (r.get("conversations") or []):
        out.append(f"{(m.get('from') or '').upper()}: {m.get('value','')}")
    return "\n".join(out)


def prepare(n_general, out_dir):
    """Re-tokenised with SmolLM2's own tokenizer; the Stage 0 corpora used GPT-2
    and its ids mean nothing to this model."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    os.makedirs(out_dir, exist_ok=True)
    tk = AutoTokenizer.from_pretrained(MODEL)
    eos = tk.eos_token_id

    g = os.path.join(out_dir, "general.bin")
    if not (os.path.exists(g) and os.path.getsize(g) == n_general * 4):
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
        buf = np.memmap(g, dtype=np.uint32, mode="w+", shape=(n_general,))
        i, batch = 0, []
        def flush(b, i):
            for ids in tk(b)["input_ids"]:
                ids = ids + [eos]
                if i + len(ids) > n_general: ids = ids[:n_general - i]
                if not ids: break
                buf[i:i+len(ids)] = np.array(ids, dtype=np.uint32); i += len(ids)
            return i
        for row in ds:
            batch.append(row["text"])
            if len(batch) >= 2000:
                i = flush(batch, i); batch = []
                if i >= n_general: break
        if i < n_general: i = flush(batch, i)
        buf.flush(); del buf
        print(f"general: {i/1e6:.0f}M tokens", flush=True)

    t = os.path.join(out_dir, "tool.bin")
    if not os.path.exists(t):
        texts = []
        try:
            d = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
            texts += [_render_glaive(r) for r in d]; print(f"glaive {len(d)}", flush=True)
        except Exception as e: print("glaive:", str(e)[:60], flush=True)
        for cfg in ("func_calling_singleturn", "func_calling", "glaive_func_calling"):
            try:
                d = load_dataset("NousResearch/hermes-function-calling-v1", cfg, split="train")
                texts += [_render_hermes(r) for r in d]; print(f"hermes/{cfg} {len(d)}", flush=True)
            except Exception as e: print(f"hermes/{cfg}:", str(e)[:50], flush=True)
        ids = []
        for j in range(0, len(texts), 1000):
            for e in tk(texts[j:j+1000])["input_ids"]:
                ids.extend(e); ids.append(eos)
        arr = np.array(ids, dtype=np.uint32)
        np.memmap(t, dtype=np.uint32, mode="w+", shape=arr.shape)[:] = arr
        print(f"tool: {len(arr)/1e6:.1f}M tokens from {len(texts)} traces", flush=True)


class Mix:
    def __init__(self, d, seqlen, p_tool, seed, val_frac=0.02):
        self.g = np.memmap(os.path.join(d, "general.bin"), dtype=np.uint32, mode="r")
        self.t = np.memmap(os.path.join(d, "tool.bin"), dtype=np.uint32, mode="r")
        self.gs = int(len(self.g) * (1 - val_frac)); self.ts = int(len(self.t) * (1 - val_frac))
        self.L, self.p = seqlen, p_tool
        self.rng = np.random.default_rng(seed)

    def _draw(self, src, lo, hi, n):
        ix = self.rng.integers(lo, hi - self.L - 1, size=n)
        x = np.stack([src[i:i+self.L] for i in ix]).astype(np.int64)
        y = np.stack([src[i+1:i+1+self.L] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()

    def train(self, n):
        if self.rng.random() < self.p: return self._draw(self.t, 0, self.ts, n)
        return self._draw(self.g, 0, self.gs, n)
    def _fixed(self, src, lo, hi, n, tag):
        r = np.random.default_rng(hash(tag) % (2**31))      # same batches every call
        ix = r.integers(lo, hi - self.L - 1, size=n)
        x = np.stack([src[i:i+self.L] for i in ix]).astype(np.int64)
        y = np.stack([src[i+1:i+1+self.L] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()

    def val_general(self, n): return self._fixed(self.g, self.gs, len(self.g), n, "gen")
    def val_tool(self, n):    return self._fixed(self.t, self.ts, len(self.t), n, "tool")


# ----------------------------------------------------------------- driver ---
def load_model():
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    return m.cuda()


def evaluate(model, mix, batch, n=25):
    model.eval(); lf = nn.CrossEntropyLoss(); out = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for name, fn in (("general", mix.val_general), ("tool", mix.val_tool)):
            tot = 0.0
            for _ in range(n):
                x, y = fn(batch)
                lg = model(x).logits
                tot += lf(lg.view(-1, lg.size(-1)), y.view(-1)).item()
            out[name] = tot / n
    model.train(); return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8, help="0 = bf16 control")
    ap.add_argument("--mode", default="ln", choices=["plain", "ln", "ln_all"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=100_000_000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--p-tool", type=float, default=0.25)
    ap.add_argument("--prepare", type=int, default=0)
    ap.add_argument("--ptq", action="store_true", help="quantize and evaluate, no training")
    a = ap.parse_args()

    if a.prepare:
        prepare(a.prepare, TOK); return

    disable_tf32(); torch.manual_seed(a.seed)
    os.makedirs(OUT, exist_ok=True)
    mix = Mix(TOK, a.seqlen, a.p_tool, a.seed)

    if a.ptq:
        rows = []
        for bits in (0, 16, 10, 8, 6, 5, 4, 3):
            model = load_model()
            n = install(model, bits, a.mode) if bits else 0
            r = evaluate(model, mix, a.batch)
            rows.append(dict(bits=bits, mode=a.mode if bits else "-", n_layers=n, **r))
            print(f"  PTQ {'bf16' if not bits else 'SF%d %s' % (bits, a.mode):12s} "
                  f"general={r['general']:.4f} tool={r['tool']:.4f}", flush=True)
            del model; torch.cuda.empty_cache()
        json.dump({"exp": "smol_ptq", "mode": a.mode, "rows": rows},
                  open(f"{OUT}/smol_ptq_{a.mode}.json", "w"))
        return

    tag = f"smol_" + ("bf16" if a.bits == 0 else f"sf{a.bits}_{a.mode}") + f"_s{a.seed}"
    if os.path.exists(f"{OUT}/{tag}.json"):
        print(f"[{tag}] done, skip", flush=True); return

    model = load_model()
    nq = install(model, a.bits, a.mode) if a.bits else 0
    n_ne = sum(p.numel() for p in model.parameters()) - model.model.embed_tokens.weight.numel()
    steps = a.tokens // (a.batch * a.seqlen)
    print(f"[{tag}] N={n_ne/1e6:.0f}M quantized={nq} steps={steps} "
          f"tokens={a.tokens/1e6:.0f}M seq={a.seqlen}", flush=True)

    before = evaluate(model, mix, a.batch)
    print(f"[{tag}] before training: general={before['general']:.4f} tool={before['tool']:.4f}",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0,
                            betas=(0.9, 0.95), fused=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps,
                                                pct_start=0.03)
    lf = nn.CrossEntropyLoss()
    hist, t0 = [], time.time()
    for step in range(steps):
        x, y = mix.train(a.batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(x).logits
            loss = lf(lg.view(-1, lg.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            r = evaluate(model, mix, a.batch, n=15)
            hist.append({"step": step, **r})
            print(f"[{tag}] {step}/{steps} general={r['general']:.4f} tool={r['tool']:.4f} "
                  f"({(time.time()-t0)/60:.0f}m)", flush=True)

    rec = {"exp": "smol_qat", "bits": a.bits, "mode": a.mode, "seed": a.seed,
           "n_nonembed": n_ne, "tokens": a.tokens, "seqlen": a.seqlen,
           "p_tool": a.p_tool, "steps": steps, "before": before,
           "val_general": hist[-1]["general"], "val_tool": hist[-1]["tool"],
           "minutes": (time.time()-t0)/60, "history": hist, "complete": True}
    if a.bits:
        nf, worst = fold(model)
        rec["folded"], rec["max_offgrid"] = nf, worst
        print(f"[{tag}] folded {nf}, max off-grid {worst:.2e}", flush=True)
    json.dump(rec, open(f"{OUT}/{tag}.json", "w"))
    print(f"[{tag}] DONE general={rec['val_general']:.4f} tool={rec['val_tool']:.4f}", flush=True)


if __name__ == "__main__":
    main()
