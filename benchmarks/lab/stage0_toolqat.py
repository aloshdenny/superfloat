"""Stage 0: does SF8 QAT survive modern transformer blocks and tool-call data?

Everything in the scaling study used vanilla GPT-2 blocks: LayerNorm, GELU, and
full multi-head attention. A tool-use model wants RMSNorm, SwiGLU and grouped
query attention, and those change the scale-absorption argument in ways that
matter:

  RMSNorm has gain but no bias, so folding is gamma' = gamma*g and nothing else.
  GQA gives k and v fewer heads than q, but all three still read the same norm,
    so they must share one per-input-channel g.
  SwiGLU's gate and up projections also share a norm, so they share a g. The
    down projection reads the SwiGLU product, which no norm feeds, so it is not
    absorbable without adding one -- the same boundary tier D hit with `proj`
    and `fc2`.

SF8 is the target rather than SF16 because SF8 grid values are exactly
representable in bfloat16 (7 significand bits against bf16's 8), so the whole
QAT path runs on tensor cores with no loss of fidelity. SF16 needs 15 and would
force fp32 matmuls. Verified numerically before this file was written.

Arms:
  bf16      unquantized control, same everything else
  sf8       SF8 weights, per-input-channel scale absorbed into the feeding norm
  sf6       as sf8, one grid step coarser
  sf8_plain SF8 with no normalisation, to show the fix is what carries it

Validation loss is reported separately on general text and on tool-call traces,
because structured output is the thing we actually care about and it may not
degrade at the same rate as prose.
"""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superfloat import disable_tf32, sf_params, sf_quantize_sv

OUT = os.environ.get("STAGE0_OUT", "/workspace/results")
TOK = os.environ.get("STAGE0_TOK", "/workspace/stage0")
VOCAB = 50304


# ----------------------------------------------------------------- model ----
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return self.w * (x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


def rope(q, k, cos, sin):
    def rot(t):
        a, b = t.chunk(2, dim=-1); return torch.cat((-b, a), dim=-1)
    return q * cos + rot(q) * sin, k * cos + rot(k) * sin


class Block(nn.Module):
    def __init__(self, d, nh, nkv, hidden):
        super().__init__()
        self.nh, self.nkv, self.hd = nh, nkv, d // nh
        self.n1 = RMSNorm(d)
        self.q = nn.Linear(d, nh * self.hd, bias=False)
        self.k = nn.Linear(d, nkv * self.hd, bias=False)
        self.v = nn.Linear(d, nkv * self.hd, bias=False)
        self.o = nn.Linear(nh * self.hd, d, bias=False)
        self.n2 = RMSNorm(d)
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)
        # identities unless ln_full swaps them in, so the graph and the
        # parameter count are unchanged in every other mode
        self.n_o = nn.Identity()
        self.n_d = nn.Identity()

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        h = self.n1(x)
        q = self.q(h).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k(h).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v(h).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = rope(q, k, cos, sin)
        if self.nkv != self.nh:                      # GQA: repeat kv heads
            r = self.nh // self.nkv
            k = k.repeat_interleave(r, dim=1); v = v.repeat_interleave(r, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(self.n_o(a.transpose(1, 2).reshape(B, T, -1)))
        h = self.n2(x)
        return x + self.down(self.n_d(F.silu(self.gate(h)) * self.up(h)))


class Model(nn.Module):
    def __init__(self, d=512, n_layer=8, nh=8, nkv=2, hidden=1408, seqlen=1024):
        super().__init__()
        self.wte = nn.Embedding(VOCAB, d)
        self.blocks = nn.ModuleList([Block(d, nh, nkv, hidden) for _ in range(n_layer)])
        self.nf = RMSNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        self.head.weight = self.wte.weight
        hd = d // nh
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        t = torch.arange(seqlen).float()
        f = torch.outer(t, inv)
        emb = torch.cat((f, f), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None], persistent=False)
        # SCALING_LAWS.md section 6 lists PyTorch's default N(0,1) embedding
        # init as a known defect of the tier A/D runs: loss starts near 465 and
        # spends part of the token budget recovering. Fixed here.
        self.apply(self._init)
        for n_, p_ in self.named_parameters():          # residual-path scaling
            if n_.endswith(("o.weight", "down.weight")):
                nn.init.normal_(p_, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        x = self.wte(idx)
        T = idx.shape[1]
        cos, sin = self.cos[:, :, :T], self.sin[:, :, :T]
        for b in self.blocks:
            x = b(x, cos, sin)
        return self.head(self.nf(x))


# ------------------------------------------------------------ quantization --
class SFLinear(nn.Linear):
    """Weights held on the SF grid.

    `sf_group` shares one per-input-channel scale across the matmuls that read
    the same norm, since a norm carries a single gain vector and could not fold
    three independent scales. Where `sf_group` is None the layer is fed by no
    norm, so no scale is absorbable and the weights are quantized as-is -- a
    per-output-row scale would be recoverable after a conv by BatchNorm, but
    these outputs go into a residual stream where nothing undoes a row scale.
    That is tier D's `ln` boundary, reproduced here for SwiGLU and GQA.
    """
    sf_scale = 0.0; sf_vmax = 0.0; sf_group = None

    def forward(self, x):
        w = self.weight
        if self.sf_group is None:
            return F.linear(x, sf_quantize_sv(w, self.sf_scale, self.sf_vmax))
        g = torch.stack([t.abs().amax(dim=0) for t in self.sf_group]).amax(0).clamp_min(1e-8)
        wq = sf_quantize_sv(w / g.unsqueeze(0), self.sf_scale, self.sf_vmax)
        return F.linear(x * g, wq)


def quantize(model, bits, mode="ln"):
    """Convert every block matmul to SF.

    mode `plain`   no scale anywhere: the pre-normalisation baseline
    mode `ln`      absorb into the norms that already exist (q/k/v share n1,
                   gate/up share n2); o and down stay plain
    mode `ln_full` additionally put a norm before o and down so every matmul is
                   fed by a norm and every weight is absorbable. A real
                   architecture change, so it gets its own control.

    Embeddings and the tied head stay in bf16. The head is tied to the
    embedding, so quantizing it would quantize the embedding table with it.
    """
    scale, vmax = sf_params(bits)
    n = 0
    for blk in model.blocks:
        qkv = [blk.q.weight, blk.k.weight, blk.v.weight]   # all read n1
        mlp = [blk.gate.weight, blk.up.weight]             # both read n2
        pairs = [(blk.q, qkv), (blk.k, qkv), (blk.v, qkv),
                 (blk.gate, mlp), (blk.up, mlp),
                 (blk.o, None), (blk.down, None)]
        if mode == "ln_full":
            blk.n_o = RMSNorm(blk.o.in_features).to(blk.o.weight.device)
            blk.n_d = RMSNorm(blk.down.in_features).to(blk.down.weight.device)
            pairs[5] = (blk.o, [blk.o.weight])
            pairs[6] = (blk.down, [blk.down.weight])
        for m, grp in pairs:
            m.__class__ = SFLinear
            m.sf_scale, m.sf_vmax = scale, vmax
            m.sf_group = None if mode == "plain" else grp
            n += 1
    return n


def fold_for_inference(model):
    """Fold every absorbable scale into the norm that feeds it and return the
    deployed weights. After this the matmuls hold exact SF grid values and the
    scale lives in the normalisation unit, off the systolic array.

    Returns (n_folded, max_offgrid) so the caller can assert the weights really
    are on the grid rather than take it on faith.
    """
    worst = 0.0; n = 0
    for blk in model.blocks:
        # layers with no absorbable scale are simply snapped onto the grid
        for m in (blk.q, blk.k, blk.v, blk.gate, blk.up):
            if m.sf_group is None:
                m.weight.data = sf_quantize_sv(m.weight.data, m.sf_scale, m.sf_vmax)
                k = m.weight.data.double() * m.sf_scale
                worst = max(worst, (k - k.round()).abs().max().item()); n += 1
        for norm, group in ((blk.n1, [blk.q, blk.k, blk.v]),
                            (blk.n2, [blk.gate, blk.up])):
            # only fold where training actually used a scale; in `plain` these
            # layers never saw one and inventing it here would change the model
            group = [m for m in group if m.sf_group is not None]
            if not group:
                for m in (blk.q, blk.k, blk.v, blk.gate, blk.up):
                    if m.sf_group is None and m.weight.data.abs().max() > 0:
                        pass
                continue
            srcs = [m.weight for m in group]
            g = torch.stack([t.abs().amax(dim=0) for t in srcs]).amax(0).clamp_min(1e-8)
            norm.w.data = norm.w.data * g                      # gamma' = gamma * g
            for m in group:
                m.weight.data = sf_quantize_sv(m.weight.data / g.unsqueeze(0),
                                               m.sf_scale, m.sf_vmax)
                m.sf_group = None                              # scale is gone now
                k = m.weight.data.double() * m.sf_scale
                worst = max(worst, (k - k.round()).abs().max().item()); n += 1
        for norm, m in ((blk.n_o, blk.o), (blk.n_d, blk.down)):
            if isinstance(norm, RMSNorm) and m.sf_group is not None:
                g = m.weight.abs().amax(dim=0).clamp_min(1e-8)
                norm.w.data = norm.w.data * g
                m.weight.data = sf_quantize_sv(m.weight.data / g.unsqueeze(0),
                                               m.sf_scale, m.sf_vmax)
                m.sf_group = None
            else:
                m.weight.data = sf_quantize_sv(m.weight.data, m.sf_scale, m.sf_vmax)
            k = m.weight.data.double() * m.sf_scale
            worst = max(worst, (k - k.round()).abs().max().item()); n += 1
    return n, worst


# ------------------------------------------------------------------ data ----
def _render_glaive(r):
    return (r.get("system") or "").strip() + "\n" + (r.get("chat") or "").strip()


def _render_hermes(r):
    tools = r.get("tools") or ""
    out = [f"SYSTEM: You have access to the following tools:\n{tools}"]
    for m in (r.get("conversations") or []):
        who = (m.get("from") or "").upper()
        out.append(f"{who}: {m.get('value','')}")
    return "\n".join(out)


def prepare(target_general, out_dir):
    """Two corpora: general text, and tool-call traces. Kept separate so the
    validation loss can be reported on each, since structured output is the
    capability at issue and may not degrade at the same rate as prose."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    os.makedirs(out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("gpt2"); eot = tok.eos_token_id

    gpath = os.path.join(out_dir, "general.bin")
    if not (os.path.exists(gpath) and os.path.getsize(gpath) == target_general * 2):
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
        buf = np.memmap(gpath, dtype=np.uint16, mode="w+", shape=(target_general,))
        i, batch = 0, []
        def flush(b, i):
            for ids in tok(b)["input_ids"]:
                ids = ids + [eot]
                if i + len(ids) > target_general: ids = ids[:target_general - i]
                if not ids: break
                buf[i:i+len(ids)] = np.array(ids, dtype=np.uint16); i += len(ids)
            return i
        for row in ds:
            batch.append(row["text"])
            if len(batch) >= 2000:
                i = flush(batch, i); batch = []
                if i >= target_general: break
        if i < target_general: i = flush(batch, i)
        buf.flush(); del buf
        print(f"general: {i/1e6:.0f}M tokens", flush=True)

    tpath = os.path.join(out_dir, "tool.bin")
    if not os.path.exists(tpath):
        texts = []
        try:
            g = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
            texts += [_render_glaive(r) for r in g]
            print(f"glaive: {len(g)} rows", flush=True)
        except Exception as e:
            print(f"glaive unavailable: {e}", flush=True)
        for cfg in ("func_calling_singleturn", "func_calling", "glaive_func_calling"):
            try:
                h = load_dataset("NousResearch/hermes-function-calling-v1", cfg, split="train")
                texts += [_render_hermes(r) for r in h]
                print(f"hermes/{cfg}: {len(h)} rows", flush=True)
            except Exception as e:
                print(f"hermes/{cfg} unavailable: {str(e)[:70]}", flush=True)
        if not texts:
            raise RuntimeError("no tool data available")
        ids = []
        B = 1000
        for j in range(0, len(texts), B):
            for e in tok(texts[j:j+B])["input_ids"]:
                ids.extend(e); ids.append(eot)
        arr = np.array(ids, dtype=np.uint16)
        np.memmap(tpath, dtype=np.uint16, mode="w+", shape=arr.shape)[:] = arr
        print(f"tool: {len(arr)/1e6:.1f}M tokens from {len(texts)} traces", flush=True)


class Mix:
    """Draws each batch entirely from one corpus, so a batch is never a
    frankenstein of prose and JSON spanning the sequence boundary."""
    def __init__(self, out_dir, seqlen, p_tool, seed, val_frac=0.01):
        self.g = np.memmap(os.path.join(out_dir, "general.bin"), dtype=np.uint16, mode="r")
        self.t = np.memmap(os.path.join(out_dir, "tool.bin"), dtype=np.uint16, mode="r")
        self.gs = int(len(self.g) * (1 - val_frac))
        self.ts = int(len(self.t) * (1 - val_frac))
        self.L, self.p = seqlen, p_tool
        self.rng = np.random.default_rng(seed)

    def _draw(self, src, lo, hi, n):
        ix = self.rng.integers(lo, hi - self.L - 1, size=n)
        x = np.stack([src[i:i+self.L] for i in ix]).astype(np.int64)
        y = np.stack([src[i+1:i+1+self.L] for i in ix]).astype(np.int64)
        return (torch.from_numpy(x).cuda(non_blocking=True),
                torch.from_numpy(y).cuda(non_blocking=True))

    def train(self, n):
        if self.rng.random() < self.p:
            return self._draw(self.t, 0, self.ts, n)
        return self._draw(self.g, 0, self.gs, n)

    def val_general(self, n): return self._draw(self.g, self.gs, len(self.g), n)
    def val_tool(self, n):    return self._draw(self.t, self.ts, len(self.t), n)


# ----------------------------------------------------------------- train ----
CONFIGS = {
    "5m":  dict(d=256, n_layer=6,  nh=8,  nkv=2, hidden=704),
    "25m": dict(d=512, n_layer=8,  nh=8,  nkv=2, hidden=1408),
    "50m": dict(d=640, n_layer=10, nh=10, nkv=2, hidden=1728),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="25m", choices=sorted(CONFIGS))
    ap.add_argument("--bits", type=int, default=8, help="0 = bf16 control")
    ap.add_argument("--mode", default="ln", choices=["plain", "ln", "ln_full"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=500_000_000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--p-tool", type=float, default=0.15)
    ap.add_argument("--prepare", type=int, default=0, metavar="N_GENERAL")
    ap.add_argument("--probe", action="store_true", help="measure throughput and exit")
    a = ap.parse_args()

    if a.prepare:
        prepare(a.prepare, TOK); return

    disable_tf32()                      # bf16 path does not need it, and it
    torch.manual_seed(a.seed)           # would corrupt anything above SF8
    torch.backends.cudnn.benchmark = True

    tag = f"s0_{a.size}_" + ("bf16" if a.bits == 0 else f"sf{a.bits}_{a.mode}") + f"_s{a.seed}"
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(f"{OUT}/{tag}.json"):
        print(f"[{tag}] done, skip", flush=True); return

    model = Model(**CONFIGS[a.size], seqlen=a.seqlen).cuda()
    n_ne = sum(p.numel() for p in model.parameters()) - model.wte.weight.numel()
    nq = quantize(model, a.bits, a.mode) if a.bits else 0
    expect = 7 * CONFIGS[a.size]["n_layer"]
    if a.bits and nq != expect:
        raise RuntimeError(f"quantized {nq}, expected {expect}")

    steps = a.tokens // (a.batch * a.accum * a.seqlen)
    print(f"[{tag}] N={n_ne/1e6:.1f}M quantized={nq} steps={steps} "
          f"tokens={a.tokens/1e6:.0f}M seq={a.seqlen}", flush=True)

    data = Mix(TOK, a.seqlen, a.p_tool, a.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1,
                            betas=(0.9, 0.95), fused=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps,
                                                pct_start=0.02)
    lf = nn.CrossEntropyLoss()
    amp = torch.autocast("cuda", dtype=torch.bfloat16)

    def evaluate(fn, n=20):
        model.eval(); tot = 0.0
        with torch.no_grad(), amp:
            for _ in range(n):
                x, y = fn(a.batch)
                tot += lf(model(x).view(-1, VOCAB), y.view(-1)).item()
        model.train(); return tot / n

    if a.probe:
        for _ in range(3):
            x, y = data.train(a.batch)
            with amp: lf(model(x).view(-1, VOCAB), y.view(-1)).backward()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t0 = time.time(); K = 20
        for _ in range(K):
            x, y = data.train(a.batch)
            with amp: lf(model(x).view(-1, VOCAB), y.view(-1)).backward()
            opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / K
        tps = a.batch * a.seqlen / dt
        tflops = 6 * n_ne * tps / 1e12
        print(f"[probe] {dt*1000:.0f} ms/step  {tps:,.0f} tok/s  "
              f"{tflops:.1f} TFLOP/s effective  "
              f"peak_mem={torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
        return

    hist, t0 = [], time.time()
    for step in range(steps):
        for _ in range(a.accum):
            x, y = data.train(a.batch)
            with amp:
                loss = lf(model(x).view(-1, VOCAB), y.view(-1)) / a.accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % max(1, steps // 12) == 0 or step == steps - 1:
            vg, vt = evaluate(data.val_general), evaluate(data.val_tool)
            hist.append({"step": step, "val_general": vg, "val_tool": vt})
            print(f"[{tag}] {step}/{steps} gen={vg:.4f} tool={vt:.4f} "
                  f"({(time.time()-t0)/60:.0f}m)", flush=True)

    rec = {"exp": "stage0", "size": a.size, "bits": a.bits, "mode": a.mode,
           "seed": a.seed, "n_nonembed": n_ne, "tokens": a.tokens,
           "seqlen": a.seqlen, "p_tool": a.p_tool, "steps": steps,
           "val_general": hist[-1]["val_general"], "val_tool": hist[-1]["val_tool"],
           "minutes": (time.time() - t0) / 60, "history": hist, "complete": True}
    if a.bits:                                   # deployed weights must be on-grid
        nf, worst = fold_for_inference(model)
        rec["folded"], rec["max_offgrid"] = nf, worst
        print(f"[{tag}] folded {nf} tensors, max off-grid {worst:.2e}", flush=True)
    json.dump(rec, open(f"{OUT}/{tag}.json", "w"))
    print(f"[{tag}] DONE gen={rec['val_general']:.4f} tool={rec['val_tool']:.4f} "
          f"({rec['minutes']:.0f}m)", flush=True)


if __name__ == "__main__":
    main()
