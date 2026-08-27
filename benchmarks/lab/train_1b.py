"""Prove SF QAT holds at Llama-3.2-1B from scratch.

Tier A of the scaling study stopped at 85M non-embedding params. This is the
same recipe — RMSNorm, SwiGLU, GQA, per-input-channel scale on every matmul —
at the Llama 3.2 1B shape, trained on FineWeb-Edu.

The Llama 3 tokenizer's vocabulary is 128256. That does not fit in uint16
(max 65535); token ids wrap and the run is garbage. Everything on disk is
uint32. That is the tokenizer bug from the previous attempt.

    python train_1b.py --prepare 20000000000
    python train_1b.py --bits 8 --mode ln_all --tokens 20000000000

Resume is the default: SIGINT/SIGTERM and a periodic checkpoint both write
`ckpt/latest.pt`, and the next launch picks it up. Tokenizer shards are
immutable once complete, so a killed prepare continues at the next shard.
"""
from __future__ import annotations

import argparse, gc, json, math, os, signal, sys, time, glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superfloat import disable_tf32, sf_params, sf_quantize_sv

# Llama 3.2 1B. Embeddings are tied and stay in bf16 with the head.
CFG = dict(hidden=2048, layers=16, heads=32, kv_heads=8, inter=8192,
           vocab=128256, eps=1e-5, theta=500_000.0)
assert CFG["vocab"] > 65535, "vocab must not fit uint16; that is the bug"

TOKENIZER = os.environ.get("SF1B_TOKENIZER", "meta-llama/Llama-3.2-1B")
DATA_NAME = os.environ.get("SF1B_DATA", "HuggingFaceFW/fineweb-edu")
DATA_CFG = os.environ.get("SF1B_DATA_CFG", "sample-100BT")
SHARD = 100_000_000          # tokens per immutable shard (400 MiB uint32)
VAL_TOKENS = 4_000_000
DTYPE = np.uint32


# ----------------------------------------------------------------- model ----
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        v = x.float().pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(v + self.eps)).type_as(x)


def rotate_half(t):
    a, b = t.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, nh, nkv = cfg["hidden"], cfg["heads"], cfg["kv_heads"]
        self.nh, self.nkv, self.hd = nh, nkv, d // nh
        self.input_layernorm = RMSNorm(d, cfg["eps"])
        self.q_proj = nn.Linear(d, nh * self.hd, bias=False)
        self.k_proj = nn.Linear(d, nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(d, nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(nh * self.hd, d, bias=False)
        self.post_attention_layernorm = RMSNorm(d, cfg["eps"])
        self.gate_proj = nn.Linear(d, cfg["inter"], bias=False)
        self.up_proj = nn.Linear(d, cfg["inter"], bias=False)
        self.down_proj = nn.Linear(cfg["inter"], d, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        h = self.input_layernorm(x)
        q = self.q_proj(h).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        if self.nkv != self.nh:
            r = self.nh // self.nkv
            k = k.repeat_interleave(r, dim=1)
            v = v.repeat_interleave(r, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o_proj(a.transpose(1, 2).reshape(B, T, -1))
        h = self.post_attention_layernorm(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class Llama(nn.Module):
    def __init__(self, cfg, seqlen):
        super().__init__()
        self.cfg = cfg
        d = cfg["hidden"]
        self.embed_tokens = nn.Embedding(cfg["vocab"], d)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg["layers"])])
        self.norm = RMSNorm(d, cfg["eps"])
        self.lm_head = nn.Linear(d, cfg["vocab"], bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        hd = d // cfg["heads"]
        inv = 1.0 / (cfg["theta"] ** (torch.arange(0, hd, 2).float() / hd))
        freq = torch.outer(torch.arange(seqlen).float(), inv)
        emb = torch.cat((freq, freq), -1)
        self.register_buffer("cos", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None], persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg["layers"]))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.embed_tokens(idx)
        cos = self.cos[:, :, :T].to(dtype=x.dtype)
        sin = self.sin[:, :, :T].to(dtype=x.dtype)
        for i, blk in enumerate(self.layers):
            if self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, cos, sin,
                                                      use_reentrant=False)
            else:
                x = blk(x, cos, sin)
        return self.lm_head(self.norm(x))


class SFLinear(nn.Linear):
    """Weights on the SF grid. `sf_group` shares one per-input-channel scale
    across the matmuls that read the same norm. Where the group is the layer's
    own weight, the scale folds one matmul back (o_proj, down_proj)."""
    sf_scale = 0.0
    sf_vmax = 0.0
    sf_group = None

    def forward(self, x):
        w = self.weight
        dt = x.dtype
        if self.sf_group is None:
            wq = sf_quantize_sv(w, self.sf_scale, self.sf_vmax).to(dtype=dt)
            b = None if self.bias is None else self.bias.to(dtype=dt)
            return F.linear(x, wq, b)
        g = torch.cat(self.sf_group, 0).abs().amax(0).clamp_min(1e-8).detach()
        wq = sf_quantize_sv(w / g.unsqueeze(0), self.sf_scale, self.sf_vmax).to(dtype=dt)
        b = None if self.bias is None else self.bias.to(dtype=dt)
        return F.linear(x * g.to(dtype=dt), wq, b)


def quantize(model, bits, mode="ln_all"):
    scale, vmax = sf_params(bits)
    n = 0
    for layer in model.layers:
        qkv = [layer.q_proj.weight, layer.k_proj.weight, layer.v_proj.weight]
        gu = [layer.gate_proj.weight, layer.up_proj.weight]
        pairs = [
            (layer.q_proj, qkv), (layer.k_proj, qkv), (layer.v_proj, qkv),
            (layer.gate_proj, gu), (layer.up_proj, gu),
            (layer.o_proj, None), (layer.down_proj, None),
        ]
        for mod, grp in pairs:
            mod.__class__ = SFLinear
            mod.sf_scale, mod.sf_vmax = scale, vmax
            if mode == "plain":
                mod.sf_group = None
            elif mode == "ln_all" and grp is None:
                mod.sf_group = [mod.weight]
            else:
                mod.sf_group = grp
            n += 1
    return n


def n_nonembed(model):
    tot = sum(p.numel() for p in model.parameters())
    return tot - model.embed_tokens.weight.numel()


# ------------------------------------------------------------------ data ----
def shard_path(root, i):
    return os.path.join(root, f"shard_{i:06d}.bin")


def complete_shards(root):
    """List finished shards. A shard is complete when the sidecar `.ok` exists,
    so a half-written file from a killed prepare is never trained on."""
    out = []
    i = 0
    while True:
        p = shard_path(root, i)
        if not os.path.exists(p + ".ok"):
            break
        out.append(p)
        i += 1
    return out


def tokens_ready(root):
    n = 0
    for p in complete_shards(root):
        n += os.path.getsize(p) // 4
    return n


def prepare(n_target, root, shard=SHARD):
    """Stream FineWeb-Edu through the Llama 3 tokenizer into uint32 shards.

    Resume: skip shards that already have a `.ok` sidecar, then continue the
    stream. Document-level skip is approximate (we skip `docs_done` rows stored
    in meta) so a killed shard is retokenized from that document, not replayed
    into a finished shard.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.makedirs(root, exist_ok=True)
    tk = AutoTokenizer.from_pretrained(TOKENIZER)
    if tk.vocab_size > 65535 and np.dtype(DTYPE) != np.dtype(np.uint32):
        raise RuntimeError("Llama vocab does not fit the chosen dtype")
    if tk.vocab_size >= 2 ** 32:
        raise RuntimeError(f"vocab {tk.vocab_size} does not fit uint32")
    eos = tk.eos_token_id
    print(f"tokenizer={TOKENIZER} vocab={tk.vocab_size} eos={eos} dtype=uint32",
          flush=True)

    meta_path = os.path.join(root, "meta.json")
    meta = dict(docs_done=0, shards_done=0)
    if os.path.exists(meta_path):
        meta.update(json.load(open(meta_path)))

    done = complete_shards(root)
    n_have = tokens_ready(root)
    print(f"have {n_have/1e6:.0f}M tokens in {len(done)} shards, "
          f"target {n_target/1e6:.0f}M", flush=True)
    if n_have >= n_target:
        print("prepare already complete", flush=True)
        return n_have

    ds = load_dataset(DATA_NAME, name=DATA_CFG, split="train", streaming=True)
    it = iter(ds)
    skipped = 0
    while skipped < meta.get("docs_done", 0):
        try:
            next(it)
        except StopIteration:
            break
        skipped += 1
        if skipped % 10000 == 0:
            print(f"  skip {skipped} docs", flush=True)
    print(f"resumed stream after {skipped} docs", flush=True)

    shard_i = len(done)
    buf = np.memmap(shard_path(root, shard_i), dtype=DTYPE, mode="w+", shape=(shard,))
    fill = 0
    docs = skipped
    batch = []

    def flush_batch(batch, fill, buf, shard_i, docs):
        if not batch:
            return fill, buf, shard_i
        for ids in tk(batch)["input_ids"]:
            ids = ids + [eos]
            k = 0
            while k < len(ids):
                take = min(len(ids) - k, shard - fill)
                buf[fill:fill + take] = np.array(ids[k:k + take], dtype=DTYPE)
                fill += take
                k += take
                if fill >= shard:
                    buf.flush()
                    open(shard_path(root, shard_i) + ".ok", "w").write("ok\n")
                    n = tokens_ready(root)
                    print(f"  shard {shard_i} done  total={n/1e6:.0f}M", flush=True)
                    shard_i += 1
                    buf = np.memmap(shard_path(root, shard_i), dtype=DTYPE,
                                    mode="w+", shape=(shard,))
                    fill = 0
                    if n >= n_target:
                        return fill, buf, shard_i
        return fill, buf, shard_i

    for row in it:
        batch.append(row["text"])
        docs += 1
        if len(batch) >= 512:
            fill, buf, shard_i = flush_batch(batch, fill, buf, shard_i, docs)
            batch = []
            n_now = tokens_ready(root)
            if n_now >= n_target:
                break
            if len(complete_shards(root)) > meta.get("shards_done", 0):
                meta = dict(docs_done=docs, shards_done=len(complete_shards(root)),
                            tokenizer=TOKENIZER, vocab=tk.vocab_size, dtype="uint32")
                tmp = meta_path + ".tmp"
                json.dump(meta, open(tmp, "w"))
                os.replace(tmp, meta_path)
    if batch and tokens_ready(root) < n_target:
        fill, buf, shard_i = flush_batch(batch, fill, buf, shard_i, docs)
    try:
        buf.flush(); del buf
    except Exception:
        pass
    # a trailing partial shard is left without .ok so the next prepare redoes it
    n = tokens_ready(root)
    print(f"prepare stopped at {n/1e6:.0f}M tokens, {len(complete_shards(root))} shards",
          flush=True)
    return n


class Mix:
    """Samples from completed uint32 shards without concatenating them into
    RAM — 20B uint32 is 80 GiB, which will not fit in this WSL VM."""

    def __init__(self, root, seqlen, seed, device="cuda"):
        self.root, self.L, self.dev = root, seqlen, device
        self.rng = np.random.default_rng(seed)
        self._load()
        if self.n_train < self.L + 2:
            raise RuntimeError(f"not enough train tokens ({self.n_train}) for seq {self.L}")

    def _load(self):
        paths = complete_shards(self.root)
        if not paths:
            raise RuntimeError(f"no complete shards in {self.root}")
        self._maps = [np.memmap(p, dtype=DTYPE, mode="r") for p in paths]
        self._lens = [len(m) for m in self._maps]
        self.n_total = int(sum(self._lens))
        self.n_val = min(VAL_TOKENS, max(self.L * 16, self._lens[0] // 20))
        if self.n_val + self.L + 2 > self._lens[0]:
            self.n_val = max(self.L * 4, self._lens[0] // 10)
        self.n_train = self.n_total - self.n_val
        self._n_shards = len(paths)

    def maybe_reload(self):
        n = len(complete_shards(self.root))
        if n != self._n_shards:
            print(f"mix: reload {self._n_shards} -> {n} shards "
                  f"({tokens_ready(self.root)/1e6:.0f}M tokens)", flush=True)
            self._load()

    def _from_global(self, start):
        """Read L+1 tokens starting at a global index that does not cross a
        shard boundary. Caller guarantees start+L+1 stays inside its shard."""
        remaining = start
        for m, n in zip(self._maps, self._lens):
            if remaining < n:
                sl = np.asarray(m[remaining:remaining + self.L + 1], dtype=np.int64)
                return sl[:self.L], sl[1:]
            remaining -= n
        raise RuntimeError("index past end of corpus")

    def _take_train(self, n, rng):
        # usable windows: shard 0 after the val region, every later shard fully
        spans = []
        for i, ntok in enumerate(self._lens):
            lo = self.n_val if i == 0 else 0
            hi = ntok - self.L - 1
            if hi > lo:
                spans.append((i, lo, hi))
        widths = np.array([hi - lo for _, lo, hi in spans], dtype=np.int64)
        tot = int(widths.sum())
        picks = rng.integers(0, tot, size=n)
        xs, ys = [], []
        for p in picks:
            s = 0
            while p >= widths[s]:
                p -= widths[s]; s += 1
            i, lo, _ = spans[s]
            off = int(lo + p)
            m = self._maps[i]
            sl = np.asarray(m[off:off + self.L + 1], dtype=np.int64)
            xs.append(sl[:self.L]); ys.append(sl[1:])
        x = np.stack(xs); y = np.stack(ys)
        return (torch.from_numpy(x).to(self.dev, non_blocking=True),
                torch.from_numpy(y).to(self.dev, non_blocking=True))

    def train(self, n):
        return self._take_train(n, self.rng)

    def val(self, n, i=0):
        rng = np.random.default_rng(10_007 + i)
        hi = self.n_val - self.L - 1
        ix = rng.integers(0, max(hi, 1), size=n)
        xs, ys = [], []
        m0 = self._maps[0]
        for off in ix:
            sl = np.asarray(m0[int(off):int(off) + self.L + 1], dtype=np.int64)
            xs.append(sl[:self.L]); ys.append(sl[1:])
        x = np.stack(xs); y = np.stack(ys)
        return (torch.from_numpy(x).to(self.dev, non_blocking=True),
                torch.from_numpy(y).to(self.dev, non_blocking=True))


# ---------------------------------------------------------------- train -----
def cosine_lr(step, warmup, total, min_ratio=0.1):
    if step < warmup:
        return max(step, 1) / max(warmup, 1)
    if step >= total:
        return min_ratio
    p = (step - warmup) / max(total - warmup, 1)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))


def clean_state_dict(sd):
    """Strip torch.compile's `._orig_mod` so checkpoints load into eager models."""
    return {k.replace("._orig_mod", ""): v for k, v in sd.items()}


def save_ckpt(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def prune_numbered(ckpt_dir, keep=2):
    files = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
    for p in files[:-keep]:
        try:
            os.remove(p)
        except OSError:
            pass


def append_jsonl(path, rec):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


@torch.no_grad()
def evaluate(model, mix, batch, n=16, vocab=CFG["vocab"]):
    was = model.training
    model.eval()
    lf = nn.CrossEntropyLoss()
    tot_loss = tot_acc = 0.0
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    for i in range(n):
        x, y = mix.val(batch, i)
        with amp:
            lg = model(x)
            loss = lf(lg.reshape(-1, vocab).float(), y.reshape(-1))
            acc = (lg.argmax(-1) == y).float().mean()
        tot_loss += float(loss)
        tot_acc += float(acc)
    model.train(was)
    loss = tot_loss / n
    return dict(val_loss=loss, val_ppl=math.exp(min(loss, 20)), val_acc=tot_acc / n)


def train(args):
    if args.bits and args.bits > 8:
        disable_tf32()
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        # SF8's grid is coarser than TF32's 10-bit mantissa; tensor cores on.
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.out, exist_ok=True)
    ckpt_dir = os.path.join(args.out, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    latest = os.path.join(ckpt_dir, "latest.pt")
    metrics = os.path.join(args.out, "metrics.jsonl")

    tag = ("bf16" if args.bits == 0 else f"sf{args.bits}_{args.mode}") + f"_s{args.seed}"
    print(f"[{tag}] waiting for >= {args.wait_tokens/1e6:.0f}M complete tokens in {args.data}",
          flush=True)
    while tokens_ready(args.data) < args.wait_tokens:
        print(f"  have {tokens_ready(args.data)/1e6:.1f}M, sleep 30s", flush=True)
        time.sleep(30)

    mix = Mix(args.data, args.seqlen, args.seed)
    print(f"[{tag}] corpus {mix.n_total/1e6:.1f}M  train={mix.n_train/1e6:.1f}M  "
          f"val={mix.n_val/1e6:.2f}M", flush=True)

    model = Llama(CFG, args.seqlen).cuda()
    n_ne = n_nonembed(model)
    nq = quantize(model, args.bits, args.mode) if args.bits else 0
    print(f"[{tag}] nonembed={n_ne/1e6:.1f}M quantized={nq} bits={args.bits} "
          f"mode={args.mode}", flush=True)

    tokens_per_step = args.batch * args.accum * args.seqlen
    steps = args.tokens // tokens_per_step
    warmup = max(1, int(steps * args.warmup))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                            betas=(0.9, 0.95), fused=True)
    lf = nn.CrossEntropyLoss()
    amp = torch.autocast("cuda", dtype=torch.bfloat16)

    start = 0
    tokens_seen = 0
    if args.resume and os.path.exists(latest):
        # Load on CPU. map_location="cuda" materializes the 14 GB blob next to
        # the live model+Adam state and fragments the 24 GB card so the first
        # real step after resume can spin forever at 100% SM / 1% mem.
        blob = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        start = blob["step"] + 1
        tokens_seen = blob.get("tokens_seen", start * tokens_per_step)
        if "np_rng" in blob:
            mix.rng.bit_generator.state = blob["np_rng"]
        def _as_byte(t):
            if t is None:
                return None
            if not torch.is_tensor(t):
                t = torch.tensor(list(t) if not isinstance(t, (bytes, bytearray)) else t,
                                 dtype=torch.uint8)
            return t.detach().cpu().contiguous().to(torch.uint8)
        try:
            if "torch_rng" in blob:
                torch.set_rng_state(_as_byte(blob["torch_rng"]))
        except Exception as e:
            print(f"[{tag}] cpu rng restore skipped: {e}", flush=True)
        del blob
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{tag}] resumed step {start} tokens_seen={tokens_seen/1e6:.1f}M  "
              f"mem={torch.cuda.memory_allocated()/2**30:.1f}G "
              f"reserved={torch.cuda.memory_reserved()/2**30:.1f}G",
              flush=True)

    if args.compile:
        print(f"[{tag}] torch.compile {len(model.layers)} blocks "
              f"(first step is slow)", flush=True)
        for i in range(len(model.layers)):
            model.layers[i] = torch.compile(model.layers[i])

    stop = {"flag": False}

    def request_stop(signum, _frame):
        print(f"signal {signum}, will checkpoint after this step", flush=True)
        stop["flag"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def payload(step):
        return dict(step=step, tokens_seen=tokens_seen, tag=tag,
                    model=clean_state_dict(model.state_dict()),
                    opt=opt.state_dict(),
                    np_rng=mix.rng.bit_generator.state,
                    torch_rng=torch.get_rng_state().detach().cpu().contiguous(),
                    args=vars(args), n_nonembed=n_ne)

    print(f"[{tag}] steps={steps} tokens/step={tokens_per_step} "
          f"seq={args.seqlen} batch={args.batch} accum={args.accum} lr={args.lr}",
          flush=True)

    if args.probe:
        model.train()
        for _ in range(3):
            x, y = mix.train(args.batch)
            with amp:
                lf(model(x).reshape(-1, CFG["vocab"]).float(), y.reshape(-1)).backward()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t0 = time.time(); K = 10
        for _ in range(K):
            x, y = mix.train(args.batch)
            with amp:
                lf(model(x).reshape(-1, CFG["vocab"]).float(), y.reshape(-1)).backward()
            opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / K
        tps = args.batch * args.seqlen / dt
        print(f"[probe] {dt*1000:.0f} ms/step  {tps:,.0f} tok/s  "
              f"peak={torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
        return

    t0 = time.time()
    t_ckpt = t0
    t_step = None
    model.train()
    for step in range(start, steps):
        lr = args.lr * cosine_lr(step, warmup, steps)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        train_loss = train_acc = 0.0
        verbose = step < start + 3
        if verbose:
            print(f"[{tag}] step {step} begin  "
                  f"mem={torch.cuda.memory_allocated()/2**30:.1f}G", flush=True)
        loss_sum = acc_sum = None
        want_acc = (step % args.log_every == 0) or (step == steps - 1) or stop["flag"]
        for mi in range(args.accum):
            x, y = mix.train(args.batch)
            with amp:
                lg = model(x)
                loss = lf(lg.reshape(-1, CFG["vocab"]).float(), y.reshape(-1))
            (loss / args.accum).backward()
            loss_sum = loss.detach() if loss_sum is None else loss_sum + loss.detach()
            if want_acc:
                with torch.no_grad():
                    a = (lg.argmax(-1) == y).float().mean()
                    acc_sum = a if acc_sum is None else acc_sum + a
            del lg, loss, x, y
            if verbose:
                print(f"[{tag}] step {step} accum {mi+1}/{args.accum}", flush=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tokens_seen += tokens_per_step
        train_loss = float(loss_sum / args.accum)
        train_acc = float(acc_sum / args.accum) if acc_sum is not None else 0.0

        if t_step is None and step >= start + 2:
            t_step = time.time()
            timed_from = step
        if step == start + 7 and t_step is not None:
            sec = (time.time() - t_step) / max(step - timed_from, 1)
            left = steps - step - 1
            tps = tokens_per_step / sec
            print(f"[{tag}] TIMING {sec:.2f}s/step  {tps:,.0f} tok/s  "
                  f"remaining {left} steps ≈ {left*sec/3600:.1f} h", flush=True)

        wall_ckpt = (time.time() - t_ckpt) >= args.ckpt_seconds
        do_log = (step % args.log_every == 0) or (step == steps - 1) or stop["flag"]
        do_ckpt = (step % args.ckpt_every == 0) or (step == steps - 1) or stop["flag"] \
                  or wall_ckpt
        if do_log:
            mix.maybe_reload()
            val = evaluate(model, mix, args.batch, n=args.eval_n)
            rec = dict(exp="sf1b", tag=tag, step=step, tokens_seen=tokens_seen,
                       train_loss=train_loss, train_ppl=math.exp(min(train_loss, 20)),
                       train_acc=train_acc, lr=lr, **val,
                       seconds=time.time() - t0,
                       tok_s=tokens_seen / max(time.time() - t0, 1),
                       mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
            append_jsonl(metrics, rec)
            print(f"[{tag}] {step}/{steps}  tok={tokens_seen/1e6:.1f}M  "
                  f"train {train_loss:.4f} ppl {rec['train_ppl']:.2f} acc {train_acc:.3f}  "
                  f"val {val['val_loss']:.4f} ppl {val['val_ppl']:.2f} acc {val['val_acc']:.3f}  "
                  f"lr {lr:.2e}  {(time.time()-t0)/3600:.2f}h", flush=True)
        if do_ckpt:
            save_ckpt(latest, payload(step))
            t_ckpt = time.time()
            if step > 0 and (step % args.ckpt_keep_every == 0 or stop["flag"]):
                save_ckpt(os.path.join(ckpt_dir, f"step_{step:07d}.pt"), payload(step))
                prune_numbered(ckpt_dir, keep=2)
            print(f"[{tag}] checkpoint step {step} -> {latest}", flush=True)
        if stop["flag"]:
            print(f"[{tag}] stopped at step {step}", flush=True)
            return

    save_ckpt(latest, payload(steps - 1))
    print(f"[{tag}] DONE tokens={tokens_seen/1e6:.0f}M  {(time.time()-t0)/3600:.1f}h",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", type=int, default=0, help="tokenize this many tokens and exit")
    ap.add_argument("--data", default=os.path.expanduser("~/alosh/sf-1b/data"))
    ap.add_argument("--out", default=os.path.expanduser("~/alosh/sf-1b/run"))
    ap.add_argument("--bits", type=int, default=8, help="0 = bf16 control")
    ap.add_argument("--mode", default="ln_all", choices=["plain", "ln", "ln_all"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=20_000_000_000)
    ap.add_argument("--wait-tokens", type=int, default=50_000_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=float, default=0.02)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--ckpt-keep-every", type=int, default=10000)
    ap.add_argument("--ckpt-seconds", type=int, default=600,
                    help="also write latest.pt at least this often (power outage)")
    ap.add_argument("--eval-n", type=int, default=16)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()

    if a.prepare:
        prepare(a.prepare, a.data)
        return
    train(a)


if __name__ == "__main__":
    main()
