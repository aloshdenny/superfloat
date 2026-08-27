"""The pure-SF datapath: every value that crosses a register boundary is SF.

Everything the scaling study and the tool-use study measured is weights-only.
The weights sit on the SF grid; the activations flowing between layers are fp32
or bf16, and nothing constrains the result of a matmul. That is a *mixed
precision* result, and it is not what the Atreides datapath does. Atreides
multiplies two Q1.15 operands, accumulates in 32 bits, and **saturates the
result back into Q1.15** before it reaches a register. A weights-only
simulation never sees that saturation, so nothing validated so far covers it.

Measured on SmolLM2-360M the gap is not marginal: activations entering matmuls
reach 2298, matmul outputs reach 15279 and the residual stream reaches 16056,
against a representable bound of 1.

This module makes every one of those boundaries explicit and measurable. A
LLaMA-shaped block is written out here rather than called into, so that each
register write is a named `Site` -- fourteen per layer, plus two shared:

    a_qkv   input_layernorm output, the operand q/k/v share
    o_q     q after RoPE          o_k    k after RoPE          o_v   v
    o_s     attention logits      a_p    softmax weights       o_a   weighted v
    o_o     o_proj output
    a_gu    post_attention_layernorm output, the operand gate/up share
    o_g     gate output           o_u    up output
    a_sg    SiLU(gate)            a_d    SiLU(gate) * up, the down operand
    o_d     down output
    res     the residual stream, written twice per block and once at the embedding
    a_head  final norm output, the operand the tied head reads

Every site carries a scale, and the scale's *granularity* is the whole question,
because granularity is what costs hardware:

    none        no scale at all. Literal Q1.x, which is what the FMA spec says.
    tensor      one calibrated constant. Merges into an adjacent weight or norm
                gain, so it is free at inference.
    chan        one constant per channel. For a matmul operand this is the same
                per-input-channel absorption the scaling study already uses for
                weights, so it folds into the feeding norm and is also free.
    token       one scalar per row, computed at run time from that row. A shared
                exponent per token -- block floating point -- and the first rung
                that costs silicon the format exists to remove. It needs no
                calibration, since the scale is the row's own maximum.
    chan+token  both.

Quantization is simulated as `sf_q(x / c) * c`, which is faithful: the stored
value is on the grid, and the dequantizing multiply is work the normalisation
unit does anyway. What differs between rungs is only how much metadata that unit
holds and whether it must reduce over a tensor at run time.
"""
import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superfloat import sf_params, sf_quantize_sv

MODEL = "HuggingFaceTB/SmolLM2-360M"

GRANS = ("off", "none", "tensor", "chan", "token", "chan+token")

BLOCK_SITES = ("a_qkv", "o_q", "o_k", "o_v", "o_s", "a_p", "o_a", "o_o",
               "a_gu", "o_g", "o_u", "a_sg", "a_d", "o_d")
# Tensors the 32-bit accumulator saturates into, as opposed to operands it reads.
RESULT_SITES = ("o_q", "o_k", "o_v", "o_s", "o_a", "o_o", "o_g", "o_u", "o_d")
# Which matmul consumes which operand site, for the accumulator-width analysis.
MATMULS = (("q_proj", "a_qkv", "o_q"), ("k_proj", "a_qkv", "o_k"),
           ("v_proj", "a_qkv", "o_v"), ("o_proj", "o_a", "o_o"),
           ("gate_proj", "a_gu", "o_g"), ("up_proj", "a_gu", "o_u"),
           ("down_proj", "a_d", "o_d"))


def pick_device(pref=None):
    if pref:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


# --------------------------------------------------------------------- site --
class Site(nn.Module):
    """A register boundary: a scale, a bit width, and a census.

    Calibration and use are deliberately separate. `calibrate` records the
    per-tensor maximum, the per-channel maxima and a reservoir of magnitudes;
    nothing about the granularity is baked in, so one calibration pass serves an
    entire granularity sweep. `token` granularity ignores the calibration
    entirely, since a row's scale is that row's own maximum.

    `chan_ok` is false where the last axis is a sequence position rather than a
    feature -- attention logits and softmax weights -- because a per-channel
    scale there would be a scale per key position, which changes shape with the
    context length and means nothing across batches.
    """

    def __init__(self, name, chan_ok=True):
        super().__init__()
        self.name, self.chan_ok = name, chan_ok
        self.bits, self.gran, self.pow2 = 0, "off", False
        self.sf_scale, self.sf_vmax = 0.0, 0.0
        self.register_buffer("cal_amax", torch.zeros(()), persistent=False)
        self.register_buffer("cal_lvl", torch.zeros(()), persistent=False)
        self.register_buffer("cal_chan", torch.zeros(1), persistent=False)
        self.calibrated = False
        self.mode = None                       # None | "calib" | "census"
        self.last_scale = None
        self.reset_stats()

    def configure(self, bits, gran, pow2=False):
        if gran not in GRANS:
            raise ValueError(f"granularity {gran!r} not in {GRANS}")
        if "chan" in gran and not self.chan_ok:
            gran = gran.replace("chan+", "").replace("chan", "tensor")
        self.bits, self.gran, self.pow2 = bits, gran, pow2
        self.sf_scale, self.sf_vmax = sf_params(bits) if bits else (0.0, 0.0)
        return self

    def reset_stats(self):
        self.n = 0
        self.amax = 0.0
        self.sumsq = 0.0
        self.clip_n = 0
        self.chan_amax = None
        self.row_amax_sum = 0.0
        self.row_amax_max = 0.0
        self.row_n = 0
        self.pool = []
        self.pool_batches = 0

    # ---- calibration and census -------------------------------------------
    @torch.no_grad()
    def observe(self, x, pool_batches=3, per_batch=40_000):
        a = x.detach().abs().float()
        flat = a.reshape(-1, a.shape[-1])
        self.n += a.numel()
        self.amax = max(self.amax, float(a.amax()))
        self.sumsq += float(a.pow(2).sum())
        c = flat.amax(0)
        self.chan_amax = c.clone() if self.chan_amax is None else torch.maximum(self.chan_amax, c)
        r = flat.amax(1)
        self.row_amax_sum += float(r.sum())
        self.row_amax_max = max(self.row_amax_max, float(r.amax()))
        self.row_n += r.numel()
        if self.pool_batches < pool_batches:
            k = min(per_batch, flat.numel())
            idx = torch.randint(0, flat.numel(), (k,), device=flat.device)
            self.pool.append(flat.reshape(-1)[idx].cpu())
            self.pool_batches += 1

    @torch.no_grad()
    def finalize(self, q=1.0):
        """Freeze the static scale. `q < 1` takes a percentile of the magnitude
        distribution instead of the maximum, trading a known clipping rate for
        resolution on everything that is not an outlier -- the trade the census
        says is on the table, since these tensors run 10 to 14 bits between their
        typical value and their largest."""
        if self.chan_amax is None:
            return self
        chan = self.chan_amax.clamp_min(1e-8).clone()
        if self.cal_chan.shape != chan.shape:
            self.register_buffer("cal_chan", chan, persistent=False)
        else:
            self.cal_chan.copy_(chan)
        self.cal_amax.fill_(max(self.amax, 1e-8))
        lvl = self.amax
        if q < 1.0 and self.pool:
            import numpy as np
            lvl = float(np.quantile(torch.cat(self.pool).numpy(), q))
        self.cal_lvl.fill_(max(lvl, 1e-8))
        self.calibrated = True
        return self

    # ---- forward -----------------------------------------------------------
    def scale(self, x):
        need_cal = self.gran in ("tensor", "chan")
        if need_cal and not self.calibrated:
            raise RuntimeError(f"site {self.name}: granularity {self.gran} needs calibration")
        if "chan" in self.gran:
            s = self.cal_chan
        elif "token" in self.gran:
            s = torch.ones((), device=x.device, dtype=x.dtype)
        else:
            s = self.cal_lvl
        s = s.to(x.dtype)
        if "token" in self.gran:
            s = s * (x.detach().abs() / s).amax(-1, keepdim=True).clamp_min(1e-8)
        if self.pow2:
            s = torch.exp2(torch.ceil(torch.log2(s.clamp_min(1e-30))))
        return s

    def forward(self, x):
        if self.mode is not None:
            self.observe(x)
            if self.mode == "calib":
                return x
        if not self.bits or self.gran == "off":
            return x
        # QAT: keep the deployed scale tracking the live distribution so a
        # checkpoint whose residual is 16000 can walk it down rather than
        # clipping against a frozen calibration from step 0.
        if (self.training and self.mode is None
                and self.gran in ("tensor", "chan", "chan+token")
                and self.calibrated):
            with torch.no_grad():
                a = x.detach().abs()
                m = a.amax().clamp_min(1e-8).to(dtype=self.cal_lvl.dtype)
                self.cal_lvl.mul_(0.99).add_(0.01 * m)
                if "chan" in self.gran:
                    c = a.reshape(-1, a.shape[-1]).amax(0).clamp_min(1e-8)
                    if self.cal_chan.numel() == c.numel():
                        self.cal_chan.mul_(0.99).add_(0.01 * c.to(dtype=self.cal_chan.dtype))
        if self.gran == "none":
            if self.mode == "census":
                self.clip_n += int((x.detach().abs() > self.sf_vmax).sum())
            return sf_quantize_sv(x, self.sf_scale, self.sf_vmax)
        s = self.scale(x)
        self.last_scale = s
        xs = x / s
        if self.mode == "census":
            self.clip_n += int((xs.detach().abs() > self.sf_vmax).sum())
        return sf_quantize_sv(xs, self.sf_scale, self.sf_vmax) * s

    # ---- reporting ---------------------------------------------------------
    def report(self):
        import numpy as np
        rms = math.sqrt(self.sumsq / self.n) if self.n else 0.0
        d = dict(site=self.name, n=self.n, amax=self.amax, rms=rms,
                 clip_frac=self.clip_n / self.n if self.n else 0.0)
        if self.pool:
            p = torch.cat(self.pool).numpy()
            for nm, q in (("p50", .5), ("p99", .99), ("p999", .999)):
                d[nm] = float(np.quantile(p, q))
            d["outlier_bits"] = math.log2(max(self.amax / max(d["p999"], 1e-12), 1.0))
        if self.chan_amax is not None:
            cm = self.chan_amax.float()
            d["chan_amax_max"] = float(cm.max())
            d["chan_amax_med"] = float(cm.median())
            d["chan_bits"] = math.log2(max(float(cm.max()) / max(float(cm.median()), 1e-12), 1.0))
        if self.row_n:
            mean = self.row_amax_sum / self.row_n
            d["row_amax_mean"] = mean
            d["row_amax_max"] = self.row_amax_max
            d["token_bits"] = math.log2(max(self.row_amax_max / max(mean, 1e-12), 1.0))
        if rms > 0:
            d["crest_bits"] = math.log2(max(self.amax / rms, 1.0))
        return d


# -------------------------------------------------------------------- model --
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


class SFLinear(nn.Linear):
    """A matmul whose weights sit on the SF grid.

    `sf_group` is the set of weights sharing one per-input-channel scale,
    because a norm carries a single gain vector and cannot fold three
    independent scales. Where the group is the layer's own weight alone the
    scale still folds -- one matmul back rather than into a norm -- which is the
    `ln_all` placement the tool-use study needed once trained weights turned out
    to reach |w| = 7.5.
    """
    sf_scale = 0.0
    sf_vmax = 0.0
    sf_group = None
    sf_gran = "off"

    def weight_scale(self):
        if self.sf_gran in ("off", "none"):
            return None
        if self.sf_gran == "tensor":
            return self.weight.detach().abs().amax().clamp_min(1e-8)
        src = self.sf_group if self.sf_group is not None else [self.weight]
        return torch.stack([t.abs().amax(dim=0) for t in src]).amax(0).clamp_min(1e-8)

    def weight_q(self):
        w = self.weight
        if self.sf_gran == "off" or not self.sf_scale:
            return w
        g = self.weight_scale()
        if g is None:
            return sf_quantize_sv(w, self.sf_scale, self.sf_vmax)
        if g.dim():
            g = g.unsqueeze(0)
        return sf_quantize_sv(w / g, self.sf_scale, self.sf_vmax) * g

    def forward(self, x):
        return F.linear(x, self.weight_q(), self.bias)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, nh, nkv = cfg["hidden"], cfg["heads"], cfg["kv_heads"]
        self.nh, self.nkv, self.hd = nh, nkv, d // nh
        self.q_proj = nn.Linear(d, nh * self.hd, bias=False)
        self.k_proj = nn.Linear(d, nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(d, nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(nh * self.hd, d, bias=False)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h = cfg["hidden"], cfg["inter"]
        self.gate_proj = nn.Linear(d, h, bias=False)
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)


class Block(nn.Module):
    """Module names mirror HuggingFace's LlamaDecoderLayer so a checkpoint loads
    without a key map, but the forward is spelled out because every intermediate
    needs a site and transformers' internal signatures move between releases.

    q and k are quantized after RoPE rather than before: the accumulator feeds
    the rotation unit and the rotated value is what reaches a register. Rotation
    is norm-preserving per pair, so the two differ by at most half a bit anyway.
    """

    def __init__(self, cfg):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg["hidden"], cfg["eps"])
        self.post_attention_layernorm = RMSNorm(cfg["hidden"], cfg["eps"])
        self.sites = nn.ModuleDict({k: Site(k, chan_ok=k not in ("o_s", "a_p"))
                                    for k in BLOCK_SITES})
        # One residual site per block, used when the residual scale is allowed
        # to differ by depth. A single global residual scale is the other
        # extreme -- it is what `Llama.res` measures, and it is the thing the
        # census says cannot work (amax 24000 against a typical value of 4).
        self.res = Site("res")

    def forward(self, x, cos, sin, res_site):
        S = self.sites
        B, T, _ = x.shape
        a = self.self_attn
        h = S["a_qkv"](self.input_layernorm(x))
        q = a.q_proj(h).view(B, T, a.nh, a.hd).transpose(1, 2)
        k = a.k_proj(h).view(B, T, a.nkv, a.hd).transpose(1, 2)
        v = S["o_v"](a.v_proj(h)).view(B, T, a.nkv, a.hd).transpose(1, 2)
        q = S["o_q"](q * cos + rotate_half(q) * sin)
        k = S["o_k"](k * cos + rotate_half(k) * sin)
        if a.nkv != a.nh:
            r = a.nh // a.nkv
            k = k.repeat_interleave(r, dim=1)
            v = v.repeat_interleave(r, dim=1)
        if all(S[n].gran == "off" and S[n].mode is None for n in ("o_s", "a_p")):
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            s = S["o_s"](q @ k.transpose(-1, -2) * (a.hd ** -0.5))
            s = s + torch.triu(torch.full((T, T), float("-inf"), device=s.device,
                                          dtype=s.dtype), 1)
            o = S["a_p"](s.softmax(-1)) @ v
        o = S["o_a"](o.transpose(1, 2).reshape(B, T, -1))
        r = self.res if res_site is None else res_site
        x = r(x + S["o_o"](a.o_proj(o)))
        m = self.mlp
        h = S["a_gu"](self.post_attention_layernorm(x))
        g = S["a_sg"](F.silu(S["o_g"](m.gate_proj(h))))
        u = S["o_u"](m.up_proj(h))
        return r(x + S["o_d"](m.down_proj(S["a_d"](g * u))))


class Llama(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg["hidden"]
        self.embed_tokens = nn.Embedding(cfg["vocab"], d)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg["layers"])])
        self.norm = RMSNorm(d, cfg["eps"])
        self.lm_head = nn.Linear(d, cfg["vocab"], bias=False)
        if cfg.get("tied", True):
            self.lm_head.weight = self.embed_tokens.weight
        # Global residual site: one physical buffer, one scale. Per-layer
        # residual sites live on each Block and are selected by `res_mode`.
        self.res = Site("res")
        self.res_mode = "global"
        self.grad_ckpt = False
        self.a_head = Site("a_head")
        hd = d // cfg["heads"]
        inv = 1.0 / (cfg["theta"] ** (torch.arange(0, hd, 2).float() / hd))
        f = torch.outer(torch.arange(cfg["seqlen"]).float(), inv)
        emb = torch.cat((f, f), -1)
        self.register_buffer("cos", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None], persistent=False)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.res(self.embed_tokens(idx))
        cos, sin = self.cos[:, :, :T].to(x.dtype), self.sin[:, :, :T].to(x.dtype)
        res_arg = None if self.res_mode == "layer" else self.res
        for b in self.layers:
            if self.training and self.grad_ckpt:
                x = torch.utils.checkpoint.checkpoint(
                    b, x, cos, sin, res_arg, use_reentrant=False)
            else:
                x = b(x, cos, sin, res_arg)
        return self.lm_head(self.a_head(self.norm(x)))

    # ---- site plumbing -----------------------------------------------------
    def all_sites(self):
        out = [("res", self.res), ("a_head", self.a_head)]
        for i, b in enumerate(self.layers):
            out.append((f"L{i}.res", b.res))
            for k in BLOCK_SITES:
                out.append((f"L{i}.{k}", b.sites[k]))
        return out

    def set_mode(self, mode):
        for _, s in self.all_sites():
            s.mode = mode

    def reset_stats(self):
        for _, s in self.all_sites():
            s.reset_stats()

    def configure_sites(self, bits_a=0, bits_o=None, bits_r=None, gran="tensor",
                        gran_r=None, pow2=False, only=None, off=(),
                        res_mode="global"):
        """Bit widths split by role, because the study's own finding is that the
        two matmul operands are not symmetric: weights saturate at SF3-SF4 while
        activations need SF6. `only` restricts quantization to a named subset,
        which is how one site is isolated as the cause of a collapse.

        `res_mode` is `global` (one scale for the whole residual stream, the
        hardware-literal case) or `layer` (one static scale per block -- 32
        constants, still free at inference, and the thing the depth-growth
        census says you actually need).
        """
        bits_o = bits_a if bits_o is None else bits_o
        bits_r = bits_a if bits_r is None else bits_r
        gran_r = gran if gran_r is None else gran_r
        self.res_mode = res_mode
        for name, s in self.all_sites():
            short = name.split(".")[-1]
            is_layer_res = short == "res" and name.startswith("L")
            if (only is not None and short not in only) or short in off:
                s.configure(0, "off")
            elif short == "res":
                if res_mode == "global" and is_layer_res:
                    s.configure(0, "off")
                else:
                    s.configure(bits_r, gran_r if bits_r else "off", pow2)
            elif short in RESULT_SITES:
                s.configure(bits_o, gran if bits_o else "off", pow2)
            else:
                s.configure(bits_a, gran if bits_a else "off", pow2)
        return self

    def quantize_weights(self, bits, gran="chan"):
        """`chan` reproduces the tool-use study's `ln_all`: q/k/v share one
        per-input-channel scale because they read one norm, gate/up share
        another, and o_proj/down_proj each get their own, folding one matmul back
        instead of into a norm."""
        scale, vmax = sf_params(bits) if bits else (0.0, 0.0)
        n = 0
        for b in self.layers:
            a, m = b.self_attn, b.mlp
            qkv = [a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]
            gu = [m.gate_proj.weight, m.up_proj.weight]
            for mod, grp in ((a.q_proj, qkv), (a.k_proj, qkv), (a.v_proj, qkv),
                             (m.gate_proj, gu), (m.up_proj, gu),
                             (a.o_proj, None), (m.down_proj, None)):
                mod.__class__ = SFLinear
                mod.sf_scale, mod.sf_vmax = scale, vmax
                mod.sf_gran = gran if bits else "off"
                mod.sf_group = grp
                n += 1
        return n

    @torch.no_grad()
    def calibrate(self, batches, q=1.0):
        self.eval()
        self.reset_stats()
        self.set_mode("calib")
        for x in batches:
            self(x)
        self.set_mode(None)
        for _, s in self.all_sites():
            s.finalize(q)
        self.reset_stats()
        return self


# ------------------------------------------------------------- checkpoints ---
def hf_config(name=MODEL, seqlen=8192):
    from huggingface_hub import hf_hub_download
    c = json.load(open(hf_hub_download(name, "config.json")))
    return dict(hidden=c["hidden_size"], layers=c["num_hidden_layers"],
                heads=c["num_attention_heads"], kv_heads=c["num_key_value_heads"],
                inter=c["intermediate_size"], vocab=c["vocab_size"],
                eps=c["rms_norm_eps"], theta=c["rope_theta"],
                tied=c.get("tie_word_embeddings", False), seqlen=seqlen)


def load_checkpoint(model, name=MODEL):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    sd = load_file(hf_hub_download(name, "model.safetensors"))
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    tied = model.lm_head.weight is model.embed_tokens.weight
    if tied:
        sd.pop("lm_head.weight", None)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not (tied and k == "lm_head.weight")]
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing} unexpected={unexpected}")
    return model


def build_smollm2(device, dtype=torch.float32, seqlen=2048, name=MODEL):
    cfg = hf_config(name, seqlen)
    m = Llama(cfg)
    load_checkpoint(m, name)
    return m.to(device=device, dtype=dtype).eval(), cfg


def n_nonembed(model):
    tot = sum(p.numel() for p in model.parameters())
    out = tot - model.embed_tokens.weight.numel()
    if model.lm_head.weight is not model.embed_tokens.weight:
        out -= model.lm_head.weight.numel()
    return out


# ---------------------------------------------------------------- evaluate ---
@torch.no_grad()
def evaluate(model, mix, batch, n=16, splits=("general", "tool")):
    was = model.training
    model.eval()
    lf = nn.CrossEntropyLoss()
    out = {}
    for name in splits:
        fn = mix.val_general if name == "general" else mix.val_tool
        tot = 0.0
        for i in range(n):
            x, y = fn(batch, i)
            lg = model(x)
            tot += float(lf(lg.reshape(-1, lg.shape[-1]).float(), y.reshape(-1)))
        out[name] = tot / n
    model.train(was)
    return out


# ------------------------------------------------- accumulator width bound ---
def accumulator_bits(site_stats, weight_stats, bits_a, bits_w):
    """How wide the accumulator has to be, from the census.

    Both operands enter the array as integers in [-(2^(b-1)-1), 2^(b-1)-1]. The
    accumulator therefore holds `y / (s_x * s_w)` in units of
    `2^-((b_a-1) + (b_w-1))`, so the width needed is

        1 + (b_a - 1) + (b_w - 1) + ceil(log2(max|y| / (s_x * s_w)))

    where the last term is the integer headroom the dot product actually uses.
    The FMA spec pairs a 15x15 multiply with a 32-bit accumulator, which leaves
    two bits of headroom -- enough for a dot product of length four at full
    scale. Whether that is enough for a transformer is an empirical question and
    this is the number that answers it.
    """
    rows = []
    for name, in_site, out_site in MATMULS:
        sx = site_stats.get(in_site, {}).get("scale")
        sw = weight_stats.get(name)
        ymax = site_stats.get(out_site, {}).get("amax")
        if not (sx and sw and ymax):
            continue
        head = math.log2(max(ymax / (sx * sw), 1.0))
        rows.append(dict(matmul=name, headroom_bits=head,
                         acc_bits=1 + (bits_a - 1) + (bits_w - 1) + math.ceil(head)))
    return rows


# ------------------------------------------------------------------ driver ---
OUT = os.environ.get("PSD_OUT", os.path.expanduser("~/sf-psd/results"))
DATA = os.environ.get("PSD_DATA", os.path.expanduser("~/sf-psd/data"))

# The arms that answer the last conversation's question. Weights stay SF8
# with per-input-channel scale (the tool-use study's ln_all, already known
# to be free). What varies is how activations, matmul outputs and the
# residual stream are kept inside the SF bound of +/-1.
#
#   fp32            unquantized control
#   w8_chan         weights-only SF8 -- mixed-precision, the published result
#   sat8_none       literal FMA spec: no scale, saturate every register write
#   a8t_roff        operand+result tensor scale, residual untouched
#   a8t_rt          tensor scale on everything, including the residual stream
#   a8c_rt          per-channel activations, tensor residual
#   a8tok_rtok      per-token (block-float) -- first rung that costs silicon
#   sat8_roff       saturate matmuls, leave residual in fp32 (isolates the sat)
#   r8_none         residual saturates, everything else weights-only
#   r8_tensor       residual tensor-scaled, everything else weights-only
PTQ_ARMS = (
    dict(name="fp32",        bits_w=0, bits_a=0, bits_o=0, bits_r=0,
         gran="off",    gran_w="off"),
    dict(name="w8_chan",     bits_w=8, bits_a=0, bits_o=0, bits_r=0,
         gran="off",    gran_w="chan"),
    dict(name="sat8_none",   bits_w=8, bits_a=8, bits_o=8, bits_r=8,
         gran="none",   gran_w="chan"),
    dict(name="sat8_roff",   bits_w=8, bits_a=8, bits_o=8, bits_r=0,
         gran="none",   gran_w="chan"),
    dict(name="a8t_roff",    bits_w=8, bits_a=8, bits_o=8, bits_r=0,
         gran="tensor", gran_w="chan"),
    dict(name="a8t_rt",      bits_w=8, bits_a=8, bits_o=8, bits_r=8,
         gran="tensor", gran_r="tensor", gran_w="chan"),
    dict(name="a8c_rt",      bits_w=8, bits_a=8, bits_o=8, bits_r=8,
         gran="chan",   gran_r="tensor", gran_w="chan"),
    dict(name="a8tok_rtok",  bits_w=8, bits_a=8, bits_o=8, bits_r=8,
         gran="token",  gran_r="token", gran_w="chan"),
    dict(name="a8t_rlayer",  bits_w=8, bits_a=8, bits_o=8, bits_r=8,
         gran="tensor", gran_r="tensor", gran_w="chan", res_mode="layer"),
    dict(name="r8_none",     bits_w=8, bits_a=0, bits_o=0, bits_r=8,
         gran="off",    gran_r="none", gran_w="chan"),
    dict(name="r8_tensor",   bits_w=8, bits_a=0, bits_o=0, bits_r=8,
         gran="off",    gran_r="tensor", gran_w="chan"),
    dict(name="r8_layer",    bits_w=8, bits_a=0, bits_o=0, bits_r=8,
         gran="off",    gran_r="tensor", gran_w="chan", res_mode="layer"),
    dict(name="r8_token",    bits_w=8, bits_a=0, bits_o=0, bits_r=8,
         gran="off",    gran_r="token", gran_w="chan"),
    dict(name="a8t_rtok",    bits_w=8, bits_a=8, bits_o=8, bits_r=8,
         gran="tensor", gran_r="token", gran_w="chan"),
    dict(name="a8tok_sdpa",  bits_w=8, bits_a=8, bits_o=8, bits_r=8,
         gran="token",  gran_r="token", gran_w="chan", off=("o_s", "a_p")),
)


def apply_arm(model, spec):
    model.quantize_weights(spec.get("bits_w", 0), spec.get("gran_w", "chan"))
    model.configure_sites(
        bits_a=spec.get("bits_a", 0),
        bits_o=spec.get("bits_o"),
        bits_r=spec.get("bits_r"),
        gran=spec.get("gran", "off"),
        gran_r=spec.get("gran_r"),
        pow2=spec.get("pow2", False),
        only=spec.get("only"),
        off=tuple(spec.get("off") or ()),
        res_mode=spec.get("res_mode", "global"),
    )
    return spec["name"]


def needs_calib(spec):
    if spec.get("bits_a") and spec.get("gran") in ("tensor", "chan", "chan+token"):
        return True
    if spec.get("bits_o") and spec.get("gran") in ("tensor", "chan", "chan+token"):
        return True
    if spec.get("bits_r") and spec.get("gran_r", spec.get("gran")) in (
            "tensor", "chan", "chan+token"):
        return True
    return False


def summarize_census(model):
    """Collapse per-layer sites into one row per site type, keeping the worst
    layer's amax so the depth-growth of the residual is visible."""
    by = {}
    for name, s in model.all_sites():
        d = s.report()
        short = name.split(".")[-1]
        row = by.setdefault(short, dict(site=short, amax=0.0, rms=0.0, n=0,
                                        clip_frac=0.0, layers=0, worst=None))
        row["amax"] = max(row["amax"], d["amax"])
        row["n"] += d["n"]
        row["clip_frac"] = max(row["clip_frac"], d.get("clip_frac", 0.0))
        row["layers"] += 1
        if row["worst"] is None or d["amax"] > by[short].get("_worst_amax", 0):
            row["worst"] = name
            row["_worst_amax"] = d["amax"]
        for k in ("p50", "p99", "p999", "crest_bits", "outlier_bits",
                  "chan_bits", "token_bits"):
            if k in d:
                row[k] = max(row.get(k, 0.0), d[k])
    for row in by.values():
        row.pop("_worst_amax", None)
    return [by[k] for k in ("res", "a_head") + BLOCK_SITES if k in by]


def append_jsonl(path, rec):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def done_names(path):
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("complete"):
                out.add(rec.get("name"))
    return out


def calib_batches(mix, n, batch):
    xs = []
    for i in range(n):
        x, _ = mix.val_general(batch, i)
        xs.append(x)
    return xs


@torch.no_grad()
def run_census(model, mix, batch, n=8):
    model.eval()
    model.reset_stats()
    model.set_mode("census")
    for i in range(n):
        x, _ = mix.val_general(batch, i)
        model(x)
    model.set_mode(None)
    return summarize_census(model)


def run_ptq(model, mix, args, device):
    path = os.path.join(args.out, "ptq.jsonl")
    skip = done_names(path) if not args.force else set()
    calibrated_key = None
    arms = [a for a in PTQ_ARMS if a["name"] in args.arms.split(",")] if args.arms else list(PTQ_ARMS)
    print(f"PTQ {len(arms)} arms, seq={args.seqlen} batch={args.batch} "
          f"eval_n={args.eval_n} skip={sorted(skip)}", flush=True)

    for spec in arms:
        name = spec["name"]
        if name in skip:
            print(f"  [{name}] already done", flush=True)
            continue
        apply_arm(model, spec)
        if needs_calib(spec):
            key = (spec.get("res_mode", "global"), args.q)
            if calibrated_key != key:
                print("  calibrating...", flush=True)
                t0 = time.time()
                model.calibrate(calib_batches(mix, args.calib_n, args.batch), q=args.q)
                calibrated_key = key
                print(f"  calibrated in {time.time()-t0:.1f}s", flush=True)
                apply_arm(model, spec)
        t0 = time.time()
        sync(device)
        loss = evaluate(model, mix, args.batch, n=args.eval_n)
        sync(device)
        rec = dict(exp="psd_ptq", name=name, spec=spec, **loss,
                   seconds=time.time() - t0, seqlen=args.seqlen,
                   batch=args.batch, eval_n=args.eval_n, q=args.q,
                   complete=True)
        append_jsonl(path, rec)
        print(f"  [{name:12s}] general={loss['general']:.4f} tool={loss['tool']:.4f}  "
              f"{rec['seconds']:.0f}s", flush=True)
    return path


def run_qat(model, mix, args, device):
    spec = dict(name=args.qat_arm)
    match = [a for a in PTQ_ARMS if a["name"] == args.qat_arm]
    if match:
        spec = dict(match[0])
    else:
        raise SystemExit(f"unknown QAT arm {args.qat_arm!r}; choose from "
                         + ", ".join(a["name"] for a in PTQ_ARMS))

    tag = f"qat_{spec['name']}_s{args.seed}"
    ckpt_dir = os.path.join(args.out, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, tag + ".pt")
    hist_path = os.path.join(args.out, tag + ".json")

    apply_arm(model, spec)
    if needs_calib(spec):
        print("calibrating for QAT...", flush=True)
        model.calibrate(calib_batches(mix, args.calib_n, args.batch), q=args.q)
        apply_arm(model, spec)

    model.train()
    model.grad_ckpt = True
    steps = args.tokens // (args.batch * args.seqlen)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.03)
    start = 0
    hist = []
    if os.path.exists(ckpt_path) and not args.force:
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        sched.load_state_dict(blob["sched"])
        start = blob["step"] + 1
        hist = blob.get("hist", [])
        print(f"resumed {tag} from step {start}", flush=True)

    print(f"[{tag}] steps={steps} tokens={args.tokens/1e6:.2f}M "
          f"seq={args.seqlen} batch={args.batch} lr={args.lr} "
          f"start={start}", flush=True)

    lf = nn.CrossEntropyLoss()
    t0 = time.time()
    warmup = 2
    timed_start = None
    reported_eta = False
    for step in range(start, steps):
        x, y = mix.train(args.batch)
        lg = model(x)
        loss = lf(lg.reshape(-1, lg.shape[-1]).float(), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

        if step == start + warmup:
            timed_start = time.time()
        n_timed = step - (start + warmup)
        if (not reported_eta and timed_start is not None
                and n_timed >= args.timing_after):
            sec_step = (time.time() - timed_start) / n_timed
            left = steps - (step + 1)
            eta_h = left * sec_step / 3600.0
            tok_s = args.batch * args.seqlen / sec_step
            print(f"[{tag}] TIMING after {step+1} steps: {sec_step:.2f}s/step  "
                  f"{tok_s:.0f} tok/s  remaining {left} steps ≈ {eta_h:.1f} h  "
                  f"({eta_h*60:.0f} min)", flush=True)
            reported_eta = True

        if step % args.log_every == 0 or step == steps - 1:
            sync(device)
            with torch.no_grad():
                val = evaluate(model, mix, args.batch, n=max(4, args.eval_n // 2))
            rec = dict(step=step, train_loss=float(loss.detach()), **val,
                       minutes=(time.time() - t0) / 60.0)
            hist.append(rec)
            print(f"[{tag}] {step}/{steps} train={rec['train_loss']:.4f} "
                  f"general={val['general']:.4f} tool={val['tool']:.4f} "
                  f"({rec['minutes']:.1f}m)", flush=True)
            if step % max(args.log_every * 4, 100) == 0 or step == steps - 1:
                torch.save(dict(step=step, model=model.state_dict(),
                                opt=opt.state_dict(), sched=sched.state_dict(),
                                hist=hist, spec=spec), ckpt_path)

    rec = dict(exp="psd_qat", name=spec["name"], spec=spec, seed=args.seed,
               tokens=args.tokens, steps=steps, seqlen=args.seqlen,
               batch=args.batch, lr=args.lr, history=hist,
               minutes=(time.time() - t0) / 60.0, complete=True)
    if hist:
        rec["val_general"] = hist[-1]["general"]
        rec["val_tool"] = hist[-1]["tool"]
    json.dump(rec, open(hist_path, "w"), indent=2)
    print(f"[{tag}] DONE {hist_path}", flush=True)
    return rec


def main():
    import argparse
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTHONHASHSEED", "0")

    ap = argparse.ArgumentParser(description="Pure-SF datapath study on SmolLM2-360M")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--eval-n", type=int, default=8)
    ap.add_argument("--calib-n", type=int, default=8)
    ap.add_argument("--q", type=float, default=1.0, help="calibration percentile")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--ptq", action="store_true")
    ap.add_argument("--arms", default="", help="comma-separated PTQ arm names")
    ap.add_argument("--qat", action="store_true")
    ap.add_argument("--qat-arm", default="a8t_rt",
                    help="PTQ arm to train: default tensor-scaled residual")
    ap.add_argument("--tokens", type=int, default=2_000_000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--timing-after", type=int, default=5,
                    help="print remaining-time estimate after this many steps")
    ap.add_argument("--dtype", default="float32", choices=("float32", "float16"))
    a = ap.parse_args()

    from psd_data import Mix

    device = pick_device(a.device)
    dtype = torch.float32 if a.dtype == "float32" else torch.float16
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)

    print(f"device={device} dtype={dtype} data={a.data} out={a.out}", flush=True)
    t_load = time.time()
    model, cfg = build_smollm2(device, dtype=dtype, seqlen=a.seqlen)
    print(f"loaded SmolLM2-360M in {time.time()-t_load:.1f}s  "
          f"nonembed={n_nonembed(model)/1e6:.1f}M  layers={cfg['layers']}  "
          f"d={cfg['hidden']}", flush=True)

    mix = Mix(a.data, seqlen=a.seqlen, p_tool=0.25, seed=a.seed, device=str(device))
    print(f"corpus general={len(mix.g)/1e6:.0f}M tool={len(mix.t)/1e6:.1f}M", flush=True)

    if not (a.census or a.ptq or a.qat):
        a.census = a.ptq = True

    if a.census:
        print("census...", flush=True)
        t0 = time.time()
        rows = run_census(model, mix, a.batch, n=a.eval_n)
        rec = dict(exp="psd_census", seconds=time.time() - t0, rows=rows)
        json.dump(rec, open(os.path.join(a.out, "census.json"), "w"), indent=2)
        print(f"{'site':8s} {'amax':>10s} {'p99':>10s} {'crest_bits':>11s}  worst")
        for r in rows:
            print(f"{r['site']:8s} {r['amax']:10.2f} {r.get('p99', float('nan')):10.2f} "
                  f"{r.get('crest_bits', float('nan')):11.2f}  {r.get('worst')}")
        print(f"census done in {rec['seconds']:.1f}s", flush=True)

    if a.ptq:
        run_ptq(model, mix, a, device)

    if a.qat:
        run_qat(model, mix, a, device)


if __name__ == "__main__":
    main()

