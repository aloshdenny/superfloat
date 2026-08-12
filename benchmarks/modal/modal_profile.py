"""GPU fitting for the SuperFloat scaling-law sweep.

Picks the cheapest GPU that actually fits each tier's workload, rather than
defaulting to the fastest one. Cost per unit of work, not throughput, is the
figure of merit: a B200 that is 3x faster than an L4 is still 8x worse value
if the job is dataloader-bound.

Each probe runs a handful of real training steps (or a real eval pass),
reports peak VRAM, achieved throughput and achieved TFLOP/s, then converts
those into dollars for the full tier. Probes are deliberately tiny -- a few
seconds of GPU each -- so the whole fitting exercise costs about a dollar.

Run:
    modal run modal/modal_profile.py                # all probes, all GPUs
    modal run modal/modal_profile.py --tier c       # just one tier
"""

import pathlib

import modal

BENCH_DIR = str(pathlib.Path(__file__).resolve().parent.parent)

app = modal.App("superfloat-profile")
vol = modal.Volume.from_name("sfx-baselines", create_if_missing=True)

# cu128 / torch 2.8 so the same image runs on Blackwell (sm_100/sm_120) and on
# everything older; a cu124 build cannot target B200 at all.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0", "torchvision==0.23.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("transformers", "datasets", "numpy", "pandas", "timm")
    .add_local_dir(BENCH_DIR, remote_path="/root/sfx_bench")
)

# $/hour, from modal.com/pricing. Used to turn throughput into cost.
PRICE = {
    "T4": 0.59, "L4": 0.80, "A10": 1.10, "L40S": 1.95,
    "A100-40GB": 2.10, "A100-80GB": 2.50, "H100": 3.95,
    "H200": 4.54, "B200": 6.25,
}

# Candidate GPUs per tier. Tier C is a small-model, dataloader-heavy sweep, so
# it is probed on cheap silicon; tier B needs VRAM for 12B weights; tier A is
# throughput-bound and only makes sense on the fast parts.
CANDIDATES = {
    "c": ["T4", "L4", "A10", "L40S", "A100-40GB"],
    "b": ["A10", "L40S", "A100-80GB", "H100"],
    "a": ["A100-80GB", "H100", "H200", "B200"],
}


def _sync():
    import torch
    torch.cuda.synchronize()


def _probe_header(tag):
    import torch
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n=== {tag} on {name} ({total:.0f} GB) ===", flush=True)
    return total


# ------------------------------------------------------------------ tier C --
def _resnet(width_mult, num_classes=100):
    """ResNet-20 with a width multiplier; the tier C p* sweep varies only this."""
    import torch.nn as nn

    w = [int(16 * width_mult), int(32 * width_mult), int(64 * width_mult)]

    class Block(nn.Module):
        def __init__(self, cin, cout, stride):
            super().__init__()
            self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
            self.b1 = nn.BatchNorm2d(cout, eps=0.125)
            self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
            self.b2 = nn.BatchNorm2d(cout, eps=0.125)
            self.sc = (nn.Sequential() if stride == 1 and cin == cout else
                       nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                     nn.BatchNorm2d(cout, eps=0.125)))
            self.r = nn.ReLU(inplace=True)

        def forward(self, x):
            o = self.r(self.b1(self.c1(x)))
            return self.r(self.b2(self.c2(o)) + self.sc(x))

    layers = [nn.Conv2d(3, w[0], 3, 1, 1, bias=False),
              nn.BatchNorm2d(w[0], eps=0.125), nn.ReLU(inplace=True)]
    cin = w[0]
    for stage, cout in enumerate(w):
        for blk in range(3):
            layers.append(Block(cin, cout, 2 if (stage and not blk) else 1))
            cin = cout
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(*layers)
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.head = nn.Linear(cin, num_classes)

        def forward(self, x):
            return self.head(self.pool(self.features(x)))

    return Net()


def _impl_c(width_mult: float = 4.0, batch: int = 128, steps: int = 40):
    """Tier C: widest ResNet-20 variant, CIFAR-100 shapes, SF4 QAT."""
    import sys, time
    import torch
    sys.path.insert(0, "/root/sfx_bench")
    from superfloat import disable_tf32, apply_superfloat, clamp_all

    disable_tf32()
    total = _probe_header(f"tier C  ResNet-20 w x{width_mult}  bs={batch}")
    dev = "cuda"
    model = _resnet(width_mult).to(dev)
    # weights-only, matching the scaling-law scope; the final Linear is the head
    nconv = apply_superfloat(model, bits=4, head_names=("head",),
                             quantize_activations=False)
    n = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lossf = torch.nn.CrossEntropyLoss()

    x = torch.randn(batch, 3, 32, 32, device=dev)
    y = torch.randint(0, 100, (batch,), device=dev)

    for _ in range(8):                       # warmup
        opt.zero_grad(set_to_none=True)
        lossf(model(x), y).backward()
        opt.step()
        clamp_all(model)
    _sync()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        lossf(model(x), y).backward()
        opt.step()
        clamp_all(model)
    _sync()
    dt = time.time() - t0

    ips = steps * batch / dt
    peak = torch.cuda.max_memory_allocated() / 1e9
    # 50k CIFAR images/epoch, 100 epochs per run
    epoch_s = 50000 / ips
    print(f"params={n/1e6:.2f}M  sf_layers={nconv}  "
          f"peak_vram={peak:.2f} GB  {ips:.0f} img/s  "
          f"epoch={epoch_s:.1f}s  100ep={epoch_s*100/60:.1f} min", flush=True)
    return {"tier": "c", "params_m": n / 1e6, "peak_gb": peak,
            "throughput": ips, "unit": "img/s", "total_gb": total,
            "run_minutes": epoch_s * 100 / 60}


# ------------------------------------------------------------------ tier B --
def _impl_b(model_id: str = "EleutherAI/pythia-410m", batch: int = 8,
            seqlen: int = 2048, steps: int = 12):
    """Tier B: load a Pythia checkpoint, SF-quantize weights, time eval."""
    import sys, time
    import torch
    sys.path.insert(0, "/root/sfx_bench")
    from superfloat import disable_tf32, sf_quantize

    disable_tf32()
    total = _probe_header(f"tier B  {model_id}  bs={batch} seq={seqlen}")
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, cache_dir="/vol/hf").cuda().eval()
    load_s = time.time() - t0
    n = sum(p.numel() for p in model.parameters())

    t0 = time.time()                          # weights-only PTQ, SF6
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, torch.nn.Linear):
                w = mod.weight.data.float()
                mod.weight.data = sf_quantize(w, 6).to(mod.weight.dtype)
    _sync()
    quant_s = time.time() - t0

    x = torch.randint(0, 1000, (batch, seqlen), device="cuda")
    with torch.no_grad():
        for _ in range(3):
            model(x)
    _sync()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(steps):
            model(x)
    _sync()
    dt = time.time() - t0

    tps = steps * batch * seqlen / dt
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"params={n/1e9:.2f}B  load={load_s:.0f}s  quant={quant_s:.0f}s  "
          f"peak_vram={peak:.1f} GB  {tps/1e3:.0f}k tok/s", flush=True)
    return {"tier": "b", "params_b": n / 1e9, "peak_gb": peak,
            "throughput": tps, "unit": "tok/s", "total_gb": total,
            "load_s": load_s, "quant_s": quant_s}


# ------------------------------------------------------------------ tier A --
def _impl_a(d_model: int = 768, n_layer: int = 12, batch: int = 16,
            seqlen: int = 1024, steps: int = 30):
    """Tier A: from-scratch GPT block throughput under SF6 QAT."""
    import sys, time
    import torch
    import torch.nn as nn
    sys.path.insert(0, "/root/sfx_bench")
    from superfloat import disable_tf32, apply_superfloat, clamp_all

    disable_tf32()
    total = _probe_header(
        f"tier A  GPT d={d_model} L={n_layer}  bs={batch} seq={seqlen}")

    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, d // 64, batch_first=True)
            self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                     nn.Linear(4 * d, d))

        def forward(self, x):
            h = self.ln1(x)
            x = x + self.attn(h, h, h, need_weights=False)[0]
            return x + self.mlp(self.ln2(x))

    model = nn.Sequential(*[Block(d_model) for _ in range(n_layer)]).cuda()
    # quantize the Linear layers the way the real trainer will
    apply_superfloat(model, bits=6, quantize_activations=False)
    n = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randn(batch, seqlen, d_model, device="cuda")

    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        opt.step()
        clamp_all(model)
    _sync()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        opt.step()
        clamp_all(model)
    _sync()
    dt = time.time() - t0

    tps = steps * batch * seqlen / dt
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"blocks={n/1e6:.0f}M  peak_vram={peak:.1f} GB  {tps/1e3:.1f}k tok/s",
          flush=True)
    return {"tier": "a", "params_m": n / 1e6, "peak_gb": peak,
            "throughput": tps, "unit": "tok/s", "total_gb": total}




# --------------------------------------------------------------- registry ---
# One Modal function per (tier, GPU). Each needs its own wrapper object with a
# distinct __name__, since the decorator registers under that name.
_IMPL = {"a": _impl_a, "b": _impl_b, "c": _impl_c}
_TIMEOUT = {"a": 60 * 30, "b": 60 * 40, "c": 60 * 20}
PROBES = {}


def _register(tier, gpu):
    impl = _IMPL[tier]
    name = f"probe_{tier}_{gpu.replace('-', '_').replace('.', '_').lower()}"

    def inner(**kw):
        return impl(**kw)

    # Modal resolves a non-serialized function by re-importing this module and
    # looking it up by __qualname__, so the wrapper has to be a real
    # module-scope attribute under that exact name.
    inner.__name__ = name
    inner.__qualname__ = name
    globals()[name] = inner
    fn = app.function(image=image, gpu=gpu, volumes={"/vol": vol},
                      timeout=_TIMEOUT[tier])(inner)
    PROBES[(tier, gpu)] = fn


for _t, _gpus in CANDIDATES.items():
    for _g in _gpus:
        _register(_t, _g)


# ------------------------------------------------------------------- main ---
@app.local_entrypoint()
def main(tier: str = "abc"):
    """Fan each tier's probe across its candidate GPUs, then rank by value."""
    import json

    results = {}
    for t in tier:
        handles = []
        for gpu in CANDIDATES[t]:
            try:
                handles.append((gpu, PROBES[(t, gpu)].spawn()))
            except Exception as exc:                     # noqa: BLE001
                print(f"  spawn failed {t}/{gpu}: {str(exc)[:120]}", flush=True)
        rows = []
        for gpu, h in handles:
            try:
                r = h.get()
                r["gpu"], r["price"] = gpu, PRICE[gpu]
                r["cost_per_unit"] = PRICE[gpu] / 3600.0 / r["throughput"]
                rows.append(r)
                print(f"  ok {t}/{gpu}", flush=True)
            except Exception as exc:                     # noqa: BLE001
                print(f"  {t}/{gpu} FAILED: {str(exc)[:200]}", flush=True)
        results[t] = sorted(rows, key=lambda r: r["cost_per_unit"])

    print("\n" + "=" * 78)
    for t, rows in results.items():
        print(f"\nTIER {t.upper()}  (cheapest per unit of work first)")
        if not rows:
            print("  no successful probes")
            continue
        u = rows[0]["unit"]
        print(f"  {'gpu':<12}{'$/hr':>7}{'peak GB':>9}{'total GB':>9}"
              f"{u:>13}{'rel cost':>10}")
        best = rows[0]["cost_per_unit"]
        for r in rows:
            print(f"  {r['gpu']:<12}{r['price']:>7.2f}{r['peak_gb']:>9.1f}"
                  f"{r['total_gb']:>9.0f}{r['throughput']:>13,.0f}"
                  f"{r['cost_per_unit']/best:>9.2f}x")
    print("\n" + "=" * 78)
    with open("/tmp/profile.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("wrote /tmp/profile.json")
