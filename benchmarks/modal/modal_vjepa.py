"""SuperFloat on V-JEPA 2 -- a fourth architecture family.

Everything measured so far is supervised and either convolutional (ResNet,
ConvNeXt, YOLO) or an autoregressive LM (GPT-2/3). V-JEPA 2 is none of those:
self-supervised joint-embedding prediction, ViT backbone, video input,
LayerNorm rather than BatchNorm. If SuperFloat behaves the same way here, the
architecture-agnosticity claim is much better supported.

Stage 1 (this file, `analyze`) is inference-only and costs cents. It tests a
falsifiable prediction from the detection sweep:

    SF4's zero-threshold is 0.0625. Kaiming init on YOLOv8x-OBB gives mean
    |w| = 0.0099, so 99.98% of conv weights quantize to exactly zero and the
    network is dead before training. ViT initialises with trunc_normal(0.02),
    which is *smaller still*, so SF4 should annihilate a randomly-initialised
    V-JEPA even more completely -- while the pretrained checkpoint, whose
    weights have grown during training, should survive.

Stage 2 is SFx QAT fine-tuning on aerial video, added once the dataset lands.
"""

import pathlib

import modal

# The trainers live one level up, in benchmarks/.
BENCH_DIR = str(pathlib.Path(__file__).resolve().parent.parent)

app = modal.App("superfloat-vjepa")
vol = modal.Volume.from_name("sfx-baselines", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "curl", "unzip", "git")
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    # V-JEPA 2 needs a recent transformers release
    .pip_install("transformers>=4.53", "accelerate", "numpy", "pandas",
                 "huggingface_hub")
    .env({"HF_HOME": "/vol/hf"})
    .add_local_dir(BENCH_DIR, remote_path="/root/sfx_bench")
)

MODEL = "facebook/vjepa2-vitl-fpc64-256"


@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 40, cpu=4)
def analyze():
    """Weight-distribution and SFx representability study. CPU only."""
    import sys
    sys.path.insert(0, "/root/sfx_bench")
    import torch
    import torch.nn as nn
    from superfloat import sf_params
    from transformers import AutoModel, AutoConfig

    def stats(model, tag):
        mods = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d,
                                                             nn.Conv3d))]
        w = torch.cat([m.weight.detach().float().flatten() for m in mods])
        print(f"\n=== {tag} ===", flush=True)
        print(f"  {len(mods)} Linear/Conv layers, {w.numel()/1e6:.1f}M weights")
        print(f"  |w| mean={w.abs().mean():.6f} std={w.std():.6f} "
              f"max={w.abs().max():.4f}")
        for bits in (16, 8, 4):
            scale, vmax = sf_params(bits)
            out = (w.abs() > vmax).float().mean().item() * 100
            q = torch.round(torch.clamp(w, -vmax, vmax) * scale) / scale
            zero = (q == 0).float().mean().item() * 100
            print(f"  SF{bits:<2d} step={1/scale:.6f} "
                  f"| outside +/-{vmax:.4f}: {out:7.4f}% "
                  f"| quantized to zero: {zero:6.2f}%")
        return w

    print(f"loading {MODEL} (pretrained)...", flush=True)
    pre = AutoModel.from_pretrained(MODEL)
    stats(pre, "V-JEPA 2 ViT-L  PRETRAINED")

    print("\nbuilding randomly-initialised V-JEPA 2 (same config)...", flush=True)
    cfg = AutoConfig.from_pretrained(MODEL)
    rnd = AutoModel.from_config(cfg)
    stats(rnd, "V-JEPA 2 ViT-L  RANDOM INIT")

    print("\n=== prediction check ===")
    print("  From the detection sweep: SF4 kills a network when standard init")
    print("  puts mean |w| below SF4's 0.0625 zero-threshold.")
    print("  YOLOv8x-OBB random init: mean|w|=0.0099 -> 99.98% zeroed -> dead.")
    print("  If V-JEPA random init shows the same, the law generalises across")
    print("  architecture families; if pretrained survives, SFx is confirmed")
    print("  as a deployment-phase format rather than a training format.")
    vol.commit()


@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 30, cpu=4)
def check_surgery():
    """Confirm the SFx layer surgery applies cleanly to a ViT/video model."""
    import sys
    sys.path.insert(0, "/root/sfx_bench")
    import torch
    import torch.nn as nn
    from superfloat import SFLinear, SFConv2d, apply_superfloat
    from transformers import AutoModel

    m = AutoModel.from_pretrained(MODEL)
    tot_lin = sum(1 for x in m.modules() if isinstance(x, nn.Linear))
    n = apply_superfloat(m, 8)
    q_lin = sum(1 for x in m.modules() if isinstance(x, SFLinear))
    q_cv = sum(1 for x in m.modules() if isinstance(x, SFConv2d))
    ln = sum(1 for x in m.modules() if isinstance(x, nn.LayerNorm))
    bn = sum(1 for x in m.modules() if isinstance(x, (nn.BatchNorm1d,
                                                      nn.BatchNorm2d)))
    print(f"converted={n}  SFLinear={q_lin}/{tot_lin}  SFConv2d={q_cv}")
    print(f"LayerNorm={ln}  BatchNorm={bn}   "
          f"(BN-specific eps/clamp logic is inert here, as expected for ViT)")

    # A forward pass proves the rebound classes actually execute.
    px = torch.randn(1, 16, 3, 256, 256)
    with torch.no_grad():
        out = m(pixel_values_videos=px)
    shape = tuple(out.last_hidden_state.shape)
    print(f"forward OK, last_hidden_state={shape}")
    return {"converted": n, "sf_linear": q_lin, "layernorm": ln}
