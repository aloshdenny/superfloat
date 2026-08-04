"""SuperFloat on V-JEPA 2 — attentive-probe benchmark on UCF101, on Modal.

Fourth architecture family: ViT backbone, self-supervised joint-embedding
objective, video input, LayerNorm. The backbone is frozen and quantized; only
an fp32 attentive probe is trained on its features, which is V-JEPA's own
evaluation protocol and matches SuperFloat's scoping as a deployment-time
format.

    modal deploy modal_vjepa_train.py
    python -c "import modal; f=modal.Function.from_name('superfloat-vjepa-train','train'); \\
               [print(f.spawn(x).object_id) for x in ('fp32','sf16','sf8','sf4')]"
"""

import pathlib

import modal

BENCH_DIR = str(pathlib.Path(__file__).resolve().parent.parent)

app = modal.App("superfloat-vjepa-train")
vol = modal.Volume.from_name("sfx-baselines", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    # ca-certificates: debian_slim ships without a CA bundle, so HTTPS
    # downloads fail with curl error 60 (cert verification).
    .apt_install("libgl1", "libglib2.0-0", "curl", "unzip", "unrar-free",
                 "ffmpeg", "git", "ca-certificates", "p7zip-full")
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("transformers>=4.53", "accelerate", "numpy", "pandas",
                 "huggingface_hub", "datasets", "av")
    .env({"HF_HOME": "/vol/hf"})
    .add_local_dir(BENCH_DIR, remote_path="/root/sfx_bench")
)

GPU = "H100"
HF_CACHE = "/vol/hf"
FORMATS = ("fp32", "sf16", "sf8", "sf4")


@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 90, cpu=8)
def prepare_ucf101():
    """Materialise UCF101 into the volume-backed HF cache, once.

    Uses the flwrlabs/ucf101 mirror; crcv.ucf.edu serves an incomplete TLS
    chain that fails verification in-container.
    """
    import os
    os.environ["HF_HOME"] = HF_CACHE
    from datasets import load_dataset
    ds = load_dataset("flwrlabs/ucf101", split="train")
    cols = list(ds[0])
    n_classes = len(set(ds["label"])) if "label" in cols else -1
    print(f"rows={len(ds)} columns={cols} classes={n_classes}", flush=True)
    vol.commit()
    return {"rows": len(ds), "columns": cols, "classes": n_classes}


@app.function(image=image, gpu=GPU, volumes={"/vol": vol},
              timeout=60 * 60 * 12, max_containers=4)
def train(fmt: str, classes: int = 25, epochs: int = 40, seed: int = 0):
    import os
    import subprocess
    import sys

    out = "/vol/runs_vjepa"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, PYTHONPATH="/root/sfx_bench",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    cmd = [sys.executable, "/root/sfx_bench/train_vjepa_probe.py",
           "--format", fmt, "--data", HF_CACHE, "--out", out,
           "--classes", str(classes), "--epochs", str(epochs),
           "--seed", str(seed), "--batch", "4"]
    print("RUN " + " ".join(cmd), flush=True)

    log = f"/vol/vjepa_{fmt}_s{seed}.log"
    with open(log, "w") as fh:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                cwd="/root/sfx_bench")
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            if any(k in line for k in ("[superfloat]", "DONE", "features",
                                       "Error", "Traceback", "classes=")):
                print(line.rstrip(), flush=True)
        rc = proc.wait()
    vol.commit()

    import json
    meta = os.path.join(out, f"vjepa2_{fmt}_s{seed}.json")
    best = json.load(open(meta))["best_val_acc"] if os.path.exists(meta) else None
    print(f"DONE {fmt} rc={rc} best={best}", flush=True)
    return {"format": fmt, "best": best, "rc": rc}


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=60 * 45)
def smoke():
    """Prove the loader, the SFx surgery and a ViT forward pass all work
    before four full runs are launched."""
    import os
    import sys
    os.environ["HF_HOME"] = HF_CACHE
    sys.path.insert(0, "/root/sfx_bench")
    import torch, torch.nn as nn
    from superfloat import apply_superfloat, SFLinear, sf_params
    from video_data import build_ucf101_loaders
    from transformers import AutoModel

    tr, va, ncls = build_ucf101_loaders(HF_CACHE, 16, 256, 2, 5, 0, workers=4)
    clips, y = next(iter(tr))
    print(f"clip batch {tuple(clips.shape)} labels {y.tolist()}", flush=True)

    m = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
    tot_lin = sum(1 for x in m.modules() if isinstance(x, nn.Linear))
    n = apply_superfloat(m, 8)
    q = sum(1 for x in m.modules() if isinstance(x, SFLinear))
    print(f"SF8 surgery: converted={n} SFLinear={q}/{tot_lin}", flush=True)

    scale, vmax = sf_params(8)
    w = torch.cat([x.weight.detach().flatten() for x in m.modules()
                   if isinstance(x, nn.Linear)])
    qz = torch.round(torch.clamp(w, -vmax, vmax) * scale) / scale
    print(f"mean|w|={w.abs().mean():.6f} zeroed={100*(qz==0).float().mean():.2f}%",
          flush=True)

    m = m.cuda().eval()
    with torch.no_grad():
        out = m(pixel_values_videos=clips.cuda()).last_hidden_state
    print(f"forward OK: {tuple(out.shape)}", flush=True)
    return {"tokens": tuple(out.shape), "classes": ncls}


@app.local_entrypoint()
def main():
    print("classes:", prepare_ucf101.remote())
    print("Now spawn against the DEPLOYED app (see module docstring).")
