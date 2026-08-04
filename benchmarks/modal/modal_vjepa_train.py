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
def train(fmt: str, classes: int = 25, epochs: int = 40, seed: int = 0,
          weights_only: bool = False, measure_acts: bool = False):
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
    if weights_only:
        cmd.append("--no-act-quant")
    if measure_acts:
        cmd.append("--measure-acts")
    print("RUN " + " ".join(cmd), flush=True)

    suffix = "_wonly" if weights_only else ""
    log = f"/vol/vjepa_{fmt}{suffix}_s{seed}.log"
    with open(log, "w") as fh:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                cwd="/root/sfx_bench")
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            if any(k in line for k in ("[superfloat]", "[acts]", "DONE", "features",
                                       "Error", "Traceback", "classes=")):
                print(line.rstrip(), flush=True)
        rc = proc.wait()
    vol.commit()

    import json
    meta = os.path.join(out, f"vjepa2_{fmt}{suffix}_s{seed}.json")
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


@app.function(image=image, gpu=GPU, volumes={"/vol": vol},
              timeout=60 * 60 * 20, max_containers=4)
def train_qat(fmt: str, classes: int = 25, epochs: int = 15, seed: int = 0,
              weights_only: bool = False):
    """End-to-end SFx QAT: the backbone is quantized and trained through."""
    import os
    import subprocess
    import sys

    out = "/vol/runs_vjepa_qat"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, PYTHONPATH="/root/sfx_bench",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    cmd = [sys.executable, "/root/sfx_bench/train_vjepa_qat.py",
           "--format", fmt, "--data", HF_CACHE, "--out", out,
           "--classes", str(classes), "--epochs", str(epochs),
           "--seed", str(seed), "--batch", "2", "--accum", "8"]
    if weights_only:
        cmd.append("--no-act-quant")
    print("RUN " + " ".join(cmd), flush=True)

    suffix = "_wonly" if weights_only else ""
    log = f"/vol/vjepaqat_{fmt}{suffix}_s{seed}.log"
    with open(log, "w") as fh:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                cwd="/root/sfx_bench")
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            if any(k in line for k in ("[superfloat]", "[data]", "DONE",
                                       "] ep ", "Error", "Traceback",
                                       "out of memory")):
                print(line.rstrip(), flush=True)
        rc = proc.wait()
    vol.commit()
    print(f"DONE-QAT {fmt} rc={rc}", flush=True)
    return {"format": fmt, "rc": rc}


@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 30, cpu=8)
def inspect_ids():
    """Check whether clip_id siblings share a source video -- if they do, a
    clip-level split leaks near-duplicate frames across train/val."""
    import os
    os.environ["HF_HOME"] = HF_CACHE
    import numpy as np
    from datasets import load_dataset
    ds = load_dataset("flwrlabs/ucf101", split="train")
    vid = np.asarray(ds["video_id"]); cid = np.asarray(ds["clip_id"])
    lab = np.asarray(ds["label"])
    print("unique video_id:", len(np.unique(vid)), flush=True)
    print("unique clip_id :", len(np.unique(cid)), flush=True)
    print("unique (vid,cid):", len(set(zip(vid.tolist(), cid.tolist()))), flush=True)
    print("classes:", len(np.unique(lab)), flush=True)
    # how many distinct clip_ids per video_id?
    from collections import defaultdict
    per = defaultdict(set)
    for v, c in zip(vid.tolist(), cid.tolist()):
        per[v].add(c)
    sizes = np.array([len(x) for x in per.values()])
    print(f"clips per video_id: mean={sizes.mean():.2f} max={sizes.max()}", flush=True)
    return {"videos": int(len(np.unique(vid))),
            "clips": int(len(set(zip(vid.tolist(), cid.tolist())))),
            "clips_per_video_mean": float(sizes.mean())}


@app.local_entrypoint()
def main():
    print("classes:", prepare_ucf101.remote())
    print("Now spawn against the DEPLOYED app (see module docstring).")
