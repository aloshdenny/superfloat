"""EuroSAT FP32 / FP16 baseline rows for the SuperFloat classification table.

The SF16/SF8/SF4 EuroSAT results were produced on an RTX 4090 with 3 seeds
each; without a full-precision reference their quantization tax cannot be
stated the way the detection rows' can. This fills that gap with the identical
recipe -- ConvNeXt-Tiny from scratch, 64px, batch 128, AdamW lr 4e-3 wd 0.05,
10-epoch warmup then ReduceLROnPlateau, early stop on patience 40 -- changing
only the numeric format.

Separate app from superfloat-sweep so redeploying cannot disturb runs still
in flight there.
"""

import pathlib

import modal

# The trainers live one level up, in benchmarks/.
BENCH_DIR = str(pathlib.Path(__file__).resolve().parent.parent)

app = modal.App("superfloat-eurosat")
vol = modal.Volume.from_name("sfx-baselines", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxext6",
                 "curl", "unzip")  # not present in debian_slim
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("timm", "numpy", "pandas")
    .add_local_dir(BENCH_DIR, remote_path="/root/sfx_bench")
)

GPU = "H100"
DATA = "/vol/eurosat/EuroSAT_RGB"


@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 30)
def prepare_eurosat():
    """Fetch EuroSAT RGB (~94 MB) into the shared volume once."""
    import os
    import subprocess
    if os.path.isdir(DATA):
        n = sum(len(f) for _, _, f in os.walk(DATA))
        print(f"already present: {n} files", flush=True)
        return n
    os.makedirs("/vol/eurosat", exist_ok=True)
    subprocess.run(
        "curl -sL -o /tmp/E.zip https://zenodo.org/records/7711810/files/"
        "EuroSAT_RGB.zip && unzip -q -o /tmp/E.zip -d /vol/eurosat",
        shell=True, check=True)
    vol.commit()
    n = sum(len(f) for _, _, f in os.walk(DATA))
    print(f"downloaded: {n} images", flush=True)
    return n


@app.function(image=image, gpu=GPU, volumes={"/vol": vol},
              timeout=60 * 60 * 6, max_containers=6)
def train_eurosat(fmt: str, seed: int):
    import os
    import subprocess
    import sys

    out = "/vol/runs_eurosat"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, PYTHONPATH="/root/sfx_bench")
    cmd = [sys.executable, "/root/sfx_bench/train_eurosat.py",
           "--format", fmt, "--seed", str(seed), "--data", DATA,
           "--out", out, "--batch-size", "128", "--workers", "8"]
    print("RUN " + " ".join(cmd), flush=True)

    log = f"/vol/eurosat_{fmt}_s{seed}.log"
    with open(log, "w") as fh:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                cwd="/root/sfx_bench")
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            if "DONE" in line or "ep " in line and " 0 " not in line:
                pass
        rc = proc.wait()
    vol.commit()

    import json
    meta = os.path.join(out, f"convnext_tiny_{fmt}_s{seed}.json")
    best = json.load(open(meta))["best_val_acc"] if os.path.exists(meta) else None
    print(f"DONE {fmt} seed{seed} rc={rc} best={best}", flush=True)
    return {"format": fmt, "seed": seed, "best": best, "rc": rc}


@app.local_entrypoint()
def main():
    print("preparing EuroSAT...")
    print("files:", prepare_eurosat.remote())
    jobs = [(f, s) for f in ("fp32", "fp16") for s in (0, 1, 2)]
    print(f"spawning {len(jobs)} EuroSAT baseline runs on {GPU}")
    for fmt, seed in jobs:
        h = train_eurosat.spawn(fmt, seed)
        print(f"  {fmt} seed{seed} -> {h.object_id}")
    print("\nrunning server-side; this process can exit.")
