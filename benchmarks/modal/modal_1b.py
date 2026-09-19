"""Modal port of train_1b.py -- from-scratch Llama-3.2-1B QAT (PURE_SF.md /
TOOL_USE_QAT.md's single biggest open number: no archived 1B val, no matched
bf16 control). Previously "in flight on a home 4090" with no archived result;
this is an independent run, not a resume of that job.

train_1b.py itself is untouched -- it already reads --data/--out as plain
CLI args, checkpoints resumably (SIGINT/SIGTERM-safe, periodic + step-numbered
checkpoints), and shards the tokenized corpus so a killed prepare picks up
where it left off. This file only adds the GPU container, the persistent
volume, the gated-tokenizer HF secret, and a probe-before-you-spend step,
matching the discipline modal_profile.py established for the original tier
A-D ladder.

Run:
    modal run modal/modal_1b.py::probe                  # throughput, ~2 min
    modal run modal/modal_1b.py::prepare --tokens 8000000000
    modal run modal/modal_1b.py::launch                 # bf16 control + sf8 ln_all
"""
import pathlib

import modal

BENCH_DIR = str(pathlib.Path(__file__).resolve().parent.parent)

app = modal.App("superfloat-1b")
vol = modal.Volume.from_name("sfx-baselines", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("transformers", "datasets", "numpy", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/vol/hf"})
    .add_local_dir(BENCH_DIR, remote_path="/root/sfx_bench")
)

DATA = "/vol/sf1b_data"
OUT_ROOT = "/vol/sf1b_run"
GPU = "H100"


def _env():
    import os
    return {**os.environ, "PYTHONPATH": "/root/sfx_bench:/root/sfx_bench/lab"}


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, secrets=[hf_secret], timeout=20 * 60)
def probe(bits: int = 0, mode: str = "ln_all"):
    """Confirms gated-tokenizer access and measures real tok/s before any
    prepare/train spend -- do not guess a token budget, measure it."""
    import os, subprocess
    os.makedirs(DATA, exist_ok=True)
    args = ["python", "train_1b.py", "--data", DATA, "--out", "/tmp/sf1b_probe",
            "--bits", str(bits), "--mode", mode, "--probe", "--no-resume", "--no-compile"]
    print(f"[probe] tokenizer access check + throughput: {' '.join(args)}", flush=True)
    proc = subprocess.run(args, cwd="/root/sfx_bench/lab", env=_env())
    return {"returncode": proc.returncode}


@app.function(image=image, volumes={"/vol": vol}, secrets=[hf_secret],
              timeout=24 * 60 * 60)
def prepare(tokens: int = 8_000_000_000):
    # CPU-only (tokenizer + streaming dataset, no torch.cuda calls) -- no gpu=
    # here, both to avoid needlessly queueing on GPU availability and because
    # a prior attempt with gpu="A10G" sat with 0 tasks running for 5+ minutes.
    import os, subprocess
    os.makedirs(DATA, exist_ok=True)
    args = ["python", "train_1b.py", "--data", DATA, "--prepare", str(tokens)]
    print(f"[prepare] {' '.join(args)}", flush=True)
    proc = subprocess.run(args, cwd="/root/sfx_bench/lab", env=_env())
    vol.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"prepare exited {proc.returncode}")
    return {"returncode": proc.returncode}


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, secrets=[hf_secret],
              timeout=24 * 60 * 60)
def train_arm(bits: int, mode: str, tokens: int, seed: int = 0, wait_tokens: int = 50_000_000):
    import os, subprocess, threading
    stop = threading.Event()

    def periodic_commit():
        while not stop.wait(300):
            vol.commit()
    t = threading.Thread(target=periodic_commit, daemon=True)
    t.start()

    tag = ("bf16" if bits == 0 else f"sf{bits}_{mode}") + f"_s{seed}"
    out = f"{OUT_ROOT}/{tag}"
    os.makedirs(out, exist_ok=True)
    args = ["python", "train_1b.py", "--data", DATA, "--out", out,
            "--bits", str(bits), "--mode", mode, "--tokens", str(tokens),
            "--seed", str(seed), "--wait-tokens", str(wait_tokens)]
    print(f"[runner] {' '.join(args)}", flush=True)
    proc = subprocess.run(args, cwd="/root/sfx_bench/lab", env=_env())
    stop.set()
    vol.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"{tag} exited {proc.returncode}")
    return {"tag": tag, "returncode": proc.returncode}


@app.local_entrypoint()
def launch(tokens: int = 8_000_000_000, seed: int = 0):
    """bf16 control + SF8 ln_all arm, in parallel. Data must already be
    prepared (run ::prepare first, or launch handles a slow initial ramp
    since train_arm itself waits on wait_tokens before starting)."""
    print("spawning bf16 control + sf8_ln_all", flush=True)
    handles = [
        train_arm.spawn(bits=0, mode="ln_all", tokens=tokens, seed=seed),
        train_arm.spawn(bits=8, mode="ln_all", tokens=tokens, seed=seed),
    ]
    for h in handles:
        try:
            print(h.get(), flush=True)
        except Exception as exc:                          # noqa: BLE001
            print(f"  run failed: {str(exc)[:200]}", flush=True)
