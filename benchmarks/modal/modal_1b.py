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
    modal deploy modal/modal_1b.py; modal run modal/modal_1b.py::launch   # after prepare is DONE
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
def train_arm(bits: int, mode: str, tokens: int, seed: int = 0, chain: int = 0,
              max_chain: int = 14):
    """One <=23h leg of a multi-day run. Resumes from ckpt/latest.pt, and if
    the wall-clock guard stops it before the token budget is reached, spawns
    the next leg itself, so a 6-day run survives Modal's 24h function ceiling
    without any local process babysitting it.

    Two things the first attempt at this got wrong, both fatal:
      * no vol.reload(): a container only sees another container's committed
        files after reloading the volume, so a training container that
        started while prepare was still writing shards trained on shard 0
        alone for 24h (~30 epochs over 96M tokens -> train loss 0.95, val
        loss 5.6: memorisation, not pretraining).
      * timeout=24h with no resume chain: the run simply stopped at 24.0h.
    """
    import os, subprocess, sys, threading
    vol.reload()

    # refuse to start on a partial corpus, ever again
    sys.path.insert(0, "/root/sfx_bench/lab"); sys.path.insert(0, "/root/sfx_bench")
    from train_1b import tokens_ready
    have = tokens_ready(DATA)
    if have < tokens:
        raise RuntimeError(
            f"corpus incomplete: {have/1e9:.2f}B tokens ready, run needs {tokens/1e9:.2f}B; "
            f"finish ::prepare before launching train_arm")

    stop = threading.Event()

    def periodic_commit():
        while not stop.wait(300):
            vol.commit()
    threading.Thread(target=periodic_commit, daemon=True).start()

    tag = ("bf16" if bits == 0 else f"sf{bits}_{mode}") + f"_s{seed}"
    out = f"{OUT_ROOT}/{tag}"
    os.makedirs(out, exist_ok=True)
    args = ["timeout", "-s", "TERM", "23h",
            "python", "train_1b.py", "--data", DATA, "--out", out,
            "--bits", str(bits), "--mode", mode, "--tokens", str(tokens),
            "--seed", str(seed), "--wait-tokens", str(tokens)]
    print(f"[runner] leg {chain}: {' '.join(args)}", flush=True)
    proc = subprocess.run(args, cwd="/root/sfx_bench/lab", env=_env())
    stop.set()
    vol.commit()

    if proc.returncode == 124:
        # wall-clock guard fired: train_1b.py caught SIGTERM, checkpointed,
        # and exited; this leg is done but the run is not.
        if chain + 1 >= max_chain:
            raise RuntimeError(f"{tag}: hit max_chain={max_chain} legs without finishing")
        nxt = train_arm.spawn(bits=bits, mode=mode, tokens=tokens, seed=seed,
                              chain=chain + 1, max_chain=max_chain)
        print(f"[runner] {tag} leg {chain} timed out cleanly; spawned leg {chain+1} "
              f"({nxt.object_id})", flush=True)
        return {"tag": tag, "leg": chain, "status": "continued", "next": nxt.object_id}
    if proc.returncode != 0:
        raise RuntimeError(f"{tag} leg {chain} exited {proc.returncode}")
    print(f"[runner] {tag} DONE after {chain+1} leg(s)", flush=True)
    return {"tag": tag, "leg": chain, "status": "done"}


@app.local_entrypoint()
def launch(tokens: int = 20_000_000_000, seed: int = 0):
    """Fire-and-forget: spawns the bf16 control and the SF8 ln_all arm and
    returns immediately. Each arm chains its own legs from there. Requires
    ::prepare to have finished for the full `tokens` budget first (train_arm
    refuses a partial corpus)."""
    for bits in (0, 8):
        h = train_arm.spawn(bits=bits, mode="ln_all", tokens=tokens, seed=seed)
        print(f"spawned bits={bits}: {h.object_id}", flush=True)
