"""Loss-based needle-in-a-haystack for the 1B arms, on a workspace that only
needs the checkpoints and one corpus shard (not the full 20B corpus).

Split out from modal_1b.py because the pretraining workspaces ran out of
credit; this runs anywhere a volume can hold ~11 GB.
"""
import pathlib

import modal

BENCH_DIR = str(pathlib.Path(__file__).resolve().parent.parent)

app = modal.App("superfloat-niah")
vol = modal.Volume.from_name("sfx-niah", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy")
    .add_local_dir(BENCH_DIR, remote_path="/root/sfx_bench")
)

DATA = "/vol/data"
CKPT = "/vol/ckpt"


def _env():
    import os
    return {**os.environ, "PYTHONPATH": "/root/sfx_bench:/root/sfx_bench/lab"}


@app.function(image=image, gpu="H100", volumes={"/vol": vol}, timeout=4 * 60 * 60)
def niah(ckpt_file: str, bits: int, ctx: int, samples: int = 8, seed: int = 0,
         mode: str = "ln_all", depths: str = "0.1,0.25,0.5,0.75,0.9"):
    import os, subprocess
    vol.reload()
    env = {**_env(), "NIAH_OUT": "/vol/results"}
    args = ["python", "niah_1b.py", "--ckpt", f"{CKPT}/{ckpt_file}", "--data", DATA,
            "--bits", str(bits), "--mode", mode, "--ctx", str(ctx),
            "--samples", str(samples), "--seed", str(seed), "--depths", depths]
    print(f"[niah] {' '.join(args)}", flush=True)
    proc = subprocess.run(args, cwd="/root/sfx_bench/lab", env=env)
    vol.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"niah exited {proc.returncode}")
    return {"ckpt": ckpt_file, "bits": bits, "ctx": ctx}
