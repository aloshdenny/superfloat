"""Generic Modal runner for benchmarks/lab/*.py scripts.

The lab scripts (exp3_11m.py, exp3_mixed.py, puresf_llm.py, ...) already read
their token/output/checkpoint paths from env vars (EXP3_OUT, EXP3_TOKENS,
PSF_OUT, PSF_TOK, ...) so they run unmodified on any host -- RunPod, a local
box, or here. This file adds no experiment logic; it just runs one of those
scripts as a subprocess inside a GPU container with the sfx-baselines volume
mounted, and commits the volume on a timer so a container death mid-run loses
at most one commit interval of progress instead of the whole run (the
volumeInGb: 0 mistake from the RunPod leg of this project is not repeated
here: Modal Volumes are persistent by construction, but writes are only
durable across container restarts once committed).

Run:
    modal run modal/modal_lab_runner.py::launch --job exp3_tpp40
    modal run modal/modal_lab_runner.py::launch --job puresf_renorm_sweep
    modal run modal/modal_lab_runner.py::launch --job mixed_stage3
"""

import os
import pathlib
import subprocess
import threading

import modal

BENCH_DIR = str(pathlib.Path(__file__).resolve().parent.parent)

app = modal.App("superfloat-lab-runner")
vol = modal.Volume.from_name("sfx-baselines", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("transformers", "datasets", "numpy", "hf_transfer", "huggingface_hub")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/vol/hf"})
    .add_local_dir(BENCH_DIR, remote_path="/root/sfx_bench")
)


@app.function(
    image=image,
    gpu="H100",
    volumes={"/vol": vol},
    timeout=6 * 60 * 60,
)
def run_lab_script(script: str, args: list, env: dict, commit_every: int = 300):
    stop = threading.Event()

    def periodic_commit():
        while not stop.wait(commit_every):
            vol.commit()

    t = threading.Thread(target=periodic_commit, daemon=True)
    t.start()

    # exp3_11m.py / exp3_mixed*.py only sys.path.insert their own dir (lab/),
    # not the repo root where superfloat.py actually lives -- they got away
    # with it on prior hosts because a stray copy of superfloat.py sat next
    # to them there. Rather than touch those (tested, archived-result-backed)
    # scripts, make every subprocess see both dirs via PYTHONPATH.
    full_env = {
        **os.environ,
        "PYTHONPATH": "/root/sfx_bench:/root/sfx_bench/lab",
        **env,
    }
    # *_OUT is where checkpoints and result json land. These scripts only
    # os.makedirs(OUT) right before the FINAL json.dump, not before the first
    # mid-run checkpoint save -- fine on a host where the dir already exists
    # from a prior run, fatal the first time a fresh subdir is used on a
    # fresh volume. Pre-create every *_OUT / *_TOK path here instead of
    # patching each tested script.
    for k, v in full_env.items():
        if k.endswith(("_OUT", "_TOK")) and v.startswith("/vol"):
            os.makedirs(v, exist_ok=True)
    tag = f"{script} {' '.join(args)}"
    print(f"[runner] launching: {tag}", flush=True)
    proc = subprocess.run(
        ["python", script, *args],
        cwd="/root/sfx_bench/lab",
        env=full_env,
    )
    stop.set()
    vol.commit()
    print(f"[runner] done ({proc.returncode}): {tag}", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{tag} exited {proc.returncode}")
    return {"script": script, "args": args, "returncode": proc.returncode}


def _exp3_env(out_subdir: str) -> dict:
    return {
        "EXP3_OUT": f"/vol/{out_subdir}",
        "EXP3_TOKENS": "/vol/fineweb_edu_tokens.bin",
    }


def _psf_env(out_subdir: str, tok_subdir: str) -> dict:
    return {
        "PSF_OUT": f"/vol/{out_subdir}",
        "PSF_TOK": f"/vol/{tok_subdir}",
    }


MODERN_ARMS = {
    "uniform6":   (6, 6, 6),
    "protect-C6": (5, 5, 8),
    "starve-C6":  (7, 7, 4),
    "protect-A6": (8, 5, 5),
    "uniform4":   (4, 4, 4),
    "protect-C4": (3, 3, 6),
    "starve-C4":  (5, 5, 2),
}
STAGE3_ARMS = {
    "uniform4":   (4, 4, 4),
    "protect-C4": (3, 3, 6),
    "starve-C4":  (5, 5, 2),
    "protect-A4": (6, 3, 3),
}

JOBS = {
    # exp3_11m tpp40 restart, clean, all 5 bit-widths in parallel.
    "exp3_tpp40": [
        ("exp3_11m.py", ["--size", "11m", "--tpp", "40", "--bits", str(b)],
         _exp3_env("runs_exp3_tpp40"))
        for b in (0, 2, 3, 4, 6)
    ],
    # puresf_llm: does tightening the residual-renorm interval close the
    # audit gap (max_resid=6.267 at renorm_every=4, needs <=1)? Cheapest
    # lever on the open boundedness question, no code changes needed.
    # tag = f"psf_{size}_{arm}_s{seed}" does NOT include renorm_every, so two
    # cells at the same size/arm/seed collide on {OUT}/{tag}.json AND on
    # {OUT}/{tag}.ckpt if they run concurrently -- give each renorm value its
    # own OUT subdir (r1 already ran clean into runs_puresf_renorm/ directly
    # before this was caught; don't reuse that path for r2).
    "puresf_renorm_sweep": [
        ("puresf_llm.py",
         ["--size", "11m", "--arm", "sf8_full", "--tokens", "200000000",
          "--renorm-every", str(r), "--seed", "0"],
         _psf_env(f"runs_puresf_renorm_r{r}", "psf_tokens"))
        for r in (2,)
    ],
    # mixed-alloc stage 2: same group allocation, RMSNorm/SwiGLU/GQA block
    # instead of GPT-2 style, two seeds, 11M/tpp10 -- checks stage 1's
    # protect-C win is not an artifact of the GPT-2 block shape.
    "mixed_stage2": (
        [("exp3_mixed_modern.py",
          ["--size", "11m", "--tpp", "10", "--bits", "0", "--seed", str(seed)],
          _exp3_env("runs_mixed_stage2"))
         for seed in (0, 1)]
        + [("exp3_mixed_modern.py",
            ["--size", "11m", "--tpp", "10", "--bits", "6",
             "--bits-a", str(ba), "--bits-b", str(bb), "--bits-c", str(bc),
             "--arm", arm, "--seed", str(seed)],
            _exp3_env("runs_mixed_stage2"))
           for seed in (0, 1)
           for arm, (ba, bb, bc) in MODERN_ARMS.items()]
    ),
    # mixed-alloc stage 3: 25M, GPT-2 style, 4-bit tier only ("does the best
    # allocation move with scale?" -- EXPERIMENT_PLAN_mixed.md stage 3).
    "mixed_stage3": (
        [("exp3_mixed.py",
          ["--size", "25m", "--tpp", "10", "--bits", "0"],
          _exp3_env("runs_mixed_stage3"))]
        + [("exp3_mixed.py",
            ["--size", "25m", "--tpp", "10", "--bits", "4",
             "--bits-a", str(ba), "--bits-b", str(bb), "--bits-c", str(bc),
             "--arm", arm],
            _exp3_env("runs_mixed_stage3"))
           for arm, (ba, bb, bc) in STAGE3_ARMS.items()]
    ),
}


@app.local_entrypoint()
def launch(job: str):
    if job not in JOBS:
        raise SystemExit(f"unknown job {job!r}; choices: {sorted(JOBS)}")
    calls = JOBS[job]
    if job == "puresf_renorm_sweep":
        # tokens.bin must exist under PSF_TOK before any arm can train;
        # idempotent (prepare() skips if the file is already the right size).
        # Non-fatal: seen a clean "corpus written" followed by a SIGABRT on
        # process teardown (native-extension cleanup, not our code) -- the
        # file is already on disk by then, so don't let that kill the sweep.
        print("preparing psf token corpus (blocking, shared by all arms)", flush=True)
        try:
            run_lab_script.remote(
                "puresf_llm.py", ["--prepare", "200000000"],
                _psf_env("runs_puresf_renorm", "psf_tokens"),
            )
        except Exception as exc:                          # noqa: BLE001
            print(f"  prepare step raised ({str(exc)[:200]}), "
                  f"continuing -- sweep runs will fail fast if the "
                  f"corpus is actually missing", flush=True)
    print(f"spawning {len(calls)} runs for job={job}", flush=True)
    handles = [run_lab_script.spawn(script, args, env) for script, args, env in calls]
    done = 0
    for h in handles:
        try:
            print(h.get(), flush=True)
        except Exception as exc:                          # noqa: BLE001
            print(f"  run failed: {str(exc)[:200]}", flush=True)
        done += 1
        print(f"  {done}/{len(handles)} complete", flush=True)
