"""SuperFloat detection sweep on Modal: 2 datasets x 5 formats x 2 inits.

    VisDrone  YOLO11x      imgsz 640  batch 8
    DOTAv1    YOLOv8x-OBB  imgsz 800  batch 8
    pretrained -> 150 epochs @ lr 1e-3
    random     -> 300 epochs @ lr 4e-3

Batch size is deliberately NOT raised to exploit a larger cloud GPU. Batch size
changes the effective training recipe, and a baseline trained at a different
batch than the SFx runs would not be a baseline -- the gap would be measuring
batch size, not precision.

TF32 stays disabled (superfloat.disable_tf32), so the fp32 row is genuinely
fp32 rather than TF32's 10-bit mantissa, which sits below SF16's 15 significand
bits. The fp16 row uses Ultralytics AMP.

Deploy, then spawn against the DEPLOYED app -- `modal run` creates an ephemeral
app that stops when the entrypoint returns, taking its spawned calls with it:

    modal deploy modal_sweep.py
    python -c "import modal; f=modal.Function.from_name('superfloat-sweep','train_baseline'); \
               [print(f.spawn(n).object_id) for n in ('visdrone_pretrained_sf16',)]"

Runs checkpoint to the volume and pass --resume, so a job cut off by Modal's
24h function cap continues rather than restarting.
"""

import pathlib

import modal

# The trainers live one level up, in benchmarks/.
BENCH_DIR = str(pathlib.Path(__file__).resolve().parent.parent)

app = modal.App("superfloat-sweep")
vol = modal.Volume.from_name("sfx-baselines", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxext6")
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("ultralytics==8.4.115", "timm", "pandas", "matplotlib", "pyyaml")
    .env({"YOLO_CONFIG_DIR": "/vol/ultralytics_cfg"})
    .add_local_dir(BENCH_DIR, remote_path="/root/sfx_bench")
)

GPU = "H100"
DATA_DIR = "/vol/datasets"

# name -> (cfg, data yaml, imgsz, batch, init, epochs, lr)
JOBS = {}
for _ds, _cfg, _yaml, _img in [
    ("visdrone", "yolo11x.yaml", "VisDrone.yaml", 640),
    ("dota", "yolov8x-obb.yaml", "DOTAv1.yaml", 800),
]:
    for _init, _ep, _lr in [("pretrained", 150, 1e-3), ("random", 300, 4e-3)]:
        for _fmt in ("fp32", "fp16", "sf16", "sf8", "sf4"):
            JOBS[f"{_ds}_{_init}_{_fmt}"] = (_cfg, _yaml, _img, 8, _init, _ep,
                                             _lr, _fmt, 0)

# Extra seeds for visdrone_random_sf8, which collapsed on seed 0: mAP peaked at
# epoch 18 then decayed while train box_loss climbed 1.98 -> 2.22, and early
# stopping ended it at 0.0724. SF16 under the identical config trained cleanly
# to 0.27, and SF8 from pretrained init reached 0.28, so this is the
# seed-dependent saturation collapse the paper reports as mean +/- std rather
# than a property of SF8. One seed alone would misreport it as a flat failure.
for _s in (1, 2):
    JOBS[f"visdrone_random_sf8_s{_s}"] = (
        "yolo11x.yaml", "VisDrone.yaml", 640, 8, "random", 300, 4e-3, "sf8", _s)

# Retry of the collapsed SF8 random run at a lower learning rate. All three
# seeds at lr 4e-3 diverged with *rising* train loss (1.94->2.22, 2.00->2.11,
# 1.97->2.11), which points at the step size rather than the seed: SF16
# tolerates 4e-3, but SF8's grid is 8x coarser and the update overshoots it.
# 1e-3 is the same rate the pretrained runs use successfully.
JOBS["visdrone_random_sf8_lr1e3"] = (
    "yolo11x.yaml", "VisDrone.yaml", 640, 8, "random", 300, 1e-3, "sf8", 0)

# Does the same learning-rate fix rescue SF4? dota_random_sf4 is the hard
# collapse -- 0.0000 for 263 epochs at lr 4e-3, never learning anything.
# If usable step size scales with grid resolution then SF4, whose grid is 16x
# coarser than SF8's, should need roughly 4x less than SF8's working 1e-3.
# Running both rates tests that rule rather than just trying one number.
for _tag, _lr in (("lr1e3", 1e-3), ("lr2p5e4", 2.5e-4)):
    JOBS[f"dota_random_sf4_{_tag}"] = (
        "yolov8x-obb.yaml", "DOTAv1.yaml", 800, 8, "random", 300, _lr, "sf4", 0)

# The real cause of the SF4 random-init failure: Kaiming init gives mean |w| =
# 0.0099 on this model, under SF4's 0.0625 zero-threshold, so 99.98% of conv
# weights quantize to exactly zero and no learning rate can help a dead
# network. Scaling conv weights 25x at init leaves 13.5% zeroed and 0.4%
# saturated. Safe because every conv is followed by BatchNorm, which
# renormalizes activations regardless of weight scale. lr 1e-3 is the rate
# that rescued SF8 from a comparable zero-fraction.
JOBS["dota_random_sf4_scaled"] = (
    "yolov8x-obb.yaml", "DOTAv1.yaml", 800, 8, "random", 300, 1e-3, "sf4", 0, 25.0)

# SF4 from scratch, standard Kaiming init, no rescaling -- only the optimizer
# changes. AdamW's decoupled weight decay (0.05) shrinks every weight toward
# zero each step, which is exactly wrong when the format's representable floor
# is 0.0625: it suppresses the ~14k of 69.5M weights that start above the floor
# and might otherwise grow and recruit others. wd=0 removes that pressure.
JOBS["dota_random_sf4_wd0"] = (
    "yolov8x-obb.yaml", "DOTAv1.yaml", 800, 8, "random", 300, 1e-3, "sf4", 0, 1.0, 0.0)


def _configure_datasets_dir():
    """Point Ultralytics at the volume, persistently.

    Ultralytics resolves DATASETS_DIR from SETTINGS at *import* time, so
    calling SETTINGS.update() in an already-imported process does not
    redirect downloads -- the first attempt at this silently wrote 3.5 GB to
    the container's ephemeral disk instead of the volume. The update is still
    what writes the setting to disk; it only takes effect in a process that
    imports Ultralytics afterwards. YOLO_CONFIG_DIR lives on the volume, so
    the value survives for every later container.
    """
    from ultralytics.utils import SETTINGS
    SETTINGS.update({"datasets_dir": DATA_DIR})
    vol.commit()


@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 60 * 4)
def prepare_data():
    """Download VisDrone and DOTAv1 into the shared volume, once."""
    import os
    import subprocess
    import sys

    os.makedirs(DATA_DIR, exist_ok=True)
    _configure_datasets_dir()

    # Fresh interpreter, so Ultralytics imports with datasets_dir already set.
    code = (
        "from ultralytics.utils import SETTINGS;"
        f"assert SETTINGS['datasets_dir']=='{DATA_DIR}', SETTINGS['datasets_dir'];"
        "from ultralytics.utils import DATASETS_DIR;"
        f"assert str(DATASETS_DIR)=='{DATA_DIR}', DATASETS_DIR;"
        "from ultralytics.data.utils import check_det_dataset;"
        "[print('READY', y, check_det_dataset(y)['path'], flush=True)"
        " for y in ('VisDrone.yaml','DOTAv1.yaml')]"
    )
    subprocess.run([sys.executable, "-c", code], check=True)

    vol.commit()
    listing = sorted(os.listdir(DATA_DIR))
    for d in ("VisDrone", "DOTAv1"):
        p = os.path.join(DATA_DIR, d, "images")
        if os.path.isdir(p):
            for split in sorted(os.listdir(p)):
                n = len(os.listdir(os.path.join(p, split)))
                print(f"  {d}/{split}: {n} images", flush=True)
    return listing


@app.function(image=image, gpu=GPU, volumes={"/vol": vol},
              timeout=60 * 60 * 24, retries=modal.Retries(max_retries=2),
              max_containers=10)
def train_baseline(name: str):
    import os
    import subprocess
    import sys

    _v = JOBS[name]
    cfg, data, imgsz, batch, init, epochs, lr, fmt, seed = _v[:9]
    init_scale = _v[9] if len(_v) > 9 else 1.0
    wd = _v[10] if len(_v) > 10 else 0.05
    _configure_datasets_dir()

    out = "/vol/runs"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, PYTHONPATH="/root/sfx_bench",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

    cmd = [
        sys.executable, "/root/sfx_bench/train_yolo.py",
        "--format", fmt, "--cfg", cfg, "--data", data, "--init", init,
        "--imgsz", str(imgsz), "--batch", str(batch), "--workers", "8",
        "--epochs", str(epochs), "--patience", "50", "--lr", str(lr),
        "--seed", str(seed), "--init-scale", str(init_scale), "--wd", str(wd),
        "--name", name, "--project", out,
        # Modal caps a function at 24h and can retry; the run dir lives on the
        # volume, so a relaunch picks up where the last one stopped instead of
        # restarting a 300-epoch job from zero.
        "--resume",
    ]
    print("RUN " + " ".join(cmd), flush=True)

    log_path = f"/vol/logs_{name}.log"
    os.makedirs("/vol", exist_ok=True)
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                bufsize=1, cwd="/root/sfx_bench")
        for i, line in enumerate(proc.stdout):
            log.write(line)
            # Flush every line: buffering lost the entire log of every job the
            # first time containers were killed, leaving nothing to diagnose.
            log.flush()
            if i % 200 == 0:
                vol.commit()
            # keep Modal's own log readable
            if any(k in line for k in ("[init]", "[superfloat]", "epochs completed",
                                       "Error", "Traceback", "out of memory")):
                print(line.rstrip(), flush=True)
        rc = proc.wait()

    vol.commit()

    # Report the best mAP50-95 so the driver can print a table.
    best = None
    import csv
    for root, _, files in os.walk(out):
        if "results.csv" in files and os.path.basename(root) == name:
            with open(os.path.join(root, "results.csv")) as f:
                rows = list(csv.DictReader(f))
            col = next((c for c in rows[0] if "mAP50-95" in c), None) if rows else None
            if col:
                best = max(float(r[col]) for r in rows)
    print(f"DONE {name} rc={rc} best_mAP50-95={best}", flush=True)
    return {"name": name, "rc": rc, "best_map50_95": best}


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=60 * 40)
def smoke(name: str):
    """One short epoch on 5% of the data: proves the GPU path and, for fp16,
    that AMP is actually engaged before committing to the full grid."""
    import os
    import subprocess
    import sys

    cfg, data, imgsz, batch, init, _ep, lr, fmt = JOBS[name]
    env = dict(os.environ, PYTHONPATH="/root/sfx_bench",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    cmd = [sys.executable, "/root/sfx_bench/train_yolo.py",
           "--format", fmt, "--cfg", cfg, "--data", data, "--init", init,
           "--imgsz", str(imgsz), "--batch", str(batch), "--workers", "8",
           "--epochs", "1", "--fraction", "0.05", "--lr", str(lr),
           "--name", f"smoke_{name}", "--project", "/tmp/smoke"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                       cwd="/root/sfx_bench")
    out = r.stdout + r.stderr
    import torch
    keep = [l for l in out.splitlines()
            if any(k in l for k in ("[init]", "[superfloat]", "AMP", "amp=",
                                    "Error", "Traceback", "epochs completed"))]
    return {"name": name, "rc": r.returncode, "gpu": torch.cuda.get_device_name(0),
            "lines": keep[-12:]}


@app.local_entrypoint()
def smoke_test():
    names = ["visdrone_pretrained_fp32", "visdrone_pretrained_fp16",
             "dota_pretrained_fp16"]
    for res in smoke.map(names):
        print(f"\n=== {res['name']} on {res['gpu']} rc={res['rc']} ===")
        for l in res["lines"]:
            print("   ", l)


@app.local_entrypoint()
def main(only: str = ""):
    print("preparing datasets...")
    print("dataset dir:", prepare_data.remote())

    names = [only] if only else list(JOBS)
    bad = [n for n in names if n not in JOBS]
    if bad:
        raise SystemExit(f"unknown job(s): {bad}\navailable: {list(JOBS)}")

    # spawn(), not map(): map is driven from the client, so if the local
    # process goes away mid-sweep every queued input is cancelled -- which is
    # exactly what killed the first attempt (all 20 inputs cancelled at once,
    # no job had actually failed). spawn hands each job to Modal and returns,
    # so the sweep is independent of this process surviving.
    print(f"spawning {len(names)} runs on {GPU} (max 10 concurrent):")
    handles = []
    for n in names:
        handles.append((n, train_baseline.spawn(n)))
        print(f"    {n:34s} -> {handles[-1][1].object_id}")

    ids = {n: h.object_id for n, h in handles}
    import json
    print("\nSPAWNED_CALL_IDS " + json.dumps(ids))
    print("\nJobs are running server-side; this process can exit safely.")
    print("Track with: modal app list  /  modal volume ls sfx-baselines runs")
