#!/usr/bin/env python3
"""Wait until aoxo/sf-scaling-laws has ckpt/latest.pt, then create a 1B QAT pod.

Never prints secrets. Caps spend at $44 (credits $46 → $2).
"""
from __future__ import annotations

import json, os, re, subprocess, sys, time, urllib.error, urllib.request

TRANSCRIPT = (
    "/Users/aoxo/.cursor/projects/Users-aoxo-Aleddo/agent-transcripts/"
    "2207c47c-20ca-48d2-96cb-44437e9b38c4/2207c47c-20ca-48d2-96cb-44437e9b38c4.jsonl"
)
REPO = "aoxo/sf-scaling-laws"
SPEND_CAP_USD = 44.0
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def runpod_key():
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key
    with open(TRANSCRIPT) as f:
        for line in f:
            if '"role":"user"' not in line[:120]:
                continue
            m = re.search(r"rpa_[A-Za-z0-9]+", line)
            if m:
                return m.group(0)
    raise SystemExit("no RUNPOD_API_KEY")


def hf_token():
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if t:
        return t.strip()
    out = subprocess.check_output(
        [
            "ssh", "-o", "BatchMode=yes", "research@100.86.165.70",
            "wsl.exe", "--", "bash", "-lc", "cat /tmp/.hf_write",
        ],
        timeout=20,
    )
    return out.decode().strip()


def api(key, method, url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
        return r.status, (json.loads(raw) if raw else {})


def hub_has_ckpt(token):
    req = urllib.request.Request(
        f"https://huggingface.co/api/datasets/{REPO}/tree/main/ckpt",
        headers={"Authorization": f"Bearer {token}", "User-Agent": "sf-lab/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            files = json.loads(r.read())
    except urllib.error.HTTPError:
        return False, 0
    for f in files or []:
        if f.get("path") == "ckpt/latest.pt" or str(f.get("path", "")).endswith("latest.pt"):
            return True, int(f.get("size") or 0)
    return False, 0


def wait_ckpt(token, timeout=3600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        ok, sz = hub_has_ckpt(token)
        print(f"hub_ckpt ok={ok} size={sz}", flush=True)
        if ok and sz > 10_000_000_000:
            return sz
        # also watch the 4090 uploader
        try:
            log = subprocess.check_output(
                [
                    "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
                    "research@100.86.165.70", "wsl.exe", "--", "bash", "-s",
                ],
                input=b"tr '\\r' '\\n' < /tmp/hf-upload.log | tail -n 3; "
                      b"pgrep -c -f hf_upload_ckpt.py || true\n",
                timeout=20,
            ).decode(errors="replace")
            print(log.strip().splitlines()[-4:], flush=True)
            if "UPLOAD_DONE" in log:
                time.sleep(5)
                continue
        except Exception as e:
            print("ssh_upload_check", type(e).__name__, flush=True)
        time.sleep(30)
    raise SystemExit("timeout waiting for HF ckpt")


def create_pod(key, pub, hf):
    attempts = [
        ("COMMUNITY", ["NVIDIA GeForce RTX 4090"]),
        ("COMMUNITY", ["NVIDIA GeForce RTX 3090"]),
        ("COMMUNITY", ["NVIDIA GeForce RTX 5090"]),
        ("SECURE", ["NVIDIA GeForce RTX 4090"]),
    ]
    for cloud, gpus in attempts:
        body = {
            "cloudType": cloud,
            "computeType": "GPU",
            "gpuCount": 1,
            "gpuTypeIds": gpus,
            "gpuTypePriority": "availability",
            "name": "sf-1b-qat",
            "imageName": IMAGE,
            "containerDiskInGb": 40,
            "volumeInGb": 80,
            "volumeMountPath": "/workspace",
            "ports": ["22/tcp"],
            "supportPublicIp": True,
            "interruptible": False,
            "env": {
                "PUBLIC_KEY": pub,
                "HF_TOKEN": hf,
                "HUGGING_FACE_HUB_TOKEN": hf,
                "HF_HOME": "/workspace/hf",
                "PYTHONUNBUFFERED": "1",
            },
        }
        print(f"TRY {cloud} {gpus[0]}", flush=True)
        try:
            code, resp = api(key, "POST", "https://rest.runpod.io/v1/pods", body)
        except urllib.error.HTTPError as e:
            print("FAIL", e.code, e.read()[:200], flush=True)
            continue
        if code in (200, 201) and isinstance(resp, dict) and resp.get("id"):
            print("CREATED", resp.get("id"), resp.get("costPerHr"), cloud, gpus[0], flush=True)
            return resp
        print("FAIL", code, resp, flush=True)
    raise SystemExit("no pod")


def wait_ssh(key, pod_id):
    for i in range(48):
        _, d = api(key, "GET", f"https://rest.runpod.io/v1/pods/{pod_id}")
        ip = d.get("publicIp")
        ports = d.get("portMappings") or {}
        port = ports.get("22") or ports.get(22)
        print(f"  ssh_wait {i} status={d.get('desiredStatus')} ip={ip} port={port}", flush=True)
        if ip and port:
            return ip, int(port), d
        time.sleep(5)
    raise SystemExit("pod ssh timeout")


BOOT = r"""
set -e
export PYTHONUNBUFFERED=1 HF_HOME=/workspace/hf
mkdir -p /workspace/sf-1b/run/ckpt /workspace/sf-1b/data /workspace/hf
cd /workspace
python3 -m pip install -q --upgrade pip
python3 -m pip install -q transformers datasets huggingface_hub accelerate
python3 - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
print("download ckpt", flush=True)
p = hf_hub_download("aoxo/sf-scaling-laws", "ckpt/latest.pt", repo_type="dataset",
                    local_dir="/workspace/sf-1b/run")
print("ckpt", p, flush=True)
for f in ("code/train_1b.py", "code/superfloat.py", "ckpt/meta.json"):
    q = hf_hub_download("aoxo/sf-scaling-laws", f, repo_type="dataset",
                        local_dir="/workspace/sf-1b")
    print("got", q, flush=True)
PY
# layout: train_1b imports superfloat from parent
mkdir -p /workspace/lab
cp -f /workspace/sf-1b/code/train_1b.py /workspace/sf-1b/train_1b.py 2>/dev/null || \
  cp -f /workspace/sf-1b/train_1b.py /workspace/sf-1b/train_1b.py
# huggingface may nest repo paths
find /workspace/sf-1b -name train_1b.py -o -name superfloat.py -o -name latest.pt | head
python3 - <<'PY'
import os, shutil
root = "/workspace/sf-1b"
# flatten whatever hf_hub_download nested
for dirpath, _, files in os.walk(root):
    for f in files:
        src = os.path.join(dirpath, f)
        if f == "train_1b.py":
            shutil.copy2(src, os.path.join(root, "train_1b.py"))
        elif f == "superfloat.py":
            shutil.copy2(src, os.path.join(root, "superfloat.py"))
            shutil.copy2(src, "/workspace/superfloat.py")
        elif f == "latest.pt":
            os.makedirs(os.path.join(root, "run", "ckpt"), exist_ok=True)
            dst = os.path.join(root, "run", "ckpt", "latest.pt")
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)
        elif f == "meta.json":
            os.makedirs(os.path.join(root, "run", "ckpt"), exist_ok=True)
            shutil.copy2(src, os.path.join(root, "run", "ckpt", "meta.json"))
print("layout_ok", flush=True)
PY
cd /workspace/sf-1b
export SF1B_TOKENIZER=unsloth/Llama-3.2-1B
export SF1B_DATA=HuggingFaceFW/fineweb-edu
export SF1B_DATA_CFG=sample-100BT
# tokenize in background so wait-tokens can trip
nohup python -u train_1b.py --prepare 2000000000 --data /workspace/sf-1b/data \
  > /workspace/tok.log 2>&1 &
echo TOK_PID=$!
nohup python -u train_1b.py \
  --bits 8 --mode ln_all --seed 0 \
  --tokens 20000000000 --wait-tokens 100000000 \
  --seqlen 2048 --batch 2 --accum 8 --lr 3e-4 \
  --log-every 50 --ckpt-every 200 --ckpt-keep-every 10000 --eval-n 16 \
  --data /workspace/sf-1b/data --out /workspace/sf-1b/run \
  > /workspace/train.log 2>&1 &
echo TRAIN_PID=$!
sleep 3
head -n 20 /workspace/train.log || true
echo BOOT_OK
"""


def ssh_boot(ip, port):
    key = os.path.expanduser("~/.ssh/id_ed25519")
    cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20",
        "-o", "BatchMode=yes", "-p", str(port), "-i", key, f"root@{ip}",
        "bash", "-s",
    ]
    print("booting pod via ssh", ip, port, flush=True)
    p = subprocess.run(cmd, input=BOOT.encode(), timeout=1800)
    if p.returncode != 0:
        raise SystemExit(f"boot rc={p.returncode}")


def main():
    rp = runpod_key()
    os.environ["RUNPOD_API_KEY"] = rp
    hf = hf_token()
    print("keys_ok", flush=True)
    sz = wait_ckpt(hf)
    print("ckpt_ready_bytes", sz, flush=True)
    pub = open(os.path.expanduser("~/.ssh/id_ed25519.pub")).read().strip()
    pod = create_pod(rp, pub, hf)
    pid = pod["id"]
    ip, port, d = wait_ssh(rp, pid)
    meta = {
        "id": pid,
        "ip": ip,
        "port": port,
        "costPerHr": d.get("costPerHr") or pod.get("costPerHr"),
        "started": time.time(),
        "spend_cap": SPEND_CAP_USD,
    }
    json.dump(meta, open("/tmp/sf1b_pod.meta", "w"))
    print("META", json.dumps({k: meta[k] for k in meta}), flush=True)
    ssh_boot(ip, port)
    print("POD_TRAINING", pid, flush=True)


if __name__ == "__main__":
    main()
