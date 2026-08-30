#!/usr/bin/env python3
"""Create a RunPod for exp3_11m D/N sweep. Never prints secrets."""
from __future__ import annotations

import json, os, re, subprocess, sys, time, urllib.error, urllib.request

TRANSCRIPT = (
    "/Users/aoxo/.cursor/projects/Users-aoxo-Aleddo/agent-transcripts/"
    "2207c47c-20ca-48d2-96cb-44437e9b38c4/2207c47c-20ca-48d2-96cb-44437e9b38c4.jsonl"
)
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
SPEND_CAP_USD = 4.50


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
    return open("/tmp/.hf_write").read().strip()


def api(key, method, url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
        return r.status, (json.loads(raw) if raw else {})


def create_pod(key, pub, hf):
    attempts = [
        ("COMMUNITY", ["NVIDIA GeForce RTX 3090"]),
        ("COMMUNITY", ["NVIDIA GeForce RTX 4090"]),
        ("COMMUNITY", ["NVIDIA RTX 3090"]),
        ("SECURE", ["NVIDIA GeForce RTX 3090"]),
    ]
    for cloud, gpus in attempts:
        body = {
            "cloudType": cloud,
            "computeType": "GPU",
            "gpuCount": 1,
            "gpuTypeIds": gpus,
            "gpuTypePriority": "availability",
            "name": "sf-exp3-11m",
            "imageName": IMAGE,
            "containerDiskInGb": 50,
            "volumeInGb": 0,
            "ports": ["22/tcp"],
            "supportPublicIp": True,
            "interruptible": False,
            "env": {
                "PUBLIC_KEY": pub,
                "SSH_PUBLIC_KEY": pub,
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
            print("FAIL", e.code, e.read()[:300], flush=True)
            continue
        if code in (200, 201) and isinstance(resp, dict) and resp.get("id"):
            print("CREATED", resp.get("id"), resp.get("costPerHr"), cloud, gpus[0], flush=True)
            return resp
        print("FAIL", code, resp, flush=True)
    raise SystemExit("no pod")


def wait_ssh(key, pod_id):
    for i in range(60):
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
mkdir -p /workspace/lab /workspace/results /workspace/hf
cd /workspace
python3 -m pip install -q --upgrade pip
python3 -m pip install -q transformers datasets huggingface_hub numpy accelerate
# clone lab scripts from HF dataset if present, else wait for scp
python3 - <<'PY'
import os, base64, pathlib
# scripts embedded at boot time via heredoc below
PY
echo BOOT_DEPS_OK
"""


def ssh_boot(ip, port, lab_files: dict[str, bytes]):
    key = os.path.expanduser("~/.ssh/id_ed25519")
    # upload files via base64 in boot script
    upload = []
    for name, data in lab_files.items():
        b64 = __import__("base64").b64encode(data).decode()
        upload.append(f"""
python3 - <<'PY'
import base64, pathlib
p = pathlib.Path('/workspace/lab/{name}')
p.parent.mkdir(parents=True, exist_ok=True)
p.write_bytes(base64.b64decode('{b64}'))
print('wrote', p, p.stat().st_size)
PY""")
    boot = BOOT + "\n".join(upload) + r"""
cd /workspace/lab
export EXP3_OUT=/workspace/results
export EXP3_TOKENS=/workspace/fineweb_edu_tokens.bin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u exp3_11m.py --prepare > /workspace/prepare.log 2>&1
echo PREPARE_DONE
nohup bash -c 'python3 -u exp3_11m.py --queue > /workspace/exp3.log 2>&1; echo EXP3_DONE >> /workspace/exp3.log' &
echo QUEUE_PID=$!
sleep 5
tail -n 20 /workspace/exp3.log 2>/dev/null || tail -n 10 /workspace/prepare.log
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo BOOT_OK
"""
    cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30",
        "-o", "BatchMode=yes", "-p", str(port), "-i", key, f"root@{ip}",
        "bash", "-s",
    ]
    print("booting pod via ssh", ip, port, flush=True)
    p = subprocess.run(cmd, input=boot.encode(), timeout=7200)
    if p.returncode != 0:
        raise SystemExit(f"boot rc={p.returncode}")


def main():
    rp = runpod_key()
    os.environ["RUNPOD_API_KEY"] = rp
    hf = hf_token()
    root = os.path.dirname(os.path.abspath(__file__))
    bench_root = os.path.dirname(root)
    lab_files = {
        "exp3_11m.py": open(os.path.join(root, "exp3_11m.py"), "rb").read(),
        "superfloat.py": open(os.path.join(bench_root, "superfloat.py"), "rb").read(),
    }
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
        "job": "exp3_11m",
    }
    meta_path = "/tmp/sf_exp3_pod.meta"
    json.dump(meta, open(meta_path, "w"))
    print("META", json.dumps(meta), flush=True)
    ssh_boot(ip, port, lab_files)
    print("POD_TRAINING", pid, flush=True)


if __name__ == "__main__":
    main()
