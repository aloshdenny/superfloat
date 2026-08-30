#!/usr/bin/env bash
# Runs ON the RunPod box. Pause 1B QAT, upload ckpt+metrics to HF.
set -euo pipefail
export HF_TOKEN=$(tr -d '\n' < /root/.hf_write)
pkill -TERM -f 'train_1b.py --bits' || true
for i in $(seq 1 12); do
  if ! pgrep -f 'train_1b.py --bits' >/dev/null; then
    echo TRAIN_STOPPED
    break
  fi
  echo wait_stop "$i"
  sleep 15
done
pkill -TERM -f 'train_1b.py --bits' || true
sleep 5
ls -lh /workspace/sf-1b/run/ckpt/latest.pt /workspace/sf-1b/run/metrics.jsonl
python3 - <<'ENDPY'
import json, os, re, time
from huggingface_hub import HfApi
log = open("/workspace/train.log").read().splitlines()
last = next((l for l in reversed(log) if "/305175" in l), "")
m = re.search(r"(\d+)/305175\s+tok=([0-9.]+)M.*?val [0-9.]+ ppl ([0-9.]+)", last)
meta = {
    "run": "sf8_ln_all_s0",
    "host": "runpod-h100-nvl",
    "handoff": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "last_log": last[-400:],
}
if m:
    meta.update(step=int(m.group(1)), tokens_seen_m=float(m.group(2)), val_ppl=float(m.group(3)))
os.makedirs("/workspace/sf-1b/run/ckpt", exist_ok=True)
json.dump(meta, open("/workspace/sf-1b/run/ckpt/meta.json", "w"), indent=2)
print("meta", meta, flush=True)
api = HfApi(token=os.environ["HF_TOKEN"])
repo, rtype = "aoxo/sf-scaling-laws", "dataset"
api.upload_file(path_or_fileobj="/workspace/sf-1b/run/ckpt/latest.pt",
                path_in_repo="ckpt/latest.pt", repo_id=repo, repo_type=rtype)
print("HF_CKPT_UP", flush=True)
api.upload_file(path_or_fileobj="/workspace/sf-1b/run/ckpt/meta.json",
                path_in_repo="ckpt/meta.json", repo_id=repo, repo_type=rtype)
print("HF_META_UP", flush=True)
if os.path.exists("/workspace/sf-1b/run/metrics.jsonl"):
    api.upload_file(path_or_fileobj="/workspace/sf-1b/run/metrics.jsonl",
                    path_in_repo="run/metrics.jsonl", repo_id=repo, repo_type=rtype)
    print("HF_METRICS_UP", flush=True)
print("HF_HANDOFF_DONE", flush=True)
ENDPY
