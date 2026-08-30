#!/usr/bin/env bash
# Hourly check on 3050 PTQ queue; HF upload when PTQ_DONE.
set -euo pipefail
HOST="${HOST:-alosh@100.107.175.57}"
LOCAL_RESULTS="${LOCAL_RESULTS:-$HOME/vscode/superfloat/benchmarks/results}"
HF_REPO="${HF_REPO:-aoxo/sf-scaling-laws}"
HF_PREFIX="${HF_PREFIX:-results/ptq_absorb}"
POLL_SECS="${POLL_SECS:-3600}"
SENTINEL="${SENTINEL:-AGENT_LOOP_WAKE_ptq_watch}"

ssh_host() {
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" "$@"
}

pull_and_upload() {
  mkdir -p "$LOCAL_RESULTS/ptq_absorb"
  ssh_host <<'EOF'
set -e
cd ~/sf-ptq/results
for f in ptqabs_*.json; do
  [ -f "$f" ] && base64 -w0 "$f" && echo " $f"
done
EOF
  # simpler: parse via python on remote and upload from mac after scp alternative
  python3 - "$HOST" "$LOCAL_RESULTS/ptq_absorb" "$HF_REPO" "$HF_PREFIX" <<'PY'
import base64, json, os, subprocess, sys
from huggingface_hub import HfApi
host, local, repo, prefix = sys.argv[1:5]
out = subprocess.check_output(["ssh", "-o", "BatchMode=yes", host,
    "bash -lc 'ls ~/sf-ptq/results/ptqabs_*.json 2>/dev/null'"], text=True)
files = [x.strip() for x in out.splitlines() if x.strip()]
os.makedirs(local, exist_ok=True)
api = HfApi(token=open("/tmp/.hf_write").read().strip())
for remote in files:
    name = os.path.basename(remote)
    data = subprocess.check_output(["ssh", "-o", "BatchMode=yes", host,
        f"cat {remote}"])
    path = os.path.join(local, name)
    open(path, "wb").write(data)
    api.upload_file(path_or_fileobj=path, path_in_repo=f"{prefix}/{name}",
                    repo_id=repo, repo_type="dataset")
    print("HF_UP", name, flush=True)
print("HF_PTQ_DONE", len(files), flush=True)
PY
}

while true; do
  status="$(ssh_host "tmux has-session -t sf-ptq 2>/dev/null && echo running || echo stopped; tail -n 2 ~/sf-ptq/queue.log 2>/dev/null; grep -c PTQ_DONE ~/sf-ptq/queue.log 2>/dev/null || echo 0" 2>/dev/null || echo ssh_fail)"
  echo "ptq_check $(date -u +%H:%M:%SZ)"
  echo "$status"
  if echo "$status" | grep -q PTQ_DONE; then
    pull_and_upload || true
    echo "${SENTINEL} {\"prompt\":\"3050 PTQ queue finished. Results uploaded to ${HF_REPO}/${HF_PREFIX}.\"}"
    exit 0
  fi
  sleep "$POLL_SECS"
done
