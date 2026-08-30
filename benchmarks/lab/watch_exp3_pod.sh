#!/usr/bin/env bash
# Poll exp3_11m RunPod: GPU check during boot, hourly after secure, HF upload + terminate on DONE.
set -euo pipefail

META="${META:-/tmp/sf_exp3_pod.meta}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
DONE_MARK="${DONE_MARK:-EXP3_DONE}"
REMOTE_LOG="${REMOTE_LOG:-/workspace/exp3.log}"
REMOTE_RESULTS="${REMOTE_RESULTS:-/workspace/results}"
LOCAL_RESULTS="${LOCAL_RESULTS:-$HOME/vscode/superfloat/benchmarks/results}"
HF_REPO="${HF_REPO:-aoxo/sf-scaling-laws}"
HF_PREFIX="${HF_PREFIX:-results/exp3_11m}"
BOOT_POLL_SECS="${BOOT_POLL_SECS:-60}"
STEADY_POLL_SECS="${STEADY_POLL_SECS:-3600}"
IDLE_POLLS="${IDLE_POLLS:-3}"
SENTINEL="${SENTINEL:-AGENT_LOOP_WAKE_exp3_watch}"

if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  if [[ -f /tmp/.runpod_api_key ]]; then
    RUNPOD_API_KEY="$(tr -d '\n' < /tmp/.runpod_api_key)"
    export RUNPOD_API_KEY
  fi
fi
if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  echo "RUNPOD_API_KEY is unset" >&2
  exit 1
fi
if [[ ! -f "$META" ]]; then
  echo "missing $META" >&2
  exit 1
fi

POD_ID="$(python3 -c "import json; print(json.load(open('$META'))['id'])")"
SSH_HOST="$(python3 -c "import json; print(json.load(open('$META'))['ip'])")"
SSH_PORT="$(python3 -c "import json; print(json.load(open('$META'))['port'])")"

ssh_base() {
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes \
      -p "$SSH_PORT" -i "$SSH_KEY" "root@$SSH_HOST" "$@"
}

refresh_ssh() {
  python3 - "$POD_ID" <<'PY' || true
import json, os, sys, urllib.request
pod_id = sys.argv[1]
req = urllib.request.Request(
    f"https://rest.runpod.io/v1/pods/{pod_id}",
    headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"},
)
try:
    with urllib.request.urlopen(req, timeout=20) as r:
        d = json.load(r)
except Exception as e:
    print(f"refresh_ssh failed: {e}", flush=True)
    sys.exit(0)
ip = d.get("publicIp") or ""
ports = d.get("portMappings") or {}
port = ports.get("22") or ports.get(22) or ""
print(f"POD_META ip={ip} port={port} status={d.get('desiredStatus')}", flush=True)
if ip:
    open(f"/tmp/pod_{pod_id}.meta", "w").write(f"{ip}\n{port}\n")
PY
  m="/tmp/pod_${POD_ID}.meta"
  if [[ -f "$m" ]]; then
    SSH_HOST="$(sed -n '1p' "$m")"
    p="$(sed -n '2p' "$m")"
    [[ -n "$p" && "$p" != "None" ]] && SSH_PORT="$p"
  fi
}

terminate_pod() {
  echo "terminating $POD_ID" >&2
  curl -sS -o /tmp/rp_term_body -w "terminate=%{http_code}\n" -X DELETE \
    "https://rest.runpod.io/v1/pods/${POD_ID}" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" || true
}

upload_hf() {
  HF_TOKEN="$(tr -d '\n' < /tmp/.hf_write)"
  export HF_TOKEN
  mkdir -p "$LOCAL_RESULTS"
  scp -o StrictHostKeyChecking=no -o ConnectTimeout=30 -o BatchMode=yes \
      -P "$SSH_PORT" -i "$SSH_KEY" \
      "root@${SSH_HOST}:${REMOTE_RESULTS}/*.json" "$LOCAL_RESULTS/" 2>/dev/null || true
  python3 - "$LOCAL_RESULTS" "$HF_REPO" "$HF_PREFIX" <<'PY'
import glob, os, sys
from huggingface_hub import HfApi
local, repo, prefix = sys.argv[1:4]
api = HfApi(token=os.environ["HF_TOKEN"])
for path in sorted(glob.glob(os.path.join(local, "exp3_11m*.json")) + glob.glob(os.path.join(local, "exp3_*.json"))):
    name = os.path.basename(path)
    dest = f"{prefix}/{name}"
    api.upload_file(path_or_fileobj=path, path_in_repo=dest, repo_id=repo, repo_type="dataset")
    print("HF_UP", dest, flush=True)
print("HF_EXP3_DONE", flush=True)
PY
}

finish() {
  local why="$1"
  echo "DONE_REASON=$why"
  refresh_ssh
  upload_hf || true
  terminate_pod
  echo "${SENTINEL} {\"prompt\":\"exp3_11m pod ${POD_ID} finished (${why}). Results on HF ${HF_REPO}/${HF_PREFIX}. Pod terminated.\"}"
  exit 0
}

secure=0
idle=0
poll="$BOOT_POLL_SECS"
echo "watching $POD_ID ssh=$SSH_HOST:$SSH_PORT boot_poll=${BOOT_POLL_SECS}s steady=${STEADY_POLL_SECS}s"
while true; do
  refresh_ssh
  if ! ssh_base "true" 2>/dev/null; then
    echo "ssh_fail $(date -u +%H:%M:%SZ)"
    sleep "$poll"
    continue
  fi
  gpu="$(ssh_base "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits" 2>/dev/null || echo '?')"
  train="$(ssh_base 'pgrep -fc "exp3_11m.py" || true' 2>/dev/null || echo 0)"
  log_tail="$(ssh_base "tail -n 3 '$REMOTE_LOG' 2>/dev/null" 2>/dev/null || true)"
  echo "check $(date -u +%H:%M:%SZ) gpu=[$gpu] procs=$train secure=$secure"
  echo "$log_tail" | tail -n 2

  if ssh_base "test -f '$REMOTE_LOG' && grep -q '$DONE_MARK' '$REMOTE_LOG'" 2>/dev/null; then
    finish "marker:$DONE_MARK"
  fi

  util="$(echo "$gpu" | cut -d, -f1 | tr -d ' ')"
  mem="$(echo "$gpu" | cut -d, -f2 | tr -d ' ')"
  if [[ "$train" -gt 0 ]] && [[ "$util" =~ ^[0-9]+$ ]] && (( util > 5 || mem > 500 )); then
    secure=1
    poll="$STEADY_POLL_SECS"
    idle=0
    echo "GPU_ACTIVE util=${util}% mem=${mem}MiB -> steady poll ${STEADY_POLL_SECS}s"
  elif [[ "$train" -gt 0 ]]; then
    idle=0
  else
    if ssh_base "test -f /workspace/prepare.log && ! test -f '$REMOTE_LOG'" 2>/dev/null; then
      idle=0
      echo "still preparing corpus"
    else
      idle=$((idle + 1))
      echo "idle_poll=$idle/$IDLE_POLLS"
      if [[ "$idle" -ge "$IDLE_POLLS" ]]; then
        finish "idle-no-train"
      fi
    fi
  fi
  sleep "$poll"
done
