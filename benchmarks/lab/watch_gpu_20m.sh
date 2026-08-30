#!/usr/bin/env bash
# Poll GPU util on 3050 SAM + RunPod exp3 every 20 minutes. Log + sentinel on idle-while-training.
set -euo pipefail

POLL_SECS="${POLL_SECS:-1200}"
HOST_3050="${HOST_3050:-alosh@100.107.175.57}"
META="${META:-/tmp/sf_exp3_pod.meta}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
LOG="${LOG:-/tmp/gpu_watch_20m.log}"
IDLE_WARN="${IDLE_WARN:-3}"  # consecutive low-GPU polls while train proc alive

if [[ -f /tmp/.runpod_api_key ]]; then
  export RUNPOD_API_KEY="$(tr -d '\n' < /tmp/.runpod_api_key)"
fi

pod_ssh() {
  local ip port
  ip="$(python3 -c "import json; print(json.load(open('$META'))['ip'])")"
  port="$(python3 -c "import json; print(json.load(open('$META'))['port'])")"
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes \
      -p "$port" -i "$SSH_KEY" "root@$ip" "$@"
}

refresh_pod_ssh() {
  [[ -n "${RUNPOD_API_KEY:-}" ]] || return 0
  python3 - "$META" <<'PY' || true
import json, os, sys, urllib.request
meta_path = sys.argv[1]
pod_id = json.load(open(meta_path))["id"]
req = urllib.request.Request(
    f"https://rest.runpod.io/v1/pods/{pod_id}",
    headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"},
)
try:
    with urllib.request.urlopen(req, timeout=20) as r:
        d = json.load(r)
except Exception:
    sys.exit(0)
ip, ports = d.get("publicIp") or "", d.get("portMappings") or {}
port = ports.get("22") or ports.get(22) or ""
if ip and port:
    m = json.load(open(meta_path))
    m.update(ip=ip, port=int(port))
    json.dump(m, open(meta_path, "w"))
PY
}

check_3050() {
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST_3050" bash -s <<'EOS'
gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null || echo "? ?")
util=$(echo "$gpu" | cut -d, -f1 | tr -d ' ')
mem=$(echo "$gpu" | cut -d, -f2 | tr -d ' ')
train=0
pgrep -f 'sam_scratch.py|sam_pretrain.py' >/dev/null && train=1
done=0
grep -qE 'SAM_FOUNDATION_DONE|SAM_PRETRAIN_LADDER_DONE' ~/sf-sam/foundation.log 2>/dev/null && done=1
tail1=$(tail -n 1 ~/sf-sam/foundation.log 2>/dev/null | tr -d '\r')
echo "3050 util=${util}% mem=${mem}MiB train=${train} done=${done} log=${tail1:0:120}"
EOS
}

check_runpod() {
  [[ -f "$META" ]] || { echo "runpod no_meta"; return; }
  refresh_pod_ssh
  pod_ssh bash -s <<'EOS' 2>/dev/null || echo "runpod ssh_fail"
gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null || echo "? ?")
util=$(echo "$gpu" | cut -d, -f1 | tr -d ' ')
mem=$(echo "$gpu" | cut -d, -f2 | tr -d ' ')
train=0
pgrep -f 'exp3_11m.py' >/dev/null && train=1
done=0
grep -q EXP3_DONE /workspace/exp3.log 2>/dev/null && done=1
tail1=$(tail -n 1 /workspace/exp3.log 2>/dev/null | tr -d '\r')
echo "runpod util=${util}% mem=${mem}MiB train=${train} done=${done} log=${tail1:0:120}"
EOS
}

idle_3050=0
idle_rp=0
echo "gpu_watch start poll=${POLL_SECS}s log=$LOG" | tee -a "$LOG"
while true; do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  line3050="$(check_3050 2>/dev/null || echo '3050 ssh_fail')"
  line_rp="$(check_runpod 2>/dev/null || echo 'runpod check_fail')"
  echo "[$ts] $line3050" | tee -a "$LOG"
  echo "[$ts] $line_rp" | tee -a "$LOG"

  u3050="$(echo "$line3050" | sed -n 's/.*util=\([0-9]*\)%.*/\1/p')"
  tr3050="$(echo "$line3050" | sed -n 's/.*train=\([01]\).*/\1/p')"
  d3050="$(echo "$line3050" | sed -n 's/.*done=\([01]\).*/\1/p')"
  if [[ "$d3050" == "1" ]]; then idle_3050=0
  elif [[ "$tr3050" == "1" ]] && [[ "$u3050" =~ ^[0-9]+$ ]] && (( u3050 < 5 )); then
    idle_3050=$((idle_3050 + 1))
    echo "[$ts] WARN 3050 low_gpu idle_poll=$idle_3050/$IDLE_WARN" | tee -a "$LOG"
  else idle_3050=0; fi

  urp="$(echo "$line_rp" | sed -n 's/.*util=\([0-9]*\)%.*/\1/p')"
  trrp="$(echo "$line_rp" | sed -n 's/.*train=\([01]\).*/\1/p')"
  drp="$(echo "$line_rp" | sed -n 's/.*done=\([01]\).*/\1/p')"
  if [[ "$drp" == "1" ]]; then idle_rp=0
  elif [[ "$trrp" == "1" ]] && [[ "$urp" =~ ^[0-9]+$ ]] && (( urp < 5 )); then
    idle_rp=$((idle_rp + 1))
    echo "[$ts] WARN runpod low_gpu idle_poll=$idle_rp/$IDLE_WARN" | tee -a "$LOG"
  else idle_rp=0; fi

  if (( idle_3050 >= IDLE_WARN )); then
    echo "AGENT_LOOP_WAKE_gpu_watch {\"prompt\":\"3050 GPU idle ${idle_3050}x20m while SAM train proc alive. Check ~/sf-sam/foundation.log and tmux sf-sam.\"}" | tee -a "$LOG"
    idle_3050=0
  fi
  if (( idle_rp >= IDLE_WARN )); then
    echo "AGENT_LOOP_WAKE_gpu_watch {\"prompt\":\"RunPod GPU idle ${idle_rp}x20m while exp3_11m alive. Check /workspace/exp3.log pod $(python3 -c \"import json;print(json.load(open('$META'))['id'])\" 2>/dev/null).\"}" | tee -a "$LOG"
    idle_rp=0
  fi

  if [[ "$d3050" == "1" ]] && [[ "$drp" == "1" ]]; then
    echo "[$ts] both jobs done, watcher exit" | tee -a "$LOG"
    exit 0
  fi
  sleep "$POLL_SECS"
done
