#!/usr/bin/env bash
# Poll a RunPod job. On DONE or crashed-idle: scp results, then terminate.
# Relies on RUNPOD_API_KEY in the environment. Never prints the key.
set -euo pipefail

POD_ID="${POD_ID:-b391evfuro5n4j}"
SSH_HOST="${SSH_HOST:-174.94.157.109}"
SSH_PORT="${SSH_PORT:-36682}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
DONE_MARK="${DONE_MARK:-REMAINING_DONE}"
REMOTE_LOG="${REMOTE_LOG:-/workspace/sf-domain-pod.log}"
REMOTE_RESULTS="${REMOTE_RESULTS:-/workspace/results}"
LOCAL_RESULTS="${LOCAL_RESULTS:-$HOME/vscode/superfloat/benchmarks/results/domain}"
POLL_SECS="${POLL_SECS:-120}"
IDLE_POLLS="${IDLE_POLLS:-2}"
SENTINEL="${SENTINEL:-AGENT_LOOP_WAKE_runpod_watch}"

if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  echo "RUNPOD_API_KEY is unset" >&2
  exit 1
fi

ssh_base() {
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes \
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
if isinstance(d, list):
    d = next((x for x in d if x.get("id") == pod_id), {})
ip = d.get("publicIp") or ""
ports = (d.get("portMappings") or {})
port = ports.get("22") or ports.get(22) or ""
status = d.get("desiredStatus") or d.get("lastStatusChange") or ""
print(f"POD_META ip={ip} port={port} status={status}", flush=True)
if ip:
    open(f"/tmp/pod_{pod_id}.meta", "w").write(f"{ip}\n{port}\n")
PY
  meta="/tmp/pod_${POD_ID}.meta"
  if [[ -f "$meta" ]]; then
    SSH_HOST="$(sed -n '1p' "$meta")"
    p="$(sed -n '2p' "$meta")"
    [[ -n "$p" && "$p" != "None" ]] && SSH_PORT="$p"
  fi
}

terminate_pod() {
  echo "terminating $POD_ID" >&2
  code="$(curl -sS -o /tmp/rp_term_body -w "%{http_code}" -X POST \
    "https://api.runpod.io/v2/pods/${POD_ID}/action" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{"action":"terminate"}' || true)"
  if [[ "$code" != "200" && "$code" != "204" && "$code" != "404" ]]; then
    curl -sS -o /tmp/rp_term_body -w " v1=%{http_code}\n" -X DELETE \
      "https://rest.runpod.io/v1/pods/${POD_ID}" \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}" || true
  else
    echo "terminate http=$code"
  fi
}

pull_results() {
  mkdir -p "$LOCAL_RESULTS"
  echo "pulling $REMOTE_RESULTS -> $LOCAL_RESULTS"
  scp -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes \
      -P "$SSH_PORT" -i "$SSH_KEY" \
      "root@${SSH_HOST}:${REMOTE_RESULTS}/*.json" "$LOCAL_RESULTS/" || true
}

finish() {
  local why="$1"
  echo "DONE_REASON=$why"
  refresh_ssh
  pull_results || true
  terminate_pod
  echo "${SENTINEL} {\"prompt\":\"RunPod ${POD_ID} finished (${why}). Results pulled to ${LOCAL_RESULTS}. Pod terminated. Report scores and confirm billing stopped.\"}"
  exit 0
}

idle=0
echo "watching $POD_ID ssh=$SSH_HOST:$SSH_PORT poll=${POLL_SECS}s marker=$DONE_MARK"
while true; do
  refresh_ssh
  if ! ssh_base "true" 2>/dev/null; then
    echo "ssh_fail $(date -u +%H:%M:%SZ) host=$SSH_HOST — not counting as idle; keeping last-good host"
    sleep "$POLL_SECS"
    continue
  fi
  if ssh_base "test -f '$REMOTE_LOG' && grep -q '$DONE_MARK' '$REMOTE_LOG'" 2>/dev/null; then
    finish "marker:$DONE_MARK"
  fi
  if ssh_base 'pgrep -f "bfcl_eval.py|vision_ptq.py|code_xlat.py|run_pod_remaining.sh|run_domain.sh" >/dev/null' 2>/dev/null; then
    idle=0
    echo "alive $(date -u +%H:%M:%SZ) ssh=$SSH_HOST:$SSH_PORT"
  else
    idle=$((idle + 1))
    echo "idle_poll=$idle/$IDLE_POLLS $(date -u +%H:%M:%SZ)"
    if [[ "$idle" -ge "$IDLE_POLLS" ]]; then
      finish "idle-no-eval"
    fi
  fi
  sleep "$POLL_SECS"
done
