#!/usr/bin/env bash
# Watch RunPod 1B QAT. Near $44 spend: SIGTERM, upload ckpt+metrics to HF, terminate.
set -euo pipefail
META="${META:-/tmp/sf1b_pod.meta}"
POLL="${POLL_SECS:-120}"
SENTINEL="${SENTINEL:-AGENT_LOOP_WAKE_sf1b_pod}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
HANDOFF="${HANDOFF:-/tmp/sf1b_handoff.ok}"
POD_SCRIPT="${POD_SCRIPT:-$(dirname "$0")/hf_handoff_pod.sh}"

if [[ ! -f "$META" ]]; then
  echo "no $META yet" >&2
  exit 0
fi

python3 - "$META" "$POLL" "$SENTINEL" "$SSH_KEY" "$HANDOFF" "$POD_SCRIPT" <<'PY'
import json, os, re, subprocess, sys, time, urllib.request, urllib.error

meta_path, poll, sentinel, ssh_key, handoff, pod_script = sys.argv[1:]
poll = int(poll)
trans = (
    "/Users/aoxo/.cursor/projects/Users-aoxo-Aleddo/agent-transcripts/"
    "2207c47c-20ca-48d2-96cb-44437e9b38c4/2207c47c-20ca-48d2-96cb-44437e9b38c4.jsonl"
)
key = os.environ.get("RUNPOD_API_KEY")
if not key:
    with open(trans) as f:
        for line in f:
            if '"role":"user"' not in line[:120]:
                continue
            m = re.search(r"rpa_[A-Za-z0-9]+", line)
            if m:
                key = m.group(0)
os.environ["RUNPOD_API_KEY"] = key

def get(url):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.load(r)

def terminate(pid):
    req = urllib.request.Request(
        f"https://rest.runpod.io/v1/pods/{pid}",
        headers={"Authorization": f"Bearer {key}"},
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            print("terminate", pid, r.status, flush=True)
    except urllib.error.HTTPError as e:
        print("terminate_err", e.code, flush=True)

def ssh_cmd(ip, port, *extra, timeout=120):
    return subprocess.check_output(
        [
            "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20",
            "-o", "BatchMode=yes", "-p", str(port), "-i", ssh_key,
            f"root@{ip}", *extra,
        ],
        timeout=timeout,
    ).decode(errors="replace")

meta = json.load(open(meta_path))
pid = meta["id"]
cap = float(meta.get("spend_cap") or 44)
# Leave ~1h for SIGTERM + 14GB HF upload so we do not die at the cap mid-push.
trigger = float(os.environ.get("SPEND_TRIGGER") or max(cap - 2.5, cap * 0.9))
started = float(meta.get("started") or time.time())
idle = 0
print(f"watching {pid} cap=${cap} trigger=${trigger:.1f} poll={poll}s", flush=True)

def handoff_and_die(ip, port, reason):
    print("HANDOFF_BEGIN", reason, flush=True)
    up_ok = False
    if ip and port:
        try:
            subprocess.check_call(
                [
                    "scp", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20",
                    "-P", str(port), "-i", ssh_key, pod_script,
                    f"root@{ip}:/root/hf_handoff_pod.sh",
                ],
                timeout=60,
            )
            out = ssh_cmd(ip, port, "bash /root/hf_handoff_pod.sh", timeout=2400)
            print(out[-4000:], flush=True)
            up_ok = "HF_HANDOFF_DONE" in out or "HF_CKPT_UP" in out
        except Exception as e:
            print("final_upload_fail", type(e).__name__, e, flush=True)
            try:
                out = ssh_cmd(ip, port, "bash /root/hf_handoff_pod.sh", timeout=2400)
                print(out[-4000:], flush=True)
                up_ok = "HF_HANDOFF_DONE" in out or "HF_CKPT_UP" in out
            except Exception as e2:
                print("retry_fail", type(e2).__name__, e2, flush=True)
    if up_ok:
        open(handoff, "w").write(reason + "\n")
        print("handoff_marker", handoff, flush=True)
    terminate(pid)
    print(
        f'{sentinel} {{"prompt":"RunPod {pid} handoff reason={reason} hf_ok={up_ok}. '
        f'Terminated. Resume 1B on WSL 4090 from aoxo/sf-scaling-laws."}}',
        flush=True,
    )

while True:
    try:
        d = get(f"https://rest.runpod.io/v1/pods/{pid}")
    except Exception as e:
        print("api_fail", e, flush=True)
        time.sleep(poll)
        continue
    ip = d.get("publicIp") or meta.get("ip")
    ports = d.get("portMappings") or {}
    port = ports.get("22") or ports.get(22) or meta.get("port")
    cost = float(d.get("costPerHr") or meta.get("costPerHr") or 0)
    hours = (time.time() - started) / 3600.0
    spent = cost * hours
    status = d.get("desiredStatus")
    print(f"pod {status} ${cost}/hr spent=${spent:.2f}/{cap:.0f} trigger=${trigger:.1f} {hours:.2f}h", flush=True)
    if status in ("EXITED", "TERMINATED") or not d:
        print(f'{sentinel} {{"prompt":"RunPod {pid} gone status={status}. Confirm billing stopped. Resume 1B on 4090 from HF if handoff.ok exists."}}')
        break
    if ip and port:
        try:
            log = ssh_cmd(
                ip, port,
                "tail -n 6 /workspace/train.log; pgrep -c -f 'train_1b.py --bits' || true; "
                "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader",
                timeout=25,
            )
            print(log.strip()[-800:], flush=True)
            if "DONE tokens=" in log:
                handoff_and_die(ip, port, "train_done")
                break
            if re.search(r"(^|\n)0\s*$", log.strip()):
                idle += 1
            else:
                idle = 0
            if idle >= 2:
                print("idle_two_polls", flush=True)
                handoff_and_die(ip, port, "idle")
                break
        except Exception as e:
            print("ssh_fail", type(e).__name__, flush=True)
            idle = 0
    if spent >= trigger:
        handoff_and_die(ip, port, f"spend_trigger_{spent:.2f}")
        break
    time.sleep(poll)
PY
