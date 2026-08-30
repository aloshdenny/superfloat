#!/usr/bin/env bash
# Poll home 4090 WSL until it is up, then resume 1B QAT from Hugging Face ckpt.
# Waits for /tmp/sf1b_handoff.ok so we do not resume the stale 312M snapshot.
set -euo pipefail
HOST="${HOST:-research@100.86.165.70}"
HANDOFF="${HANDOFF:-/tmp/sf1b_handoff.ok}"
POLL="${POLL_SECS:-60}"
SENTINEL="${SENTINEL:-AGENT_LOOP_WAKE_wsl_1b}"
HF_FILE="${HF_FILE:-/tmp/.hf_write}"

echo "waiting WSL on $HOST; handoff=$HANDOFF poll=${POLL}s" >&2

wsl() {
  ssh -o BatchMode=yes -o ConnectTimeout=12 "$HOST" wsl.exe -- bash -s
}

while true; do
  if [[ ! -f "$HANDOFF" ]]; then
    echo "no_handoff_yet $(date -u +%H:%M:%SZ)"
  fi
  out=$(wsl <<'EOF' 2>&1 || true
echo WSL_OK
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader
pgrep -c -f 'train_1b.py --bits' || echo 0
pgrep -c -f sam_pretrain.py || echo 0
EOF
)
  echo "$out" | tr -d '\0' | tail -n 8
  if echo "$out" | tr -d '\0' | grep -q WSL_OK; then
    if [[ -f "$HANDOFF" ]]; then
      echo "WSL_UP_AND_HANDOFF $(date -u +%H:%M:%SZ)"
      break
    fi
    echo "WSL_UP waiting_handoff $(date -u +%H:%M:%SZ)"
  else
    echo "WSL_DOWN $(date -u +%H:%M:%SZ)"
  fi
  sleep "$POLL"
done

# Push HF write token into WSL without printing it.
# Windows OpenSSH + wsl.exe -lc '...' eats quotes; copy via a Windows path instead.
if [[ -f "$HF_FILE" ]]; then
  scp -o BatchMode=yes -o ConnectTimeout=15 "$HF_FILE" "$HOST:C:/Users/research/.hf_write"
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" wsl.exe -- bash -c "cp /mnt/c/Users/research/.hf_write /tmp/.hf_write && chmod 600 /tmp/.hf_write && rm -f /mnt/c/Users/research/.hf_write"
fi

wsl <<'EOF'
set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_TOKEN=$(tr -d '\n' < /tmp/.hf_write 2>/dev/null || true)
source /home/research/miniconda3/etc/profile.d/conda.sh
conda activate sfx
# 1B takes the 4090; pause BoxSeg if it came back with the machine.
tmux has-session -t sf-pretrain 2>/dev/null && tmux send-keys -t sf-pretrain C-c || true
pkill -TERM -f sam_pretrain.py || true
sleep 2
tmux has-session -t sf1b-train 2>/dev/null && tmux kill-session -t sf1b-train || true
pkill -TERM -f 'train_1b.py --bits' || true
sleep 2

mkdir -p /home/research/alosh/sf-1b/run/ckpt
cd /home/research/alosh/sf-1b
python - <<'PY'
from huggingface_hub import hf_hub_download
import json, os, shutil
print("download ckpt from aoxo/sf-scaling-laws", flush=True)
p = hf_hub_download("aoxo/sf-scaling-laws", "ckpt/latest.pt", repo_type="dataset",
                    local_dir="/home/research/alosh/sf-1b/run")
dst = "/home/research/alosh/sf-1b/run/ckpt/latest.pt"
os.makedirs(os.path.dirname(dst), exist_ok=True)
if os.path.abspath(p) != os.path.abspath(dst):
    shutil.copy2(p, dst)
print("ckpt", os.path.getsize(dst), flush=True)
# Keep the home-4090 metrics (pre-0.31B) if present; HF only has the H100 tail.
home_metrics = "/home/research/alosh/sf-1b/run/metrics.jsonl"
if os.path.exists(home_metrics) and os.path.getsize(home_metrics) > 0:
    bak = "/home/research/alosh/sf-1b/run/metrics-4090-pre-h100.jsonl"
    shutil.copy2(home_metrics, bak)
    print("kept", bak, os.path.getsize(bak), flush=True)
for f, dest in (
    ("ckpt/meta.json", "/home/research/alosh/sf-1b/run/ckpt/meta.json"),
    ("run/metrics.jsonl", "/home/research/alosh/sf-1b/run/metrics-h100.jsonl"),
):
    try:
        q = hf_hub_download("aoxo/sf-scaling-laws", f, repo_type="dataset",
                            local_dir="/tmp/sf1b_hf")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(q, dest)
        print("got", f, "->", dest, flush=True)
    except Exception as e:
        print("skip", f, type(e).__name__, flush=True)
# One file: pre-H100 4090 evals + H100 tail. Later 4090 resume appends here.
pre = "/home/research/alosh/sf-1b/run/metrics-4090-pre-h100.jsonl"
h100p = "/home/research/alosh/sf-1b/run/metrics-h100.jsonl"
outp = "/home/research/alosh/sf-1b/run/metrics.jsonl"
by_step = {}
for src in (pre, h100p):
    if not os.path.exists(src):
        continue
    with open(src) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_step[int(rec["step"])] = rec
rows = [by_step[s] for s in sorted(by_step)]
with open(outp, "w") as fh:
    for rec in rows:
        fh.write(json.dumps(rec) + "\n")
print("MERGED_METRICS", outp, "n=", len(rows),
      "steps", rows[0]["step"] if rows else None, "->",
      rows[-1]["step"] if rows else None, flush=True)
print("HF_PULL_OK", flush=True)
PY

export SF1B_TOKENIZER=unsloth/Llama-3.2-1B
export SF1B_DATA=HuggingFaceFW/fineweb-edu
export SF1B_DATA_CFG=sample-100BT
# Home 4090 recipe: batch 2 accum 8. --no-compile matches the working H100 run.
tmux new-session -d -s sf-1b-resume "cd /home/research/alosh/sf-1b && python -u train_1b.py \
  --bits 8 --mode ln_all --seed 0 \
  --tokens 20000000000 --wait-tokens 50000000 \
  --seqlen 2048 --batch 2 --accum 8 --lr 3e-4 \
  --log-every 50 --ckpt-every 200 --ckpt-keep-every 10000 --eval-n 16 \
  --no-compile \
  --data /home/research/alosh/sf-1b/data --out /home/research/alosh/sf-1b/run \
  2>&1 | tee -a /home/research/alosh/sf-1b/run/train-4090.log"
sleep 8
tmux ls
pgrep -af 'train_1b.py --bits' || true
tail -n 20 /home/research/alosh/sf-1b/run/train-4090.log || true
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo RESUME_STARTED
EOF

echo "${SENTINEL} {\"prompt\":\"WSL 4090 is up and 1B QAT resume launched from HF ckpt. Confirm train-4090.log resumed step and GPU util. Do not restart RunPod.\"}"
