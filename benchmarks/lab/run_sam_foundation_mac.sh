#!/usr/bin/env bash
# 1) coco128 SF2–SF16 probe (skip finished json)
# 2) wait for COCO train2017
# 3) BoxSeg-S pretrain: fp32, SF16, SF8, SF4
set -euo pipefail
cd "$(dirname "$0")"
export PYTHON="${PYTHON:-/opt/homebrew/Caskroom/miniconda/base/envs/sf-scaling-laws/bin/python}"
export PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1
export DOMAIN_OUT="${DOMAIN_OUT:-$(pwd)/../results/domain}"
COCO_ROOT="${COCO_ROOT:-/Users/aoxo/sfx_data/coco}"
mkdir -p "$DOMAIN_OUT"

echo "=== coco128 SF2-16 probe ==="
bash ./run_sam_scratch_mac.sh || true

echo "=== wait for COCO train2017 ==="
for i in $(seq 1 240); do
  if [[ -f "$COCO_ROOT/.ready" ]]; then
    echo "coco ready"
    break
  fi
  echo "waiting coco ($i/240)"
  sleep 60
done
if [[ ! -f "$COCO_ROOT/.ready" ]]; then
  echo "COCO not ready — skip pretrain"
  exit 0
fi

skip() { [[ -f "$1" ]] && echo "skip $(basename "$1")" && return 0 || return 1; }

for bits in 0 16 8 4; do
  tag=fp32
  [[ "$bits" -gt 0 ]] && tag="sf${bits}"
  skip "$DOMAIN_OUT/sam_pretrain_${tag}.json" && continue
  echo "=== sam_pretrain bits=$bits ==="
  "$PYTHON" -u sam_pretrain.py --bits "$bits" --coco "$COCO_ROOT" --epochs 6 --patience 2 --batch 8 || true
done

skip "$DOMAIN_OUT/sam_pretrain_fp16.json" || {
  echo "=== sam_pretrain fp16 ==="
  "$PYTHON" -u sam_pretrain.py --bits 0 --fp16 --coco "$COCO_ROOT" --epochs 6 --patience 2 --batch 0 || true
}
echo SAM_PRETRAIN_DONE
