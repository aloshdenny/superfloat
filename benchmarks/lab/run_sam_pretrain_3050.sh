#!/usr/bin/env bash
# RTX 3050 4GB: COCO BoxSeg-S pretrain ladder (fp32 → fp16 → SF16 → SF8 → SF4).
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sf
cd ~/sf-sam/lab
export PYTHONUNBUFFERED=1
export DOMAIN_OUT="${DOMAIN_OUT:-$HOME/sf-sam/results}"
export COCO_ROOT="${COCO_ROOT:-$HOME/sfx_data/coco}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=python
mkdir -p "$DOMAIN_OUT"

skip() { [[ -f "$1" ]] && echo "skip $(basename "$1")" && return 0; return 1; }

run_arm() {
  local bits="$1" fp16="${2:-0}"
  local tag=fp32
  [[ "$fp16" == "1" ]] && tag=fp16
  [[ "$bits" -gt 0 ]] && tag="sf${bits}"
  skip "$DOMAIN_OUT/sam_pretrain_${tag}.json" && return 0
  echo "=== sam_pretrain ${tag} $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  local extra=()
  [[ "$fp16" == "1" ]] && extra+=(--fp16)
  "$PY" -u sam_pretrain.py --bits "$bits" "${extra[@]}" \
    --coco "$COCO_ROOT" --epochs 6 --patience 2 --batch 0 --threads 8 --device cuda \
    || true
}

if [[ ! -f "$COCO_ROOT/.ready" ]]; then
  echo "=== COCO download ==="
  COCO_ROOT="$COCO_ROOT" PYTHON="$PY" bash ./download_coco_seg.sh
fi

for spec in "0 0" "0 1" "16 0" "8 0" "4 0"; do
  set -- $spec
  run_arm "$1" "$2"
done
echo SAM_PRETRAIN_3050_DONE
