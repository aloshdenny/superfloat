#!/usr/bin/env bash
# RTX 3050: coco128 probe only. COCO pretrain runs on RunPod.
set -euo pipefail
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sf
export PYTHONUNBUFFERED=1
export DOMAIN_OUT="${DOMAIN_OUT:-$HOME/sf-sam/results}"
export COCO128="${COCO128:-$HOME/sfx_data/coco128-seg}"
PY=python
mkdir -p "$DOMAIN_OUT"

skip() {
  if [[ -f "$1" ]]; then echo "skip $(basename "$1")"; return 0; fi
  return 1
}

run_scratch_arm() {
  local bits="$1" fp16="${2:-0}"
  local tag=fp32
  [[ "$fp16" == "1" ]] && tag=fp16
  [[ "$bits" -gt 0 ]] && tag="sf${bits}"
  skip "$DOMAIN_OUT/sam_scratch_${tag}.json" && return 0
  echo "=== sam_scratch ${tag} ==="
  local extra=()
  [[ "$fp16" == "1" ]] && extra+=(--fp16)
  "$PY" -u sam_scratch.py --bits "$bits" "${extra[@]}" \
    --images "$COCO128/images" --labels "$COCO128/labels" \
    --device cuda --batch 16 || true
}

fetch_coco128() {
  if [[ -d "$COCO128/images" ]] && ls "$COCO128/images"/*.jpg &>/dev/null; then return 0; fi
  mkdir -p "$(dirname "$COCO128")"
  "$PY" - <<'PY'
from ultralytics.utils.downloads import download
from pathlib import Path
import os
p = Path(os.environ["COCO128"]).parent
download("https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128-seg.zip", dir=p)
PY
}

echo "=== phase 1: coco128 ladder ==="
fetch_coco128
for spec in "0 0" "0 1" "16 0" "8 0" "4 0"; do
  set -- $spec
  run_scratch_arm "$1" "$2"
done
echo SAM_SCRATCH_LADDER_DONE
echo "=== phase 2: dense PTQ (parallel fill) ==="
bash "$(dirname "$0")/run_ptq_dense_3050.sh" || true
echo SAM_3050_FILL_DONE
