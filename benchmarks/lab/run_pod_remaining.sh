#!/usr/bin/env bash
# Resume after the Qwen3 ladder: skip json that already exists.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/workspace/hf}"
export BFCL_OUT="${BFCL_OUT:-/workspace/results}"
export DOMAIN_OUT="${DOMAIN_OUT:-/workspace/results}"
export CXX="${CXX:-g++}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PYTHON:-python}"
mkdir -p "$BFCL_OUT"

skip() {
  if [[ -f "$1" ]]; then
    echo "skip $(basename "$1")"
    return 0
  fi
  return 1
}

for bits in 8 6; do
  out="$BFCL_OUT/bfcl_Qwen3-8B_sf${bits}_ln_all.json"
  skip "$out" && continue
  echo "=== BFCL Qwen/Qwen3-8B bits=$bits ==="
  "$PY" -u bfcl_eval.py --model Qwen/Qwen3-8B --bits "$bits" --mode ln_all --batch 2
done

# coco128-seg images for SAM (ultralytics also pulls this on yolo val)
SAM_IMGS="${SAM_IMAGES:-/workspace/datasets/coco128-seg/images}"
if [[ ! -d "$SAM_IMGS" ]]; then
  echo "=== fetch coco128-seg for SAM ==="
  "$PY" - <<'PY'
from ultralytics.utils.downloads import download
from pathlib import Path
p = Path("/workspace/datasets")
p.mkdir(parents=True, exist_ok=True)
download("https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128-seg.zip", dir=p)
PY
  SAM_IMGS="/workspace/datasets/coco128-seg/images"
fi

for bits in 0 8 6; do
  skip "$DOMAIN_OUT/yolo_seg_b${bits}.json" || {
    echo "=== yolo-seg bits=$bits ==="
    "$PY" -u vision_ptq.py --task yolo --bits "$bits"
  }
  skip "$DOMAIN_OUT/sam_box_b${bits}.json" || {
    echo "=== sam bits=$bits ==="
    "$PY" -u vision_ptq.py --task sam --bits "$bits" --sam-images "$SAM_IMGS"
  }
done

for model in Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B; do
  name="${model##*/}"
  for bits in 0 8 6; do
    tag=bf16
    [[ "$bits" -gt 0 ]] && tag="sf${bits}"
    skip "$DOMAIN_OUT/xlat_${name}_${tag}.json" && continue
    echo "=== code xlat $model bits=$bits ==="
    "$PY" -u code_xlat.py --model "$model" --bits "$bits"
  done
done

echo REMAINING_DONE
