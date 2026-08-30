#!/usr/bin/env bash
# M4 leftover: SAM + C→C++. YOLO PTQ already showed the cliff (fp32 0.39, SF 0.0).
# Full BFCL generate jetsams 16 GB — do not rerun it here.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
export PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1
export DOMAIN_OUT="${DOMAIN_OUT:-$(pwd)/../results/domain}"
mkdir -p "$DOMAIN_OUT"
SAM_IMGS="${SAM_IMAGES:-/Users/aoxo/sfx_data/coco128-seg/images}"

skip() {
  if [[ -f "$1" ]]; then
    echo "skip $(basename "$1")"
    return 0
  fi
  return 1
}

for bits in 0 8 6; do
  skip "$DOMAIN_OUT/sam_box_b${bits}.json" && continue
  echo "=== sam bits=$bits ==="
  # MPS dual-ViT jetsams 16 GB. CPU is slow but finishes.
  HF_HUB_OFFLINE=1 "$PY" -u vision_ptq.py --task sam --bits "$bits" \
      --sam-images "$SAM_IMGS" --device cpu || true
done

for bits in 0 8 6; do
  tag=bf16
  [[ "$bits" -gt 0 ]] && tag="sf${bits}"
  skip "$DOMAIN_OUT/xlat_Qwen3-0.6B_${tag}.json" && continue
  echo "=== code xlat Qwen3-0.6B bits=$bits ==="
  "$PY" -u code_xlat.py --model Qwen/Qwen3-0.6B --bits "$bits" --device cpu || true
done

echo MAC_REMAINING_DONE
