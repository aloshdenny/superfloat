#!/usr/bin/env bash
# M4: vision + C→C++ only. Full BFCL generate OOMs/jetsams 16 GB unified memory.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
export PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1
export DOMAIN_OUT="${DOMAIN_OUT:-$(pwd)/../results/domain}"
mkdir -p "$DOMAIN_OUT"
for bits in 0 8 6; do
  echo "=== yolo-seg bits=$bits ==="
  "$PY" -u vision_ptq.py --task yolo --bits "$bits" || true
done
for bits in 0 8 6; do
  echo "=== sam bits=$bits ==="
  "$PY" -u vision_ptq.py --task sam --bits "$bits" || true
done
for bits in 0 8 6; do
  echo "=== code xlat bits=$bits ==="
  "$PY" -u code_xlat.py --bits "$bits" || true
done
echo MAC_VISION_DONE
