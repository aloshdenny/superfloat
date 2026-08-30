#!/usr/bin/env bash
# From-scratch box-prompt SF2–SF16 QAT on M4. Sequential: MPS hates 15 copies.
# fp32 control first so a broken task is visible before the sweep burns an hour.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-/opt/homebrew/Caskroom/miniconda/base/envs/sf-scaling-laws/bin/python}"
export PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1
export DOMAIN_OUT="${DOMAIN_OUT:-$(pwd)/../results/domain}"
mkdir -p "$DOMAIN_OUT"

skip() {
  if [[ -f "$1" ]]; then
    echo "skip $(basename "$1")"
    return 0
  fi
  return 1
}

for bits in 0 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2; do
  tag=fp32
  [[ "$bits" -gt 0 ]] && tag="sf${bits}"
  skip "$DOMAIN_OUT/sam_scratch_${tag}.json" && continue
  echo "=== sam_scratch bits=$bits ==="
  "$PY" -u sam_scratch.py --bits "$bits" || true
done
skip "$DOMAIN_OUT/sam_scratch_fp16.json" || {
  echo "=== sam_scratch fp16 ==="
  "$PY" -u sam_scratch.py --bits 0 --fp16 || true
}
echo SAM_SCRATCH_DONE
