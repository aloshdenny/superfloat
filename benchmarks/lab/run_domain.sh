#!/usr/bin/env bash
# Domain-knowledge SF evals. Safe to interrupt: each arm writes its own json.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export BFCL_OUT="${BFCL_OUT:-$HERE/../results/domain}"
export DOMAIN_OUT="${DOMAIN_OUT:-$HERE/../results/domain}"
mkdir -p "$BFCL_OUT"
cd "$HERE"
PY="${PYTHON:-python}"

host="${1:-mac}"   # mac | pod
echo "host=$host  py=$PY  out=$BFCL_OUT"

run_bfcl() {
  local model="$1" bits="$2"
  extra=()
  if [[ "${host}" == "mac" ]]; then
    extra+=(--max-len 1536 --max-new 96 --batch 1)
  fi
  echo "=== BFCL $model bits=$bits ==="
  "$PY" -u bfcl_eval.py --model "$model" --bits "$bits" --mode ln_all "${extra[@]}"
}

run_vis() {
  local bits="$1"
  echo "=== vision bits=$bits ==="
  "$PY" -u vision_ptq.py --task both --bits "$bits"
}

run_xlat() {
  local bits="$1"
  echo "=== code xlat bits=$bits ==="
  "$PY" -u code_xlat.py --bits "$bits"
}

# Qwen3 dense (later than 2.5, same Llama-shaped matmuls so ln_all surgery
# applies). Qwen3.5 is a linear-attention + vision hybrid — not this recipe.
if [[ "$host" == "mac" ]]; then
  for bits in 0 8 6; do
    run_vis "$bits"
    run_xlat "$bits"
  done
fi

if [[ "$host" == "pod" ]]; then
  for model in \
      Qwen/Qwen3-0.6B \
      Qwen/Qwen3-1.7B \
      Qwen/Qwen3-4B \
      Qwen/Qwen3-8B
  do
    for bits in 0 8 6; do
      run_bfcl "$model" "$bits"
    done
  done
  for bits in 0 8 6; do
    run_vis "$bits"
    "$PY" -u code_xlat.py --model Qwen/Qwen3-0.6B --bits "$bits"
    "$PY" -u code_xlat.py --model Qwen/Qwen3-1.7B --bits "$bits"
  done
fi

echo DONE
