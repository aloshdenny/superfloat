#!/usr/bin/env bash
# Dense PTQ+absorb across Pythia training steps (§6 extension). 160m+410m on 4GB.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sf
cd ~/sf-ptq/lab
export EXP2_OUT=~/sf-ptq/results-dense EXP2_EVAL=~/sf-ptq/eval_tokens.npy
export HF_HOME=~/hf-cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$EXP2_OUT"

for size in 160m 410m; do
  for step in 1000 3000 8000 20000 39000 78000 143000; do
    tag="ptqabs_${size}_step${step}_fp16.json"
    if [[ -f "$EXP2_OUT/$tag" ]]; then echo skip "$tag"; continue; fi
    echo "=== ${size} step${step} fp16 ==="
    python -u exp_ptq_absorb.py --size "$size" --step "$step" --bits 0 --batch 1 || true
    for bits in 4 6 8; do
      for absorb in "" "--absorb"; do
        python -u exp_ptq_absorb.py --size "$size" --step "$step" --bits "$bits" $absorb --batch 1 || true
      done
    done
  done
done
echo PTQ_DENSE_DONE
