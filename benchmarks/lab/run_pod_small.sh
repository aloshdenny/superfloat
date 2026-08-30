#!/usr/bin/env bash
# After the 3B/7B queue, run the small BFCL models too (M4 cannot finish generate).
set -euo pipefail
cd /workspace/sf/benchmarks/lab
export PYTHONUNBUFFERED=1 HF_HOME=/workspace/hf BFCL_OUT=/workspace/results DOMAIN_OUT=/workspace/results CXX=g++ PYTHON=python
while pgrep -f 'run_domain.sh pod' >/dev/null; do sleep 30; done
echo "=== small BFCL follow-on ==="
for model in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct; do
  for bits in 0 8 6; do
    python -u bfcl_eval.py --model "$model" --bits "$bits" --mode ln_all
  done
done
echo SMALL_BFCL_DONE
