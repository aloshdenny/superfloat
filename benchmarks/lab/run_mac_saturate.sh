#!/usr/bin/env bash
# Pack the M4: GPU (MPS) = YOLO11n-seg QAT from scratch; CPU = 4-wide TinyBoxSeg
# SF2–SF16; network = COCO-2017 download. 16 GB unified — no SAM-ViT dual load.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH="$(cd "$HERE/.." && pwd)"
PY="${PYTHON:-/opt/homebrew/Caskroom/miniconda/base/envs/sf-scaling-laws/bin/python}"
export PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1
export DOMAIN_OUT="${DOMAIN_OUT:-$BENCH/results/domain}"
mkdir -p "$DOMAIN_OUT" /tmp/sf-sat /Users/aoxo/sfx_data/coco

YAML="$DOMAIN_OUT/coco128-seg-local.yaml"
SRC="/opt/homebrew/Caskroom/miniconda/base/envs/sf-scaling-laws/lib/python3.11/site-packages/ultralytics/cfg/datasets/coco128-seg.yaml"
sed 's|^path: coco128-seg|path: /Users/aoxo/sfx_data/coco128-seg|' "$SRC" > "$YAML"

skip() { [[ -f "$1" ]]; }

echo "=== network: COCO-2017 download ==="
if [[ ! -f /Users/aoxo/sfx_data/coco/.ready ]]; then
  nohup bash "$HERE/download_coco_seg.sh" >> /tmp/sf-coco-dl.log 2>&1 &
  echo DL_PID=$!
fi

echo "=== GPU/MPS: YOLO11n-seg from-scratch QAT ==="
(
  cd "$BENCH"
  for spec in "fp32:1.0" "sf8:2.0" "sf4:8.0"; do
    fmt="${spec%%:*}"
    scale="${spec##*:}"
    out="$DOMAIN_OUT/yolo_seg_scratch_${fmt}.json"
    skip "$out" && echo "skip $(basename "$out")" && continue
    echo "=== yolo-seg scratch $fmt ==="
    "$PY" -u train_yolo.py \
      --format "$fmt" --cfg yolo11n-seg.yaml --init random \
      --data "$YAML" --name "yolo11n-seg_${fmt}" \
      --project "$DOMAIN_OUT/yolo_runs" --device mps \
      --imgsz 640 --batch 8 --epochs 40 --patience 8 \
      --workers 0 --warmup 3 --lr 4e-3 --init-scale "$scale" \
      >> /tmp/sf-sat/yolo-${fmt}.log 2>&1 || true
    # Ultralytics writes results.csv; copy a json summary if present
    csv="$DOMAIN_OUT/yolo_runs/yolo11n-seg_${fmt}/results.csv"
    if [[ -f "$csv" ]]; then
      "$PY" - "$csv" "$out" "$fmt" <<'PY'
import csv, json, sys
rows=list(csv.DictReader(open(sys.argv[1])))
last=rows[-1] if rows else {}
json.dump({"exp":"yolo_seg_scratch","format":sys.argv[3],"n":len(rows),"last":last},
          open(sys.argv[2],"w"), indent=2)
print("wrote", sys.argv[2])
PY
    fi
  done
  echo GPU_YOLO_DONE
) >> /tmp/sf-sat/yolo.log 2>&1 &
echo YOLO_PID=$!

echo "=== CPU: TinyBoxSeg SF2-16, 4-wide ==="
cpu_one() {
  bits="$1"
  tag=fp32
  [[ "$bits" -gt 0 ]] && tag="sf${bits}"
  out="$DOMAIN_OUT/sam_scratch_${tag}.json"
  [[ -f "$out" ]] && echo "skip $tag" && return 0
  echo "cpu sam_scratch bits=$bits"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
    "$PY" -u "$HERE/sam_scratch.py" --bits "$bits" --device cpu --batch 16 \
    --epochs 40 --patience 8 >> "/tmp/sf-sat/scratch-b${bits}.log" 2>&1 || true
}
export -f cpu_one
export PY HERE DOMAIN_OUT
printf '%s\n' 0 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 \
  | xargs -P 3 -n 1 bash -c 'cpu_one "$0"'
echo CPU_SCRATCH_DONE
wait || true
echo MAC_SATURATE_DONE
