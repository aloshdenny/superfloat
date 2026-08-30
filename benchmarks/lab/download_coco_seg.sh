#!/usr/bin/env bash
# Download COCO 2017 train/val images + YOLO-seg labels (~20 GB) into $COCO_ROOT.
set -euo pipefail
COCO_ROOT="${COCO_ROOT:-/Users/aoxo/sfx_data/coco}"
PY="${PYTHON:-/opt/homebrew/Caskroom/miniconda/base/envs/sf-scaling-laws/bin/python}"
mkdir -p "$COCO_ROOT/images" "$COCO_ROOT/labels"
"$PY" - "$COCO_ROOT" <<'PY'
import sys
from pathlib import Path
from ultralytics.utils import ASSETS_URL
from ultralytics.utils.downloads import download

root = Path(sys.argv[1])
img = root / "images"
# labels zip extracts a `coco/` folder next to images if we download into parent
parent = root.parent
print("ASSETS_URL", ASSETS_URL, flush=True)
print("labels…", flush=True)
download([ASSETS_URL + "/coco2017labels-segments.zip"], dir=parent)
print("images (19G train + 1G val)…", flush=True)
download(
    [
        "http://images.cocodataset.org/zips/train2017.zip",
        "http://images.cocodataset.org/zips/val2017.zip",
    ],
    dir=img,
    threads=2,
)
print("COCO_DOWNLOAD_DONE", flush=True)
PY
# ultralytics labels zip is `coco/labels/{train,val}2017` under parent
if [[ -d "$COCO_ROOT/labels/train2017" ]]; then
  echo labels_ok
elif [[ -d "$COCO_ROOT/coco/labels/train2017" ]]; then
  mv "$COCO_ROOT/coco/labels" "$COCO_ROOT/labels"
  rmdir "$COCO_ROOT/coco" 2>/dev/null || true
  echo labels_moved
elif [[ -d "$(dirname "$COCO_ROOT")/coco/labels/train2017" ]]; then
  mv "$(dirname "$COCO_ROOT")/coco/labels" "$COCO_ROOT/labels"
  rmdir "$(dirname "$COCO_ROOT")/coco" 2>/dev/null || true
  echo labels_moved
fi
ls "$COCO_ROOT/images/train2017" | wc -l
ls "$COCO_ROOT/labels/train2017" 2>/dev/null | wc -l
touch "$COCO_ROOT/.ready"
echo COCO_READY
