"""Domain-vision PTQ: known-class segmentation (YOLO11n-seg) and promptable
unknown objects (SAM ViT-B, box prompt). Weights-only SF, head left in fp32.

YOLO reports mask mAP on coco128-seg. SAM reports mean IoU against the same
model's fp32 masks on those images — that is the PTQ fidelity question, not
a COCO-trained SAM leaderboard number.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superfloat import apply_superfloat, disable_tf32

_HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("DOMAIN_OUT", _HERE.parent / "results" / "domain"))


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def head_prefixes(model):
    seq = getattr(model, "model", None)
    if isinstance(seq, nn.Sequential) and len(seq):
        return [f"model.{len(seq) - 1}"]
    return []


def unwrap(m):
    if isinstance(m, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)):
        m = m.module
    inner = getattr(m, "model", None)
    if isinstance(inner, nn.Module) and not isinstance(inner, nn.Sequential):
        return unwrap(inner)
    return m


def yolo_seg(bits, device, data, weights, imgsz):
    from ultralytics import YOLO
    disable_tf32()
    m = YOLO(weights)
    if bits:
        n = apply_superfloat(unwrap(m.model), bits, head_names=head_prefixes(unwrap(m.model)),
                             quantize_activations=False)
        print(f"yolo quantized {n} conv/linear layers bits={bits}", flush=True)
    t0 = time.time()
    r = m.val(data=data, imgsz=imgsz, device=device, plots=False, verbose=False)
    rec = dict(exp="yolo_seg", weights=weights, data=data, bits=bits,
               device=device, seconds=time.time() - t0)
    # Ultralytics SegMetrics: box and mask
    for k in ("map", "map50", "map75"):
        if hasattr(r.box, k):
            rec[f"box_{k}"] = float(getattr(r.box, k))
        if hasattr(r.seg, k):
            rec[f"mask_{k}"] = float(getattr(r.seg, k))
    return rec


def _box_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        h, w = mask.shape
        return [0, 0, w - 1, h - 1]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def sam_ptq(bits, device, n_images, data_root):
    """Box-prompt SAM. Teacher = fp32 masks. Student = SF-quantized image encoder."""
    images = sorted(Path(data_root).rglob("*.jpg"))[:n_images]
    if not images:
        raise SystemExit(f"no jpg under {data_root}")
    if not bits:
        # Identity: student is the teacher. Skip the dual load (16 GB M4 jetsams it)
        # and skip the HF download entirely.
        return dict(exp="sam_box", model="facebook/sam-vit-base", bits=0,
                    device=device, n=len(images), mean_iou=1.0, p50_iou=1.0,
                    quantized=0, seconds=0.0, note="bf16 identity")

    from PIL import Image
    from transformers import SamModel, SamProcessor

    disable_tf32()
    proc = SamProcessor.from_pretrained("facebook/sam-vit-base")
    teacher = SamModel.from_pretrained("facebook/sam-vit-base").eval()
    student = SamModel.from_pretrained("facebook/sam-vit-base").eval()
    # image encoder is the ViT; mask decoder stays fp32 (the "head")
    n = apply_superfloat(student.vision_encoder, bits, quantize_activations=False)
    print(f"sam vision_encoder quantized {n} layers bits={bits}", flush=True)

    dev = torch.device(device)
    # Two ViT-B copies on MPS jetsam 16 GB unified memory. Teacher stays on CPU.
    teacher = teacher.cpu()
    student = student.to(dev)

    ious, t0 = [], time.time()
    for p in images:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        # one full-frame box: "segment whatever is in view" — unknown object
        boxes = [[[0, 0, w - 1, h - 1]]]
        inp = proc(im, input_boxes=boxes, return_tensors="pt")
        t_inp = {k: v.cpu() if torch.is_tensor(v) else v for k, v in inp.items()}
        s_inp = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in inp.items()}
        with torch.no_grad():
            t_out = teacher(**t_inp, multimask_output=False)
            s_out = student(**s_inp, multimask_output=False)
        t_mask = proc.post_process_masks(t_out.pred_masks.cpu(),
                                         t_inp["original_sizes"].cpu(),
                                         t_inp["reshaped_input_sizes"].cpu())[0][0, 0]
        s_mask = proc.post_process_masks(s_out.pred_masks.cpu(),
                                         s_inp["original_sizes"].cpu(),
                                         s_inp["reshaped_input_sizes"].cpu())[0][0, 0]
        a = t_mask.bool().numpy()
        b = s_mask.bool().numpy()
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        ious.append(float(inter / union) if union else 1.0)
    rec = dict(exp="sam_box", model="facebook/sam-vit-base", bits=bits,
               device=device, n=len(images), mean_iou=float(np.mean(ious)),
               p50_iou=float(np.median(ious)), quantized=n,
               seconds=time.time() - t0)
    return rec


def append(rec):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{rec['exp']}_b{rec.get('bits', 0)}.json"
    json.dump(rec, open(path, "w"), indent=2)
    print(json.dumps({k: rec[k] for k in rec if k != "ious"}), flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["yolo", "sam", "both"], default="both")
    ap.add_argument("--bits", type=int, default=0)
    ap.add_argument("--yolo-weights", default="yolo11n-seg.pt")
    ap.add_argument("--data", default="coco128-seg.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--sam-n", type=int, default=16)
    ap.add_argument("--sam-images", default="")
    ap.add_argument("--device", default="", help="cuda | mps | cpu (default: auto)")
    a = ap.parse_args()
    device = a.device or pick_device()
    print(f"device={device}", flush=True)
    if a.task in ("yolo", "both"):
        rec = yolo_seg(a.bits, device, a.data, a.yolo_weights, a.imgsz)
        append(rec)
        img_root = Path("datasets/coco128-seg/images")
        if not a.sam_images and img_root.exists():
            a.sam_images = str(img_root)
    if a.task in ("sam", "both"):
        root = a.sam_images or "datasets/coco128-seg/images"
        rec = sam_ptq(a.bits, device, a.sam_n, root)
        append(rec)


if __name__ == "__main__":
    main()
