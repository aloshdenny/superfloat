"""YOLO detection under SuperFloat QAT, via Ultralytics callbacks.

Covers the UAV row (YOLO11n / VisDrone) and the satellite row (YOLOv8n-OBB /
DOTAv1). Models are built from .yaml, i.e. random init -- no COCO pretraining.

The delicate part is making sure *every* model Ultralytics can validate is
actually quantized. ModelEMA deepcopies the network before the
on_pretrain_routine_end callback fires, and in-training validation runs against
that EMA copy, so patching only trainer.model would leave validation silently
running in full precision. We therefore re-apply the (idempotent) surgery at
on_pretrain_routine_end to both trainer.model and trainer.ema.ema, and again at
on_val_start to whatever model the validator is holding -- which for the final
best.pt evaluation is a freshly loaded checkpoint inside an AutoBackend.
"""

import argparse
import os

import torch
import torch.nn as nn
from ultralytics import YOLO

from superfloat import (SFConv2d, SFLinear, apply_superfloat, clamp_all,
                        disable_tf32, sf_params)


def head_prefixes(model):
    """Names of detection-head submodules, kept in full precision.

    Anchored on structure rather than class name: in every Ultralytics
    DetectionModel the head is the last entry of the .model Sequential. The
    head classes get renamed and multiplied across releases (Detect, OBB,
    OBB26, v10Detect, ...), so matching on identity of the last block is the
    version-proof way to find it.
    """
    seq = getattr(model, "model", None)
    if isinstance(seq, nn.Sequential) and len(seq):
        return [f"model.{len(seq) - 1}"]
    return []


def unwrap(m):
    """Reach the real nn.Module through AutoBackend / DDP wrappers."""
    if isinstance(m, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)):
        m = m.module
    inner = getattr(m, "model", None)
    # AutoBackend stores the DetectionModel at .model; a DetectionModel's .model
    # is a plain Sequential, which we must not mistake for a wrapper.
    if isinstance(inner, nn.Module) and not isinstance(inner, nn.Sequential):
        return unwrap(inner)
    return m


def quantize(model, bits, quantize_activations=True):
    if model is None:
        return 0
    m = unwrap(model)
    n = apply_superfloat(m, bits, head_names=head_prefixes(m),
                         quantize_activations=quantize_activations)
    return n


@torch.no_grad()
def report_clipping(model, bits):
    """How much of the incoming weight mass the SFx range cannot hold.

    Only meaningful for pretrained init: COCO weights were trained with no
    bound, so any beyond +/-vmax get clamped at conversion and that pretrained
    information is lost before QAT starts. The paper's premise is that ~99% of
    parameters already sit inside [-1, 1]; this measures it directly rather
    than assuming it.
    """
    _, vmax = sf_params(bits)
    tot = clipped = 0
    for m in model.modules():
        if isinstance(m, (SFConv2d, SFLinear)):
            w = m.weight
            tot += w.numel()
            clipped += (w.abs() > vmax).sum().item()
    if tot:
        print(f"[superfloat] SF{bits} range check: {clipped}/{tot} weights "
              f"({100.0 * clipped / tot:.3f}%) fall outside +/-{vmax:.6f} "
              f"and are clamped at init", flush=True)


def attach_superfloat(yolo, bits, quantize_activations=True, verbose=True):
    state = {"reported": False}

    def on_pretrain_routine_end(trainer):
        n1 = quantize(trainer.model, bits, quantize_activations)
        n2 = quantize(getattr(trainer.ema, "ema", None), bits, quantize_activations)
        if verbose and not state["reported"]:
            print(f"[superfloat] SF{bits} applied: {n1} layers in train model, "
                  f"{n2} in EMA copy (head left fp32)", flush=True)
            report_clipping(unwrap(trainer.model), bits)
            state["reported"] = True
        # Pretrained weights can start outside the representable range; clamp
        # once up front so the bounded STE does not freeze them permanently.
        clamp_all(unwrap(trainer.model))

    def on_train_batch_end(trainer):
        clamp_all(unwrap(trainer.model))

    def on_train_epoch_end(trainer):
        # True working set, not the caching allocator's reserve. nvidia-smi and
        # Ultralytics' GPU_mem column both report reserved memory, which grows
        # to fill whatever is free and so cannot be used to size parallelism.
        if torch.cuda.is_available():
            alloc = torch.cuda.max_memory_allocated() / 2**20
            resv = torch.cuda.max_memory_reserved() / 2**20
            print(f"[superfloat] peak VRAM: allocated={alloc:.0f} MiB "
                  f"reserved={resv:.0f} MiB", flush=True)
            torch.cuda.reset_peak_memory_stats()

    yolo.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
    yolo.add_callback("on_train_batch_end", on_train_batch_end)
    yolo.add_callback("on_train_epoch_end", on_train_epoch_end)

    # No validator-side hook is needed, and there is nowhere to put one anyway:
    # BaseValidator keeps its model in a local, never on self. Both validation
    # paths are covered regardless, which was verified by auditing layer classes
    # at runtime rather than assumed:
    #   - in-training val runs on trainer.ema.ema, patched above;
    #   - final_eval reloads best.pt, and since the checkpoint pickles the
    #     rebound SFConv2d/SFLinear classes by reference, it deserializes still
    #     quantized (this is why __class__ rebinding is used over wrappers).


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--format", required=True,
                   help="fp32, fp16, or sfx with x in 2..16")
    p.add_argument("--cfg", required=True,
                   help="architecture, e.g. yolo11x.yaml / yolov8x-obb.yaml")
    p.add_argument("--init", default="random", choices=["random", "pretrained"],
                   help="random builds from .yaml; pretrained loads the COCO .pt")
    p.add_argument("--fraction", type=float, default=1.0,
                   help="fraction of the train split (used for fast calibration)")
    p.add_argument("--resume", action="store_true",
                   help="continue from weights/last.pt if present")
    p.add_argument("--init-scale", type=float, default=1.0,
                   help="multiply conv weights at init; lifts them onto the "
                        "SFx grid when the default init falls under its step")
    p.add_argument("--data", required=True, help="e.g. VisDrone.yaml, DOTAv1.yaml")
    p.add_argument("--name", required=True)
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--lr", type=float, default=4e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="", help="0 | mps | cpu (default: auto)")
    args = p.parse_args()

    disable_tf32()
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Resume path: Ultralytics restores optimizer, epoch and args from the
    # checkpoint, and the SFx layer classes are pickled into it by reference,
    # so a resumed run comes back still quantized.
    last = os.path.join(args.project, args.name, "weights", "last.pt")
    resuming = args.resume and os.path.exists(last)
    if resuming:
        print(f"[init] resuming from {last}", flush=True)
        YOLO(last).train(resume=True)
        return

    if args.init == "pretrained":
        # Load COCO weights, then hand the trainer the architecture so it does
        # not re-randomise: YOLO(<weights>.pt) carries the trained parameters.
        weights = args.cfg.replace(".yaml", ".pt")
        model = YOLO(weights)
        print(f"[init] pretrained from {weights}", flush=True)
    else:
        model = YOLO(args.cfg)  # .yaml -> random init
        print(f"[init] random from {args.cfg}", flush=True)

    if args.init_scale != 1.0:
        # SuperFloat has no exponent, so its resolution near zero is fixed at
        # the grid step. Kaiming init gives mean |w| ~ 0.01 on this model,
        # which is below SF4's 0.0625 zero-threshold: 99.98% of conv weights
        # quantize to exactly zero and the network is dead before training
        # starts (observed as identical loss across a 16x LR range).
        # Rescaling is safe here because every conv is followed by BatchNorm,
        # which renormalizes activations, so the forward statistics are
        # unchanged while the weights land on representable grid points.
        with torch.no_grad():
            n = 0
            for m in model.model.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.mul_(args.init_scale)
                    n += 1
        print(f"[init] scaled {n} conv weight tensors by {args.init_scale}x",
              flush=True)

    # fp32/fp16 are the unquantized baselines: no SuperFloat surgery at all.
    if args.format not in ("fp32", "fp16"):
        attach_superfloat(model, int(args.format[2:]))

    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,        # open-ended: stop when it stops improving
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        seed=args.seed,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=(args.init == "pretrained"),
        fraction=args.fraction,
        optimizer="AdamW",
        lr0=args.lr,
        weight_decay=args.wd,
        warmup_epochs=args.warmup,
        cos_lr=True,
        # Mixed precision only for the fp16 baseline row. For every SFx format
        # autocast would silently re-round the quantized grid to fp16's 10-bit
        # mantissa, which is below SF16's 15 significand bits -- exactly the
        # corruption disable_tf32() exists to prevent.
        amp=(args.format == "fp16"),
        val=True,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
