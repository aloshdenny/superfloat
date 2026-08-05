"""SuperFloat QAT on V-JEPA 2 — end-to-end, backbone unfrozen.

Companion to `train_vjepa_probe.py`, which is post-training quantization: there
the pretrained FP32 backbone is quantized once and frozen, and only an fp32
probe is trained. That measures how much of a *trained* representation survives
quantization, which matches the paper's framing of SFx as a deployment format.

This script asks the other question. The backbone is quantized and then
*trained through* -- gradients flow across the SFx grid via the bounded STE,
exactly as in the ConvNeXt and YOLO benchmarks. It exists to test a specific
prediction from the PTQ result:

    Under PTQ, quantizing activations cost V-JEPA ~48 points while bit-width
    made almost no difference (SF16 48.8, SF8 51.0, SF4 47.9). That points at
    the [-1, 1] activation bound rather than significand width: BatchNorm CNNs
    keep activations near unit scale, but a ViT residual stream accumulates
    across 24 blocks and exceeds it.

    If that is right, QAT should partially recover -- the network can learn to
    keep its residual stream inside the representable range, which a frozen
    network cannot. If QAT does *not* recover, the bound is incompatible with
    the architecture rather than merely unadapted.

Head stays fp32, as everywhere else in this suite.
"""

import argparse
import csv
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from superfloat import apply_superfloat, clamp_all, disable_tf32
from train_vjepa_probe import MODEL_ID, AttentiveProbe


class VJepaClassifier(nn.Module):
    """Quantized V-JEPA backbone + full-precision attentive head."""

    def __init__(self, backbone, dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = AttentiveProbe(dim, num_classes)

    def forward(self, clips):
        tokens = self.backbone(pixel_values_videos=clips).last_hidden_state
        return self.head(tokens)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--format", required=True,
                   choices=["fp32", "sf16", "sf8", "sf4"])
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="runs/vjepa_qat")
    p.add_argument("--classes", type=int, default=25)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=8,
                   help="gradient accumulation; effective batch = batch*accum")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="BACKBONE rate. A pretrained 300M ViT-L needs ~1e-5; "
                        "at 1e-4 the fp32 control collapsed to chance (5.12%% "
                        "on 25 classes) within 15 epochs, destroying features "
                        "that score 97%% under a frozen probe.")
    p.add_argument("--head-lr", type=float, default=1e-3,
                   help="the head is randomly initialised and needs a much "
                        "larger rate than the pretrained backbone")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wd", type=float, default=0.05,
                   help="Backbone weight decay. Set 0 for SF4: decay pulls "
                        "weights toward zero, and anything dropping below "
                        "SF4's 0.0625 floor is annihilated permanently since "
                        "the bounded STE then gives it no gradient. SF4-QAT "
                        "starts at 12.2%% train accuracy and decays to ~3%%, "
                        "while SF16 under the identical config climbs.")
    p.add_argument("--no-act-quant", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    disable_tf32()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    tag = f"vjepa2qat_{args.format}{'_wonly' if args.no_act_quant else ''}"
    if args.wd != 0.05:
        tag += f"_wd{args.wd:g}"
    tag += f"_s{args.seed}"

    from transformers import AutoModel
    from video_data import build_ucf101_loaders

    backbone = AutoModel.from_pretrained(MODEL_ID)
    n_q = 0
    if args.format != "fp32":
        bits = int(args.format[2:])
        n_q = apply_superfloat(backbone, bits,
                               quantize_activations=not args.no_act_quant)
        print(f"[superfloat] SF{bits} QAT: {n_q} layers quantized "
              f"(activations {'off' if args.no_act_quant else 'on'})", flush=True)

    dim = backbone.config.hidden_size
    train_ld, val_ld, ncls = build_ucf101_loaders(
        args.data, args.frames, args.size, args.batch, args.classes, args.seed)

    model = VJepaClassifier(backbone, dim, ncls).to(device)
    model.backbone.gradient_checkpointing_enable()   # 300M params on video

    # Separate rates: the backbone is pretrained and must be nudged, the head
    # is random and must be learned. One shared rate cannot serve both.
    opt = torch.optim.AdamW(
        [{"params": model.backbone.parameters(), "lr": args.lr,
          "weight_decay": args.wd},
         {"params": model.head.parameters(), "lr": args.head_lr,
          "weight_decay": 0.05}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log_path = os.path.join(args.out, f"{tag}.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc",
                                "val_loss", "val_acc", "lr", "secs"])

    best = 0.0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tl = tc = tn = 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(train_ld):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss = F.cross_entropy(out, y, label_smoothing=0.1) / args.accum
            loss.backward()
            if (i + 1) % args.accum == 0:
                # Clip the two groups separately. Clipping model.parameters()
                # to norm 1.0 lets the 300M-parameter backbone dominate the
                # global norm and scales the randomly-initialised head's
                # gradient to near nothing -- the fp32 control then sits at
                # chance (7.5% on 25 classes) with train accuracy equally flat.
                torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), 5.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                # Keep shadow weights representable after every update.
                if args.format != "fp32":
                    clamp_all(model.backbone)
            tl += loss.item() * args.accum * y.size(0)
            tc += out.argmax(1).eq(y).sum().item()
            tn += y.size(0)
            if i % 100 == 0:
                print(f"  ep{ep} step {i}/{len(train_ld)} "
                      f"loss={loss.item()*args.accum:.3f}", flush=True)

        model.eval()
        vl = vc = vn = 0
        with torch.no_grad():
            for x, y in val_ld:
                x, y = x.to(device), y.to(device)
                o = model(x)
                vl += F.cross_entropy(o, y).item() * y.size(0)
                vc += o.argmax(1).eq(y).sum().item()
                vn += y.size(0)
        sched.step()
        va = 100.0 * vc / vn
        best = max(best, va)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{tl/tn:.4f}", f"{100.0*tc/tn:.3f}",
                                    f"{vl/vn:.4f}", f"{va:.3f}",
                                    f"{opt.param_groups[0]['lr']:.2e}",
                                    f"{time.time()-t0:.1f}"])
        print(f"[{tag}] ep {ep:3d} | train {tl/tn:.3f}/{100.0*tc/tn:5.2f}% | "
              f"val {vl/vn:.3f}/{va:5.2f}% | best {best:5.2f}%", flush=True)
        json.dump({"tag": tag, "format": args.format, "regime": "qat",
                   "model": "vjepa2-vitl", "dataset": "ucf101",
                   "classes": ncls, "quantized_layers": n_q,
                   "epochs_run": ep, "best_val_acc": best, "seed": args.seed},
                  open(os.path.join(args.out, f"{tag}.json"), "w"), indent=2)

    print(f"[{tag}] DONE best_val_acc={best:.2f}", flush=True)


if __name__ == "__main__":
    main()
