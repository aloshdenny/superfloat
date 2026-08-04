"""SuperFloat on V-JEPA 2: does the format preserve a learned representation?

Every other benchmark here trains a supervised CNN. V-JEPA 2 is a ViT trained
with a self-supervised joint-embedding objective on video, normalised with
LayerNorm rather than BatchNorm -- a family sharing essentially nothing with
ResNet/ConvNeXt/YOLO. If SuperFloat behaves the same way here, architecture
agnosticity is demonstrated rather than asserted.

Protocol is V-JEPA's own: freeze the backbone, quantize it to SFx, and train
only an attentive probe on the frozen features. This is deliberately not a
fine-tune. SuperFloat is scoped by the paper as an inference-time deployment
format, so the question that matters is how much of a *trained* representation
survives quantization -- not whether the backbone can be retrained through the
grid, which the detection sweep already covers.

Features are extracted once per format and cached, so the probe sweep is cheap:
the expensive ViT-L forward pass happens a single time per numeric format.
"""

import argparse
import csv
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from superfloat import apply_superfloat, disable_tf32, sf_params

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"


class AttentiveProbe(nn.Module):
    """Cross-attention pooling + linear classifier, in full precision.

    Mirrors V-JEPA's attentive-probe evaluation head. Kept unquantized for the
    same reason detection heads are: the paper's recipe keeps output logits in
    full precision, so the probe measures representation quality rather than
    head quantization.
    """

    def __init__(self, dim, num_classes, heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, tokens):                      # tokens: (B, N, D)
        q = self.query.expand(tokens.size(0), -1, -1)
        pooled, _ = self.attn(q, tokens, tokens)
        return self.fc(self.norm(pooled.squeeze(1)))


@torch.no_grad()
def extract_features(backbone, loader, device, pool_tokens=256):
    """Run the (quantized) frozen backbone once and cache pooled tokens."""
    backbone.eval()
    feats, labels = [], []
    for i, (clips, y) in enumerate(loader):
        clips = clips.to(device, non_blocking=True)
        out = backbone(pixel_values_videos=clips).last_hidden_state  # (B,N,D)
        # Adaptive-pool the token axis so the probe sees a fixed length and the
        # cache stays small; D is preserved.
        if out.size(1) > pool_tokens:
            out = F.adaptive_avg_pool1d(out.transpose(1, 2),
                                        pool_tokens).transpose(1, 2)
        feats.append(out.float().cpu())
        labels.append(y)
        if i % 20 == 0:
            print(f"  extracted {i * loader.batch_size} clips", flush=True)
    return torch.cat(feats), torch.cat(labels)


def run_probe(train_f, train_y, val_f, val_y, num_classes, device,
              epochs, lr, log_path, tag):
    dim = train_f.size(-1)
    probe = AttentiveProbe(dim, num_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tr = torch.utils.data.TensorDataset(train_f, train_y)
    va = torch.utils.data.TensorDataset(val_f, val_y)
    trl = torch.utils.data.DataLoader(tr, batch_size=64, shuffle=True)
    val = torch.utils.data.DataLoader(va, batch_size=64)

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc",
                                "val_loss", "val_acc", "lr", "secs"])

    best = 0.0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        probe.train()
        tl = tc = tn = 0
        for x, y in trl:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(probe(x), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                tl += loss.item() * y.size(0)
                tc += probe(x).argmax(1).eq(y).sum().item()
                tn += y.size(0)
        probe.eval()
        vl = vc = vn = 0
        with torch.no_grad():
            for x, y in val:
                x, y = x.to(device), y.to(device)
                o = probe(x)
                vl += F.cross_entropy(o, y).item() * y.size(0)
                vc += o.argmax(1).eq(y).sum().item()
                vn += y.size(0)
        sched.step()
        va_acc = 100.0 * vc / vn
        best = max(best, va_acc)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{tl/tn:.4f}", f"{100.0*tc/tn:.3f}",
                                    f"{vl/vn:.4f}", f"{va_acc:.3f}",
                                    f"{opt.param_groups[0]['lr']:.2e}",
                                    f"{time.time()-t0:.1f}"])
        print(f"[{tag}] ep {ep:3d} | train {tl/tn:.3f}/{100.0*tc/tn:5.2f}% | "
              f"val {vl/vn:.3f}/{va_acc:5.2f}% | best {best:5.2f}%", flush=True)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--format", required=True,
                   choices=["fp32", "sf16", "sf8", "sf4"])
    p.add_argument("--data", required=True, help="UCF101 root (class subdirs)")
    p.add_argument("--out", default="runs/vjepa")
    p.add_argument("--classes", type=int, default=101)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    disable_tf32()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    tag = f"vjepa2_{args.format}_s{args.seed}"

    from transformers import AutoModel
    from video_data import build_ucf101_loaders          # local helper

    backbone = AutoModel.from_pretrained(MODEL_ID)

    n_q = 0
    if args.format != "fp32":
        bits = int(args.format[2:])
        n_q = apply_superfloat(backbone, bits)
        scale, vmax = sf_params(bits)
        # How much of the pretrained representation the grid cannot hold.
        tot = clipped = zeroed = 0
        with torch.no_grad():
            for m in backbone.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                    w = m.weight
                    tot += w.numel()
                    clipped += (w.abs() > vmax).sum().item()
                    q = torch.round(torch.clamp(w, -vmax, vmax) * scale) / scale
                    zeroed += (q == 0).sum().item()
        print(f"[superfloat] SF{bits}: {n_q} layers quantized | "
              f"{100.0*clipped/tot:.5f}% outside +/-{vmax:.4f} | "
              f"{100.0*zeroed/tot:.2f}% quantized to zero", flush=True)

    backbone.to(device).eval()
    for prm in backbone.parameters():
        prm.requires_grad_(False)

    train_ld, val_ld, ncls = build_ucf101_loaders(
        args.data, args.frames, args.size, args.batch, args.classes, args.seed)
    print(f"[{tag}] classes={ncls} train={len(train_ld.dataset)} "
          f"val={len(val_ld.dataset)}", flush=True)

    t0 = time.time()
    trf, trY = extract_features(backbone, train_ld, device)
    vaf, vaY = extract_features(backbone, val_ld, device)
    print(f"[{tag}] features {tuple(trf.shape)} in {time.time()-t0:.0f}s",
          flush=True)

    best = run_probe(trf, trY, vaf, vaY, ncls, device, args.epochs, args.lr,
                     os.path.join(args.out, f"{tag}.csv"), tag)

    json.dump({"tag": tag, "format": args.format, "model": "vjepa2-vitl",
               "dataset": "ucf101", "classes": ncls, "quantized_layers": n_q,
               "best_val_acc": best, "seed": args.seed},
              open(os.path.join(args.out, f"{tag}.json"), "w"), indent=2)
    print(f"[{tag}] DONE best_val_acc={best:.2f}", flush=True)


if __name__ == "__main__":
    main()
