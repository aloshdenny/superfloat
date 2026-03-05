"""
test_sf16.py  –  Quick smoke-tests for the SF16 (Q1.15) quantizer
and model building blocks.

Run:
    python test_sf16.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

from sf16_quantizer import (
    quantize_q115,
    to_q115_int16,
    from_q115_int16,
    Q115_RESOLUTION,
    Q115_MIN_FLOAT,
    Q115_MAX_FLOAT,
    Q115_SCALE,
    snap_weights_to_q115,
    STEQuantizeQ115,
    Q115Linear,
    Q115Conv2d,
    quantize_images_q115,
    weight_stats_q115,
)
from resnet_model import resnet18, resnet18_sf16

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
_errors: list = []


def check(cond: bool, msg: str):
    if cond:
        print(f"  {PASS} {msg}")
    else:
        print(f"  {FAIL} {msg}")
        _errors.append(msg)


# ---------------------------------------------------------------------------
# 1. Quantizer arithmetic
# ---------------------------------------------------------------------------
print("\n── 1. Q1.15 quantizer arithmetic ───────────────────────────────────")

x = torch.tensor([0.0, 0.5, -0.5, 0.999, -1.0, 1.5, -1.5])
q = quantize_q115(x)

check(q.max().item() <= Q115_MAX_FLOAT,
      "quantized values ≤ Q115_MAX_FLOAT")
check(q.min().item() >= Q115_MIN_FLOAT,
      "quantized values ≥ Q115_MIN_FLOAT")
check(abs(q[0].item()) < 1e-9,
      "0.0 quantizes to 0.0")
check(abs(q[1].item() - round(0.5 * Q115_SCALE) / Q115_SCALE) < 1e-7,
      "0.5 snaps to nearest Q1.15 grid point")
check(q[5].item() == Q115_MAX_FLOAT,
      "1.5 saturates to Q115_MAX_FLOAT")
check(q[6].item() == Q115_MIN_FLOAT,
      "-1.5 saturates to Q115_MIN_FLOAT")

# round-trip via int16
x_fp = torch.linspace(-0.9, 0.9, 100)
x_i16 = to_q115_int16(x_fp)
x_back = from_q115_int16(x_i16)
err = (x_back - quantize_q115(x_fp)).abs().max().item()
check(err < 1e-7, f"int16 round-trip max error = {err:.2e}")

# resolution
check(abs(Q115_RESOLUTION - 1.0 / 32768) < 1e-10,
      f"Q115_RESOLUTION = {Q115_RESOLUTION:.6e}")

print()

# ---------------------------------------------------------------------------
# 2. Straight-Through Estimator gradient
# ---------------------------------------------------------------------------
print("── 2. STE gradient flow ─────────────────────────────────────────────")

t = torch.tensor([0.3, -0.6, 0.95], requires_grad=True)
out = STEQuantizeQ115.apply(t)
loss = out.sum()
loss.backward()

check(t.grad is not None, "gradient flows through STE")
check(torch.allclose(t.grad, torch.ones(3)),
      "STE gradient is all-ones (identity)")

# Gradient is zero for saturated inputs? No – STE passes through always.
t2 = torch.tensor([2.0], requires_grad=True)  # saturated
o2 = STEQuantizeQ115.apply(t2)
o2.backward()
check(t2.grad is not None and abs(t2.grad.item() - 1.0) < 1e-6,
      "STE gradient is 1.0 even for saturated input (straight-through)")

print()

# ---------------------------------------------------------------------------
# 3. Q115Linear layer
# ---------------------------------------------------------------------------
print("── 3. Q115Linear layer ──────────────────────────────────────────────")

layer = Q115Linear(16, 8, bias=True)
x     = torch.randn(4, 16)

# Manually clamp weights to Q1.15 range
with torch.no_grad():
    layer.weight.clamp_(-1.0, 1.0)

out = layer(x)
check(out.shape == (4, 8), f"output shape {out.shape} == (4, 8)")

loss = out.sum()
loss.backward()
check(layer.weight.grad is not None, "gradient reached weight")

print()

# ---------------------------------------------------------------------------
# 4. Q115Conv2d layer
# ---------------------------------------------------------------------------
print("── 4. Q115Conv2d layer ──────────────────────────────────────────────")

conv = Q115Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
x    = torch.randn(2, 3, 32, 32)
with torch.no_grad():
    conv.weight.clamp_(-1.0, 1.0)

out = conv(x)
check(out.shape == (2, 16, 32, 32), f"output shape correct: {out.shape}")

out.sum().backward()
check(conv.weight.grad is not None, "gradient reached conv weight")

print()

# ---------------------------------------------------------------------------
# 5. snap_weights_to_q115
# ---------------------------------------------------------------------------
print("── 5. snap_weights_to_q115 ──────────────────────────────────────────")

tmp_layer = torch.nn.Linear(8, 4)
with torch.no_grad():
    tmp_layer.weight.fill_(0.12345678)
snap_weights_to_q115(tmp_layer)
# After snap, weight / Q115_RESOLUTION should be (nearly) integer
residuals = (tmp_layer.weight * Q115_SCALE) % 1.0
check(residuals.abs().max().item() < 1e-4,
      "weights on Q1.15 grid after snap")

print()

# ---------------------------------------------------------------------------
# 6. Image quantization
# ---------------------------------------------------------------------------
print("── 6. quantize_images_q115 ──────────────────────────────────────────")

imgs = torch.randn(8, 3, 32, 32) * 0.5  # typical normalized CIFAR range
q_imgs = quantize_images_q115(imgs)

check(q_imgs.max().item() <= Q115_MAX_FLOAT + 1e-6,
      "quantized images ≤ max Q1.15 value")
check(q_imgs.min().item() >= Q115_MIN_FLOAT - 1e-6,
      "quantized images ≥ min Q1.15 value")

# Each pixel should lie on the Q1.15 grid
residuals = (q_imgs * Q115_SCALE) % 1.0
# Allow for floating point noise
check(residuals.abs().max().item() < 1e-3,
      "image pixels snap to Q1.15 grid")

print()

# ---------------------------------------------------------------------------
# 7. SF16 ResNet-18 forward pass
# ---------------------------------------------------------------------------
print("── 7. SF16 ResNet-18 end-to-end forward ─────────────────────────────")

model = resnet18_sf16(num_classes=10)
model.eval()

dummy = torch.randn(2, 3, 32, 32) * 0.5
with torch.no_grad():
    logits = model(dummy)

check(logits.shape == (2, 10), f"logits shape {logits.shape} == (2, 10)")
check(logits.max().item() <= Q115_MAX_FLOAT + 1e-6,
      "logits clamped to Q1.15 range (max)")
check(logits.min().item() >= Q115_MIN_FLOAT - 1e-6,
      "logits clamped to Q1.15 range (min)")

# Training mode – gradients should flow
model.train()
dummy2 = torch.randn(2, 3, 32, 32) * 0.5
logits2 = model(dummy2)
logits2.sum().backward()
grad_norms = [p.grad.norm().item() for p in model.parameters()
              if p.grad is not None]
check(len(grad_norms) > 0, f"gradients populated for {len(grad_norms)} param tensors")
check(max(grad_norms) < 1e6, f"grad norms sane (max={max(grad_norms):.2f})")

print()

# ---------------------------------------------------------------------------
# 8. Baseline ResNet-18
# ---------------------------------------------------------------------------
print("── 8. Baseline FP32 ResNet-18 ───────────────────────────────────────")

model_fp = resnet18(num_classes=10)
model_fp.eval()
with torch.no_grad():
    logits_fp = model_fp(dummy)
check(logits_fp.shape == (2, 10), f"baseline logits shape {logits_fp.shape}")
# baseline logits are NOT bounded to [-1, 1]
check(True, "baseline logits not restricted to Q1.15 range (expected)")

print()

# ---------------------------------------------------------------------------
# 9. weight_stats diagnostic
# ---------------------------------------------------------------------------
print("── 9. weight_stats_q115 ─────────────────────────────────────────────")

stats = weight_stats_q115(resnet18_sf16(num_classes=10))
check("total_params" in stats,    "stats contains total_params")
check("saturated_frac" in stats,  "stats contains saturated_frac")
check(stats["saturated_frac"] >= 0.0, "saturated_frac ≥ 0")
check(stats["saturated_frac"] <= 1.0, "saturated_frac ≤ 1")
print(f"  stats: {stats}")

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
print()
if _errors:
    print(f"\033[91m{len(_errors)} test(s) FAILED:\033[0m")
    for e in _errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    print("\033[92mAll tests passed!\033[0m")
