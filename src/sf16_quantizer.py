"""
sf16_quantizer.py  –  Q1.15 (SF16) and Q1.31 quantization primitives for PyTorch

Q1.15 format:
──────────────────────────────────────────────────────────────────────────────
  Stored representation:  signed 16-bit integer (int16)
  Logical meaning:        int16_value / 2^15   ∈ [-1, 1)
  Resolution:             1 / 2^15  ≈  3.05e-5

Q1.31 format (accumulator):
──────────────────────────────────────────────────────────────────────────────
  Stored representation:  signed 32-bit integer (int32)
  Logical meaning:        int32_value / 2^31  ∈ [-1, 1)
  Resolution:             1 / 2^31  ≈  4.65e-10

  In real hardware, Q1.15 × Q1.15 = Q2.30 per product; summing N products
  into an int32 accumulator and then right-shifting by 1 gives Q1.31.
  We simulate this by: clamp(x, -1, 1), then round to the Q1.31 grid.

Block Floating Point (BFP) scaling:
──────────────────────────────────────────────────────────────────────────────
  Real activations after BatchNorm are ~N(0,1) and can reach ±3.  Storing
  them naively in Q1.15 (range [-1,1)) saturates 32% of values, collapsing
  the network.

  The solution (same as q115_common.cuh in superfloat.cuda) is per-tensor
  *scale factors*.  A scaled Q1.15 tensor with scale s represents the range
  [-s, s):

      stored int16  =  round(x / s * 32768)
      recovered x   =  int16 / 32768 * s

  In float simulation:  quantize_bfp(x, s) = quantize_q115(x/s) * s

  The hardware accumulator is always wider (int32 in fixed-point, float32
  in simulation) — even real Q1.15 chips do not accumulate in Q1.15.

Scale choices (mirroring q115_common.cuh):
──────────────────────────────────────────────────────────────────────────────
  Q115_ACT_SCALE   = 8.0   # post-BN+ReLU activations, covers ±3σ safely
  Q115_INPUT_SCALE = 3.0   # CIFAR images after standard normalization
  Q115_LOGIT_SCALE = 1.0   # final logits (kept small, cross-entropy is fine)

Training strategy:
──────────────────────────────────────────────────────────────────────────────
  FORWARD : inputs       → BFP Q1.15 (scale 3) once at network entry
            weights      → Q1.15 (scale 1) before every matmul
            accumulator  → Q1.31 (scale 1) after every Conv/Linear output
            activations  → BFP Q1.15 (scale 8) at each block boundary after BN+ReLU
            logits       → Q1.15 (scale 1) at network exit
  BACKWARD: STE passes gradients through all Q boundaries in FP32
  OPTIMIZER: AdamW on FP32 master weights; clamp to Q1.15 range after each step
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
Q115_SCALE      = 32768.0               # 2^15
Q115_MAX_FLOAT  =  32767.0 / Q115_SCALE # ≈  0.999969  (int16 max / 2^15)
Q115_MIN_FLOAT  = -32768.0 / Q115_SCALE # = -1.0       (int16 min / 2^15)
Q115_RESOLUTION =  1.0    / Q115_SCALE  # ≈  3.05e-5

# Q1.31 accumulator constants (simulates int32 accumulator output)
Q131_SCALE      = 2_147_483_648.0                   # 2^31
Q131_MAX_FLOAT  =  2_147_483_647.0 / Q131_SCALE     # ≈  1.0 - 2^-31
Q131_MIN_FLOAT  = -2_147_483_648.0 / Q131_SCALE     # = -1.0
Q131_RESOLUTION =  1.0             / Q131_SCALE     # ≈  4.65e-10

# Per-tensor Block Floating Point scale factors
# (mirror q115_common.cuh: Q115_FFN_SCALE, Q115_ATTENTION_SCALE, etc.)
Q115_ACT_SCALE   = 1.0   # Standard residual stream is Scale-1 (strictly [-1, 1))
Q115_INPUT_SCALE = 3.0   # CIFAR-10 images: ~[-2.5, 2.5]
Q115_LOGIT_SCALE = 24.0  # Final logits: match superfloat.cuda (allows confident softmax)


# ---------------------------------------------------------------------------
# Core BFP-aware quantize/dequantize  (float simulation of Q1.15 hardware)
# ---------------------------------------------------------------------------

def quantize_q115(x: torch.Tensor) -> torch.Tensor:
    """
    Quantize to Q1.15 grid, representable range [-1, 1).
    Values outside the range are saturated (clipped).
    """
    x = x.clamp(Q115_MIN_FLOAT, Q115_MAX_FLOAT)
    return (x * Q115_SCALE).round() / Q115_SCALE


def quantize_q131(x: torch.Tensor) -> torch.Tensor:
    """
    Quantize to Q1.31 grid, representable range [-1, 1).
    Simulates the int32 accumulator output of a Q1.15 × Q1.15 MAC array.
    Values outside [-1, 1) are saturated — BN must learn to keep outputs in range.
    Resolution is 2^-31 ≈ 4.65e-10 (vastly finer than Q1.15).
    """
    x = x.clamp(Q131_MIN_FLOAT, Q131_MAX_FLOAT)
    return (x * Q131_SCALE).round() / Q131_SCALE


def quantize_bfp(x: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Block Floating Point (BFP) Q1.15 quantization with per-tensor scale.

    Represents the range [-scale, scale) using Q1.15 format:
        stored   = round(x / scale * 32768) / 32768   [Q1.15 value in [-1,1)]
        logical  = stored * scale                      [float back in [-s, s)]

    This is exactly what q115_to_float_scaled / float_to_q115_scaled do in
    q115_common.cuh.
    """
    normalized = x / scale
    snapped    = quantize_q115(normalized)   # clamp + round to Q1.15 grid
    return snapped * scale                   # back to original scale


# ---------------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------------

def to_q115_int16(x: torch.Tensor) -> torch.Tensor:
    """Convert float tensor to int16 Q1.15 representation (scale=1)."""
    x_clamped = x.clamp(Q115_MIN_FLOAT, Q115_MAX_FLOAT)
    return (x_clamped * Q115_SCALE).round().to(torch.int16)


def from_q115_int16(x: torch.Tensor) -> torch.Tensor:
    """Convert int16 Q1.15 representation back to float (scale=1)."""
    return x.to(torch.float32) / Q115_SCALE


def quantize_images_q115(images: torch.Tensor,
                          scale: float = Q115_INPUT_SCALE) -> torch.Tensor:
    """
    Quantize input images to Q1.15 using BFP scaling.

    CIFAR-10 images normalized with standard mean/std span roughly [-2.5, 2.5].
    With scale=3.0, the Q1.15 representable range covers [-3, 3):
        • ~99.7% of pixels land inside the range (3σ coverage)
        • No meaningful information is destroyed
    """
    return quantize_bfp(images, scale)


def quantize_activations_q115(x: torch.Tensor,
                               scale: float = Q115_ACT_SCALE) -> torch.Tensor:
    """
    Quantize intermediate activations to Q1.15 using BFP scaling.

    Post-BN activations are ~N(0,1), after ReLU they are in [0, ~3].
    Residual connections can sum two such branches → [0, ~6].
    With scale=8.0, the representable range [-8, 8) covers all of this
    with <0.1% saturation.

    This is the equivalent of q115_to_float_scaled(x, Q115_FFN_SCALE) in
    q115_common.cuh.

    The multiply-accumulate inside Conv/Linear still uses float32
    (the hardware equivalent is an int32 accumulator, widened from Q1.15).
    """
    return quantize_bfp(x, scale)


def quantize_outputs_q115(logits: torch.Tensor,
                           scale: float = Q115_LOGIT_SCALE) -> torch.Tensor:
    """Quantize final output logits to Q1.15 (BFP with given scale)."""
    return quantize_bfp(logits, scale)


# ---------------------------------------------------------------------------
# Straight-Through Estimator autograd functions
# ---------------------------------------------------------------------------

class STEQuantizeQ115(torch.autograd.Function):
    """
    Forward : quantise to Q1.15 grid (float representation, scale=1).
    Backward: pass gradients straight through (identity / STE).
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return quantize_q115(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class STEQuantizeBFP(torch.autograd.Function):
    """
    Forward : BFP-scaled Q1.15 quantization (scale stored in ctx).
    Backward: straight-through.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(scale)
        s = scale.item()
        return quantize_bfp(x, s)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None   # STE for x; None for scale (not learned)


ste_quantize_q115 = STEQuantizeQ115.apply


def ste_quantize_q131(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Forward : quantize to Q1.31 grid (clamped to [-scale, scale)).
    Backward: straight-through (STE).
    Simulates the int32 accumulator truncation.
    """
    class STEQuantizeQ131(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor) -> torch.Tensor:
            # normalize by scale, quantize to Q1.31 unit range, then rescale
            norm = x / scale
            snapped = quantize_q131(norm)
            return snapped * scale

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
            return grad_output

    return STEQuantizeQ131.apply(x)


class Q115Snap(nn.Module):
    """Module wrapper for Q1.15 snapping in nn.Sequential."""
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 1.0:
            return ste_quantize_q115(x)
        return ste_quantize_bfp(x, self.scale)


def ste_quantize_bfp(x: torch.Tensor, scale: float) -> torch.Tensor:
    """STE-wrapped BFP quantization. Use this anywhere in the forward graph."""
    s = torch.tensor(scale, dtype=torch.float32)
    return STEQuantizeBFP.apply(x, s)


# ---------------------------------------------------------------------------
# Q1.15-aware Conv2d  (weights in Q1.15, accumulator in float32)
# ---------------------------------------------------------------------------

class Q115Conv2d(nn.Conv2d):
    """
    Conv2d where:
      - weights are quantized to Q1.15 before forward
      - accumulator output is quantized to Q1.31 (snapped to accum_scale)
    """
    def __init__(self, *args, accum_scale: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.accum_scale = accum_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = ste_quantize_q115(self.weight)
        out = self._conv_forward(x, w_q, self.bias)
        return ste_quantize_q131(out, self.accum_scale)


class Q115Linear(nn.Linear):
    """Linear where weights are Q1.15 and accumulator is Q1.31."""
    def __init__(self, *args, accum_scale: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.accum_scale = accum_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = ste_quantize_q115(self.weight)
        out = nn.functional.linear(x, w_q, self.bias)
        return ste_quantize_q131(out, self.accum_scale)


# ---------------------------------------------------------------------------
# Utility: snap master weights back to Q1.15 grid after optimizer step
# ---------------------------------------------------------------------------

def snap_weights_to_q115(model: nn.Module) -> None:
    """
    After an optimizer step, clamp Conv2d/Linear master weights to the Q1.15
    representable range so they don't diverge too far.
    We DO NOT round them here, as that would destroy the high-precision gradient
    accumulation needed by the Straight-Through Estimator!
    We also skip BatchNorm parameters, as they are kept in full precision.
    """
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (Q115Conv2d, Q115Linear)):
                m.weight.clamp_(Q115_MIN_FLOAT, Q115_MAX_FLOAT)
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.clamp_(Q115_MIN_FLOAT, Q115_MAX_FLOAT)


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def weight_stats_q115(model: nn.Module) -> dict:
    """Return statistics about weight Q1.15 representation quality."""
    total: int = 0
    saturated: int = 0
    zero: int = 0
    for m in model.modules():
        if isinstance(m, (Q115Conv2d, Q115Linear)):
            w = getattr(m, "weight", None)
            if w is not None:
                total += w.numel()
                saturated += int((w.abs() >= Q115_MAX_FLOAT).sum().item())
                zero += int((w == 0).sum().item())
            b = getattr(m, "bias", None)
            if b is not None:
                total += b.numel()
                saturated += int((b.abs() >= Q115_MAX_FLOAT).sum().item())
                zero += int((b == 0).sum().item())
    return {
        "total_params":   total,
        "saturated_frac": saturated / max(total, 1),
        "zero_frac":      zero      / max(total, 1),
        "resolution":     Q115_RESOLUTION,
    }
