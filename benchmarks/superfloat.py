"""SuperFloat (SFx) quantization-aware training primitives.

SFx is a signed fractional fixed-point schema: 1 sign bit, (x-1) fractional
bits, no exponent and no integer bit. Representable set is

    { k / 2^(x-1)  :  k integer, |k| <= 2^(x-1) - 1 }

so the range is symmetric and bounded by +/- (1 - 2^-(x-1)).

    SF16 -> scale 32768, max  0.999969482421875
    SF8  -> scale   128, max  0.9921875
    SF4  -> scale     8, max  0.875

These match the constants tabulated in the SuperFloat paper.

Everything here is fp32. The reference implementation in cifar_modular/model.py
casts to float64 for the accumulation, which Metal cannot do -- MPS has no
float64 at all. fp32 is exact for the SF grid regardless: the coarsest
requirement is SF16, whose scale is 2^15, and fp32 represents every integer up
to 2^24 exactly, so round(x * scale) / scale lands on the same grid points in
fp32 as it does in fp64. Only the conv accumulation order differs, which is a
hardware-emulation detail rather than a property of the numeric format.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Constant BatchNorm epsilon used for every format, per the paper's recipe.
BN_EPS = 0.125


def disable_tf32():
    """Turn off TF32 on NVIDIA hardware.

    Ampere/Ada default to TF32 for cuDNN convolutions and (historically) matmul.
    TF32 keeps only 10 mantissa bits, so it would silently round SF16 values --
    which carry 15 significand bits -- back to below SF8 fidelity inside the
    accumulation. The paper's recipe assumes full-precision accumulate, so this
    must stay off for the numbers to mean anything. Costs speed; correctness
    wins.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def sf_params(bits: int):
    """Return (scale, max_value) for the SFx member of the schema."""
    if not (2 <= bits <= 16):
        raise ValueError(f"SF{bits} outside the schema's 2..16 range")
    scale = float(2 ** (bits - 1))
    return scale, (scale - 1.0) / scale


def sf_quantize(x, bits):
    """Real grid quantization with a bounded straight-through estimator.

    Forward snaps to the SFx grid. Backward passes the gradient through
    unchanged where the input was inside the representable range and zeroes it
    outside, so saturated values stop receiving updates.

    Written as pure differentiable ops rather than an autograd.Function on
    purpose. clamp's own backward already zeroes the gradient outside
    [-vmax, vmax], which is exactly the bounded STE, and the residual term is
    detached so round contributes no gradient. Keeping it op-level lets
    TorchInductor trace straight through and fuse the clamp/round/div chain
    into the surrounding kernels; a custom Function would force a graph break
    at every quantized layer.
    """
    scale, vmax = sf_params(bits)
    return sf_quantize_sv(x, scale, vmax)


def sf_quantize_sv(x, scale, vmax):
    """sf_quantize with the grid constants already resolved.

    The hot path calls this twice per layer per forward, so the scale/vmax
    lookup is hoisted to layer-conversion time rather than recomputed here.
    """
    xc = torch.clamp(x, -vmax, vmax)
    return xc + (torch.round(xc * scale) / scale - xc).detach()


# ---------------------------------------------------------------------------
# Layer surgery
#
# Rather than swapping in new module objects, we rebind __class__ on the
# existing Conv2d / Linear instances. The parameter objects, the state_dict
# keys and every isinstance(m, nn.Conv2d) check downstream stay exactly as they
# were, which matters because Ultralytics introspects and deep-copies its model
# graph (EMA, fusing, checkpointing) and would trip over wrapper modules.
# ---------------------------------------------------------------------------


class SFConv2d(nn.Conv2d):
    sf_bits = 16
    sf_act = True
    sf_scale = 32768.0
    sf_vmax = 32767.0 / 32768.0

    def forward(self, x):
        s, v = self.sf_scale, self.sf_vmax
        w = sf_quantize_sv(self.weight, s, v)
        b = sf_quantize_sv(self.bias, s, v) if self.bias is not None else None
        out = self._conv_forward(x, w, b)
        return sf_quantize_sv(out, s, v) if self.sf_act else out


class SFLinear(nn.Linear):
    sf_bits = 16
    sf_act = True
    sf_scale = 32768.0
    sf_vmax = 32767.0 / 32768.0

    def forward(self, x):
        s, v = self.sf_scale, self.sf_vmax
        w = sf_quantize_sv(self.weight, s, v)
        b = sf_quantize_sv(self.bias, s, v) if self.bias is not None else None
        out = F.linear(x, w, b)
        return sf_quantize_sv(out, s, v) if self.sf_act else out


def _is_head(name, head_names):
    return any(name == h or name.startswith(h + ".") for h in head_names)


def apply_superfloat(model, bits, head_names=(), quantize_activations=True,
                     set_bn_eps=True):
    """Convert a model in place to SFx quantization-aware training.

    head_names lists module prefixes to leave in full precision, so that output
    logits stay fp32 as the paper's recipe requires.

    Returns the number of layers converted.
    """
    converted = 0
    for name, m in model.named_modules():
        if _is_head(name, head_names):
            continue
        if isinstance(m, nn.Conv2d) and not isinstance(m, SFConv2d):
            m.__class__ = SFConv2d
        elif isinstance(m, nn.Linear) and not isinstance(m, SFLinear):
            m.__class__ = SFLinear
        else:
            if set_bn_eps and isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eps = BN_EPS
            continue
        m.sf_bits = bits
        m.sf_act = quantize_activations
        m.sf_scale, m.sf_vmax = sf_params(bits)
        converted += 1
    # Layer set changed, so any cached clamp tensor lists are stale.
    if hasattr(model, "_sf_clamp_cache"):
        del model._sf_clamp_cache
    return converted


def _clamp_cache(model):
    """Tensors needing post-step clamping, grouped by bound, collected once.

    Called after every optimizer step, so walking model.modules() here would
    put a few hundred Python-level isinstance checks and individual clamp_
    launches on the hot path each iteration. The layer set is fixed after
    surgery, so the lists are built once and cached on the model;
    apply_superfloat invalidates the cache if the set ever changes.
    """
    cache = getattr(model, "_sf_clamp_cache", None)
    if cache is None:
        by_bound = {}
        norm = []
        for m in model.modules():
            if isinstance(m, (SFConv2d, SFLinear)):
                group = by_bound.setdefault(m.sf_vmax, [])
                group.append(m.weight)
                if m.bias is not None:
                    group.append(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if m.weight is not None:
                    norm.append(m.weight)
                if m.bias is not None:
                    norm.append(m.bias)
        cache = (list(by_bound.items()), norm)
        model._sf_clamp_cache = cache
    return cache


@torch.no_grad()
def clamp_all(model):
    """Clamp shadow weights and BatchNorm affine params back into range.

    Two things are being held in place:

    Weights -- the bounded STE zeroes gradients for saturated entries, so a
    weight that leaves [-vmax, vmax] could never come back on its own.
    Clamping after each step keeps it recoverable.

    BatchNorm affine -- the reference CIFAR trainer clamps these symmetrically
    to stop BN scale walking outside what the format can hold; without it the
    quantized forward and the fp32 shadow weights drift apart.

    Uses the fused multi-tensor ops so each bound group is a single kernel
    launch rather than one per parameter.
    """
    groups, norm = _clamp_cache(model)
    for vmax, tensors in groups:
        torch._foreach_clamp_min_(tensors, -vmax)
        torch._foreach_clamp_max_(tensors, vmax)
    if norm:
        torch._foreach_clamp_min_(norm, -1.0)
        torch._foreach_clamp_max_(norm, 1.0)




def storage_reduction(bits):
    """Per-weight storage saving relative to the FP32 baseline."""
    return 1.0 - bits / 32.0
