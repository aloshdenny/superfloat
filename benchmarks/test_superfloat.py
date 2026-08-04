"""Verify the optimized SuperFloat path is numerically identical to the
straightforward reference implementation.

Run on CPU; touches no GPU.
"""

import torch
import torch.nn as nn

from superfloat import (SFConv2d, apply_superfloat, clamp_all, sf_params,
                        sf_quantize, sf_quantize_sv)

torch.manual_seed(0)
ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


# ---- 1. grid constants match the paper -----------------------------------
print("grid constants")
for bits, want in [(16, 0.999969482421875), (8, 0.9921875), (4, 0.875)]:
    check(f"SF{bits} vmax == {want}", sf_params(bits)[1] == want)

# ---- 2. hoisted scale/vmax gives identical values -------------------------
print("sf_quantize_sv == sf_quantize")
x = torch.randn(10000) * 2
for bits in (16, 8, 4):
    s, v = sf_params(bits)
    check(f"SF{bits} bit-identical", torch.equal(sf_quantize(x, bits),
                                                 sf_quantize_sv(x, s, v)))

# ---- 3. bounded STE gradient unchanged ------------------------------------
print("bounded STE gradient")
for bits in (16, 8, 4):
    s, v = sf_params(bits)
    a = x.clone().requires_grad_(True)
    sf_quantize_sv(a, s, v).sum().backward()
    expect = (x.abs() <= v).float()
    check(f"SF{bits} grad == inside-range mask", torch.equal(a.grad, expect))

# ---- 4. quantized output lands exactly on the grid ------------------------
print("outputs lie on the SFx grid")
for bits in (16, 8, 4):
    s, v = sf_params(bits)
    q = sf_quantize_sv(x, s, v)
    check(f"SF{bits} q*scale integral", torch.equal(q * s, torch.round(q * s)))
    check(f"SF{bits} within range", q.abs().max().item() <= v)


# ---- 5. fused clamp_all == naive per-module clamping ----------------------
def naive_clamp(model):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, SFConv2d):
                _, v = sf_params(m.sf_bits)
                m.weight.clamp_(-v, v)
                if m.bias is not None:
                    m.bias.clamp_(-v, v)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if m.weight is not None:
                    m.weight.clamp_(-1.0, 1.0)
                if m.bias is not None:
                    m.bias.clamp_(-1.0, 1.0)


print("clamp_all == naive per-module clamp")


def make_net():
    torch.manual_seed(7)
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, bias=True), nn.BatchNorm2d(8), nn.ReLU(),
        nn.Conv2d(8, 8, 3, bias=True), nn.BatchNorm2d(8),
    )


for bits in (16, 8, 4):
    a, b = make_net(), make_net()
    apply_superfloat(a, bits)
    apply_superfloat(b, bits)
    # push params well outside the representable range
    with torch.no_grad():
        for p in list(a.parameters()) + list(b.parameters()):
            p.mul_(5.0)
    clamp_all(a)
    naive_clamp(b)
    same = all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    check(f"SF{bits} identical after clamp", same)
    # cache must stay correct across repeated calls
    clamp_all(a)
    naive_clamp(b)
    same2 = all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    check(f"SF{bits} identical after second clamp (cache reuse)", same2)

# ---- 6. layer forward matches an explicit reference -----------------------
print("SFConv2d forward == explicit reference")
for bits in (16, 8, 4):
    conv = nn.Conv2d(3, 4, 3, bias=True)
    ref_w, ref_b = conv.weight.detach().clone(), conv.bias.detach().clone()
    apply_superfloat(conv, bits)
    inp = torch.randn(2, 3, 16, 16)
    got = conv(inp)
    s, v = sf_params(bits)
    exp = torch.nn.functional.conv2d(inp, sf_quantize_sv(ref_w, s, v),
                                     sf_quantize_sv(ref_b, s, v))
    exp = sf_quantize_sv(exp, s, v)
    check(f"SF{bits} forward identical", torch.equal(got, exp))

# ---- 7. head really stays full precision ----------------------------------
print("head exclusion")
net = nn.Sequential(nn.Conv2d(3, 4, 3), nn.Conv2d(4, 4, 3))
n = apply_superfloat(net, 8, head_names=("1",))
check("only non-head converted", n == 1)
check("head[1] left as plain Conv2d",
      type(net[1]) is nn.Conv2d and isinstance(net[0], SFConv2d))

print("\nALL PASS" if ok else "\nFAILURES PRESENT")
raise SystemExit(0 if ok else 1)
