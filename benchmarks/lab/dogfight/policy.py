"""The decision model: engagement state in, typed decisions out.

Deliberately System-One shaped, following Jev and Laya: one forward pass, no
autoregression, and a fixed output schema. Three heads mirror the three
question types those models expose:

  choice  which manoeuvre to fly        (categorical over pilot.MANEUVERS)
  score   how threatened am I           (ordinal, 3 levels)
  noul    P(I am in his weapons envelope within the next second)

Architecture is a pre-norm residual MLP rather than an attention stack, since
the state is a fixed-size vector and there is no sequence. What it does keep is
the structure the scale-absorption result in SCALING_LAWS.md 4.2 depends on:
every block's INPUT projection is fed by an RMSNorm and is therefore absorbable,
while the block's OUTPUT projection writes straight into the residual stream
and is not. That is the group A/B vs group C split of the mixed-allocation
study, so a per-group bit budget can be tested on the same model.
"""
from __future__ import annotations

import math

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_here, "..", ".."), os.path.join(_here, ".."), _here):
    sys.path.insert(0, os.path.abspath(_p))
from superfloat import sf_quantize_sv

from pilot import MANEUVERS

# The features a BFM decision keys on. Order is fixed: checkpoints depend on it.
FEATURES = [
    "range_m", "ata_deg", "aa_deg", "red_ata_deg", "closure_mps", "dz_m",
    "hca_deg", "blue_v_mps", "red_v_mps", "dv_mps", "blue_mach", "red_mach",
    "blue_nz", "blue_phi_deg", "blue_alt_ft", "blue_gamma_deg",
    "los_fwd", "los_right", "los_down",
]
N_FEAT = len(FEATURES)
N_MAN = len(MANEUVERS)
N_THREAT = 3

# Rough scales so the input is O(1) without a learned normaliser, which would
# itself need quantizing and would confound the study.
_SCALE = {
    "range_m": 3000.0, "ata_deg": 90.0, "aa_deg": 90.0, "red_ata_deg": 90.0,
    "closure_mps": 200.0, "dz_m": 1000.0, "hca_deg": 90.0,
    "blue_v_mps": 300.0, "red_v_mps": 300.0, "dv_mps": 100.0,
    "blue_mach": 1.0, "red_mach": 1.0, "blue_nz": 5.0,
    "blue_phi_deg": 90.0, "blue_alt_ft": 20000.0, "blue_gamma_deg": 30.0,
    "los_fwd": 1.0, "los_right": 1.0, "los_down": 1.0,
}


def encode(obs):
    """Observation dict -> feature list, in FEATURES order."""
    return [float(obs.get(k, 0.0)) / _SCALE[k] for k in FEATURES]


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class Block(nn.Module):
    """Pre-norm residual MLP. `fc_in` is norm-fed (absorbable); `fc_out` is not.

    `res_scale`/`res_vmax`, when set, saturate the residual SUM -- the register
    a real datapath writes the accumulator back into. Leaving them None is the
    default everywhere else in this repo and is what the datapath audit found
    escaping the grid.
    """

    def __init__(self, d, mult=4):
        super().__init__()
        self.norm = RMSNorm(d)
        self.fc_in = nn.Linear(d, mult * d, bias=False)
        self.fc_out = nn.Linear(mult * d, d, bias=False)
        self.res_scale = None
        self.res_vmax = None

    def forward(self, x):
        h = x + self.fc_out(F.silu(self.fc_in(self.norm(x))))
        if self.res_scale is not None:
            h = sf_quantize_sv(h, self.res_scale, self.res_vmax)
        return h


class Policy(nn.Module):
    def __init__(self, d=256, depth=4, mult=4):
        super().__init__()
        self.stem = nn.Linear(N_FEAT, d, bias=False)
        self.blocks = nn.ModuleList([Block(d, mult) for _ in range(depth)])
        self.norm = RMSNorm(d)
        self.head_choice = nn.Linear(d, N_MAN, bias=False)
        self.head_score = nn.Linear(d, N_THREAT, bias=False)
        self.head_noul = nn.Linear(d, 1, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        h = self.norm(h)
        return self.head_choice(h), self.head_score(h), self.head_noul(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs):
        """One decision. Returns (manoeuvre, threat level, P(in his WEZ))."""
        x = torch.tensor([encode(obs)], dtype=torch.float32,
                         device=next(self.parameters()).device)
        c, s, n = self(x)
        return MANEUVERS[int(c.argmax(-1))], int(s.argmax(-1)), float(torch.sigmoid(n))


def residual_format(total_bits, int_bits):
    """Q(int_bits+1).(total-1-int_bits) as a (scale, vmax) pair.

    int_bits=0 is the plain SF grid this repo uses everywhere: SF8 -> Q1.7,
    step 1/128, range +-127/128. int_bits=2 gives Q3.5: step 1/32, range
    +-3.97, which is the headroom the audit says a 4-block residual needs.
    """
    scale = 2.0 ** (total_bits - 1 - int_bits)
    vmax = (2.0 ** (total_bits - 1) - 1) / scale
    return scale, vmax


def set_residual_format(model, total_bits, int_bits):
    scale, vmax = residual_format(total_bits, int_bits)
    for b in model.blocks:
        b.res_scale, b.res_vmax = scale, vmax
    return scale, vmax


def param_count(m):
    return sum(p.numel() for p in m.parameters())
