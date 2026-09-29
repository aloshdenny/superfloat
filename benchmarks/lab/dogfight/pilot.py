"""Inner loop: a typed manoeuvre choice becomes control-surface commands.

This is the half of the system that is NOT learned, and keeping it separate is
the whole point of the experiment. The decision model emits a manoeuvre from a
fixed set -- the typed `choice` of a System One model -- and this deterministic
code flies it. Quantizing the model therefore changes which manoeuvre is
selected, never how well it is executed, so a closed-loop loss is attributable
to the decision and not to a degraded controller.

JSBSim sign conventions, measured rather than assumed (see git history):
  elevator-cmd-norm  negative = nose up (pull), Nz +4.8 at -0.5
  aileron-cmd-norm   positive = roll right,     phi +78 deg at +0.5
"""
from __future__ import annotations

import math

from geom import los_body

MANEUVERS = ["pursue", "lag", "lead", "break_left", "break_right",
             "extend", "high_yoyo", "low_yoyo", "recover"]
MAN_IDX = {m: i for i, m in enumerate(MANEUVERS)}

G_LIMIT = 9.0
_PULL_GAIN = 2.2
_ROLL_GAIN = 2.0


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _aim(b, up_bias=0.0):
    """Roll error and off-nose angle for a line-of-sight in body axes.

    `up_bias` tilts the aim point above (+) or below (-) the target, which is
    what separates a yo-yo from pure pursuit.
    """
    fwd, right, down = b
    down = down - up_bias
    # roll so the aim point lies straight "up" in the canopy
    phi_err = math.degrees(math.atan2(right, -down)) if (right or down) else 0.0
    off_nose = math.degrees(math.atan2(math.hypot(right, down), fwd))
    return phi_err, off_nose


def _pursuit(b, pull_scale=1.0, up_bias=0.0, throttle=1.0):
    phi_err, off_nose = _aim(b, up_bias)
    ail = _clamp(_ROLL_GAIN * phi_err / 90.0, -1.0, 1.0)
    # only pull once roughly rolled into the plane, or the lift vector is wasted
    gate = max(0.0, math.cos(math.radians(phi_err)))
    pull = _clamp(_PULL_GAIN * pull_scale * gate * off_nose / 90.0, 0.0, 1.0)
    return ail, -pull, throttle


def _bank_to(phi_now, phi_target, pull, throttle):
    err = phi_target - phi_now
    while err > 180.0:
        err -= 360.0
    while err < -180.0:
        err += 360.0
    return _clamp(_ROLL_GAIN * err / 90.0, -1.0, 1.0), -pull, throttle


def commands(man, blue, red, nz):
    """(aileron, elevator, throttle) for one manoeuvre, one frame."""
    b = los_body(blue, red)
    phi = blue.get("phi", 0.0)

    if man == "pursue":
        ail, ele, thr = _pursuit(b, 1.0, 0.0, 1.0)
    elif man == "lag":
        # aim behind him: ease the pull, preserve energy, avoid an overshoot
        ail, ele, thr = _pursuit(b, 0.55, 0.0, 0.85)
    elif man == "lead":
        ail, ele, thr = _pursuit(b, 1.45, 0.0, 1.0)
    elif man == "high_yoyo":
        # pull up out of plane to kill closure and trade speed for position
        ail, ele, thr = _pursuit(b, 1.15, 0.6, 0.8)
    elif man == "low_yoyo":
        # drop below to regain speed and cut the corner
        ail, ele, thr = _pursuit(b, 1.15, -0.6, 1.0)
    elif man == "break_left":
        ail, ele, thr = _bank_to(phi, -80.0, 1.0, 1.0)
    elif man == "break_right":
        ail, ele, thr = _bank_to(phi, 80.0, 1.0, 1.0)
    elif man == "recover":
        # ground collision avoidance: roll upright and pull to the horizon.
        # Kept in the manoeuvre set rather than hidden in a safety layer, so
        # failing to call it is a decision error the study can see.
        ail, ele, thr = _bank_to(phi, 0.0, 0.75, 1.0)
    elif man == "extend":
        # unload, roll level, run: rebuild energy
        ail, ele, thr = _bank_to(phi, 0.0, 0.05, 1.0)
    else:
        raise ValueError(f"unknown manoeuvre {man!r}")

    # G limiter: the airframe, not the policy, decides how hard it can pull
    if nz > G_LIMIT and ele < 0:
        ele *= max(0.0, 1.0 - (nz - G_LIMIT) / 2.0)
    return ail, ele, thr
