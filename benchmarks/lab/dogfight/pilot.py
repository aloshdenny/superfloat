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

MANEUVERS = [
    # offensive: pursuit curves and out-of-plane repositioning
    "pursue", "lag", "lead", "high_yoyo", "low_yoyo", "lag_displacement_roll",
    # defensive: deny the solution, force the overshoot
    "break_left", "break_right", "hard_turn", "jink", "barrel_roll_defense",
    "defensive_spiral", "last_ditch",
    # reversals: trade energy for nose position
    "split_s", "immelmann", "pitchback", "flat_scissors", "rolling_scissors",
    # separate / survive
    "extend", "notch", "recover",
]
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


def _reverse_dir(b):
    """Which way to turn: toward him if he is off to one side."""
    return 1.0 if b[1] > 0 else -1.0


def commands(man, blue, red, nz, t=0.0):
    """(aileron, elevator, throttle) for one manoeuvre, one frame.

    `t` is engagement time in seconds; only the manoeuvres whose whole purpose
    is to be unpredictable (jink) use it. Everything else is a pure function of
    the geometry, so a decision is reproducible from the state alone.

    Multi-phase manoeuvres (split_s, immelmann, scissors) are written
    statelessly: they key on current attitude, so re-selecting the same
    manoeuvre each decision continues it, and selecting a different one
    abandons it cleanly. That matches how the policy actually runs -- one
    choice every 100 ms, no memory between them.
    """
    b = los_body(blue, red)
    phi = blue.get("phi", 0.0)
    theta = blue.get("theta", 0.0)

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
    elif man == "hard_turn":
        # sustained turn, not a break: generates angles without emptying the
        # energy tank. The manoeuvre a break turn should usually have been.
        ail, ele, thr = _pursuit(b, 0.75, 0.0, 1.0)
    elif man == "jink":
        # aperiodic out-of-plane displacement. The point is unpredictability,
        # not angles: a tracking gun solution needs a predictable target.
        ph = math.sin(t * 2.3) + 0.6 * math.sin(t * 5.7 + 1.1)
        ail = _clamp(ph, -1.0, 1.0)
        ele = -_clamp(0.45 + 0.35 * math.sin(t * 3.9), 0.0, 1.0)
        thr = 1.0
    elif man == "barrel_roll_defense":
        # high-G barrel roll across his flight path: kills forward velocity and
        # puts an overshooting attacker out in front
        ail, ele, thr = _clamp(0.85, -1, 1), -0.85, 0.9
    elif man == "defensive_spiral":
        # descending max-rate turn: trades altitude for turn rate and dares him
        # to follow into a rate fight at low level
        d = _reverse_dir(b)
        ail, ele, thr = _bank_to(phi, 75.0 * d, 0.9, 1.0)
    elif man == "last_ditch":
        # inside guns range with the pipper tracking: violent out-of-plane
        # displacement, accepting total energy loss to make him miss now
        d = _reverse_dir(b)
        ail, ele, thr = _clamp(1.0 * d, -1, 1), -1.0, 1.0
    elif man == "lag_displacement_roll":
        # roll around his flight path to bleed closure without crossing his 3/9
        ail, ele, thr = _pursuit(b, 0.35, 0.35, 0.7)
    elif man == "split_s":
        # half roll inverted, then pull through: 180 deg of heading for a lot of
        # altitude. Stateless: roll until inverted, then pull.
        if abs(phi) < 150.0:
            ail, ele, thr = _bank_to(phi, 180.0, 0.05, 0.9)
        else:
            ail, ele, thr = 0.0, -1.0, 0.9
    elif man == "immelmann":
        # half loop then roll upright: 180 deg of heading, trading speed for
        # altitude -- the opposite trade to split_s
        if theta > -60.0 and abs(phi) < 120.0:
            ail, ele, thr = -0.05 * phi / 90.0, -0.95, 1.0
        else:
            ail, ele, thr = _bank_to(phi, 0.0, 0.15, 1.0)
    elif man == "pitchback":
        # nose-high reversal: pull into the vertical, roll toward him, come back
        d = _reverse_dir(b)
        ail, ele, thr = _clamp(0.6 * d, -1, 1), -0.85, 1.0
    elif man == "flat_scissors":
        # horizontal reversals after an overshoot. Won by being SLOWER with the
        # smaller radius, so the throttle is deliberately back.
        d = _reverse_dir(b)
        ail, ele, thr = _bank_to(phi, 70.0 * d, 0.85, 0.35)
    elif man == "rolling_scissors":
        # the vertical version: barrel rolls, each trying to end up behind
        d = _reverse_dir(b)
        ail, ele, thr = _clamp(0.75 * d, -1, 1), -0.7, 0.55
    elif man == "notch":
        # put him on the 3/9 line to sit in his radar's Doppler notch. Pure
        # geometry here; the sensor model that would make it pay off is not
        # implemented yet (see TACTICS.md section 8).
        want = 90.0 if b[1] > 0 else -90.0
        cur = math.degrees(math.atan2(b[1], b[0]))
        err = want - cur
        ail, ele, thr = _clamp(err / 60.0, -1.0, 1.0), -0.25, 1.0
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
