"""Two-aircraft closed-loop engagement on JSBSim.

The loop runs the flight dynamics at 60 Hz and asks each side for a manoeuvre
at a lower decision rate (default 10 Hz), which is the rate a real decision
model would have to sustain. Decisions are typed -- one of `pilot.MANEUVERS`
-- and the inner loop flies them.

Scoring is a guns-only weapons engagement zone: inside range, pipper on, held
long enough to count. Missiles are deliberately out of scope; a gun solution
is a pure test of whether the policy can fly to a position.
"""
from __future__ import annotations

import math
import os
import random

import jsbsim

from geom import engagement, los_body
from pilot import commands

FT2M = 0.3048
DT = 1.0 / 60.0

WEZ_MIN_M = 150.0      # inside this you are overshooting / colliding
WEZ_MAX_M = 900.0      # guns, not missiles
WEZ_ATA_DEG = 8.0      # pipper on
WEZ_HOLD_S = 0.5       # tracking time to count as a kill
FLOOR_FT = 3000.0      # hard deck; below it you have lost the fight
CEILING_FT = 45000.0


# JSBSim finds its aircraft/engine data automatically when pip-installed
# normally, but not when installed with --target into a lab box's lib dir, so
# allow the root to be given explicitly.
JSBSIM_ROOT = os.environ.get("JSBSIM_ROOT") or None


def _aircraft(model, alt_ft, kts, psi, lat, lon):
    f = jsbsim.FGFDMExec(JSBSIM_ROOT)
    f.set_debug_level(0)
    f.load_model(model)
    f.set_dt(DT)
    f["ic/h-sl-ft"] = alt_ft
    f["ic/vc-kts"] = kts
    f["ic/psi-true-deg"] = psi
    f["ic/lat-gc-deg"] = lat
    f["ic/long-gc-deg"] = lon
    f["ic/gamma-deg"] = 0.0
    f.run_ic()
    f["propulsion/set-running"] = -1
    f["fcs/throttle-cmd-norm"] = 0.9
    f["simulation/do_simple_trim"] = 1
    return f


def _state(f):
    return dict(lat=f["position/lat-gc-deg"], lon=f["position/long-gc-deg"],
                alt=f["position/h-sl-ft"], psi=f["attitude/psi-deg"],
                theta=f["attitude/theta-deg"], phi=f["attitude/phi-deg"],
                vt=f["velocities/vt-fps"], gamma=f["flight-path/gamma-deg"],
                nz=f["accelerations/Nz"], mach=f["velocities/mach"])


KM = 1000.0 / 111120.0


class Engagement:
    """One fight. `reset` randomises the setup so arms see the same spread."""

    def __init__(self, model="f16", decision_hz=10.0, max_seconds=90.0, seed=0):
        self.model = model
        self.every = max(1, int(round((1.0 / decision_hz) / DT)))
        self.max_frames = int(max_seconds / DT)
        self.rng = random.Random(seed)

    def reset(self, setup=None):
        """Randomised start. Symmetric by construction.

        An earlier version drew blue's heading from +/-20 deg of north while
        red's was uniform over the circle, which handed red a systematic
        angular advantage: expert-vs-expert self-play ran 10-16 instead of
        even. Both sides now get the same treatment, so a surviving asymmetry
        is the policy's, not the setup's.
        """
        r = self.rng
        setup = setup or r.choice(["neutral", "offensive", "defensive", "head_on"])
        alt = r.uniform(12000, 22000)
        jitter = lambda: r.uniform(-25, 25)
        bpsi, rpsi = jitter(), jitter()
        bn = be = 0.0
        if setup == "offensive":        # blue behind red, both roughly co-heading
            rn, re_ = r.uniform(0.8, 2.0), r.uniform(-0.4, 0.4)
        elif setup == "defensive":      # red behind blue
            rn, re_ = -r.uniform(0.8, 2.0), r.uniform(-0.4, 0.4)
        elif setup == "head_on":
            rn, re_ = r.uniform(3.0, 6.0), r.uniform(-0.5, 0.5)
            rpsi = 180.0 + jitter()
        else:                            # neutral: offset beam, crossing angle
            rn, re_ = r.uniform(-1.0, 1.0), r.uniform(1.5, 3.0)
            rpsi = r.choice([90.0, -90.0]) + jitter()

        # same speed and altitude draw for both: no free energy advantage
        bkts, rkts = r.uniform(380, 520), r.uniform(380, 520)
        dalt = r.uniform(-1500, 1500)
        self.blue = _aircraft(self.model, alt, bkts, bpsi, bn * KM, be * KM)
        self.red = _aircraft(self.model, alt + dalt, rkts, rpsi, rn * KM, re_ * KM)
        self.frame = 0
        self.blue_lock = 0
        self.red_lock = 0
        self.setup = setup
        return self.observe()

    def _obs_from(self, me, him, me_state, him_state):
        g = engagement(me_state, him_state)
        g["blue_nz"] = me_state["nz"]
        g["blue_mach"] = me_state["mach"]
        g["red_mach"] = him_state["mach"]
        g["blue_phi_deg"] = me_state["phi"]
        g["blue_alt_ft"] = me_state["alt"]
        g["blue_gamma_deg"] = me_state["gamma"]
        lb = los_body(me_state, him_state)
        g["los_fwd"], g["los_right"], g["los_down"] = lb
        return g

    def step(self, blue_man, red_man):
        """Hold both manoeuvres for one decision interval.

        One engagement() per frame serves both weapon checks: ata_deg is blue's
        nose on red and red_ata_deg is red's nose on blue, so the reciprocal
        geometry is free. Returns (blue_obs, red_obs, done, info).
        """
        hold = int(WEZ_HOLD_S / DT)
        for _ in range(self.every):
            bs, rs = _state(self.blue), _state(self.red)
            t = self.frame * DT
            ba, be, bt = commands(blue_man, bs, rs, bs["nz"], t)
            ra, re_, rt = commands(red_man, rs, bs, rs["nz"], t)
            self.blue["fcs/aileron-cmd-norm"] = ba
            self.blue["fcs/elevator-cmd-norm"] = be
            self.blue["fcs/throttle-cmd-norm"] = bt
            self.red["fcs/aileron-cmd-norm"] = ra
            self.red["fcs/elevator-cmd-norm"] = re_
            self.red["fcs/throttle-cmd-norm"] = rt
            if not (self.blue.run() and self.red.run()):
                return self.observe() + (True, {"outcome": "sim_error"})
            self.frame += 1

            bs, rs = _state(self.blue), _state(self.red)
            g = engagement(bs, rs)
            in_band = WEZ_MIN_M < g["range_m"] < WEZ_MAX_M
            self.blue_lock = self.blue_lock + 1 if (in_band and g["ata_deg"] < WEZ_ATA_DEG) else 0
            self.red_lock = self.red_lock + 1 if (in_band and g["red_ata_deg"] < WEZ_ATA_DEG) else 0

            if self.blue_lock >= hold or self.red_lock >= hold:
                bw, rw = self.blue_lock >= hold, self.red_lock >= hold
                out = "mutual" if bw and rw else ("blue_win" if bw else "red_win")
                return self._both(bs, rs) + (True, {"outcome": out, "t": self.frame * DT})
            if bs["alt"] < FLOOR_FT or rs["alt"] < FLOOR_FT:
                who = "blue_crash" if bs["alt"] < FLOOR_FT else "red_crash"
                return self._both(bs, rs) + (True, {"outcome": who, "t": self.frame * DT})
            if self.frame >= self.max_frames:
                return self._both(bs, rs) + (True, {"outcome": "timeout", "t": self.frame * DT})
        return self._both(bs, rs) + (False, {})

    def _both(self, bs, rs):
        return self._obs_from(self.blue, self.red, bs, rs), self._obs_from(self.red, self.blue, rs, bs)

    def observe(self):
        return self._both(_state(self.blue), _state(self.red))
