"""Which airframes fight, and how: the E-M characterisation.

The one-circle / two-circle decision in TACTICS.md is decided by the two
aircraft's turn rate and turn radius curves, so a matchup matrix needs those
measured rather than assumed. This probes each JSBSim airframe for:

  max Mach          is it actually a supersonic fighter
  corner velocity   lowest speed that still reaches the G limit -- the speed
                    every BFM pilot fights at
  instantaneous rate  best turn rate available, deg/s, ignoring energy cost
  sustained rate    turn rate at which specific excess power is zero, i.e.
                    the rate you can hold forever

A naive probe (full throttle, zero elevator, 120 s) makes the f15 and f22
diverge to absurd Mach numbers, because nothing holds them level. Everything
here flies under a pitch damper.
"""
from __future__ import annotations

import argparse, json, math, os

import jsbsim

FT2M = 0.3048
ROOT = os.environ.get("JSBSIM_ROOT") or None

# Everything in the stock tree that is plausibly a jet fighter.
CANDIDATES = ["f16", "f15", "f22", "f104", "F4N", "T38", "A4", "F80C"]


def _throttle(f, v):
    """Set every engine. The f15, F4N and f22 are twins, and setting only
    `fcs/throttle-cmd-norm` leaves the second engine at idle -- which is why
    they first measured as unable to pass Mach 0.8."""
    # `propulsion/engine-count` reads back 0.0; the API is the truth.
    try:
        n = f.get_propulsion().get_num_engines()
    except Exception:
        n = 1
    for i in range(max(1, int(n))):
        f["fcs/throttle-cmd-norm[%d]" % i] = v


def _mk(model, alt_ft, kts, dt=1/120.):
    f = jsbsim.FGFDMExec(ROOT)
    f.set_debug_level(0)
    if not f.load_model(model):
        return None
    f.set_dt(dt)
    f["ic/h-sl-ft"] = alt_ft
    if kts is None:
        f["ic/mach"] = 0.80
    else:
        f["ic/vc-kts"] = kts
    f["ic/psi-true-deg"] = 0
    f["ic/gamma-deg"] = 0
    f.run_ic()
    f["propulsion/set-running"] = -1
    _throttle(f, 1.0)
    try:
        f["simulation/do_simple_trim"] = 1
    except Exception:
        pass
    return f


def _hold_level(f, ail=0.0, kp=0.06, kd=0.02):
    """Proportional-derivative on flight path angle: keeps the probe honest."""
    gam = f["flight-path/gamma-deg"]
    q = f["velocities/q-rad_sec"] * 57.2958
    ele = kp * gam + kd * q
    f["fcs/elevator-cmd-norm"] = max(-1.0, min(1.0, ele))
    f["fcs/aileron-cmd-norm"] = ail
    _throttle(f, 1.0)
    f["fcs/throttle-cmd-norm"] = 1.0


def max_mach(model, alt_ft=36000, seconds=180, max_loss_ft=1500):
    """Top speed in LEVEL flight.

    The first version held flight-path angle with a PD loop and reported Mach
    1.14 for an A-4 Skyhawk, which is subsonic: the loop was not holding, the
    aircraft was descending, and the probe was measuring dive speed. Altitude
    is now held directly and any run that loses more than `max_loss_ft` is
    rejected rather than reported.
    """
    f = _mk(model, alt_ft, None)     # start subsonic; see note below
    if f is None:
        return None
    best = 0.0
    settle = int(10.0 / f.get_delta_t())
    n = int(seconds / f.get_delta_t())
    for step in range(n):
        h = f["position/h-sl-ft"]
        hdot = f["velocities/h-dot-fps"]
        # altitude hold: drive (h - target) and its rate to zero
        ele = 0.0012 * (h - alt_ft) + 0.010 * hdot
        f["fcs/elevator-cmd-norm"] = max(-0.6, min(0.6, ele))
        f["fcs/aileron-cmd-norm"] = -0.02 * f["attitude/phi-deg"]
        _throttle(f, 1.0)
        if not f.run():
            break
        m = f["velocities/mach"]
        if not math.isfinite(m) or m > 6.0 or abs(f["aero/alpha-deg"]) > 40:
            return None
        if alt_ft - f["position/h-sl-ft"] > max_loss_ft:
            return None                      # it dived; the number would be a lie
        # 400 KCAS at 36000 ft IS Mach 1.14, so the first version was reporting
        # its own initial condition as a top speed -- an A-4 "went supersonic".
        # Start at Mach 0.80 and ignore the acceleration transient.
        if step > settle:
            best = max(best, m)
    return best


def turn_probe(model, alt_ft=15000, kts=None, g_limit=9.0, seconds=12, alpha_limit=25.0):
    """Roll hard and pull to the G limit; report achieved rate and energy rate."""
    f = _mk(model, alt_ft, kts)
    if f is None:
        return None
    dt = f.get_delta_t()
    # settle
    for _ in range(int(2 / dt)):
        _hold_level(f)
        if not f.run():
            return None
    v0 = f["velocities/vt-fps"]
    h0 = f["position/h-sl-ft"]
    psi_prev = f["attitude/psi-deg"]
    swept = 0.0
    nz_peak = 0.0
    n = int(seconds / dt)
    for _ in range(n):
        # bank hard, then pull to the limit
        phi = f["attitude/phi-deg"]
        f["fcs/aileron-cmd-norm"] = max(-1.0, min(1.0, (80.0 - phi) / 40.0))
        nz = f["accelerations/Nz"]
        nz_peak = max(nz_peak, nz)
        # Real fly-by-wire limits BOTH G and angle of attack. Without the alpha
        # limit the f22 departs under max pull and "turns" at 183 deg/s, which
        # is a tumble, not a turn.
        alpha = f["aero/alpha-deg"]
        pull = 1.0 if nz < g_limit else max(0.0, 1.0 - (nz - g_limit))
        if alpha > alpha_limit:
            pull = min(pull, max(0.0, 1.0 - (alpha - alpha_limit) / 5.0))
        f["fcs/elevator-cmd-norm"] = -pull
        _throttle(f, 1.0)
        if not f.run():
            return None
        psi = f["attitude/psi-deg"]
        d = psi - psi_prev
        while d > 180: d -= 360
        while d < -180: d += 360
        swept += abs(d)
        psi_prev = psi
        if not math.isfinite(psi) or abs(f["aero/alpha-deg"]) > 60.0:
            return None          # departed controlled flight
    v1 = f["velocities/vt-fps"]
    h1 = f["position/h-sl-ft"]
    rate = swept / seconds                                   # deg/s
    # specific energy rate, ft/s: dEs/dt with Es = h + V^2/2g
    es0 = h0 + v0 * v0 / 64.34
    es1 = h1 + v1 * v1 / 64.34
    ps = (es1 - es0) / seconds
    v_mps = (v0 + v1) / 2 * FT2M
    radius = v_mps / math.radians(rate) if rate > 0.1 else float("inf")
    if rate > 40.0:              # nothing with a wing sustains this; it tumbled
        return None
    return {"kts": kts, "rate_dps": rate, "radius_m": radius, "ps_fps": ps,
            "nz_peak": nz_peak, "mach": f["velocities/mach"]}


def characterise(model, alt_ft=15000):
    # Some stock models raise from run_ic (f104 references a property its own
    # systems never define). One bad airframe must not kill the survey.
    try:
        mm = max_mach(model)
    except Exception as e:
        return {"model": model, "error": "load/init failed: %s" % str(e)[:70]}
    if mm is None:
        return {"model": model, "error": "diverged or failed to load"}
    grid = [250, 300, 350, 400, 450, 500, 550, 600, 700]
    pts = []
    for k in grid:
        try:
            r = turn_probe(model, alt_ft, k)
        except Exception:
            r = None
        if r:
            pts.append(r)
    if not pts:
        return {"model": model, "max_mach": mm, "error": "no turn data"}
    best = max(pts, key=lambda r: r["rate_dps"])
    # sustained: best rate among points that are not losing energy
    holding = [r for r in pts if r["ps_fps"] > -50]
    sust = max(holding, key=lambda r: r["rate_dps"]) if holding else None
    return {
        "model": model, "max_mach": mm, "supersonic": mm >= 1.0,
        "corner_kts": best["kts"], "inst_rate_dps": best["rate_dps"],
        "min_radius_m": min(r["radius_m"] for r in pts),
        "sustained_rate_dps": sust["rate_dps"] if sust else None,
        "sustained_kts": sust["kts"] if sust else None,
        "points": pts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(CANDIDATES))
    ap.add_argument("--alt", type=int, default=15000)
    ap.add_argument("--out", default=os.path.join(os.environ.get("DOGFIGHT_OUT", "runs"), "airframes.json"))
    a = ap.parse_args()
    out = []
    print("%-7s %9s %8s %11s %10s %11s" % ("model", "max Mach", "corner", "inst rate", "min radius", "sust rate"))
    for m in a.models.split(","):
        r = characterise(m, a.alt)
        out.append(r)
        if "error" in r:
            print("%-7s  %s" % (m, r["error"]))
        else:
            print("%-7s %9.2f %6dkt %8.1f/s %8.0f m %8s/s" % (
                m, r["max_mach"], r["corner_kts"], r["inst_rate_dps"], r["min_radius_m"],
                ("%.1f" % r["sustained_rate_dps"]) if r["sustained_rate_dps"] else "n/a"))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
