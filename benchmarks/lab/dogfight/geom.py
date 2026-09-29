"""Air-combat geometry: the state a BFM decision actually keys on.

JSBSim reports geodetic position and Euler attitude. Basic fighter manoeuvre
logic is expressed in relative terms instead -- range, who is pointing at whom,
and how fast the gap is closing -- so everything downstream reads these.

Conventions. Local frame is NED (north, east, down) centred on the blue
aircraft, which is close enough to flat-earth for engagements of a few km.
Angles are degrees, distances metres, speeds m/s.

  ATA (antenna train angle): angle between MY nose and the line of sight to
      him. 0 means I am pointing at him. This is the gun-tracking term.
  AA  (aspect angle): angle between HIS tail and the line of sight from him to
      me. 0 means I am directly behind him (his six). 180 means I am head-on.

The pair (ATA, AA) is what distinguishes the four classic positions: offensive
(both small), defensive (both large), head-on (ATA small, AA large) and
neutral. A policy that cannot see both cannot tell whether it is winning.
"""
from __future__ import annotations

import math

FT2M = 0.3048
DEG = math.pi / 180.0
R_EARTH = 6378137.0


def ned_offset(lat0, lon0, alt0_ft, lat1, lon1, alt1_ft):
    """Position of 1 relative to 0 in metres, NED."""
    dlat = (lat1 - lat0) * DEG
    dlon = (lon1 - lon0) * DEG
    n = dlat * R_EARTH
    e = dlon * R_EARTH * math.cos(lat0 * DEG)
    d = -(alt1_ft - alt0_ft) * FT2M          # down is positive
    return n, e, d


def body_axis(psi_deg, theta_deg):
    """Unit vector along the nose, in NED. Roll does not move the nose."""
    p, t = psi_deg * DEG, theta_deg * DEG
    return (math.cos(t) * math.cos(p), math.cos(t) * math.sin(p), -math.sin(t))


def _angle(u, v):
    du = math.sqrt(sum(c * c for c in u)) or 1e-9
    dv = math.sqrt(sum(c * c for c in v)) or 1e-9
    c = sum(a * b for a, b in zip(u, v)) / (du * dv)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def velocity_ned(vt_fps, psi_deg, theta_deg, gamma_deg=None):
    """Velocity vector in NED from speed and flight path, metres/second."""
    v = vt_fps * FT2M
    t = (gamma_deg if gamma_deg is not None else theta_deg) * DEG
    p = psi_deg * DEG
    return (v * math.cos(t) * math.cos(p), v * math.cos(t) * math.sin(p), -v * math.sin(t))


def engagement(blue, red):
    """Relative geometry between two aircraft state dicts.

    Each dict needs lat, lon, alt (ft), psi, theta (deg), vt (fps) and
    optionally gamma (deg). Returns the BFM terms plus the raw components a
    network can learn from.
    """
    n, e, d = ned_offset(blue["lat"], blue["lon"], blue["alt"],
                         red["lat"], red["lon"], red["alt"])
    los = (n, e, d)                                   # blue -> red
    rng = math.sqrt(n * n + e * e + d * d)

    blue_nose = body_axis(blue["psi"], blue["theta"])
    red_nose = body_axis(red["psi"], red["theta"])

    ata = _angle(blue_nose, los)                      # blue pointing at red
    # aspect: red's TAIL vs the line of sight red -> blue
    red_tail = tuple(-c for c in red_nose)
    aa = _angle(red_tail, tuple(-c for c in los))

    # closure: positive means the gap is shrinking
    vb = velocity_ned(blue["vt"], blue["psi"], blue["theta"], blue.get("gamma"))
    vr = velocity_ned(red["vt"], red["psi"], red["theta"], red.get("gamma"))
    dv = tuple(a - b for a, b in zip(vr, vb))
    closure = -sum(a * b for a, b in zip(dv, los)) / (rng or 1e-9)

    return {
        "range_m": rng,
        "ata_deg": ata,
        "aa_deg": aa,
        "closure_mps": closure,
        "dz_m": -d,                                   # red above blue is positive
        "blue_v_mps": blue["vt"] * FT2M,
        "red_v_mps": red["vt"] * FT2M,
        "dv_mps": (red["vt"] - blue["vt"]) * FT2M,
        "blue_alt_m": blue["alt"] * FT2M,
        "hca_deg": _angle(blue_nose, red_nose),       # heading crossing angle
        # his nose on me: the threat term. Identically 180 - aa, but a policy
        # reads it constantly and the derivation is easy to get backwards.
        "red_ata_deg": 180.0 - aa,
    }


def dcm_ned_to_body(phi_deg, theta_deg, psi_deg):
    """Rotation NED -> body, standard aerospace 3-2-1 (yaw, pitch, roll)."""
    p, t, s = phi_deg * DEG, theta_deg * DEG, psi_deg * DEG
    cp, sp = math.cos(p), math.sin(p)
    ct, st = math.cos(t), math.sin(t)
    cs, ss = math.cos(s), math.sin(s)
    return (
        (ct * cs,                 ct * ss,                 -st),
        (sp * st * cs - cp * ss,  sp * st * ss + cp * cs,  sp * ct),
        (cp * st * cs + sp * ss,  cp * st * ss - sp * cs,  cp * ct),
    )


def los_body(blue, red):
    """Line of sight to red expressed in blue's body axes (x fwd, y right, z down).

    Roll matters here, unlike for ATA: to point the nose at something you first
    roll so it lies in the pull plane, which needs to know where it sits in the
    canopy.
    """
    n, e, d = ned_offset(blue["lat"], blue["lon"], blue["alt"],
                         red["lat"], red["lon"], red["alt"])
    R = dcm_ned_to_body(blue.get("phi", 0.0), blue["theta"], blue["psi"])
    v = (n, e, d)
    b = tuple(sum(R[i][j] * v[j] for j in range(3)) for i in range(3))
    m = math.sqrt(sum(c * c for c in b)) or 1e-9
    return tuple(c / m for c in b)
