"""A rule-based BFM pilot: the teacher for behaviour cloning, and the opponent.

Deliberately simple and readable. It is not a good fighter pilot; it only has
to be consistent, because its job is to provide a stable target function whose
degradation under quantization can be measured. An RL policy would fly better
and would also make every arm noisier, which is the wrong trade for a
precision study.

Priority order is the standard one taught for guns-only BFM: survive first,
then avoid an overshoot, then convert position, then rebuild energy.
"""
from __future__ import annotations

from pilot import MANEUVERS


def expert(g):
    """Observation dict -> manoeuvre name.

    `pursue` already rolls toward the target and pulls, so it is the correct
    "turn toward him" primitive. An earlier version reached for blind
    break_left/break_right in neutral geometry; those manoeuvres ignore the
    enemy entirely, so the expert bled energy in max-G turns and lost to naive
    pursuit. Breaks are now reserved for the case they exist for -- being shot
    at -- and are directed into the threat.
    """
    rng = g["range_m"]
    ata = g["ata_deg"]            # my nose on him
    threat = g["red_ata_deg"]     # his nose on me
    closure = g["closure_mps"]
    dz = g["dz_m"]
    mach = g["blue_mach"]

    # 0. the floor beats every other consideration. FLOOR_FT is 3000; start
    #    recovering with room to spare, because a 9G pull takes altitude.
    if g["blue_alt_ft"] < 5000.0 and g["blue_gamma_deg"] < 0.0:
        return "recover"

    defensive = threat < 30.0 and rng < 1800.0 and ata > 90.0

    # 1. he is shooting: break into him, hardest turn available
    if defensive:
        return "break_right" if g.get("los_right", 0.0) > 0 else "break_left"

    # 2. too slow to fight and not currently threatened: rebuild energy
    if mach < 0.50 and threat > 50.0:
        return "extend"

    # 3. closing too fast inside guns range: kill closure or overshoot
    if ata < 50.0 and rng < 1000.0 and closure > 140.0:
        return "high_yoyo"

    # 4. in the saddle: take the shot
    if ata < 25.0 and 150.0 < rng < 900.0:
        return "lead"

    # 5. well below him and far out: trade height for speed on the way in
    if ata < 60.0 and rng > 2500.0 and dz > 400.0:
        return "low_yoyo"

    # 6. anything else: point at him. Lag instead if the angles are poor and
    #    he is close, which prevents a flight-path overshoot.
    if ata > 90.0 and rng < 1200.0:
        return "lag"
    return "pursue"


def _left_of(g):
    """Crude: turn the shorter way toward him using the sign of the body-y LOS."""
    return g.get("los_right", 0.0) < 0.0


def naive_pursue(g):
    return "pursue"


def naive_break(g):
    return "break_left"


def naive_extend(g):
    return "extend"


def make_random(seed=0):
    import random
    r = random.Random(seed)
    return lambda g: r.choice(MANEUVERS)
