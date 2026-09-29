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
    """Observation dict -> manoeuvre, over the 21-manoeuvre vocabulary.

    Ordered the way guns-only BFM is taught: survive, then deny the shot, then
    avoid the overshoot, then convert, then rebuild energy. The earlier
    9-manoeuvre version answered `pursue` to 71% of states, which flattened the
    decision distribution so far that the precision study could not resolve the
    SF8-to-SF4 band on win rate. A wider vocabulary spreads it.
    """
    rng = g["range_m"]
    ata = g["ata_deg"]            # my nose on him
    threat = g["red_ata_deg"]     # his nose on me
    closure = g["closure_mps"]
    dz = g["dz_m"]
    mach = g["blue_mach"]
    alt = g["blue_alt_ft"]
    right = g.get("los_right", 0.0)

    # 0. the floor outranks every tactical consideration
    if alt < 5000.0 and g["blue_gamma_deg"] < 0.0:
        return "recover"

    # 1. he is shooting, inside guns range, tracking: last ditch
    if threat < 12.0 and rng < 900.0 and ata > 100.0:
        return "last_ditch"

    # 2. he is threatening from close: deny the solution.
    #    Kept deliberately narrow. A wider gate (threat<30, rng<1800) diverted
    #    the expert into energy-bleeding defence often enough that it lost to
    #    naive pursuit, which simply points at the enemy and shoots.
    if threat < 25.0 and rng < 1200.0 and ata > 110.0:
        if rng < 1200.0 and mach < 0.65:
            # slow and defensive: force the overshoot rather than out-turn him
            return "barrel_roll_defense"
        # a spiral trades altitude for rate, so it needs altitude to trade.
        # Triggering it at 12k drove a 20% crash rate.
        if alt > 18000.0 and mach > 0.75:
            return "defensive_spiral"
        return "break_right" if right > 0 else "break_left"

    # 3. he is actually tracking me and close: unpredictable, not maximal.
    #    A wider trigger (threat<45, rng<3000) spent 13.6% of all decisions
    #    jinking, which gains no angles and loses fights.
    if threat < 25.0 and rng < 1500.0 and ata > 90.0:
        return "jink"

    # 4. post-overshoot: whoever gets slow and small wins the scissors
    if rng < 900.0 and abs(closure) < 40.0 and 60.0 < ata < 140.0:
        return "rolling_scissors" if abs(dz) > 200.0 else "flat_scissors"

    # 5. offensive, closing too fast: kill closure before the overshoot
    if ata < 50.0 and rng < 1000.0 and closure > 140.0:
        return "lag_displacement_roll" if rng < 600.0 else "high_yoyo"

    # 6. in the saddle: take the shot
    if ata < 25.0 and 150.0 < rng < 900.0:
        return "lead"

    # 7. offensive, needs to close
    if ata < 60.0:
        if rng > 2500.0 and dz > 400.0:
            return "low_yoyo"
        return "pursue" if ata < 35.0 else "lag"

    # 8. neutral, high aspect, plenty of energy: reverse in the vertical
    if ata > 120.0 and mach > 0.85 and alt > 14000.0:
        return "split_s" if dz < -300.0 else "immelmann"

    # 9. slow and not threatened: rebuild energy
    if mach < 0.55 and threat > 60.0:
        return "extend"

    # 10. he is off the nose but not a threat: turn toward him efficiently
    if ata > 110.0 and rng < 1200.0 and mach > 0.8:
        return "pitchback"
    return "hard_turn" if ata > 75.0 else "pursue"


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


# --------------------------------------------------------------------------
# Opponent league.
#
# The sweep previously trained against one scripted expert, which lets the
# policy overfit a single opponent's habits. These are the archetypes from
# TACTICS.md section 3.1, written as the minimum that forces genuinely
# different counters: an energy fighter must be denied separation, an angles
# fighter must not be followed slow, and a defender must be converted before
# it regenerates energy.
# --------------------------------------------------------------------------

def energy_fighter(g):
    """P4: refuses an angles fight. Extends, uses the vertical, re-attacks."""
    if g["blue_alt_ft"] < 5000.0 and g["blue_gamma_deg"] < 0.0:
        return "recover"
    if g["blue_mach"] < 0.80:
        return "extend"
    if g["ata_deg"] < 30.0 and 150.0 < g["range_m"] < 900.0:
        return "lead"
    if g["red_ata_deg"] < 25.0 and g["range_m"] < 1200.0:
        return "break_right" if g.get("los_right", 0) > 0 else "break_left"
    if g["range_m"] > 3000.0:
        return "pursue"
    return "high_yoyo" if g["closure_mps"] > 100.0 else "lag"


def angles_fighter(g):
    """P5: wants a one-circle fight. Gets slow, points, forces the scissors."""
    if g["blue_alt_ft"] < 5000.0 and g["blue_gamma_deg"] < 0.0:
        return "recover"
    if g["range_m"] < 900.0 and abs(g["closure_mps"]) < 60.0:
        return "flat_scissors"
    if g["ata_deg"] < 25.0 and 150.0 < g["range_m"] < 900.0:
        return "lead"
    if g["ata_deg"] > 90.0:
        return "pitchback" if g["blue_mach"] > 0.8 else "hard_turn"
    return "pursue"


def defensive_turner(g):
    """P3/D-side: always defensive, predictable enough to be exploitable."""
    if g["blue_alt_ft"] < 5000.0 and g["blue_gamma_deg"] < 0.0:
        return "recover"
    if g["red_ata_deg"] < 40.0 and g["range_m"] < 2500.0:
        return "jink" if g["range_m"] < 1200.0 else "break_left"
    return "extend"


def vertical_fighter(g):
    """Fights in the vertical: yo-yos, Immelmanns, split-S."""
    if g["blue_alt_ft"] < 6000.0 and g["blue_gamma_deg"] < 0.0:
        return "recover"
    if g["ata_deg"] < 25.0 and 150.0 < g["range_m"] < 900.0:
        return "lead"
    if g["ata_deg"] > 100.0 and g["blue_mach"] > 0.85:
        return "split_s" if g["dz_m"] < -200.0 else "immelmann"
    if g["closure_mps"] > 120.0 and g["range_m"] < 1200.0:
        return "high_yoyo"
    return "low_yoyo" if g["dz_m"] > 300.0 else "pursue"


LEAGUE = {
    "expert": expert,
    "energy": energy_fighter,
    "angles": angles_fighter,
    "defensive": defensive_turner,
    "vertical": vertical_fighter,
    "pursue": naive_pursue,
    "break": naive_break,
    "extend": naive_extend,
    "random": make_random(7),
}
