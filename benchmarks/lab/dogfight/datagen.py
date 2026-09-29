"""Expert rollouts -> a behaviour-cloning dataset.

Why cloning and not RL. The subject of this study is precision, not policy
search. An RL policy would fly better and would also make every arm noisier,
and the seed spread of RL in a two-player game is far larger than the effect
being measured. Cloning a fixed, deterministic teacher gives a stable target
function whose degradation under quantization is attributable to the
quantization. RL is the obvious follow-up once the precision picture is clear.

Labels, one per decision (10 Hz):
  choice  the manoeuvre the expert flew
  score   threat level, 3 ordinal bins from the geometry
  noul    did blue actually enter red's gun envelope within the next second?
          This one is a lookahead label taken from the realised trajectory, so
          it is a genuine prediction target rather than a relabelled input.
"""
from __future__ import annotations

import argparse, os

import numpy as np

from env import Engagement, WEZ_MIN_M, WEZ_MAX_M, WEZ_ATA_DEG
from policy import encode, FEATURES
from pilot import MAN_IDX
import expert as E

OUT = os.environ.get("DOGFIGHT_OUT", "runs")
LOOKAHEAD = 10          # decisions, = 1.0 s at 10 Hz


def threat_label(o):
    if o["red_ata_deg"] < 30.0 and o["range_m"] < 1800.0:
        return 2
    if o["red_ata_deg"] < 60.0 and o["range_m"] < 4000.0:
        return 1
    return 0


def in_his_wez(o):
    """Blue inside red's gun envelope, read from blue's own observation."""
    return WEZ_MIN_M < o["range_m"] < WEZ_MAX_M and o["red_ata_deg"] < WEZ_ATA_DEG


OPPONENTS = {"expert": E.expert, "pursue": E.naive_pursue, "break": E.naive_break,
             "extend": E.naive_extend, "random": E.make_random(7)}


def generate(episodes=500, seed=0, model="f16"):
    eng = Engagement(model=model, seed=seed)
    names = list(OPPONENTS)
    X, Yc, Ys, danger, ep_id = [], [], [], [], []
    outcomes = {}
    for ep in range(episodes):
        red_fn = OPPONENTS[names[ep % len(names)]]
        bo, ro = eng.reset()
        done, info = False, {}
        while not done:
            man = E.expert(bo)
            X.append(encode(bo))
            Yc.append(MAN_IDX[man])
            Ys.append(threat_label(bo))
            danger.append(1.0 if in_his_wez(bo) else 0.0)
            ep_id.append(ep)
            bo, ro, done, info = eng.step(man, red_fn(ro))
        outcomes[info.get("outcome", "?")] = outcomes.get(info.get("outcome", "?"), 0) + 1

    X = np.asarray(X, dtype=np.float32)
    Yc = np.asarray(Yc, dtype=np.int64)
    Ys = np.asarray(Ys, dtype=np.int64)
    danger = np.asarray(danger, dtype=np.float32)
    ep_id = np.asarray(ep_id, dtype=np.int64)

    # noul: was blue in red's WEZ at any point in the next LOOKAHEAD decisions,
    # without leaking across an episode boundary
    n = len(danger)
    Yn = np.zeros(n, dtype=np.float32)
    for i in range(n):
        j = min(n, i + LOOKAHEAD + 1)
        same = ep_id[i:j] == ep_id[i]
        Yn[i] = danger[i:j][same].max()
    return X, Yc, Ys, Yn, ep_id, outcomes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    X, Yc, Ys, Yn, ep, outcomes = generate(a.episodes, a.seed)
    os.makedirs(OUT, exist_ok=True)
    path = a.out or os.path.join(OUT, f"bc_s{a.seed}_e{a.episodes}.npz")
    np.savez_compressed(path, X=X, Yc=Yc, Ys=Ys, Yn=Yn, ep=ep, features=np.array(FEATURES))
    print(f"{path}: {len(X)} decisions from {a.episodes} episodes")
    print("  manoeuvre mix:", {k: int(v) for k, v in
                               zip(*np.unique(Yc, return_counts=True))})
    print("  threat mix   :", {k: int(v) for k, v in
                               zip(*np.unique(Ys, return_counts=True))})
    print("  noul positive: %.1f%%" % (100 * Yn.mean()))
    print("  outcomes     :", outcomes)


if __name__ == "__main__":
    main()
