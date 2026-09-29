"""Does the environment discriminate? Round-robin of scripted policies.

If a good policy does not beat a bad one here, no quantization result measured
in this environment means anything, so this runs first and its numbers are the
licence for everything downstream.

Both sides get a correctly-derived observation from their own cockpit via
Engagement.observe(); an earlier version hand-patched red's view and produced
a table in which random beat the expert, which was the harness lying rather
than the expert being bad.
"""
from __future__ import annotations

import argparse, collections, time

from env import Engagement
import expert as E


def score(res):
    """(wins, losses, draws). Driving the opponent into the ground is a win:
    the hard deck is part of the fight, not an accounting oddity."""
    w = res["blue_win"] + res["red_crash"]
    l = res["red_win"] + res["blue_crash"]
    return w, l, sum(res.values()) - w - l


def duel(blue_fn, red_fn, n=40, seed=0, model="f16"):
    eng = Engagement(model=model, seed=seed)
    res = collections.Counter()
    times = []
    for _ in range(n):
        bo, ro = eng.reset()
        done, info = False, {}
        while not done:
            bo, ro, done, info = eng.step(blue_fn(bo), red_fn(ro))
        res[info.get("outcome", "?")] += 1
        if "t" in info:
            times.append(info["t"])
    return res, times


POLICIES = {
    "expert": E.expert,
    "pursue": E.naive_pursue,
    "break": E.naive_break,
    "extend": E.naive_extend,
    "random": E.make_random(0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    names = [n for n in POLICIES if not a.only or n in a.only.split(",")]
    t0 = time.time()
    print("blue \\ red    " + "".join("%12s" % n for n in names))
    for bn in names:
        row = "%-14s" % bn
        for rn in names:
            res, _ = duel(POLICIES[bn], POLICIES[rn], a.n, a.seed)
            row += "%12s" % ("%d-%d-%d" % score(res))
        print(row, flush=True)
    print("\n(cell = blue wins - blue losses - draws, out of %d; %.0fs)" % (a.n, time.time() - t0))


if __name__ == "__main__":
    main()
