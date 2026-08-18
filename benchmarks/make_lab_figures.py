"""Figures for the second scaling-law round (experiments 1-6).

  1  activation precision vs weight precision -- the frontier that decides
     whether an SF-only array is viable, since the systolic array consumes
     activations in SF too
  2  the D/N law: PTQ damage against tokens-per-parameter
  3  does the coarse-grid advantage survive more data?
  4  depth, the axis tier C never varied
  5  per-layer dead fraction, the signal a bit-allocation rule would key on
  6  the learning-rate law under quantization

    python make_lab_figures.py --results-dir <dir> --out figures/
"""
import argparse, collections, glob, json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def load(d, exp):
    """Rows for one experiment, from `<exp>.jsonl` or from loose JSON files.

    The repo ships one JSONL per experiment; a live run directory holds one
    JSON per configuration. Both are accepted so figures can be regenerated
    from the archive or straight off a pod.
    """
    arch = os.path.join(d, f"{exp}.jsonl")
    if os.path.exists(arch):
        with open(arch) as f:
            return [json.loads(l) for l in f if l.strip()]
    out = []
    for f in glob.glob(os.path.join(d, "*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if r.get("exp") == exp and r.get("complete"):
            out.append(r)
    return out


def fig_act(rows, out):
    """exp1: the (bits_w, bits_a) frontier, plus the clipping mechanism."""
    if not rows: return None
    g = {(r["bits_w"], r["bits_a"]): r for r in rows}
    W = sorted({w for w, _ in g}); A = sorted({a for _, a in g if a})
    fig, ax = plt.subplots(1, 3, figsize=(19.5, 4.8))

    for i, w in enumerate(W):
        xs = [a for a in A if (w, a) in g]
        if not xs: continue
        ax[0].plot(xs, [g[(w, a)]["best_acc"] for a in xs], marker="o", ms=5,
                   lw=1.8, color=C[i % len(C)], label=f"weights SF{w}")
        if (w, 0) in g:
            ax[0].axhline(g[(w, 0)]["best_acc"], color=C[i % len(C)], ls=":", lw=1, alpha=0.6)
    ax[0].set_xlabel("activation precision (bits)"); ax[0].set_ylabel("best top-1 (%)")
    ax[0].set_title("(a) activations saturate well above weights\n(dotted = unquantized activations)", fontsize=10)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    # heatmap of the frontier
    M = np.full((len(W), len(A)), np.nan)
    for i, w in enumerate(W):
        for j, a in enumerate(A):
            if (w, a) in g: M[i, j] = g[(w, a)]["best_acc"]
    im = ax[1].imshow(M, cmap="viridis", aspect="auto", origin="lower")
    ax[1].set_xticks(range(len(A))); ax[1].set_xticklabels(A)
    ax[1].set_yticks(range(len(W))); ax[1].set_yticklabels(W)
    ax[1].set_xlabel("activation bits"); ax[1].set_ylabel("weight bits")
    ax[1].set_title("(b) accuracy over the joint grid", fontsize=10)
    for i in range(len(W)):
        for j in range(len(A)):
            if not np.isnan(M[i, j]):
                ax[1].text(j, i, f"{M[i,j]:.0f}", ha="center", va="center",
                           color="w", fontsize=8)
    fig.colorbar(im, ax=ax[1], fraction=0.046)

    mech = [r for r in rows if r.get("bits_a") and "clip_frac_mean" in r]
    if mech:
        for i, w in enumerate(sorted({r["bits_w"] for r in mech})):
            pts = sorted([(r["bits_a"], 100 * r["clip_frac_mean"]) for r in mech
                          if r["bits_w"] == w])
            if pts:
                ax[2].plot(*zip(*pts), marker="o", ms=5, lw=1.8,
                           color=C[i % len(C)], label=f"weights SF{w}")
        ax[2].set_yscale("log")
        ax[2].set_xlabel("activation precision (bits)")
        ax[2].set_ylabel("activations clipped at the SF bound (%)")
        ax[2].set_title("(c) mechanism: clipping, not grid coarseness", fontsize=10)
        ax[2].legend(fontsize=8)
    else:
        ax[2].text(.5, .5, "clip_frac recorded only for cells\nstarted after the "
                   "mechanism patch", ha="center", va="center", fontsize=9)
        ax[2].set_axis_off()
    ax[2].grid(alpha=0.3)
    fig.tight_layout(); p = os.path.join(out, "lab_exp1_activation.png")
    fig.savefig(p, dpi=180); plt.close(fig); return p


def fig_dn(rows, out):
    """exp2: penalty against tokens-per-parameter, the proposed governing axis."""
    if not rows: return None
    fig, ax = plt.subplots(1, 2, figsize=(13.6, 4.8))
    bysz = collections.defaultdict(dict)
    for r in rows: bysz[(r["size"], r["step"])][r["bits"]] = r
    sizes = ["160m", "410m", "1.4b"]
    for i, s in enumerate(sizes):
        pts = []
        for (sz, st), d in bysz.items():
            if sz != s or 0 not in d or 6 not in d: continue
            dn = d[6]["tokens"] / d[6]["params"]
            pts.append((dn, min(d[6]["val_loss"] - d[0]["val_loss"], 20)))
        if pts:
            pts.sort()
            ax[0].plot(*zip(*pts), marker="o", ms=5, lw=1.8, color=C[i], label=s)
    ax[0].set_xscale("log"); ax[0].set_xlabel("tokens per parameter (D/N)")
    ax[0].set_ylabel("SF6 penalty vs FP16 (nats)")
    ax[0].set_title("(a) U-shaped in D/N, but the curves do not collapse:\n"
                    "D/N is not the governing variable", fontsize=10)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    # p_min: cheapest precision holding the penalty under 0.1 nats.  Reading
    # off the tested grid alone quantizes this to the sampled precisions, so
    # interpolate the crossing in (p, log penalty) between adjacent points.
    THR = 0.1
    for i, s in enumerate(sizes):
        pts = []
        for (sz, st), d in bysz.items():
            if sz != s or 0 not in d:
                continue
            ref = d[0]["val_loss"]
            bs = sorted(b for b in d if b)
            pen = [(b, max(d[b]["val_loss"] - ref, 1e-6)) for b in bs]
            cross = None
            for (b0, e0), (b1, e1) in zip(pen, pen[1:]):
                if e0 > THR >= e1:                     # crosses between b0 and b1
                    f = (math.log(e0) - math.log(THR)) / (math.log(e0) - math.log(e1))
                    cross = b0 + f * (b1 - b0)
                    break
            if cross is None and pen and pen[0][1] <= THR:
                cross = pen[0][0]                      # already under at cheapest
            if cross is not None:
                dn = d[bs[0]]["tokens"] / d[bs[0]]["params"]
                pts.append((dn, cross))
        if pts:
            pts.sort()
            ax[1].plot(*zip(*pts), marker="s", ms=6, lw=1.8, color=C[i], label=s)
    ax[1].set_xscale("log"); ax[1].set_xlabel("tokens per parameter (D/N)")
    ax[1].set_ylabel("cheapest precision holding +0.1 nats (interpolated)")
    ax[1].set_title("(b) precision needed, vs how hard the model was trained", fontsize=10)
    ax[1].legend(fontsize=7.5, loc="upper left"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); p = os.path.join(out, "lab_exp2_dn_law.png")
    fig.savefig(p, dpi=180); plt.close(fig); return p


def fig_reg(rows, out):
    """exp3: does coarse-is-better die as tokens-per-parameter grows?

    Tier D found lower precision beating higher under scale absorption. If that
    inversion is a regularisation effect it should be a small-data artefact:
    give the model more tokens per parameter and the coarser grid should lose
    its advantage and then fall behind.
    """
    if not rows: return None
    g = collections.defaultdict(dict)
    for r in rows: g[r["tpp"]][r["bits"]] = r["final_val_loss"]
    fig, ax = plt.subplots(1, 2, figsize=(13.6, 4.8))
    tpps = sorted(g)

    # (a) absolute loss -- needs no control, so it is readable while the sweep
    # is still filling in
    for i, b in enumerate([0, 2, 3, 4, 6]):
        pts = [(t, g[t][b]) for t in tpps if b in g[t]]
        if pts:
            ax[0].plot(*zip(*pts), marker="o", ms=6, lw=1.8, color=C[i],
                       label="FP32" if b == 0 else f"SF{b}")
    ax[0].set_xscale("log"); ax[0].set_xlabel("tokens per parameter")
    ax[0].set_ylabel("final val loss (nats)")
    ax[0].set_title("(a) loss vs data, one curve per precision", fontsize=10)
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

    # (b) penalty against each cell's own FP32 control
    for i, b in enumerate([2, 3, 4, 6]):
        pts = [(t, g[t][b] - g[t][0]) for t in tpps if b in g[t] and 0 in g[t]]
        if pts:
            ax[1].plot(*zip(*pts), marker="o", ms=6, lw=1.8, color=C[i],
                       label=f"SF{b}")
    ax[1].axhline(0, color="k", lw=0.8, ls=":")
    ax[1].set_xscale("log"); ax[1].set_xlabel("tokens per parameter")
    ax[1].set_ylabel("penalty vs FP32 control (nats)")
    ax[1].set_title("(b) if the inversion is regularisation, the gaps\n"
                    "should widen as data grows", fontsize=10)
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)

    fig.tight_layout(); p = os.path.join(out, "lab_exp3_regularisation.png")
    fig.savefig(p, dpi=180); plt.close(fig); return p


def fig_depth(rows, out):
    """exp4: the axis tier C never varied."""
    if not rows: return None
    g = collections.defaultdict(list)
    for r in rows: g[(r["depth"], r["bits_w"])].append(r["best_acc"])
    D = sorted({d for d, _ in g}); B = sorted({b for _, b in g})
    fig, ax = plt.subplots(1, 2, figsize=(13.6, 4.8))
    for i, d in enumerate(D):
        xs = [b for b in B if (d, b) in g]
        ys = [np.mean(g[(d, b)]) for b in xs]
        es = [np.ptp(g[(d, b)]) / 2 if len(g[(d, b)]) > 1 else 0 for b in xs]
        ax[0].errorbar(xs, ys, yerr=es, marker="o", ms=5, lw=1.8, capsize=3,
                       color=C[i % len(C)], label=f"ResNet-{d}")
    ax[0].set_xlabel("SuperFloat precision (bits)"); ax[0].set_ylabel("best top-1 (%)")
    ax[0].set_title("(a) precision vs depth", fontsize=10)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    for i, b in enumerate(B):
        xs = [d for d in D if (d, b) in g]
        ax[1].plot(xs, [np.ptp(g[(d, b)]) if len(g[(d, b)]) > 1 else 0 for d in xs],
                   marker="o", ms=5, lw=1.8, color=C[i % len(C)], label=f"SF{b}")
    ax[1].set_xlabel("depth"); ax[1].set_ylabel("seed spread (pp)")
    ax[1].set_title("(b) the paper's SF16@R56 spread was 12.0 -- does it reappear?", fontsize=10)
    ax[1].legend(fontsize=7.5, loc="upper left"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); p = os.path.join(out, "lab_exp4_depth.png")
    fig.savefig(p, dpi=180); plt.close(fig); return p


def fig_alloc(rows, out):
    """exp5: per-layer dead fraction, the signal an allocation rule keys on."""
    if not rows: return None
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i, r in enumerate(sorted(rows, key=lambda r: r["bits"])):
        L = r["layers"]
        ax.plot([l["fan_in"] for l in L], [100 * l["dead_plain"] for l in L],
                "o", ms=6, color=C[i % len(C)], label=f"SF{r['bits']} plain")
        ax.plot([l["fan_in"] for l in L], [100 * l["dead_norm"] for l in L],
                "^", ms=6, mfc="none", color=C[i % len(C)], label=f"SF{r['bits']} + norm")
    ax.set_xscale("log"); ax.set_xlabel("layer fan_in")
    ax.set_ylabel("weights dead at init (%)")
    ax.set_title("Per-layer dead fraction: plain SF tracks fan_in,\n"
                 "normalisation removes the dependence", fontsize=10)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); p = os.path.join(out, "lab_exp5_per_layer.png")
    fig.savefig(p, dpi=180); plt.close(fig); return p


def fig_lr(rows, out):
    """exp6: does usable step size depend on precision?  It does not.

    The hypothesis was eta*(p) ~ 2^p: one fewer bit halves the usable learning
    rate, so SF3 should tolerate roughly 1/8000 of SF16's step.  Across a 600x
    sweep nothing diverged at any precision, and the accuracy optimum sits at
    4e-3 for SF4, SF6, SF8 and SF16 alike.  The curves differ in height, not
    in position.
    """
    if not rows: return None
    g = {(r["bits"], r["lr"]): r for r in rows}
    B = sorted({b for b, _ in g}); L = sorted({l for _, l in g})
    fig, ax = plt.subplots(1, 3, figsize=(19.5, 4.8))

    M = np.full((len(B), len(L)), np.nan)
    for i, b in enumerate(B):
        for j, l in enumerate(L):
            if (b, l) in g: M[i, j] = g[(b, l)]["best_acc"]
    im = ax[0].imshow(M, cmap="viridis", aspect="auto", origin="lower")
    ax[0].set_xticks(range(len(L))); ax[0].set_xticklabels([f"{l:.0e}" for l in L], rotation=45)
    ax[0].set_yticks(range(len(B))); ax[0].set_yticklabels([f"SF{b}" for b in B])
    ax[0].set_xlabel("learning rate"); ax[0].set_ylabel("precision")
    ax[0].set_title("(a) accuracy over the (p, lr) grid", fontsize=10)
    fig.colorbar(im, ax=ax[0], fraction=0.046)

    for i, b in enumerate(B):
        xs = [l for l in L if (b, l) in g]
        ax[1].plot(xs, [g[(b, l)]["best_acc"] for l in xs], marker="o", ms=4,
                   lw=1.7, color=C[i % len(C)], label=f"SF{b}")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("learning rate"); ax[1].set_ylabel("best top-1 (%)")
    ax[1].set_title("(b) the curves differ in height, not in position", fontsize=10)
    ax[1].legend(fontsize=8, ncol=2); ax[1].grid(alpha=0.3)

    opt = [(b, max((l for l in L if (b, l) in g),
                   key=lambda l: g[(b, l)]["best_acc"])) for b in B]
    ax[2].plot(*zip(*opt), marker="o", ms=8, lw=2.0, color=C[0],
               label="measured optimum")
    bs = np.array(B, float)
    a0 = opt[0]
    ax[2].plot(bs, a0[1] * 2.0 ** (bs - a0[0]), ls="--", color=C[3],
               label=r"hypothesis $\eta^*\propto 2^{\,p}$")
    div = sum(1 for k in g if g[k]["diverged"])
    ax[2].set_yscale("log"); ax[2].set_ylim(1e-4, 1e2)
    ax[2].set_xlabel("precision (bits)"); ax[2].set_ylabel("learning rate")
    ax[2].set_title("(c) the optimum does not move with precision", fontsize=10)
    ax[2].text(0.5, 0.06, f"{div}/{len(g)} cells diverged: none, anywhere,\n"
                          f"across a {max(L)/min(L):.0f}x range of step size",
               transform=ax[2].transAxes, ha="center", fontsize=8.5,
               bbox=dict(fc="#fff3cd", ec="#d39e00", alpha=0.9))
    ax[2].legend(fontsize=8, loc="upper left"); ax[2].grid(alpha=0.3)

    fig.tight_layout(); p = os.path.join(out, "lab_exp6_lr_law.png")
    fig.savefig(p, dpi=180); plt.close(fig); return p


def fig_plain(rows4, rows7, out):
    """exp7 against exp4: the same depth sweep with and without normalisation.

    exp4 runs the channel-normalised condition, so on its own it cannot speak
    to the paper's SF16@ResNet-56 seed spread of 12.0 points, which came from a
    pipeline that had no per-channel scale. exp7 reruns the two precisions that
    spread was reported at, without normalisation, at three seeds.
    """
    if not rows7: return None
    def bucket(rows):
        g = collections.defaultdict(list)
        for r in rows:
            g[(r["depth"], r["bits_w"])].append(r["best_acc"])
        return g
    g4, g7 = bucket(rows4), bucket(rows7)
    B = sorted({b for _, b in g7})
    D = sorted({d for d, _ in g7})
    fig, ax = plt.subplots(1, 2, figsize=(13.6, 4.8))

    for i, b in enumerate(B):
        for g, ls, lab in ((g4, "-", "+ channel norm"), (g7, "--", "plain SF")):
            xs = [d for d in D if (d, b) in g]
            if not xs:
                continue
            ys = [np.mean(g[(d, b)]) for d in xs]
            es = [np.ptp(g[(d, b)]) / 2 if len(g[(d, b)]) > 1 else 0 for d in xs]
            ax[0].errorbar(xs, ys, yerr=es, marker="o", ms=5, lw=1.8, ls=ls,
                           capsize=3, color=C[i % len(C)], label=f"SF{b}, {lab}")
    ax[0].set_xlabel("depth"); ax[0].set_ylabel("best top-1 (%)")
    ax[0].set_title("(a) accuracy vs depth, with and without normalisation",
                    fontsize=10)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    for i, b in enumerate(B):
        for g, ls, lab in ((g4, "-", "+ channel norm"), (g7, "--", "plain SF")):
            xs = [d for d in D if (d, b) in g and len(g[(d, b)]) > 1]
            if not xs:
                continue
            ax[1].plot(xs, [np.ptp(g[(d, b)]) for d in xs], marker="o", ms=5,
                       lw=1.8, ls=ls, color=C[i % len(C)], label=f"SF{b}, {lab}")
    ax[1].axhline(12.0, color="#d62728", lw=1.6, ls=":")
    ax[1].text(0.02, 12.3, "spread reported in the paper (SF16 @ ResNet-56)",
               fontsize=8, color="#d62728", transform=ax[1].get_yaxis_transform())
    ax[1].set_xlabel("depth"); ax[1].set_ylabel("seed spread (pp)")
    ax[1].set_title("(b) does the reported instability reproduce?", fontsize=10)
    ax[1].legend(fontsize=8, loc="upper left"); ax[1].grid(alpha=0.3)

    fig.tight_layout(); p = os.path.join(out, "lab_exp7_plain_depth.png")
    fig.savefig(p, dpi=180); plt.close(fig); return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="."); ap.add_argument("--out", default="figures")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    made = []
    got = {}
    for exp, fn in (("exp1", fig_act), ("exp2", fig_dn), ("exp3", fig_reg),
                    ("exp4", fig_depth), ("exp5_profile", fig_alloc), ("exp6", fig_lr)):
        rows = load(a.results_dir, exp)
        got[exp] = rows
        print(f"  {exp}: {len(rows)} rows")
        p = fn(rows, a.out)
        if p: made.append(p)
    rows7 = load(a.results_dir, "exp7")
    print(f"  exp7: {len(rows7)} rows")
    p = fig_plain(got.get("exp4", []), rows7, a.out)
    if p: made.append(p)
    for p in made: print("  " + p)


if __name__ == "__main__":
    main()
