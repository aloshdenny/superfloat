#!/usr/bin/env python3
"""Optimistic log–log regression of SF8 1B val ppl/loss out to 3 epochs.

Fits val_ppl = A * T^b on the last 100 observed evals (combined 4090+H100
metrics.jsonl). Val loss is log(ppl). Emits a step-cadence jsonl matching
the live 4090 recipe (log every 50 steps, 32768 tok/step) and a smooth
continuous plot of training tokens / epochs vs ppl and loss only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OBS = HERE / "metrics-observed.jsonl"
EPOCH = 20_000_000_000
NOW_TOK = 1_204_060_160
NOW_STEP = 23_134
TOK_PER_STEP = 32_768  # live 4090: batch 2, accum 8, seq 2048
LOG_EVERY = 50
END_TOK = 3 * EPOCH  # 60B = 3 epochs of the 20B FineWeb-Edu shard
LAST_N = 100

OUT_JSONL = HERE / "metrics-optimistic-3ep.jsonl"
OUT_PNG = HERE / "tokens_epochs_ppl_loss.png"
OUT_SVG = HERE / "tokens_epochs_ppl_loss.svg"
OUT_FIT = HERE / "fit.json"


def load_obs(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def tok(r):
    return float(r.get("tokens_seen") or r.get("tokens"))


def fit_power(t, y):
    b, a = np.polyfit(np.log(t), np.log(y), 1)
    return float(np.exp(a)), float(b)


def ppl_at(A, b, t):
    t = np.asarray(t, dtype=float)
    return A * t**b


def main():
    rows = load_obs(OBS)
    fit_rows = [r for r in rows if tok(r) >= 50e6 and r["val_ppl"] < 200]
    t = np.array([tok(r) for r in fit_rows[-LAST_N:]], dtype=float)
    p = np.array([r["val_ppl"] for r in fit_rows[-LAST_N:]], dtype=float)
    A, b = fit_power(t, p)
    pred = ppl_at(A, b, t)
    rms = float(np.sqrt(np.mean((pred - p) ** 2)))

    fit = {
        "model": "val_ppl = A * tokens_seen ** b",
        "loss": "val_loss = log(val_ppl)",
        "A": A,
        "b": b,
        "fit_on": f"last {LAST_N} evals of combined metrics.jsonl",
        "fit_tokens": [float(t[0]), float(t[-1])],
        "in_sample_rms_ppl": rms,
        "now_tokens": NOW_TOK,
        "now_step": NOW_STEP,
        "end_tokens": END_TOK,
        "epoch_tokens": EPOCH,
        "tok_per_step": TOK_PER_STEP,
        "log_every": LOG_EVERY,
        "at_1ep_20B": float(ppl_at(A, b, EPOCH)),
        "at_2ep_40B": float(ppl_at(A, b, 2 * EPOCH)),
        "at_3ep_60B": float(ppl_at(A, b, 3 * EPOCH)),
    }
    OUT_FIT.write_text(json.dumps(fit, indent=2) + "\n")

    last = rows[-1]
    n_proj = 0
    with OUT_JSONL.open("w") as f:
        rec = {
            "exp": "sf1b",
            "tag": "sf8_ln_all_s0",
            "source": "observed",
            "step": last["step"],
            "tokens_seen": last["tokens_seen"],
            "epoch": last["tokens_seen"] / EPOCH,
            "val_ppl": last["val_ppl"],
            "val_loss": last["val_loss"],
            "train_ppl": last["train_ppl"],
            "train_loss": last["train_loss"],
        }
        f.write(json.dumps(rec) + "\n")
        # Next log-aligned step on the 4090 cadence, through 3 epochs.
        step = ((NOW_STEP // LOG_EVERY) + 1) * LOG_EVERY
        while True:
            tokens_seen = NOW_TOK + (step - NOW_STEP) * TOK_PER_STEP
            if tokens_seen > END_TOK:
                break
            vp = float(ppl_at(A, b, tokens_seen))
            vl = math.log(vp)
            rec = {
                "exp": "sf1b",
                "tag": "sf8_ln_all_s0_optimistic",
                "source": "optimistic_regression",
                "step": step,
                "tokens_seen": int(tokens_seen),
                "epoch": tokens_seen / EPOCH,
                "val_ppl": vp,
                "val_loss": vl,
            }
            f.write(json.dumps(rec) + "\n")
            n_proj += 1
            step += LOG_EVERY

    # Smooth continuous curves (not the discrete log grid).
    ts = np.linspace(NOW_TOK, END_TOK, 4000)
    vp = ppl_at(A, b, ts)
    vl = np.log(vp)
    ep = ts / EPOCH
    tb = ts / 1e9

    plt.rcParams.update({
        "font.size": 11,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": True,
    })
    fig, (ax_p, ax_l) = plt.subplots(
        2, 1, sharex=True, figsize=(9.5, 6.4),
        gridspec_kw={"hspace": 0.08},
    )
    ax_p.plot(tb, vp, color="#1f4e79", lw=2.0, solid_capstyle="round")
    ax_l.plot(tb, vl, color="#7a3e1d", lw=2.0, solid_capstyle="round")
    ax_p.set_ylabel("perplexity")
    ax_l.set_ylabel("loss")
    ax_l.set_xlabel("training tokens (B)  ·  epochs of 20B")

    for ax in (ax_p, ax_l):
        for e in (1, 2, 3):
            ax.axvline(e * EPOCH / 1e9, color="#888888", lw=0.7, ls="--")
        ax.axvline(NOW_TOK / 1e9, color="#888888", lw=0.7, ls=":")
        ax.set_xlim(0, END_TOK / 1e9)
        ax.grid(False)

    # Epoch labels on the shared x as a second line of ticks.
    ax_l.set_xticks([NOW_TOK / 1e9, 20, 40, 60])
    ax_l.set_xticklabels([
        f"{NOW_TOK/1e9:.1f}B\n0.06 ep",
        "20B\n1 epoch",
        "40B\n2 epochs",
        "60B\n3 epochs",
    ])
    ax_p.set_ylim(bottom=min(vp) * 0.92, top=max(vp) * 1.06)
    ax_l.set_ylim(bottom=min(vl) * 0.985, top=max(vl) * 1.02)

    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")
    plt.close(fig)

    print("fit A", A, "b", b, "rms", rms)
    print("ppl 20/40/60B", fit["at_1ep_20B"], fit["at_2ep_40B"], fit["at_3ep_60B"])
    print("projected logs", n_proj)
    print("wrote", OUT_JSONL)
    print("wrote", OUT_PNG)
    print("wrote", OUT_SVG)


if __name__ == "__main__":
    main()
