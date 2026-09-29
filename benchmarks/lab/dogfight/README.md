# Closed-loop dogfight: SuperFloat in a control loop

Every SuperFloat result so far scores a model on a static corpus. This asks a
different question: **does a quantized policy still win the fight?** The model
makes a bounded decision many times a second, acts on the world, and then sees
the consequences of its own earlier decisions. Errors compound instead of
averaging out, which is the failure mode a perplexity number cannot show.

## Why a decision model, not an LLM

The subjects are System One / typed-decision models in the sense of TypeSafe's
Jev and Convai's Laya: unstructured state in, typed decisions out, one parallel
forward pass, no autoregression. That shape is the only one that fits a control
loop at all -- a token-by-token decode cannot close a 10 Hz loop, let alone
60 Hz -- and it is the shape SuperFloat's fixed-point datapath targets.

`policy.py` mirrors the three question types those models expose:

| head | type | question |
| --- | --- | --- |
| `choice` | categorical over 9 manoeuvres | what do I fly now? |
| `score` | ordinal, 3 levels | how threatened am I? |
| `noul` | P(true) | am I about to be inside his gun envelope? |

## Split between decision and execution

The model chooses a manoeuvre. `pilot.py` -- ordinary deterministic code --
flies it. Quantization therefore changes *which* manoeuvre is chosen and never
how well it is executed, so a closed-loop loss is attributable to the decision.
This is the same separation a real system would use, and it is what makes the
experiment interpretable.

## Environment

JSBSim 1.3.1, two F-16s, flight dynamics at 60 Hz, decisions at 10 Hz. Scoring
is guns-only: 150-900 m, pipper within 8 deg, held 0.5 s. The hard deck at
3000 ft is part of the fight -- driving an opponent into the ground is a win,
and failing to call `recover` is a decision error the study can see.

Sim cost is negligible (~1300x realtime for two aircraft), so the model forward
passes dominate, which is the regime we want to measure.

## Validation: the harness discriminates

Round-robin, 40 engagements per cell, blue wins - losses - draws:

```
              expert      pursue      break      extend     random
expert       10-16-14    16-12-12     1-0-39     8-0-32    18-0-22
pursue        9-16-15    10-13-17     2-0-38     4-0-36    10-5-25
break          1-1-38      1-1-38     1-1-38     0-0-40     0-1-39
extend         2-8-30      1-5-34     0-0-40     0-1-39     1-2-37
random         1-14-25     5-14-21     4-0-36     1-0-39     2-6-32
```

The scripted expert beats random 18-0 and extend 8-0, and beats naive pursuit
from both sides. After symmetrising the start geometry (see `reset`), self-play
runs 21-17 over 60 fights -- near even, as it must be.

Two bugs found on the way to that table, both recorded because they produced
confident nonsense rather than errors: red was being handed a hand-patched
observation instead of its own, and an opponent flying into the ground was
scored a draw rather than a loss.

## Files

```
geom.py        BFM geometry: range, ATA, aspect, closure, body-frame LOS
pilot.py       manoeuvre -> control surfaces (JSBSim signs measured, not assumed)
env.py         two-aircraft engagement, WEZ scoring, hard deck
expert.py      rule-based BFM teacher and opponent
policy.py      the decision model: 2.11M params, choice/score/noul heads
tournament.py  round-robin validation of the harness
```

## Results

Three seeds per arm, 30 epochs, identical data in identical order. fp32 is the
100% baseline; published numbers for other models are never the reference.

### Accuracy and calibration disagree, and that is the finding

| arm | balanced acc (rel) | ECE (rel) | closed-loop agreement |
| --- | --- | --- | --- |
| fp16 / bf16 | 100.0% | 1.0x | 0.947 / 0.949 |
| SF8 | 99.9% | 1.1x | 0.945 |
| SF6 | 99.9% | 1.3x | 0.929 |
| SF4 | 100.0% | 1.9x | 0.910 |
| SF3 | 95.5% | 5.5x | 0.692 |
| SF2 | 95.0% | 5.6x | 0.678 |
| SF8 + saturated outputs | 99.6% | **0.5x** | **0.961** |
| SF6 + saturated outputs | 99.3% | 0.6x | 0.936 |
| SF4 + saturated outputs | 99.4% | 1.0x | 0.901 |

SF4 is free on accuracy while already costing nearly double the calibration
error. Accuracy alone would have called it safe, and for a typed-decision model
whose output *is* a probability that would have been the wrong call. The cliff
sits between SF4 and SF3 on every axis.

**Saturating layer outputs improves calibration.** That is the Atreides
datapath, not the weights-only recipe the rest of the repo measures, and at SF8
it halves ECE against fp32 for 0.4 points of balanced accuracy. Bounding every
layer output bounds the hidden state, which shrinks logits and cures
overconfidence -- an implicit confidence regulariser.

It contrasts with [PURE_SF.md](../../../PURE_SF.md) section 1, where the same
literal saturate-every-register recipe destroyed SmolLM2-360M (13.42 against
2.67 nats). It does not overturn that. The regimes differ -- normalised O(1)
inputs and a four-block residual here, against a 360M LM whose residual reaches
tens of thousands -- so the reading is that the datapath question has a
different answer at control-model scale, which is the scale the hardware
targets.

### Closed loop: agreement is the usable metric, win rate is not

Agreement degrades monotonically with seed spreads of 0.001-0.02 and reproduces
the SF4/SF3 cliff exactly. Win rate carries 31-37% draws and spreads up to
0.029, so although SF4 scores 113% of fp32 it is under 3 sigma on three seeds
and **is not claimed**.

**Crash rate falls as the model degrades** -- 10.5% at SF2 against 26.5% at
fp32 -- and that is a trap, not a win. A degraded policy becomes passive, stops
chasing, and so neither hits the ground nor wins. This is the same artefact as
the BFCL irrelevance scores in [TOOL_USE_QAT.md](../../../TOOL_USE_QAT.md)
section 6, where a model that had stopped emitting tool calls scored 100% on
correctly declining to call them. A crash rate without a win rate beside it is
not interpretable.

The cloned policy wins about 29% against its own teacher and crashes a quarter
of the time. Behaviour cloning is not expected to match the expert; what the
study needs is a stable reference, and the relative comparison across
precisions is the result.

### Latency: the GPU is the denominator, not the answer

Measured on an RTX 4090, batch 1, deployed weights (rounded to the grid once,
then an ordinary Linear with no quantizer in the loop):

| | fp32 | SF8 | SF6 | SF4 | SF3 | SF2 |
| --- | --- | --- | --- | --- | --- | --- |
| deployed p50 | 0.33 ms | 0.33 | 0.33 | 0.33 | 0.34 | 0.32 |
| weights | 8229 KiB | 2057 | 1543 | 1029 | 771 | 514 |

Latency is flat across precisions, as it must be: a GPU has no narrow datapath
to exploit. A first pass showed a 1.294x spread, which an order control
disproved -- measuring fp32 both first and last gives as much difference as
fp32 against SF4 (1.034x at 4000 iterations). It was warmup.

Two things do not come from the GPU. Timing an `SFLinear` model measures the
cost of *simulating* the format, which is 2-10x and points the wrong way.
And the real per-precision difference is the 16x weight footprint, which
decides whether the policy streams from DRAM or sits in on-chip SRAM.

At 2.11M MACs per decision the bridge to hardware is

    cycles ~= MACs / PEs,  latency = cycles / Fmax

giving 8224 cycles on a 256-PE array, or 0.082 ms at 100 MHz -- about 4x
faster than the 4090. That is not a surprising result so much as a statement
about the regime: at batch 1 on a 2.11M-parameter model the GPU is bound by
kernel launch overhead, not arithmetic, which is exactly the case a small
fixed-point accelerator is for. Replace the projection with measured RTL
numbers before quoting it.

## What is not established

- **Behaviour cloning, not RL.** The policy inherits the teacher's ceiling.
  An RL policy would fly better and would also carry seed variance far larger
  than the effect being measured, which is the wrong trade here but the
  obvious follow-up.
- **One aircraft, one weapon.** F-16 against F-16, guns only, no missiles, no
  sensors, no countermeasures.
- **The opponent is scripted and fixed.** Nothing here is self-play, so the
  policies are not being pushed to exploit each other.
- **Win rate does not resolve arms.** 200 fights per seed with a third drawn
  leaves it too noisy to rank the SF8-SF4 band. Agreement does resolve them.
- **The Atreides latency is a projection**, one MAC per PE per cycle with no
  memory model.

## Files

```
geom.py        BFM geometry: range, ATA, aspect, closure, body-frame LOS
pilot.py       manoeuvre -> control surfaces (JSBSim signs measured, not assumed)
env.py         two-aircraft engagement, WEZ scoring, hard deck
expert.py      rule-based BFM teacher and opponent
policy.py      the decision model: 2.11M params, choice/score/noul heads
datagen.py     expert rollouts -> behaviour-cloning dataset
train.py       one arm per precision, QAT
eval_closed.py closed-loop win rate, agreement, crash rate
latency.py     deployed vs simulated latency, MACs, Atreides projection
tournament.py  round-robin validation of the harness
```

Results land in `benchmarks/results/dogfight_{qat,closed_loop}.jsonl` and
`dogfight_latency.json`; the figure is `benchmarks/figures/lab_dogfight_qat.png`.
