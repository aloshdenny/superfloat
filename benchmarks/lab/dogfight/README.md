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

## Status

Harness validated. Data generation, QAT across precisions, and the latency
testbench are not yet written.
