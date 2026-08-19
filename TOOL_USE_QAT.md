# SuperFloat for tool-use models

A feasibility study for pretraining or converting a tool-calling model to run
on SF-only hardware. 18 runs, all archived under `benchmarks/results`.

The scaling study established that scale placement, not precision, sets the
usable floor. Everything there used vanilla GPT-2 blocks and models trained
from scratch. A tool-use model is neither: it wants RMSNorm, SwiGLU and grouped
query attention, and the practical path starts from a checkpoint that already
cost trillions of tokens to make. This asks whether the result survives both
changes.

---

## 1. The headline

**For SF8, no training is needed at all.** A trained checkpoint quantizes to
SF8 post-training for free, provided every matmul gets a scale. That removes
the entire pretraining budget from the plan.

| | general | tool |
| --- | --- | --- |
| SmolLM2-360M, untouched | 2.5300 | 1.1539 |
| SF8 PTQ, no training | **2.4493** | **1.0575** |
| SF16 PTQ, `o_proj`/`down_proj` unscaled | 7.7309 | 6.4452 |

The third row is the finding that matters, and it is a correction to a claim
this repository makes elsewhere.

---

## 2. The range assumption fails on trained checkpoints

The README states that ~99.999% of trained weights fall inside SF's
representable range, measured on YOLO11x where 0 of 53.6M weights fell outside
SF16's bound. That does not generalise. SmolLM2-360M:

| group | max abs w | std | % outside +/-1 |
| --- | --- | --- | --- |
| q/k/v | 5.94 | 0.181 | 0.041% |
| gate/up | 7.44 | 0.178 | 0.009% |
| o_proj | 7.47 | 0.164 | 0.014% |
| down_proj | 7.00 | 0.178 | 0.011% |

Only about one weight in ten thousand exceeds the bound, but those are the
largest-magnitude weights in the network, and clipping them is enough to
destroy the model **even at SF16**, where the grid step is 3e-5 and precision
is plainly not the issue.

Under `ln` absorption q/k/v and gate/up are divided by their per-input-channel
maximum, so they land inside the bound by construction. `o_proj` and
`down_proj` are fed by no norm, so the original design quantized them as they
stood -- straight into the clipping. Giving every matmul a scale recovers the
model completely.

This is the study's own thesis reappearing rather than contradicting it: with
no scale, a fixed-range format is simply wrong for the weights it has to hold.
It bites harder here than in the from-scratch experiments because a trained
checkpoint has std 0.18 against an initialised model's 0.005 to 0.02, which is
why section 3 never surfaced it.

**Absorbing the two extra scales.** Neither layer is fed by a norm, but both
scales still fold one layer back rather than into one:

    scaling o_proj's input channel j  ==  scaling v_proj's output row j
    scaling down_proj's input channel j == scaling up_proj's output row j,
      since (gate * up) * g == gate * (up * g)

so the deployed matmul still holds exact SF grid values. The scaling is
implemented and measured; the fold-back is derived but **not yet verified in
code**, and should be before anything ships.

---

## 3. Stage 0: SF8 is free on modern blocks, from scratch

22.6M non-embedding parameters, RMSNorm/SwiGLU/GQA, 500M tokens, 15% tool
traces, two seeds. Penalty against a bf16 control trained identically; the
control's own seed spread is 0.011 general and 0.021 tool.

| bits | `ln` general | `ln` tool | `plain` general | `plain` tool |
| --- | --- | --- | --- | --- |
| SF8 | **-0.0006** | **-0.0024** | +0.0072 | +0.0081 |
| SF6 | +0.0622 | +0.0572 | -- | -- |
| SF4 | +0.1418 | +0.1541 | +0.3971 | +0.4272 |
| SF3 | +0.2534 | +0.2668 | +2.3844 | +3.4391 |
| SF2 | +0.3862 | +0.4093 | +2.3844 | +3.4391 |

SF8's penalty is an order of magnitude below the seed spread: free, not merely
small. SF4 under absorption costs +0.14, which is usable and better than the
SF6 floor the from-scratch language models in the scaling study reported.

**The identical SF2 and SF3 plain rows are not a copy-paste error.** They agree
bitwise, at 6.087118 and 4.835851, and both plateau by step 7626 while the
absorbed runs keep improving. With every `o_proj`/`down_proj` weight dead at
initialisation, each block outputs exactly zero and the residual stream carries
the embedding straight to the head. The network degenerates to embedding, norm,
tied head -- the same degenerate model whether two bits were requested or
three, which is why the numbers match exactly.

Dead weights at initialisation, measured directly, predicted this before the
runs finished:

| bits | q/k/v absorbed | gate/up absorbed | o/down, not absorbable |
| --- | --- | --- | --- |
| SF8 | 1.0% | 1.1% | 56.5% |
| SF6 | 3.9% | 4.4% | 99.8% |
| SF4 | 15.4% | 17.5% | 100% |
| SF3 | 30.3% | 34.1% | 100% |
| SF2 | 56.2% | 62.1% | 100% |

`o_proj` and `down_proj` carry residual-scaled initialisation, std 0.005
against 0.02 elsewhere, so they are both the most vulnerable to a coarse grid
and the only two layers `ln` cannot absorb a scale into.

**Absorption is exact on these blocks.** Folding every scale into the feeding
norm leaves 0 of 2,818,048 weights off the grid and reproduces the training
forward pass bitwise, across `plain`/`ln`/`ln_full` and SF6/SF8. RMSNorm has no
bias term, so the algebra is exact rather than approximate.

---

## 4. QAT from a checkpoint recovers what PTQ cannot

SF6 PTQ collapses to 7.89. Sixty million tokens of quantization-aware
continued pretraining recovers it:

| step | tokens | general | tool |
| --- | --- | --- | --- |
| PTQ, before training | 0 | 7.8858 | 5.6057 |
| 0 | 130K | 5.6602 | 3.6326 |
| 366 | 6M | **2.6412** | **0.6925** |
| 3660 | 60M | 2.6303 | 0.5391 |
| bf16 control | 60M | 2.5265 | 0.6758 |

**Most of the recovery happens in the first six million tokens** -- 0.01% of
what the checkpoint originally cost. The remaining 54M buy 0.01 nats on general
and 0.15 on tool.

Against the bf16 control, SF6 QAT ends +0.10 nats worse on general and 0.14
nats **better** on tool. The tool advantage replicates the direction of the
inversion in section 4.1 of the scaling study, but this is one seed and the
general/tool trade could be a training-dynamics artefact rather than a property
of the format. Recorded, not claimed.

The continued-pretraining recipe is sound independently of precision: the bf16
control moved tool loss from 1.1539 to 0.6758 while leaving general at 2.5265
against 2.5300, so it adapts the model to tool-call format without eroding
what it already knew.

---

## 5. Structured output is not more precision-sensitive, until it is

The worry for a tool model was that JSON and function-call syntax might degrade
faster than prose. In the safe regime they do not:

| | general | tool |
| --- | --- | --- |
| SF8 | -0.0006 | -0.0024 |
| SF6 | +0.0622 | +0.0572 |
| SF4 | +0.1418 | +0.1541 |

Past the cliff the picture reverses. At SF3 `plain` the tool loss blows out to
+3.44 against general's +2.38. Structured output is the first thing to break
once the format stops working, which is an argument for keeping a margin rather
than running at the edge.

---

## 6. What this means for a build

- **Want SF8?** Quantize an existing tool-capable checkpoint. No training, no
  GPU-weeks, no new credit. Scale every matmul, including `o_proj` and
  `down_proj`.
- **Want SF6?** Budget roughly 10M tokens of QAT from the checkpoint, not a
  pretraining run. On a single RTX 4090 that is hours, not weeks.
- **Pretraining from scratch is the wrong instrument.** Measured throughput put
  a month of 4090 time at roughly 300M parameters on 9B tokens. The smallest
  models with credible tool calling saw three orders of magnitude more data,
  and no precision or kernel work closes that gap.

Measured cost of SF8 QAT against a bf16 control, same model and batch:
**188k against 230k tokens/second**, an 18% throughput penalty for
quantize-in-forward. PTQ costs nothing, since there is no forward to pay for.

---

## 7. What is not established

- **Single seed** on every SmolLM2 run and on the SF2/SF3/SF4 arms of section
  3. The two-seed arms give a spread of 0.011 to 0.027, so differences below
  about 0.03 nats are not resolved.
- **The fold-back for `o_proj`/`down_proj`** is derived in section 2 but not
  verified in code. Every other absorption path in this document is checked to
  0 off-grid weights; that one is not yet.
- **Loss is not capability.** Nothing here measures whether the model calls the
  right function with the right arguments. BFCL, tau-bench or API-Bank would,
  and none were run. A 0.5 nat improvement in tool loss is not a claim about
  tool-use accuracy.
- **360M is not 1B.** Tier A found the QAT penalty grows with model size below
  SF6, so the SF6 result in particular may not hold at the scale a real tool
  model needs.
- **No 8K context run.** SmolLM2-360M supports 8192 positions natively and the
  runs here used 2048. Long-context behaviour under SF is untested.
- **PTQ sweep eval noise.** The section 1 sweep drew different random batches
  per configuration, so differences below about 0.02 nats there are not
  meaningful. The large gaps are unaffected, and the section 4 runs use fixed
  evaluation batches.

---

## 8. Files

```
benchmarks/lab/
  stage0_toolqat.py   modern-block model, SF surgery, from-scratch sweep
  smol_qat.py         SmolLM2-360M PTQ sweep and QAT continued pretraining
benchmarks/results/
  stage0.jsonl        14 runs, section 3
  smol_ptq.jsonl      2 PTQ sweeps, ln and ln_all, section 1 and 2
  smol_qat.jsonl      2 continued-pretraining runs, section 4
```

```bash
cd benchmarks/lab
python stage0_toolqat.py --prepare 900000000
python stage0_toolqat.py --size 25m --bits 8 --mode ln --seed 0
python smol_qat.py --prepare 250000000
python smol_qat.py --ptq --mode ln_all
python smol_qat.py --bits 6 --mode ln_all --tokens 60000000
```
