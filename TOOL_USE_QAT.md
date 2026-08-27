# SuperFloat for tool-use models

A feasibility study for pretraining or converting a tool-calling model to run
on SF-only hardware. 18 loss runs and 16 BFCL arms, all archived under
`benchmarks/results`.

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

## 6. Capability, not loss: BFCL

Everything above is validation loss. A model can lose 0.1 nats and still call
the right function every time, or gain nothing and start emitting malformed
JSON. Qwen2.5-1.5B-Instruct on the Berkeley Function Calling Leaderboard, 840
prompts per arm, greedy decode, weights-only SF, scored by AST match:

| arm | simple | multiple | irrelevance | call rate |
| --- | --- | --- | --- | --- |
| bf16 | 83.5 +/-3.6 | 80.0 +/-5.5 | 72.1 +/-5.7 | 99.5% |
| **SF8 PTQ** | **83.2 +/-3.7** | **81.0 +/-5.4** | **76.2 +/-5.4** | 97.5% |
| SF6 PTQ | 0.0 | 0.5 | 100.0 | 1.2% |
| SF4 PTQ | 0.0 | 0.0 | 100.0 | 0.0% |

SF8 differs from bf16 by -0.2, +1.0 and +4.2 points, all inside the roughly
+/-5pp confidence interval on the difference. **Post-training quantization to
SF8 costs no measurable tool-use accuracy.**

**The 100% irrelevance figures are an artifact and must not be quoted.**
Irrelevance is scored as correctly declining to call a function, so a model
that has stopped emitting tool calls passes every case for free. SF6 and SF4
call on 1.2% and 0.0% of prompts respectively: they are not cautious, they are
broken. Aggregating the three categories would report them at about 28.7%,
which is why this table gives categories separately with the call rate beside
them. A BFCL number without a call rate is not interpretable.

**Loss was a valid screen.** SF8 PTQ at 2.4493 against bf16's 2.4680 predicted
intact capability; SF6 PTQ at 7.17 predicted the collapse. Loss costs about a
hundredth of a benchmark run and located the cliff correctly, which is worth
knowing before paying for evaluation sweeps.

---

## 6b. One number is a trap: Qwen3 0.6B–8B

Section 6 is Qwen2.5-1.5B. Section 8 listed that as a limitation: 360M is not
1B, and a single 1.5B row does not say whether the SF8 result survives scale
or whether the SF6 cliff is a small-model artefact. Same 840 prompts, same
AST matcher, same `ln_all` recipe, thinking off (`enable_thinking=False`),
Qwen3 dense 0.6B / 1.7B / 4B / 8B. Qwen3.5 was not run: it is a
linear-attention plus vision hybrid, and this surgery does not apply.

Overall accuracy, 840 prompts. Call rates sit in the jsonl; they stay high
except where noted.

| model | bf16 | SF8 PTQ | SF6 PTQ |
| --- | --- | --- | --- |
| Qwen3-0.6B | 81.9% | 78.7% | 27.3% |
| Qwen3-1.7B | 88.7% | **88.6%** | 78.9% |
| Qwen3-4B | 89.6% | **90.4%** | 62.9% |
| Qwen3-8B | 91.3% | **90.2%** | **88.0%** |

**SF8 is free once the model is not tiny.** From 1.7B up it is within a point
of bf16, matching the 1.5B result. 0.6B pays −3.2 points, which is a small
model being small, not the format falling over.

**The SF6 cliff is the small models.** At 8B, SF6 holds at 88.0 against 91.3
(−3.3 points) with call rates intact (simple 99.0%, multiple 97.5%). At 4B
it is already a real hit (62.9, and simple call rate drops to 84%). At 0.6B
it is 27.3%. That 0.6B failure is not the silent-model artefact from section
6: simple still calls on 20.8% of prompts, but the arguments are garbage.
The 1.5B SF6 run that scored ~0 with a 1.2% call rate was a model that had
stopped emitting tools. These still try.

4B SF8 at 90.4 against 89.6 is one seed and is not a claim that SF8 is
better. 8B SF8 ran 105 minutes, SF6 113 minutes, on a 24 GB card after
quantize-on-CPU then bf16-to-GPU; in-place `model.float()` OOMs.

---

## 7. What this means for a build

- **Want SF8?** Quantize an existing tool-capable checkpoint. No training, no
  GPU-weeks, no new credit. Scale every matmul, including `o_proj` and
  `down_proj`. Do not quote a 0.6B number as the format's cost; from 1.7B
  the PTQ result is free.
- **Want SF6?** At 8B, PTQ already holds at −3.3 points. Below that, budget
  roughly 10M tokens of QAT from the checkpoint, not a pretraining run — and
  treat that QAT as untested on capability until section 8's item is closed.
  On a single RTX 4090 that is hours, not weeks.
- **Pretraining from scratch is the wrong instrument.** Measured throughput put
  a month of 4090 time at roughly 300M parameters on 9B tokens. The smallest
  models with credible tool calling saw three orders of magnitude more data,
  and no precision or kernel work closes that gap.

Measured cost of SF8 QAT against a bf16 control, same model and batch:
**188k against 230k tokens/second**, an 18% throughput penalty for
quantize-in-forward. PTQ costs nothing, since there is no forward to pay for.

---

## 8. What is not established

- **Single seed** on every SmolLM2 run and on the SF2/SF3/SF4 arms of section
  3. The two-seed arms give a spread of 0.011 to 0.027, so differences below
  about 0.03 nats are not resolved.
- **The fold-back for `o_proj`/`down_proj`** is derived in section 2 but not
  verified in code. Every other absorption path in this document is checked to
  0 off-grid weights; that one is not yet.
- **SF6 capability after QAT is untested.** Section 4 shows SF6 QAT recovers
  loss to near-control; section 6 shows SF6 *without* QAT scores zero on BFCL
  at 1.5B. Section 6b shows 8B SF6 PTQ already holds at −3.3 points, so the
  QAT question is now "does it close the last few points on mid-size models",
  not "does anything work at SF6". It was still not run.
- **BFCL was run on Qwen2.5-1.5B-Instruct and on Qwen3 dense 0.6B–8B, not on
  the SmolLM2 models of sections 3-4.** It measures whether SF preserves
  capability a model already has. It does not measure the continued-
  pretraining runs.
- **Only three AST-checkable categories.** No multi-turn, no live, no
  executable categories, and no tau-bench or API-Bank.
- **PTQ scale is measured; QAT scale is not.** Section 6b closes the "one
  1.5B number" gap for post-training quantization. From-scratch 1B QAT is
  in flight on a home 4090 and has no archived val number yet. A matched
  bf16 1B control has not been started.
- **No 8K context run.** SmolLM2-360M supports 8192 positions natively and the
  runs here used 2048. Long-context behaviour under SF is untested.
- **PTQ sweep eval noise.** The section 1 sweep drew different random batches
  per configuration, so differences below about 0.02 nats there are not
  meaningful. The large gaps are unaffected, and the section 4 runs use fixed
  evaluation batches.

---

## 9. Files

```
benchmarks/lab/
  stage0_toolqat.py   modern-block model, SF surgery, from-scratch sweep
  smol_qat.py         SmolLM2-360M PTQ sweep and QAT continued pretraining
  bfcl_eval.py        BFCL v3 runner and AST matcher
  train_1b.py         Llama-3.2-1B from-scratch QAT (in flight; no archive yet)
benchmarks/results/
  stage0.jsonl        14 runs, section 3
  smol_ptq.jsonl      2 PTQ sweeps, ln and ln_all, section 1 and 2
  smol_qat.jsonl      2 continued-pretraining runs, section 4
  bfcl.jsonl          16 arms: 4 Qwen2.5-1.5B (section 6) + 12 Qwen3 (section 6b)
```

```bash
cd benchmarks/lab
python stage0_toolqat.py --prepare 900000000
python stage0_toolqat.py --size 25m --bits 8 --mode ln --seed 0
python smol_qat.py --prepare 250000000
python smol_qat.py --ptq --mode ln_all
python smol_qat.py --bits 6 --mode ln_all --tokens 60000000
python bfcl_eval.py --model Qwen/Qwen3-1.7B --bits 8 --mode ln_all
```
