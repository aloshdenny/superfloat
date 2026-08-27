# A pure Superfloat datapath

Everything the scaling study and the tool-use study measured is
**weights-only**. The weights sit on the SF grid; activations between
layers stay fp32 or bf16, and nothing saturates the result of a matmul.
That is mixed precision. It is not what the Atreides datapath does.
Atreides multiplies two Q1.15 operands, accumulates in 32 bits, and
**saturates the result back into Q1.15** before it reaches a register.

Measured on SmolLM2-360M the gap is not marginal. Activations entering
matmuls reach thousands; matmul outputs and the residual stream reach
tens of thousands, against a representable bound of 1.

---

## 1. The headline

**Literal saturate-every-register FMA destroys the model. A per-token
scale is the first granularity that preserves it.**

PTQ, SmolLM2-360M, general / tool loss, eval_n=8. `w8_chan` is the
weights-only result this repository already trusts. `sat8_none` is
literal Q1.7 on every named register write. `a8tok_rtok` is SF8 weights
plus per-token (block-float) scales on activations and residual.

| arm | what it is | general | tool |
| --- | --- | --- | --- |
| fp32 | untouched | 2.669 | 1.482 |
| w8_chan | weights-only SF8, per-channel | 2.686 | 1.489 |
| sat8_none | every site SF8, no scale | 13.42 | 13.05 |
| a8t_roff | tensor scale, residual off | 7.16 | 4.10 |
| a8tok_rtok | **per-token act + residual** | **2.946** | **1.778** |
| r8_token | residual-only, per-token | 2.841 | 1.702 |

Tensor and channel activation scales do not recover it (several land at
the dead 10.80 that is a collapsed residual). Per-token residual alone
is already close. The combination that matches the hardware question —
operand and result both on the grid, with a runtime row scale — is
`a8tok_rtok`, +0.28 nats general against fp32, against +10.8 with no
scale.

That per-token scale is block floating point. It is the first rung that
costs silicon the format exists to remove. Tensor and channel scales
fold into a neighbouring weight or norm and are free at inference.

---

## 2. QAT on that rung does not close general

2M tokens of quantization-aware continued pretraining on `a8tok_sdpa`
(per-token act/residual, attention softmax and logits left in fp32),
seed 0, lr 2e-5:

| | general | tool |
| --- | --- | --- |
| PTQ, before training | 2.96 | 1.85 |
| 2M tokens QAT | 2.93 | 0.82 |
| fp32 reference (section 1) | 2.67 | 1.48 |

General does not move. Tool falls because the mix contains tool traces,
which is adaptation, not QAT recovery of a precision cliff. This is one
seed and 0.01% of a pretraining budget; it is recorded, not claimed as
a ceiling.

---

## 3. What this means for a build

- A weights-only SF8 number is not a datapath number. Quote it as
  mixed-precision PTQ.
- If every register write must be SF, budget a per-token scale (or
  accept a dead model). Do not expect tensor/channel activation scales
  to substitute.
- From-scratch 1B QAT on the Llama-3.2-1B shape (`train_1b.py`, SF8
  `ln_all`, embeddings and tied head in bf16) is the scale-up of this
  question. It is in flight on a home 4090. No val number is archived,
  and there is no matched bf16 control yet.

---

## 4. What is not established

- **Eval_n=8** on the PTQ table. Direction is not in doubt; the 0.02-nat
  gaps are.
- **Softmax / attention logits left in fp32** on the QAT run (`o_s`,
  `a_p` off). A fully-saturated attention datapath was PTQ-probed and
  is worse; it was not QAT'd.
- **No 1B val.** `train_1b.py` writes `ckpt/latest.pt` locally. That
  checkpoint is not in this repository.
- **No matched bf16 1B control.** Do not claim QAT recovered 1B until
  that row exists.

---

## 5. Files

```
benchmarks/lab/
  psd.py          named-site Llama block, granularity sweep, census, QAT
  psd_data.py     FineWeb / tool-mix loaders for the 360M runs
  train_1b.py     Llama-3.2-1B from-scratch QAT, uint32 shards
benchmarks/results/
  psd_census.jsonl   activation dynamic range by site
  psd_ptq.jsonl      18 PTQ arms, section 1
  psd_qat.jsonl      1 QAT run, 2M tokens, section 2
```

```bash
cd benchmarks/lab
python psd.py --census
python psd.py --ptq
python psd.py --qat --name a8tok_sdpa --tokens 2000000 --seed 0
```

The Llama 3 tokenizer vocabulary is 128256. That does not fit in
uint16; token ids wrap and the run is garbage. `train_1b.py` shards are
uint32. That is the bug from the previous attempt, not a new finding.
