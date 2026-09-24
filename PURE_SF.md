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

## 3. From scratch, the residual renorm bounds the residual and nothing else

`psd.py`'s granularity ablation is PTQ on a pretrained model. `puresf_llm.py`
asks the harder version: train an 11M RMSNorm/SwiGLU/GQA model from scratch
with weights, activations, AND the residual stream all SF8, using the
`a8tok_rtok` recipe above plus periodic residual renormalisation (divide the
residual stream by its own per-token max every `renorm_every` blocks --
sound because every downstream read goes through a norm, so `RMSNorm(x/s) ==
RMSNorm(x)`). Sweeping the interval on the same 11M/200M-token config
(`puresf_llm.jsonl`) isolates what the mechanism actually fixes:

| renorm_every | val loss | max residual | max layer output | audit pass |
| --- | --- | --- | --- | --- |
| 4 | 4.2632 | 6.27 | 4.42 | no |
| 2 | 4.2570 | 4.30 | 4.76 | no |
| 1 | 4.2632 | **1.00** | 4.74 | no |

Renorming every block pins the residual stream at exactly the SF bound, as
designed -- that half of the problem is solved. `max_layer_output` does not
move with it (4.42 to 4.76, no trend), because nothing in the mechanism
touches it: it is the *output* of `down_proj` (or `gate_proj`, per-layer)
before that value is added to the residual, and no norm or clamp sits
between the matmul and that addition. This is the literal Atreides
requirement PURE_SF.md opened with -- saturate the FMA result itself, not
just the operands -- and it is still missing. The residual renorm was never
going to close it; it closes a different half of the same audit.

---

## 4. What this means for a build

- A weights-only SF8 number is not a datapath number. Quote it as
  mixed-precision PTQ.
- If every register write must be SF, budget a per-token scale (or
  accept a dead model). Do not expect tensor/channel activation scales
  to substitute.
- The residual renorm from section 3 is necessary but not sufficient: it
  bounds the residual stream, not the per-matmul output that feeds it. A
  build needs both, and only the first exists right now.
- From-scratch 1B QAT on the Llama-3.2-1B shape (`train_1b.py`, SF8
  `ln_all`, embeddings and tied head in bf16) is the scale-up of the
  section-1 question. It is now measured: at 8K over 6.4B matched tokens
  SF8 and bf16 are indistinguishable, and the 8K -> 32K -> 128K context
  ladder costs SF8 a flat ~1% rather than a compounding penalty
  (section 5). Chunked cross-entropy is what makes the long rungs fit --
  full-sequence fp32 logits are 16.8 GiB at 32K and 67 GiB at 128K, which
  OOMs an 80 GiB H100 before attention is the constraint.

---

## 5. What is not established

- **Eval_n=8** on the PTQ table. Direction is not in doubt; the 0.02-nat
  gaps are.
- **Softmax / attention logits left in fp32** on the QAT run (`o_s`,
  `a_p` off). A fully-saturated attention datapath was PTQ-probed and
  is worse; it was not QAT'd.
- **The matmul-output saturation gap (section 3).** Three renorm
  frequencies, one architecture, one size, one seed. Whether hard-clamping
  `down_proj`/`gate_proj` output itself (rather than just its norm-fed
  input) closes `max_layer_output` is untested, not just unsolved.
- **The 1B run is cut short, not finished.** The corrected run (full corpus
  verified on disk before start, 8K context) reached 6.4B of a planned 20B
  tokens before the compute workspace hit its spend limit. What it does
  establish, on identical data in identical order over 971 matched
  evaluation points, is that **SF8 `ln_all` and bf16 are indistinguishable at
  the Llama-3.2-1B shape**: final matched val loss 2.8100 (bf16) vs 2.8098
  (SF8), ppl 16.61 both, with the gap over the last 40 matched evals at
  -0.013%. An earlier attempt trained both arms on a single 100M-token shard
  for 24h (a `vol.reload()` omission) and is memorisation, not pretraining;
  it is kept only as `sf1b_run0_*.jsonl` because both arms saw identical
  tokens there too. `sf1b_8k_*.jsonl`, `lab_sf1b_8k_matched.png`.
- **Context extension holds, and the SF8 penalty does not compound.**
  Continuing those 8K checkpoints at longer context, each stage with its own
  short cosine schedule at lr 5e-5:

  | rung | tokens | bf16 val ppl | SF8 val ppl | gap |
  | --- | --- | --- | --- | --- |
  | 8K (pretrain) | 6.4B | 16.61 | 16.61 | -0.02% |
  | 32K (extension) | +60M | 15.77 | 15.97 | +1.27% |
  | 128K (extension) | +50M | **15.45** | **15.61** | +1.05% |

  Both arms end each rung below the previous one, which is the expected
  benefit of longer context. Two things are worth separating. First, before
  adaptation SF8 *extrapolates* to unseen RoPE positions better: at the first
  32K evaluation bf16 sits at 111.9 ppl against SF8's 86.6 (-22.6%), and both
  then fall to ~16 within 100 steps. Second, after adaptation SF8 settles
  about 1% behind, and that gap is flat across rungs (+1.27% at 32K, +1.05%
  at 128K) rather than growing with context length -- the quantisation cost
  of extension is paid once, not per doubling. `sf1b_ext32k_*.jsonl`,
  `sf1b_ext128k_*.jsonl`, `lab_sf1b_context_ladder.png`.

- **Longer context improved perplexity without buying retrieval.** The ladder
  above is perplexity, which a model can improve using only local context. A
  loss-based needle test (`niah_1b.py`) settles what it actually uses: plant a
  random token sequence at depth *d*, re-present its first 16 tokens at the
  end, and score NLL on the remaining 48 against a control that plants a
  *different* needle, so the only variable is whether the answer is
  retrievable. Perfect copying would score ~11.76 nats (log 128256); 0 means
  no retrieval. **Our own bf16 control is the baseline at 100%** -- Meta's
  published Llama numbers are never the reference, since corpus, budget and
  recipe all differ.

  | evaluation context | bf16 (baseline) | SF8 | SF8 relative |
  | --- | --- | --- | --- |
  | 8K | 2.908 | 3.091 | 106.3% |
  | 32K | 0.212 | 0.192 | 90.6% |
  | 128K | 0.005 | 0.004 | n/a (both at noise) |
  | 32K, 8K-only model (control) | 0.004 | 0.005 | n/a (both at noise) |

  Higher is better here, the opposite of the perplexity tables. Two readings.
  First, SF8 tracks bf16 at every context: quantisation does not cost
  retrieval, and at 8K it is marginally ahead. Second, and more important,
  **neither arm retrieves beyond ~8K**. At 128K both sit at 0.004-0.005 nats,
  indistinguishable from an 8K-only model evaluated at 32K, which never saw
  those positions at all.

  The per-depth data shows this is a *distance* effect rather than a context
  one: at 8K a needle 819 tokens back scores 7.08 nats, 2048 back scores 4.42,
  4096 back scores 1.20, and 7372 back scores 0.19. The effective retrieval
  range is roughly two thousand tokens and decays sharply past four, so at
  128K even the nearest tested needle (13k back) is out of range. The
  perplexity gains at 32K and 128K are therefore local-context gains, and the
  extension stages did not extend what the model can actually reach.
  `benchmarks/results/niah/`, `lab_sf1b_niah.png`.

  Read narrowly. One seed; the 8K base is 6.4B tokens rather than a
  completed budget; the two arms entered extension from slightly different
  8K steps (48800 bf16 vs 48400 SF8); and each extension stage is 50-60M
  tokens against the six-stage, far longer recipe Llama 3 uses. Long-context
  *retrieval* behaviour is not measured at all here -- this is perplexity on
  held-out FineWeb-Edu, which a model can improve without actually using the
  far context.

---

## 6. Files

```
benchmarks/lab/
  psd.py          named-site Llama block, granularity sweep, census, QAT
  psd_data.py     FineWeb / tool-mix loaders for the 360M runs
  puresf_llm.py   11M from-scratch, weights+activations+residual all SF8,
                  residual-renorm-interval sweep, section 3
  train_1b.py     Llama-3.2-1B from-scratch QAT, uint32 shards
benchmarks/results/
  psd_census.jsonl   activation dynamic range by site
  psd_ptq.jsonl      18 PTQ arms, section 1
  psd_qat.jsonl      1 QAT run, 2M tokens, section 2
  puresf_llm.jsonl   renorm-interval sweep, section 3
  sf1b_run0_*.jsonl  first 1B attempt, shard-0-only (see section 5); SF8 vs bf16 tracking
```

```bash
cd benchmarks/lab
python psd.py --census
python psd.py --ptq
python psd.py --qat --name a8tok_sdpa --tokens 2000000 --seed 0
python puresf_llm.py --arm sf8_full --renorm-every 1 --tokens 200000000
```

The Llama 3 tokenizer vocabulary is 128256. That does not fit in
uint16; token ids wrap and the run is garbage. `train_1b.py` shards are
uint32. That is the bug from the previous attempt, not a new finding.
