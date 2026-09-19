# Mixed-precision allocation study

## The question

Every SF result so far assigns one bit width to every matmul. That is almost
certainly not optimal, because the study has already shown the layers are not
equally fragile:

- **exp5**: dead-weight fraction at init scales with `fan_in`. Wide layers are
  the ones that die.
- **section 5.2**: `o_proj` and `down_proj` are fed by no norm, so their scale
  is **not absorbable**. They are structurally the weakest layers, and they
  also carry residual-scaled init (std 0.005 vs 0.02), making them the most
  vulnerable to a coarse grid.
- **section 5.1**: weights saturate at SF3-SF4 while activations need SF6.

So the hypothesis: **at a fixed average bit budget, spending bits where they
are structurally needed and saving them where they are not should beat uniform
allocation.** If true, "SF8" as a single number is the wrong way to specify a
deployment, and the datapath should support per-layer widths.

What is NOT being tested: staged-in-time recipes (SF16 for a while, then SF8).
Appendix A already showed there is no steeper SF16 curve to harvest -- early
slope is noise around 1x across the whole ladder and SF8 finishes ahead at 25M.

## Design

Every arm is compared against a uniform arm **at the same average bits per
weight**, so a win is an allocation win rather than a bit-count win. Average is
weighted by parameter count, not layer count, since `mlp` dominates.

Layer groups in a GPT block, by absorbability:

| group | layers | scale absorbable? | share of block params |
| --- | --- | --- | --- |
| A | q, k, v | yes, into the norm feeding them | ~33% |
| B | gate/up (or fc1) | yes, into the norm feeding them | ~33% |
| C | o, down (or proj, fc2) | **no** -- fed by no norm | ~33% |

Arms at a 6-bit average:

| arm | A | B | C | avg bits |
| --- | --- | --- | --- | --- |
| uniform6 | 6 | 6 | 6 | 6.0 |
| protect-C | 5 | 5 | 8 | 6.0 |
| starve-C | 7 | 7 | 4 | 6.0 |
| protect-A | 8 | 5 | 5 | 6.0 |

`protect-C` is the hypothesis. `starve-C` is its opposite and should be worst
if the mechanism is real. `protect-A` controls for "any non-uniformity helps".

Arms at a 4-bit average, where the effect should be larger because the grid is
coarse enough for scale placement to decide whether weights survive at all:

| arm | A | B | C | avg bits |
| --- | --- | --- | --- | --- |
| uniform4 | 4 | 4 | 4 | 4.0 |
| protect-C | 3 | 3 | 6 | 4.0 |
| starve-C | 5 | 5 | 2 | 4.0 |

## Scales and architectures

1. **11M GPT-2 style** (LayerNorm/GELU) -- comparable to exp3, tpp10
2. **11M modern** (RMSNorm/SwiGLU/GQA) -- comparable to stage0, checks that the
   allocation result is not an artifact of one block design
3. **25M GPT-2 style** -- does the best allocation move with scale?

Every cell gets its own fp32 control at the same size and token budget, because
comparing across sizes without one is what produced the wrong DOMAIN ladder.

## Cost

11M at tpp10 is 6,498 steps, about 1h on an A100 at micro-batch 1, and cells
parallelise well (the job is launch-bound at ~3% FLOP utilisation). Roughly:

- stage 1: 11M GPT-2, 7 arms + control  = 8 cells
- stage 2: 11M modern, 7 arms + control  = 8 cells
- stage 3: 25M GPT-2, 4 arms + control   = 5 cells (2.5h each)

~21 cells. At 6-way parallelism on one A100 that is roughly 8-10 wall hours,
about $12. The budget allows repeating the whole thing at a second seed, which
matters because the coco128 replicates showed single-seed SF runs can spread
0.28 -- differences below the seed spread are not results.
