# SuperFloat (SFx) — Full Evaluation Results

Evaluation of the SuperFloat numeric schema across four architecture families,
five numeric formats, two initialisation regimes, and two quantization regimes
(quantization-aware training and post-training quantization).

Detection heads and classifier logits are kept in full precision throughout, as
the paper's recipe specifies.

## 1. Numeric formats

SFx = 1 sign bit + (x−1) fractional bits, no exponent, no integer bit.

| Format | Significand bits | Grid step | Max value | Zero-threshold | Storage vs FP32 |
|---|---|---|---|---|---|
| SF16 | 15 | 0.000031 | 0.999969482421875 | 0.000015 | 50.0% |
| SF8 | 7 | 0.007812 | 0.9921875 | 0.003906 | 75.0% |
| SF4 | 3 | 0.125000 | 0.875 | 0.062500 | 87.5% |

Max values verified identical to the paper's tabulated constants.

## 2. Training recipe

Identical across formats within each group; only the numeric format changes.

| | Classification | Detection (pretrained) | Detection (scratch) | V-JEPA 2 |
|---|---|---|---|---|
| Model | ConvNeXt-Tiny (28M) | YOLO11x (57M) / YOLOv8x-OBB (69M) | same | ViT-L (300M) |
| Init | random | COCO-pretrained | random | V-JEPA 2 pretrained |
| Epochs | ≤300, early stop @40 | 150, early stop @50 | 300, early stop @50 | 15 (QAT) / 40 (probe) |
| LR (AdamW) | 4e-3 | 1e-3 | 4e-3 | 1e-5 backbone / 1e-3 head |
| Weight decay | 0.05 | 0.05 | 0.05 | 0.05 |
| Batch | 128 | 8 | 8 | 16 |
| Seeds | 3 | 1 | 1 | 1 |

**TF32 disabled throughout.** Ada/Hopper/Blackwell default to TF32 for cuDNN
convolutions, whose 10-bit mantissa is below SF16's 15 significand bits and
would silently degrade SF16 to worse-than-SF8 fidelity inside every
convolution. FP16 rows use AMP; all SFx and FP32 rows use true fp32 accumulate.

## 3. Remote sensing classification — EuroSAT / ConvNeXt-Tiny

27,000 images, 10 classes, 64px, stratified 80/20 split, from scratch, 3 seeds.

| Format | Top-1 (%) | Per-seed | Best epoch (med) | Train loss | Val loss |
|---|---|---|---|---|---|
| FP32 | **96.31 ±0.55** | 96.06, 95.81, 97.07 | 104 | 0.5269 | 0.2084 |
| FP16 | **96.19 ±0.45** | 96.20, 95.63, 96.72 | 144 | 0.5193 | 0.2094 |
| SF16 | **96.43 ±0.06** | 96.41, 96.52, 96.37 | 199 | 0.5668 | 0.1938 |
| SF8 | **96.27 ±0.19** | 96.00, 96.37, 96.43 | 236 | 0.5413 | 0.1999 |
| SF4 | **96.27 ±0.03** | 96.30, 96.30, 96.22 | 271 | 0.5435 | 0.2012 |

SF4 vs FP32: **−0.04 points** at 87.5% weight-storage reduction — within seed
noise. Every SuperFloat format matches full precision on this task.

Note the SuperFloat runs show *lower* seed variance than the full-precision
ones (SF4 ±0.03, SF16 ±0.06 vs FP32 ±0.55), suggesting the bounded grid acts
as a regulariser in this regime.

## 4. UAV detection — VisDrone / YOLO11x

6,471 train / 548 val images, 640px, batch 8.

### Pretrained init

| Format | mAP50-95 | mAP50 | Precision | Recall | vs FP32 | Epochs | Status |
|---|---|---|---|---|---|---|---|
| FP32 | **0.2942** | 0.4827 | 0.6070 | 0.4812 | +0.0% | 127 (best 77) | final |
| FP16 | **0.2947** | 0.4825 | 0.6025 | 0.4760 | +0.2% | 150 (best 74) | final |
| SF16 | **0.2817** | 0.4676 | 0.5826 | 0.4627 | −4.3% | 150 (best 113) | final |
| SF8 | **0.2836** | 0.4690 | 0.5681 | 0.4681 | −3.6% | 150 (best 104) | final |
| SF4 | **0.2501** | 0.4208 | 0.5298 | 0.4249 | −15.0% | 150 (best 140) | final |

### Random init

| Format | mAP50-95 | mAP50 | vs FP32 | Epochs | Status |
|---|---|---|---|---|---|
| FP32 | **0.3011** | 0.4890 | +0.0% | 280 (best 233) | final (3 short of patience stop) |
| FP16 | **0.2998** | 0.4904 | −0.4% | 274 (best 224) | final (early-stopped) |
| SF16 | **0.2948** | 0.4818 | −2.1% | 283 (best 260) | final |
| SF8 @ lr 4e-3 | **0.0748 ±0.0017** | 0.1387 | −75.2% | 68 (best 18) | final — collapsed, 3 seeds |
| SF8 @ lr 1e-3 | **0.2361** | 0.3991 | −21.6% | 87 (best 86) | **truncated; ~0.26 projected** |
| SF4 | **0.2587** | 0.4292 | −14.1% | 244 (best 194) | final (early-stopped) |

The SF8 @ lr 1e-3 run was stopped at epoch 87 of 300 for budget reasons. At the
matched epoch it tracks SF16 at a ratio of 0.888; SF16 gained a further 10.9%
between epoch 87 and its final value, which projects SF8 to **≈0.26**. This is
an *estimate, not a measurement*, and should be reported as such. The finding
it supports does not depend on the projection: 0.2361 against 0.0748 ±0.0017 is
a 3× separation no reasonable extrapolation closes.

## 5. Satellite detection — DOTAv1 / YOLOv8x-OBB

1,411 train / 458 val images (untiled), 800px, batch 8.

### Pretrained init

| Format | mAP50-95 | mAP50 | Precision | Recall | vs FP32 | Epochs | Status |
|---|---|---|---|---|---|---|---|
| FP32 | **0.4489** | 0.5942 | 0.7863 | 0.5818 | +0.0% | 150 (best 129) | final |
| FP16 | **0.4496** | 0.5994 | 0.8170 | 0.5693 | +0.2% | 150 (best 103) | final |
| SF16 | **0.4298** | 0.5753 | 0.7947 | 0.5479 | −4.2% | 150 (best 139) | final |
| SF8 | **0.4254** | 0.5705 | 0.7657 | 0.5544 | −5.2% | 150 (best 139) | final |
| SF4 | **0.3488** | 0.4891 | 0.7507 | 0.4623 | −22.3% | 150 (best 137) | final |

### Random init

| Format | mAP50-95 | mAP50 | vs FP32 | Epochs | Status |
|---|---|---|---|---|---|
| FP32 | **0.4088** | 0.5465 | +0.0% | 295 (best 245) | final |
| FP16 | **0.4091** | 0.5489 | +0.1% | 283 (best 233) | final |
| SF16 | **0.3162** | 0.4446 | −22.7% | 300 (best 283) | final |
| SF8 | **0.3111** | 0.4388 | −23.9% | 300 (best 288) | final |
| SF4 | **0.0000** | 0.0000 | −100.0% | 263 (best 1) | final — see §8.2 |

## 6. Video transformer — V-JEPA 2 ViT-L / UCF101

25-class subset, 16 frames at 256px, split 80/20 **over source videos**.

UCF101 cuts each source video into ~5.25 clips (1,818 videos → 9,537 clips),
and sibling clips are near-duplicate segments. A clip-level split leaks ~4 of
every 5 siblings into training and scores an fp32 probe at **100.00%**. All
figures below use the video-level split.

Two regimes are reported. **PTQ** freezes the pretrained backbone, quantizes it
once, and trains only an fp32 attentive probe on its features — V-JEPA's own
evaluation protocol, and the one matching the paper's scoping of SFx as a
deployment-phase format. **QAT** unfreezes the backbone and trains through the
grid via the bounded STE, as the CNN benchmarks do. The two regimes are not
comparable to each other; each carries its own FP32 control.

### Weights **and** activations quantized

| Format | PTQ (probe) | QAT |
|---|---|---|
| FP32 | **97.07** | **77.51** |
| SF16 | 48.81 | 5.12 |
| SF8 | 51.01 | 5.12 |
| SF4 | 47.90 | 5.12 |

### Weights only

| Format | PTQ (probe) | QAT | vs FP32 (QAT) |
|---|---|---|---|
| FP32 | **97.07** | **77.51** | — |
| SF16 | 96.53 | **85.56** | **+8.05** |
| SF8 | **98.90** | **83.91** | **+6.40** |
| SF4 | 74.22 | 5.12 | −72.4 |

**SF16 and SF8 both exceed FP32 under QAT**, on identical schedules, data and
learning rates — the only difference is the numeric format. SF8 also exceeds
FP32 under PTQ (98.90 vs 97.07) at 75% storage reduction.

FP32-QAT (77.51) sits below FP32-PTQ (97.07) because 15 epochs of end-to-end
fine-tuning with a randomly-initialised head is a harder optimisation than 40
epochs of probe training on cached frozen features. Part of SFx's margin over
FP32-QAT is plausibly the bounded grid acting as a regulariser on a short
schedule — the same effect visible in EuroSAT's seed variance. The comparison
is valid as a same-schedule control; it does not establish that SFx is
inherently more accurate than full precision.

## 7. Activation quantization is architecture-bound

Quantizing activations costs V-JEPA ~48 points under PTQ and drives QAT to
chance, while **bit-width barely matters** (PTQ: SF16 48.8, SF8 51.0, SF4 47.9;
QAT: all three at exactly 5.12). If this were a precision effect, SF4 — whose
grid is 4096× coarser than SF16's — would be far worse. It is not. The damage
comes from the one property all three share: the **[−1, 1] bound**.

Measured directly on V-JEPA 2 across 292 Linear/LayerNorm outputs:

| Statistic | Value |
|---|---|
| max \|activation\| | **256.14** |
| mean fraction with \|a\| > 1 | **26.26%** |
| layers with >10% of activations over 1 | **193 / 292** |

BatchNorm renormalises CNN activations to roughly unit scale, so the ±1 grid is
nearly free there. A ViT residual stream accumulates across 24 blocks and spans
±256; clamping truncates the representation regardless of how finely the
survivors are gridded. **QAT does not recover it** — the forward signal is
destroyed thoroughly enough that no gradient survives to adapt the network, and
train accuracy sits at chance alongside validation accuracy.

> **Finding.** SuperFloat *weight* quantization is architecture-agnostic across
> all four families tested. SuperFloat *activation* quantization is not: it
> assumes the network's normalisation confines activations to [−1, 1], which
> BatchNorm CNNs satisfy and transformer residual streams violate. This is a
> property of the architecture, not a training deficiency.

**Hardware consequence for Atreides.** The exponent-free SF16 weight datapath
generalises to transformers. A transformer accelerator would additionally need
a per-tensor activation scale factor or wider activation storage; the saturating
fixed-point accumulate is safe for weights but not for a residual stream.

## 8. Two failure modes, and their causes

### 8.1 SF8 from random init — learning rate must match grid resolution

| Run | LR | Best mAP50-95 | Peak epoch | Stopped | Outcome |
|---|---|---|---|---|---|
| visdrone_random_sf8 | 4e-3 | 0.0724 | 18 | 68 | collapsed |
| visdrone_random_sf8_s1 | 4e-3 | 0.0755 | 9 | 59 | collapsed |
| visdrone_random_sf8_s2 | 4e-3 | 0.0764 | 11 | 61 | collapsed |
| visdrone_random_sf8_lr1e3 | 1e-3 | 0.2361 | 86 | 87 (truncated) | trains normally |

Three seeds at 4e-3: **0.0748 ±0.0017**. All peak by epoch 9–18 then decay
while *training* loss rises (1.94→2.22, 2.00→2.11, 1.97→2.11) — divergence, not
underfitting. The spread is ~4000× tighter than the paper's seed-dependent
CIFAR collapse (±6.7), indicating a systematic cause rather than seed luck. At
**lr 1e-3 the same configuration trains normally**, confirming step size as the
cause: SF8's grid is 256× coarser than SF16's, so an update sized for SF16
overshoots the representable grid and the bounded STE strands weights in
saturation.

### 8.2 SF4 — the representable floor, from both directions

**From random init the network is dead at step 0.** `dota_random_sf4` never
left 0.0000 mAP. Four configurations were tried:

| Config | LR | Weight decay | Result | box_loss @ep15 |
|---|---|---|---|---|
| original | 4e-3 | 0.05 | 0.0000 (263 ep) | 3.766 |
| lower LR | 1e-3 | 0.05 | 0.0000 | 3.768 |
| lowest LR | 2.5e-4 | 0.05 | 0.0000 | 3.766 |
| no decay | 1e-3 | 0.00 | 0.0000 | 3.769 |

`box_loss` is identical to three decimals across a **16× learning-rate range**
and with weight decay removed — the signature of a network with no gradient
signal. The optimiser is irrelevant because there is nothing to optimise.

**Cause: standard initialisation falls below the representable floor.** SF4's
grid step is 0.125, so any |w| < 0.0625 quantizes to **exactly zero**.

| Model / init | mean \|w\| | SF16 zeroed | SF8 zeroed | SF4 zeroed |
|---|---|---|---|---|
| YOLOv8x-OBB, Kaiming random | 0.0099 | 0.08% | 21.0% | **99.98%** |
| YOLOv8x-OBB, COCO-pretrained | 0.0058 | 0.40% | 56.6% | 99.79% |
| V-JEPA 2 ViT-L, random | 0.0160 | 0.06% | 15.5% | **99.82%** |
| V-JEPA 2 ViT-L, pretrained | 0.0494 | 0.02% | 5.79% | 69.8% |

Standard initialisation places weights an order of magnitude below SF4's floor,
annihilating the network before the first update. This is a property of a
uniform-grid format: SuperFloat has no exponent, so unlike floating point it
has **no fine resolution near zero**.

The V-JEPA rows were measured *after* predicting this outcome from the YOLO
result — the law generalises across unrelated architecture families. Note also
that YOLO11x under the identical from-scratch protocol trains fine (0.2587), so
the from-scratch failure is architecture-specific, not universal.

**From pretrained init the network starts alive but does not learn.** SF4-QAT
weights-only on V-JEPA scores 5.12 (chance = 4.0) at both `wd=0.05` and
`wd=0` — weight decay is not the cause. The likely mechanism is the mirror
image of §8.1: with a backbone LR of 1e-5 and a grid step of 0.125, a weight
needs ~12,500 consistent steps to traverse a single grid point, against ~1,785
updates available in 15 epochs. Total movement is ~0.018, far below one grid
step, so the backbone is effectively frozen *and* 69.8% zeroed, leaving a random
head to learn from degraded features. The PTQ probe reaches 74.22 from those
same features given 40 epochs at lr 1e-3. **This explanation is inferred, not
yet tested**; the experiment that would settle it is SF4-QAT at backbone
lr ~1e-3.

> **Design rule.** Usable learning rate must be matched to grid resolution from
> *both* sides. Too large and updates overshoot the grid (SF8 at 4e-3); too
> small and they never traverse a grid point at all (SF4 at 1e-5). SF16, whose
> grid is finest, is the most forgiving and trains at the FP32 recipe unchanged.

## 9. The [−1, 1] premise, measured

The paper asserts ~99% of parameters lie in [−1, 1]. Measured on trained
checkpoints:

| Model | Format | Weights outside range | In range |
|---|---|---|---|
| YOLO11x | SF16 | 0 / 53,646,624 | 100.00000% |
| YOLO11x | SF4 | 15 / 56,872,240 | 99.99997% |
| YOLOv8x-OBB | SF4 | 51 / 69,433,776 | 99.99993% |
| V-JEPA 2 ViT-L | SF4 | 0.0002% | 99.9998% |

The true figure is five nines, not ~99% — a materially stronger claim.

The premise is true but incomplete: weights lie in [−1, 1] **clustered tightly
around zero**, which is the worst case for a uniformly-spaced format (§8.2).

## 10. Architecture agnosticity

| Family | Model | Paradigm | Norm | Weight quant | Activation quant |
|---|---|---|---|---|---|
| CNN classifier | ConvNeXt-Tiny | supervised | LayerNorm | matches FP32 | works |
| CNN detector | YOLO11x | supervised | BatchNorm | −4.3% | works |
| Oriented detector | YOLOv8x-OBB | supervised | BatchNorm | −4.2% | works |
| Video transformer | V-JEPA 2 ViT-L | self-supervised JEPA | LayerNorm | beats FP32 | fails, §7 |
| Autoregressive LM | GPT-2 / GPT-3 | supervised | LayerNorm | prior work | — |

## 11. Methodology notes

**fp32 accumulate, not fp64.** The SF grid is exact in fp32: the coarsest
requirement is SF16's scale of 2^15, and fp32 represents every integer to 2^24
exactly. The reference `cifar_modular/model.py` casts to float64, which Metal
cannot do at all.

**TF32 must be disabled** on Ampere/Ada/Blackwell — see §2. Consequence for
hardware selection: with tensor cores unusable, GPU choice is governed by plain
fp32 throughput. Measured in that regime: conv 3×3 takes 3.64 ms on B200,
6.63 ms on RTX PRO 6000 Blackwell, and 20.29 ms on H100; fp32 GEMM runs at
63.9 / 78.2 / 52.1 TFLOPS respectively. Convolution-dominated workloads
therefore favour B200 by ~5.6× over H100, despite H100 being the nominally
higher-tier accelerator for mixed precision.

**Bounded STE as plain ops.** `clamp`'s own backward already zeroes gradients
outside the representable range, which *is* the bounded STE, so the quantizer is
written without a custom `autograd.Function`. This keeps it traceable for
TorchInductor; a custom Function forces a graph break at every quantized layer
and cost ~20% throughput.

**Layer conversion rebinds `__class__`** rather than wrapping modules, so
parameter objects, `state_dict` keys and `isinstance` checks are unchanged.
Checkpoints therefore deserialize still quantized (verified: 149/174 convs
remain SFConv2d after a torch 2.6 → 2.8 round trip).

**Validate what the evaluator sees.** Ultralytics constructs `ModelEMA` *before*
the `on_pretrain_routine_end` callback and validates against that copy, so
patching only `trainer.model` leaves validation running in full precision while
reporting it as quantized. Layer classes were audited at runtime rather than
assumed.

**Controls caught every harness bug in this study.** Three faults were found,
and in each case the *full-precision control* exposed them — the quantized runs
looked superficially plausible throughout:

| Symptom | Cause |
|---|---|
| FP32 probe at 100.00% | UCF101 split at clip level, leaking near-duplicate siblings |
| FP32 QAT at chance | one AdamW rate for a 300M pretrained backbone and a random head |
| FP32 QAT still at chance | `clip_grad_norm_` over all parameters, letting the backbone swamp the head's gradient |

Dropping the FP32 row to save compute would have concealed all three. A
corollary: a result in which the quantized model *beats* full precision by a
large margin should be treated as a bug signal until the control is verified —
one intermediate sweep showed exactly that, and the cause was an excessive
learning rate that quantization clamping happened to mitigate.

## 12. Reproduction

```
superfloat.py          SFx grid, bounded STE, layer surgery, fused clamping
train_eurosat.py       ConvNeXt-Tiny classification
train_yolo.py          YOLO detection via Ultralytics callbacks
train_vjepa_probe.py   V-JEPA 2 PTQ + attentive probe
train_vjepa_qat.py     V-JEPA 2 end-to-end QAT
video_data.py          UCF101 clip assembly, video-level split
modal/modal_sweep.py   20-job detection grid
modal/modal_eurosat.py EuroSAT baselines
modal/modal_vjepa.py   V-JEPA 2 weight-representability analysis
modal/modal_vjepa_train.py  V-JEPA 2 PTQ and QAT benchmarks
test_superfloat.py     26 correctness checks (grid, STE, clamp, surgery)
plot_curves.py         per-format convergence figures
plot_summary.py        cross-format comparison figures
```

Quantization correctness is asserted by `test_superfloat.py`: grid constants
match the paper, the optimized fused-clamp path is bit-identical to a naive
per-module implementation, STE gradients are exactly zero on saturated entries,
quantized outputs lie exactly on the SFx grid, and heads are provably excluded.
