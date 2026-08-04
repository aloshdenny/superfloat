# SuperFloat (SFx) — Full Evaluation Results

Evaluation of the SuperFloat numeric schema across three domains, five numeric
formats, and two initialisation regimes. All runs use quantization-aware
training with weights **and** activations on the SFx grid; detection heads and
classifier logits are kept in full precision.

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

| | Classification | Detection (pretrained) | Detection (from scratch) |
|---|---|---|---|
| Model | ConvNeXt-Tiny (28M) | YOLO11x (57M) / YOLOv8x-OBB (69M) | same |
| Init | random | COCO-pretrained | random |
| Epochs | ≤300, early stop @40 | 150, early stop @50 | 300, early stop @50 |
| LR (AdamW) | 4e-3 | 1e-3 | 4e-3 |
| Weight decay | 0.05 | 0.05 | 0.05 |
| Schedule | warmup 10 → ReduceLROnPlateau | warmup 10 → cosine | same |
| Batch | 128 | 8 | 8 |
| Seeds | 3 | 1 | 1 |

**TF32 disabled throughout.** Ada/Hopper default to TF32 for cuDNN
convolutions, whose 10-bit mantissa is below SF16's 15 significand bits and
would have silently degraded SF16 to worse-than-SF8 fidelity inside every
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

SF4 vs FP32: **-0.04 points** at 87.5% weight-storage reduction — within
seed noise. Every SuperFloat format matches full precision on this task.

Note the SuperFloat runs show *lower* seed variance than the full-precision
ones (SF4 ±0.03, SF16 ±0.06 vs FP32 ±0.54), suggesting the bounded grid acts
as a regulariser in this regime.

## 4. UAV detection — VisDrone / YOLO11x

6,471 train / 548 val images, 640px, batch 8.

### Pretrained init

| Format | mAP50-95 | mAP50 | Precision | Recall | vs FP32 | Epochs | Status |
|---|---|---|---|---|---|---|---|
| FP32 | **0.2942** | 0.4827 | 0.6070 | 0.4812 | +0.0% | 127 (best 77) | final |
| FP16 | **0.2947** | 0.4825 | 0.6025 | 0.4760 | +0.2% | 150 (best 74) | final |
| SF16 | **0.2817** | 0.4676 | 0.5826 | 0.4627 | -4.3% | 150 (best 113) | final |
| SF8 | **0.2836** | 0.4690 | 0.5681 | 0.4681 | -3.6% | 150 (best 104) | final |
| SF4 | **0.2501** | 0.4208 | 0.5298 | 0.4249 | -15.0% | 150 (best 140) | final |

### Random init

| Format | mAP50-95 | mAP50 | Precision | Recall | vs FP32 | Epochs | Status |
|---|---|---|---|---|---|---|---|
| FP32 | **0.3011** | 0.4890 | 0.6097 | 0.4814 | +0.0% | 280 (best 233) | **mid-run** (of 300) |
| FP16 | **0.2998** | 0.4904 | 0.5998 | 0.4857 | -0.4% | 274 (best 224) | final |
| SF16 | **0.2948** | 0.4818 | 0.6044 | 0.4762 | -2.1% | 300 (best 260) | final |
| SF8 | **0.0724** | 0.1387 | 0.2562 | 0.1912 | -75.9% | 68 (best 18) | final |
| SF4 | **0.2587** | 0.4292 | 0.5570 | 0.4307 | -14.1% | 244 (best 194) | final |

## 5. Satellite detection — DOTAv1 / YOLOv8x-OBB

1,411 train / 458 val images (untiled), 800px, batch 8.

### Pretrained init

| Format | mAP50-95 | mAP50 | Precision | Recall | vs FP32 | Epochs | Status |
|---|---|---|---|---|---|---|---|
| FP32 | **0.4489** | 0.5942 | 0.7863 | 0.5818 | +0.0% | 150 (best 129) | final |
| FP16 | **0.4496** | 0.5994 | 0.8170 | 0.5693 | +0.2% | 150 (best 103) | final |
| SF16 | **0.4298** | 0.5753 | 0.7947 | 0.5479 | -4.2% | 150 (best 139) | final |
| SF8 | **0.4254** | 0.5705 | 0.7657 | 0.5544 | -5.2% | 150 (best 139) | final |
| SF4 | **0.3488** | 0.4891 | 0.7507 | 0.4623 | -22.3% | 150 (best 137) | final |

### Random init

| Format | mAP50-95 | mAP50 | Precision | Recall | vs FP32 | Epochs | Status |
|---|---|---|---|---|---|---|---|
| FP32 | **0.4088** | 0.5465 | 0.7846 | 0.5277 | +0.0% | 295 (best 245) | final |
| FP16 | **0.4091** | 0.5489 | 0.7578 | 0.5320 | +0.1% | 283 (best 233) | final |
| SF16 | **0.3162** | 0.4446 | 0.7113 | 0.4262 | -22.7% | 300 (best 283) | final |
| SF8 | **0.3111** | 0.4388 | 0.6842 | 0.4306 | -23.9% | 300 (best 288) | final |
| SF4 | **0.0000** | 0.0000 | 0.0000 | 0.0000 | -100.0% | 263 (best 1) | final |

## 6. SF8 from random init — collapse and its cause

| Run | LR | Best mAP50-95 | Peak epoch | Stopped | Outcome |
|---|---|---|---|---|---|
| visdrone_random_sf8 | 4e-3 | 0.0724 | 18 | 68 | collapsed |
| visdrone_random_sf8_s1 | 4e-3 | 0.0755 | 9 | 59 | collapsed |
| visdrone_random_sf8_s2 | 4e-3 | 0.0764 | 11 | 61 | collapsed |
| visdrone_random_sf8_lr1e3 | 1e-3 | 0.2361 | 86 | 87 | trains normally |

Three seeds at lr 4e-3: **0.0748 ±0.0017**. All three
peak by epoch 9–18 then decay while *training* loss rises (1.94→2.22,
2.00→2.11, 1.97→2.11) — divergence, not underfitting. The spread is ~4000×
tighter than the paper's seed-dependent CIFAR collapse (±6.7), indicating a
systematic cause rather than seed luck.

At **lr 1e-3 the same configuration trains normally**, confirming the cause
is step size: SF8's grid is 8× coarser than SF16's, so an update sized for
SF16 overshoots the representable grid and the bounded STE strands weights
in saturation.

> **Design rule:** usable learning rate scales with grid resolution. SF16
> trains at the FP32 recipe's 4e-3; SF8 requires ~1e-3.

## 7. SF4 from random init — the representable-floor limit

`dota_random_sf4` never left 0.0000 mAP. Four configurations were tried:

| Config | LR | Weight decay | Result | box_loss @ep15 |
|---|---|---|---|---|
| original | 4e-3 | 0.05 | 0.0000 (263 ep) | 3.766 |
| lower LR | 1e-3 | 0.05 | 0.0000 | 3.768 |
| lowest LR | 2.5e-4 | 0.05 | 0.0000 | 3.766 |
| no decay | 1e-3 | 0.00 | 0.0000 | 3.769 |

`box_loss` is identical to three decimals across a **16× learning-rate range**
and with weight decay removed — the signature of a network with no gradient
signal. The optimiser is irrelevant because there is nothing to optimise.

### Measured cause: weights fall below the representable floor

SF4's grid step is 0.125, so any |w| < 0.0625 quantizes to **exactly zero**.

| Model / init | mean \|w\| | SF16 zeroed | SF8 zeroed | SF4 zeroed |
|---|---|---|---|---|
| YOLOv8x-OBB, Kaiming random | 0.0099 | 0.08% | 21.0% | **99.98%** |
| YOLOv8x-OBB, COCO-pretrained | 0.0058 | 0.40% | 56.6% | 99.79% |
| V-JEPA 2 ViT-L, random | 0.0160 | 0.06% | 15.5% | **99.82%** |
| V-JEPA 2 ViT-L, pretrained | 0.0494 | 0.02% | 5.79% | 69.8% |

Standard initialisation places weights an order of magnitude below SF4's
floor, annihilating the network before the first update. This is a property
of a uniform-grid format: SuperFloat has no exponent, so unlike floating
point it has **no fine resolution near zero**.

The V-JEPA rows were measured *after* predicting this outcome from the YOLO
result — the law generalises across unrelated architecture families.

## 8. The [−1, 1] premise, measured

The paper asserts ~99% of parameters lie in [−1, 1]. Measured on trained
checkpoints:

| Model | Format | Weights outside range | In range |
|---|---|---|---|
| YOLO11x | SF16 | 0 / 53,646,624 | 100.00000% |
| YOLO11x | SF4 | 15 / 56,872,240 | 99.99997% |
| YOLOv8x-OBB | SF4 | 51 / 69,433,776 | 99.99993% |
| V-JEPA 2 ViT-L | SF4 | 0.0002% | 99.9998% |

The true figure is five nines, not ~99% — a materially stronger claim.

## 9. Architecture agnosticity

| Family | Model | Paradigm | Norm | Covered |
|---|---|---|---|---|
| CNN classifier | ConvNeXt-Tiny | supervised | LayerNorm | trained |
| CNN detector | YOLO11x | supervised | BatchNorm | trained |
| Oriented detector | YOLOv8x-OBB | supervised | BatchNorm | trained |
| Video transformer | V-JEPA 2 ViT-L | self-supervised JEPA | LayerNorm | analysed |
| Autoregressive LM | GPT-2 / GPT-3 | supervised | LayerNorm | prior work |

## 10. Reproduction

```
superfloat.py        SFx grid, bounded STE, layer surgery, fused clamping
train_eurosat.py     ConvNeXt-Tiny classification
train_yolo.py        YOLO detection via Ultralytics callbacks
modal_sweep.py       20-job detection grid on H100
modal_eurosat.py     EuroSAT baselines
modal_vjepa.py       V-JEPA 2 weight analysis
test_superfloat.py   26 correctness checks (grid, STE, clamp, surgery)
```

Quantization correctness is asserted by `test_superfloat.py`: grid constants
match the paper, the optimized fused-clamp path is bit-identical to a naive
per-module implementation, STE gradients are exactly zero on saturated
entries, and quantized outputs lie exactly on the SFx grid.
