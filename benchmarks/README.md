# SuperFloat Benchmarks

Quantization-aware training with SuperFloat (SFx) across three domains, with
FP32 and FP16 reference rows trained in the same pipeline.

| Domain | Model | Dataset |
| --- | --- | --- |
| Remote sensing classification | ConvNeXt-Tiny | EuroSAT |
| UAV object detection | YOLO11x | VisDrone |
| Satellite object detection | YOLOv8x-OBB | DOTAv1 |

## Layout

```
benchmarks/
  superfloat.py        SFx grid, bounded STE, layer surgery, clamping
  train_eurosat.py     ConvNeXt-Tiny classification
  train_yolo.py        YOLO detection via Ultralytics callbacks
  test_superfloat.py   correctness tests (run these first)
  collect_results.py   results tables
  plot_curves.py       convergence figures
  modal/
    modal_sweep.py     detection sweep (20 runs)
    modal_eurosat.py   classification runs
```

## Running

```bash
python test_superfloat.py                      # verify the quantizer first

python train_eurosat.py --format sf8 --seed 0  # classification
python train_yolo.py --format sf8 --cfg yolo11x.yaml --data VisDrone.yaml \
    --init pretrained --imgsz 640 --batch 8

modal deploy modal/modal_sweep.py              # cloud sweep
```

Deployed Modal apps must be spawned against by name. `modal run` creates an
*ephemeral* app that stops when its entrypoint returns, cancelling every job it
spawned:

```python
import modal
fn = modal.Function.from_name("superfloat-sweep", "train_baseline")
fn.spawn("visdrone_pretrained_sf16")
```

## Results

All numbers below were produced by this code. Detection is mAP50-95 on the
validation split; classification is top-1 accuracy.

### EuroSAT — ConvNeXt-Tiny, from scratch, mean ± std over 3 seeds

| Format | Top-1 (%) | Per-weight storage saving |
| --- | --- | --- |
| SF16 | **96.43 ± 0.06** | 50% |
| FP32 | 96.31 ± 0.54 | — |
| SF8 | 96.27 ± 0.19 | 75% |
| SF4 | 96.27 ± 0.03 | 87.5% |
| FP16 | 96.18 ± 0.45 | 50% |

Every SuperFloat format matches full precision within seed noise, including SF4
at an 87.5% storage reduction. The SFx runs also show markedly *lower* seed
variance than FP32/FP16, i.e. the bounded grid behaves as a regulariser here.

### VisDrone — YOLO11x @640, COCO-pretrained, 150 epochs

| Format | mAP50-95 | mAP50 | vs FP32 |
| --- | --- | --- | --- |
| FP16 | 0.2947 | 0.4825 | +0.2% |
| FP32 | 0.2942 | 0.4827 | — |
| SF8 | 0.2836 | 0.4690 | −3.6% |
| SF16 | 0.2817 | 0.4676 | −4.2% |
| SF4 | 0.2501 | 0.4208 | −15.0% |

### DOTAv1 — YOLOv8x-OBB @800, COCO-pretrained, 150 epochs

| Format | mAP50-95 | mAP50 | vs FP32 |
| --- | --- | --- | --- |
| FP16 | 0.4496 | 0.5994 | +0.2% |
| FP32 | 0.4489 | 0.5942 | — |
| SF16 | 0.4298 | 0.5753 | −4.3% |
| SF8 | 0.4254 | 0.5705 | −5.2% |
| SF4 | 0.3488 | 0.4891 | −22.3% |

### From-scratch detection (300 epochs)

| Format | VisDrone | DOTAv1 |
| --- | --- | --- |
| FP32 | 0.3011 | 0.4088 |
| FP16 | 0.2998 | 0.4091 |
| SF16 | 0.2948 | 0.3162 |
| SF8 | see below | 0.3111 |
| SF4 | 0.2587 | 0.0000 (see below) |

## Findings

**SF8 is indistinguishable from SF16.** Across all three domains the two differ
by under 1%, and in opposite directions depending on the task — i.e. noise. Seven
significand bits are sufficient; the extra eight buy nothing.

**The SuperFloat tax is ~4% on detection and ~0% on classification.** SF16 lands
−4.2% and −4.3% below true FP32 on two unrelated detection tasks, but matches
FP32 on EuroSAT. Quantization cost is task-dependent, not format-dependent.

**SF4's floor is task-dependent.** Free on classification, −15% and −22% on
detection. Dense localisation needs finer weight resolution than 10-class
classification does.

**FP16 ≡ FP32 everywhere measured**, so the meaningful comparison is SuperFloat
against either, not FP32 against FP16.

### Two failure modes worth knowing about

**SF8 from random init diverges at lr 4e-3 but trains at 1e-3.** Three seeds at
4e-3 all collapsed to 0.0748 ± 0.0017 — peaking around epoch 10-18 then decaying
with *rising* training loss. The identical spread across seeds indicated a
systematic cause rather than seed luck. At lr 1e-3 the same configuration trains
normally (0.2312 by epoch 78 and still climbing). SF16 tolerates 4e-3 because
its grid is 256x finer; a step sized for SF16 overshoots SF8's grid.

**SF4 cannot be trained from random init on YOLOv8x-OBB.** Kaiming initialisation
gives mean |w| = 0.0099, while SF4's zero-threshold is 0.0625, so **99.98% of
conv weights quantize to exactly zero at step 0** and the network has no gradient
signal. This was verified as unrecoverable across four optimiser configurations
(lr 4e-3 / 1e-3 / 2.5e-4, and weight decay 0), all producing identical training
loss to three decimal places. It is a property of a uniform grid with no
exponent: SuperFloat has no fine resolution near zero, and standard
initialisation places weights an order of magnitude below the representable
floor. Note YOLO11x under the same protocol trains fine (0.2587), so this is
architecture-specific, not universal.

### Measured: how much of the weight distribution SuperFloat can hold

The premise that trained weights lie in [-1, 1], measured on COCO-trained
detectors rather than assumed:

| Model | Format | Weights outside range |
| --- | --- | --- |
| YOLO11x | SF16 | 0 / 53,646,624 |
| YOLO11x | SF4 | 15 / 56,872,240 (0.00003%) |
| YOLOv8x-OBB | SF4 | 51 / 69,433,776 (0.00007%) |

99.99993% of trained weights are representable even at SF4's ±0.875 bound.

## Implementation notes

**fp32 is used throughout, not fp64.** The SF grid is exact in fp32: the
coarsest requirement is SF16's scale of 2^15, and fp32 represents every integer
to 2^24 exactly. `cifar_modular/model.py` casts to float64, which Metal cannot
do at all.

**TF32 must be disabled on Ampere/Ada.** It keeps 10 mantissa bits and would
silently round SF16 (15 significand bits) to below SF8 fidelity inside every
convolution, making the formats look falsely equivalent. `disable_tf32()`
handles this.

**The STE is written as plain ops, not an `autograd.Function`.** `clamp`'s own
backward already zeroes gradients outside the range, which *is* the bounded STE,
so the whole thing is traceable and TorchInductor can fuse it. A custom Function
forces a graph break at every quantized layer and cost ~20% throughput.

**Layer conversion rebinds `__class__`** rather than wrapping modules, so
parameter objects, `state_dict` keys and `isinstance` checks are unchanged.
This matters for Ultralytics, which deep-copies its model graph for EMA — and
it means checkpoints deserialize still quantized.

**Validate what the EMA sees.** Ultralytics builds `ModelEMA` *before* the
`on_pretrain_routine_end` callback, and validates against that copy. Patching
only `trainer.model` leaves validation running in full precision while
reporting it as quantized. `test_superfloat.py` and the runtime audit in
`train_yolo.py` exist because of this.
