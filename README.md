# Superfloat: Accelerators for AI on Edge. Reimagined.

Superfloat (SFx) is a parameter-aware numeric schema for deep learning
inference: it discards the exponent field entirely and spends every bit beyond
the sign on the significand. The premise is a measured property of trained
networks rather than an approximation of floating point — weights are bounded
and concentrated near zero, so the dynamic range an exponent buys is largely
unused at inference time.

This repository holds the format definition, the training and benchmark suite,
and the measured results. The silicon, the compiler and the website live in
their own repositories:

| Repository | What it is |
| --- | --- |
| [superfloat.gpu](https://github.com/aloshdenny/superfloat.gpu) | Atreides — the Q1.15 (SF16) accelerator RTL, ISA and Sky130 hardening |
| [superfloat.llvm](https://github.com/aloshdenny/superfloat.llvm) | Clang/LLVM fork adding `sf16` as a first-class builtin type |
| [superfloat-site](https://github.com/aloshdenny/superfloat-site) | Project website |
| [superfloat.cuda](https://github.com/aloshdenny/superfloat.cuda) | CUDA/C++ kernels |

---

## What is Superfloat?

SFx allocates **1 sign bit and `x−1` significand bits. There is no exponent
field and no integer bit.** This makes SFx mathematically identical to a signed
fixed-point format, Q1.(x−1):

```
value = (−1)^sign × Σ b_i · 2^(−i)      for i = 1 … x−1
```

| Format | Notation | Scale | Representable range |
| --- | --- | --- | --- |
| SF16 | 1.15 | 2^15 = 32768 | ±0.999969482421875 |
| SF8 | 1.7 | 2^7 = 128 | ±0.9921875 |
| SF4 | 1.3 | 2^3 = 8 | ±0.875 |

The schema is defined for any bit width; intermediate members such as SF14 or
SF6 are constructed identically. SF16, SF8 and SF4 are the three points carried
through to hardware and evaluated here.

Every SFx variant is backward compatible: exact conversion without additional
quantization is possible when the source floating point format provides at
least `x−1` effective significand bits.

**Why this works.** Measured across trained model families, ~99.999% of
parameters already fall inside `[−1, 1]` — 0 of 53.6M YOLO11x weights fall
outside SF16's range, and only 15 outside SF4's tighter ±0.875. The exponent is
paying for range that trained weights do not use.

![Float vs Superfloat](assets/results/FloatvsSuperfloat.jpg)

---

## Benchmarks

[`benchmarks/`](benchmarks/) evaluates SFx across four architecture families
with FP32 and FP16 reference rows trained in the same pipeline, so the
quantization cost is measured rather than cited. Full tables and failure-mode
analyses: [SUPERFLOAT_RESULTS.md](SUPERFLOAT_RESULTS.md). Scripts and Modal
apps: [benchmarks/README.md](benchmarks/README.md).

A separate study asks how the usable precision moves with model width, depth,
parameter count and training tokens, and finds that the floor was never set by
precision but by where the scale factor lives:
[SCALING_LAWS.md](SCALING_LAWS.md).

![Validation trajectories](benchmarks/figures/format_overlay.png)

### Results measured in this repository

| Domain | Model | Dataset | SF16 vs FP32 |
| --- | --- | --- | --- |
| Remote sensing classification | ConvNeXt-Tiny | EuroSAT | **+0.1%** (96.43 vs 96.31) |
| UAV object detection | YOLO11x | VisDrone | −4.3% (0.2817 vs 0.2942) |
| Satellite object detection | YOLOv8x-OBB | DOTAv1 | −4.2% (0.4298 vs 0.4489) |
| Video, self-supervised ViT | V-JEPA 2 ViT-L | UCF101 | **+8.1%** (85.56 vs 77.51) |

- **SF8 is indistinguishable from SF16** in every domain (<1% apart, in both
  directions) — 7 significand bits suffice, for a 75% storage reduction.
- **SF8 and SF16 beat full precision on V-JEPA 2** under weights-only
  quantization: 83.91 and 85.56 against an FP32 control at 77.51, on identical
  schedules.
- **SF4 is free on classification** (96.27 ±0.03 vs FP32's 96.31 ±0.55, at
  87.5% storage reduction) but costs 15–22% on dense localisation.

### Two reproducible failure modes

**Weight quantization is architecture-agnostic; activation quantization is
not.** Clamping activations to [−1, 1] is nearly free after BatchNorm, which
holds CNN activations near unit scale. A ViT residual stream accumulates across
24 blocks: measured max |a| = **256.1**, with **26.3%** of activations outside
the bound across 193 of 292 layers. Quantization-aware training does not
recover it.

**Usable learning rate must match grid resolution, from both sides.** SF8 from
random init diverges at the FP32 recipe's 4e-3 (0.0748 ±0.0017 over 3 seeds)
and trains normally at 1e-3, because its grid is 256× coarser than SF16's. SF4
fails in the opposite direction: standard Kaiming init places weights an order
of magnitude below its 0.0625 floor, zeroing **99.98%** of them at step 0.

### Prior results from the paper

The tables below are from the SuperFloat paper, not re-measured here. They
cover ResNet depths on CIFAR and ImageNet, and the GPT-2/GPT-3 weight
distribution and convergence studies in [assets/results](assets/results/).

**CIFAR-10 top-1 (%), mean ± std over 3 seeds**

| Format | R20 | R32 | R44 | R56 |
| --- | --- | --- | --- | --- |
| FP32 | 87.4 ±0.1 | 87.7 ±0.1 | 88.4 ±0.2 | 88.5 ±0.3 |
| FP16 | 87.4 ±0.1 | 88.0 ±0.1 | 88.3 ±0.4 | 88.6 ±0.2 |
| SF16 | 86.9 ±0.3 | 86.8 ±0.3 | 71.5 ±10.4 | 47.1 ±12.0 |
| SF8 | 86.8 ±0.2 | 86.9 ±0.1 | 82.7 ±0.5 | 74.1 ±9.9 |
| SF4 | 86.5 ±0.3 | 86.2 ±0.2 | 83.7 ±0.3 | 77.7 ±0.1 |

**CIFAR-100 top-1 (%), mean ± std over 3 seeds**

| Format | R20 | R32 | R44 | R56 |
| --- | --- | --- | --- | --- |
| FP32 | 56.7 ±0.1 | 58.1 ±0.6 | 58.8 ±0.2 | 59.0 ±0.2 |
| FP16 | 57.0 ±0.2 | 58.0 ±0.3 | 58.4 ±0.1 | 58.7 ±0.2 |
| SF16 | 52.7 ±0.7 | 55.4 ±0.3 | 17.3 ±8.1 | 15.5 ±9.2 |
| SF8 | 52.7 ±0.4 | 55.8 ±0.4 | 38.4 ±6.7 | 29.3 ±6.1 |
| SF4 | 51.7 ±0.3 | 54.4 ±0.5 | 48.6 ±1.4 | 36.8 ±1.8 |

**ImageNet-1K top-1 (%), single seed, 50 epochs**

| Format | R20 | R32 | R44 | R56 |
| --- | --- | --- | --- | --- |
| FP32 | 69.4 | 74.3 | 77.3 | 72.6 |
| FP16 | 69.95 | 74.5 | 74.5 | 72.8 |
| SF16 | 69.08 | 65.1 | 62.2 | 63.8 |
| SF8 | 69.17 | 74.1 | 63.9 | 63.5 |
| SF4 | 66.25 | 69.2 | 65.5 | 64.3 |

**Language models.** GPT-2 (124M) and GPT-3 (125M) trained on Fineweb-100B for
1 epoch; weight-distribution plots for GPT-2/3, Llama-2-7B, Mistral-7B,
Qwen2-7B, MiniCPM-V and Japanese StableLM are in
[assets/results/LLM Distribution](assets/results/LLM%20Distribution/), with
YOLOv5/v7 layer-wise distributions alongside them.

The benchmarks in this repository extend that picture in one important way: the
CIFAR tables show SFx destabilising at depth with large seed variance
(SF16 at R56: 47.1 ±12.0), whereas the modern-architecture results here are
stable, and the instabilities that do appear have identified, reproducible
causes rather than seed dependence.

---

## Chip-1: Atreides

Atreides is an ASIC accelerator built for SFx inference: a Q1.15 fixed-point
GPU with a redesigned systolic array, a custom 16-bit ISA and fused
multiply-add units that carry no exponent hardware. RTL, testbenches and the
Sky130 hardening flow are in
[superfloat.gpu](https://github.com/aloshdenny/superfloat.gpu).

Silicon configuration: 2 cores × 2 threads, one 2×2 systolic array per core
(8 MAC units), 128 B address-mapped on-die scratchpad, 8×4 Tiny Tapeout tiles,
50 MHz target.

![Chip-1 Architecture](assets/results/atreides_architecture.png)

### FMA unit

The FMA is where removing the exponent pays. Sign is computed by XOR, the
significands go through a 15×15 unsigned multiply, and the result saturates
into Q1.15 with a 32-bit internal accumulator. There is no exponent-difference
barrel shifter, no leading-zero detector and no round-to-nearest-even packing.

![FMA](assets/results/FMA.png)

### Measured hardening results

Post-place-and-route, Sky130 HD, 20 ns (50 MHz) constraint. All four levels
close with zero DRC, LVS and antenna violations.

| Design | WNS (ns) | Worst slack (ns) | Implied Fmax (MHz) | Core area (µm²) | Util | Power (W) |
| --- | --- | --- | --- | --- | --- | --- |
| Processing Element | 0.0 | 6.73 | 75.3 | 26268.9 | 0.691 | 0.00273 |
| Systolic Array | 0.0 | 7.62 | 80.8 | 99382.8 | 0.865 | 0.00696 |
| Core | 0.0 | 0.93 | 52.5 | 451496 | 0.288 | 0.01224 |
| Accelerator | 0.0 | 3.95 | 62.3 | 694446 | 0.545 | 0.02165 |

Against IEEE baselines hardened through the identical flow, at iso clock
period and iso library:

| Metric | SF16 PE | IEEE FP16 PE | IEEE FP32 PE |
| --- | --- | --- | --- |
| Stdcell area (µm²) | 18147.4 | 25396.9 (1.40×) | 96919.2 (5.34×) |
| Setup slack (ns) | +6.726 | +0.177 | −5.045 |
| Implied Fmax (MHz) | 75.34 | 50.45 | 39.93 |
| Meets 50 MHz | yes | yes | no (69 violating paths) |

The advantage compounds at array granularity: a 2×2 IEEE FP16 systolic array is
1.63× the SF16 array's stdcell area (139775 vs 85931.2 µm²) and misses the
50 MHz target by 0.42 ns, while the SF16 array closes with 7.62 ns of margin.

One honest caveat: at full-chip level the critical path leaves the arithmetic
datapath and lands in memory address generation, so chip-level Fmax is bounded
by the memory subsystem rather than by the numeric format.

### Instruction set

Atreides implements a 14-instruction, 16-bit ISA in a 4-field register-register
encoding. Integer ops handle indexing, addressing and control flow; FMA and ACT
are reserved for Q1.15 matrix math so accumulation precision is never lost to
the integer path.

```
[OPCODE 15:12] [Rd 11:8] [Rs 7:4] [Rt 3:0]
```

| Opcode | Mnemonic | Operands | Description |
| --- | --- | --- | --- |
| 0000 | NOP | — | No operation |
| 0001 | BRnzp | offset9 | Branch on nzp flags to PC + 1 + sign_extend(offset9) |
| 0010 | CMP | Rd, Rs | Compare integers, set nzp flags |
| 0011 | ADD | Rd, Rs, Rt | Integer add |
| 0100 | SUB | Rd, Rs, Rt | Integer subtract |
| 0101 | MUL | Rd, Rs, Rt | Integer multiply, 16-bit zero-extended result |
| 0110 | DIV | Rd, Rs, Rt | Integer divide, hardware reciprocal |
| 0111 | LDR | Rd, Rs | Load 16-bit word from data memory at address Rs |
| 1000 | STR | Rd, Rs | Store 16-bit word from Rs to address Rd |
| 1001 | CONST | Rd, imm8 | Load 8-bit sign-extended immediate |
| 1010 | FMA | Rd, Rs, Rt | Q1.15 fused multiply-accumulate: Rd = (Rs × Rt) + Rd |
| 1011 | ACT | Rd, Rs, Rt | Bias-add + activation (passthrough / ReLU / leaky / clipped) |
| 1100 | SYS | op, idx | Systolic array control (clear / load / compute / read) |
| 1111 | RET | — | Return from kernel |

Each thread owns 16 registers: `R0`–`R12` are general purpose read/write, and
`R13`–`R15` are hardwired read-only SIMD index registers (`%blockIdx`,
`%blockDim`, `%threadIdx`) that give a thread its coordinate without dedicated
addressing hardware.

Throughput: 400 × 10⁶ MAC/s across the systolic paths (8 PEs × 1 MAC/cycle ×
50 MHz), 100 × 10⁶ FMA/s on the scalar path.

---

## Compiler

[superfloat.llvm](https://github.com/aloshdenny/superfloat.llvm) is an LLVM 18
fork (branch `my-llvm-changes`) that adds `sf16` as a builtin C type rather
than a library typedef, so the format survives the whole pipeline:

- **LLVM IR** — `sf16` as a first-class type, through `Type.h`, the AsmParser,
  AsmWriter and `ValueTypes`.
- **Clang frontend** — the `sf16` keyword in `TokenKinds.def`, a builtin type
  in `BuiltinTypes.def`, plus AST, lexer, parser, `Sema` and Itanium mangling.
- **CodeGen** — 16-bit width, 15-bit scale, lowered to `i16` with fixed-point
  intrinsics for the arithmetic.
- **Target** — RISC-V, matching Atreides' modded RV32 control path.

```c
sf16 literal_test() {
    sf16 x = 0.5;
    return x;
}
```

---

## Repository layout

```
benchmarks/         SFx training and evaluation suite (see benchmarks/README.md)
  superfloat.py       SFx grid, bounded STE, layer surgery, clamping
  train_eurosat.py    ConvNeXt-Tiny classification
  train_yolo.py       YOLO detection via Ultralytics callbacks
  train_vjepa_*.py    V-JEPA 2 PTQ probe and end-to-end QAT
  test_superfloat.py  correctness tests (run these first)
  make_figures.py     all paper figures, one uniform style
  analyze_scaling.py  logistic fit of the critical precision, width law
  make_scaling_figures.py  the four-tier scaling figures
  make_lab_figures.py      the follow-up experiment figures
  lab/                follow-up experiments (see benchmarks/lab/README.md)
  results/            every raw scaling result, one JSONL per experiment
  modal/              Modal apps for the cloud sweeps
cifar_modular/      ResNet CIFAR training used for the paper's CIFAR tables
src/
  modal/              GPT-2/GPT-3 pretraining under clamped matmul
  test/               matrix/stream generators for hardware testbenches
  verilog/            early FPGA functional units (superseded by superfloat.gpu)
Q115 layer story.py Layer-by-layer Q1.15 vs FP32 signal analysis on ResNet-20
assets/results/     weight-distribution studies and architecture figures
docs/paper/         TPAMI manuscript, anonymized main document and title page
```

## Usage

```bash
git clone https://github.com/aloshdenny/superfloat
cd superfloat/benchmarks

python test_superfloat.py                       # verify the quantizer first
python train_eurosat.py --format sf8 --seed 0   # classification
python train_yolo.py --format sf8 --cfg yolo11x.yaml --data VisDrone.yaml \
    --init pretrained --imgsz 640 --batch 8     # detection
```

Cloud sweeps run on Modal. Deployed apps must be spawned by name — `modal run`
creates an *ephemeral* app that stops when its entrypoint returns, cancelling
every job it spawned:

```bash
modal deploy modal/modal_sweep.py
```

```python
import modal
modal.Function.from_name("superfloat-sweep", "train_baseline").spawn("visdrone_pretrained_sf16")
```

---

## Contributions

Contributions are welcome. Feel free to open issues or submit pull requests.

## Sponsors

<div style="display: flex; justify-content: space-between;">
  <img src="https://pbs.twimg.com/profile_images/1848649662825406464/NFqR2OSK_400x400.jpg" width="200"/>
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ5jctHbOd3dceXJxLi7tFZ8h1tqxfOrX7YAg&s" width="200"/>
  <img src="https://pbs.twimg.com/profile_images/1247800867777994755/JjEBNHba_400x400.jpg" width="200"/>
  <img src="https://styles.redditmedia.com/t5_bxucfi/styles/profileIcon_xjiodpqbkvbd1.jpg?width=256&height=256&frame=1&auto=webp&crop=256:256,smart&s=bda66bfc6dae1682cf1e5351a48ae8e473e12203" width="200">
</div>

## License

This project is licensed under the MIT License.
