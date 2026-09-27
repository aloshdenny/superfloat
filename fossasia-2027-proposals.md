# FOSSASIA Summit 2027 — CFP drafts

## Track 1: Artificial Intelligence

### Title

**Precision Was Never the Problem: Running Tool-Calling LLMs on 8-Bit Fixed-Point With Zero Retraining**

Alternates:
- Kill the Exponent: What 878 Training Runs Taught Us About Quantizing LLMs
- Your LLM Doesn't Need an Exponent: Superfloat and the Case for Fixed-Point Edge AI

### Abstract

Every LLM you run at the edge carries an exponent field it barely uses: trained weights are bounded and cluster near zero, so most of a float's dynamic range is dead weight. Superfloat (SFx) is an open-source numeric format that drops the exponent entirely and spends every bit after the sign on the significand, making it a plain signed fixed-point number. This talk is about what happens when you push real language models onto it.

I'll walk through the open benchmark suite behind the project: an 878-run scaling study showing that the "precision floor" everyone reports is an artefact of where scales are placed, not of the format; why weights survive at 3–4 bits while activations need 6; and why the worst checkpoint to quantize is the over-trained one that ships. Then the practical result: Qwen3 1.7B–8B quantized to 8-bit SF post-training, with no retraining, holds function-calling accuracy on BFCL within a point of bf16, plus the one class of weights that silently destroys a model if you don't check the checkpoint first.

Everything is MIT-licensed, reproducible from archived results, and runs on one consumer GPU. Attendees leave with a deployment recipe and a healthy suspicion of quantization folklore.

### Experience / connection

I created Superfloat and have maintained it as an open-source project since its first paper. Over the past year I designed and ran the full study this talk draws on: 878 archived training runs across four scaling tiers and eight follow-up experiments, the tool-calling PTQ/QAT study on SmolLM2 and Qwen3 0.6B–8B evaluated on BFCL, and the pure-datapath study that showed per-token scaling is what keeps a model alive on saturating hardware. Every number in the talk traces to a script and a result file in the public repository. Professionally I head AI at Aleddo Technologies and run LLMOps at Intensors, so the edge-deployment questions are ones I face in production, not just in benchmarks. I have previously spoken on this work at IndiaFOSS and GHCI.

---

## Track 2: Open Hardware

### Title

**No Exponent, No Problem: Building an Open-Source AI Accelerator From RTL to Sky130 GDS**

Alternates:
- Atreides: A Fixed-Point AI Accelerator for Tiny Tapeout, Built Entirely With Open Tools
- Hardware–Software Co-Design in the Open: One Number Format, One Chip, One Compiler

### Abstract

What if the numeric format were designed for the silicon rather than the other way round? Atreides is an open-source AI inference accelerator built around Superfloat, a fixed-point format with no exponent. Removing the exponent removes the barrel shifter, the leading-zero detector and the rounding logic from every multiply-add unit. This talk follows the design from format to GDS with a fully open toolchain: Verilog RTL, Cocotb testbenches, Yosys, OpenROAD and LibreLane, hardened on the Sky130 PDK for a Tiny Tapeout 8×4 tile.

I'll cover the architecture: two cores, a 2×2 systolic array per core, a 14-instruction 16-bit ISA with hardwired SIMD index registers, a 128-byte on-die scratchpad and an external SRAM bus. Then the measured post-place-and-route numbers: the SF16 processing element closes at 75 MHz in 1.4× less area than an IEEE FP16 PE and 5.3× less than FP32, through the identical flow. And the honest caveat: at full-chip level the critical path leaves the arithmetic and lands in memory address generation.

Finally, the co-design loop: how the accelerator's saturating datapath sent us back to the model side, and how an LLVM fork makes `sf16` a first-class C type targeting RISC-V.

### Experience / connection

I designed Atreides end to end: the Q1.15 fused multiply-add unit, the systolic array, the 14-instruction ISA, the memory controllers and scratchpad, the Cocotb test suite, and the LibreLane/OpenROAD hardening flow that closes timing on Sky130 with zero DRC, LVS or antenna violations. I also ran the IEEE FP16 and FP32 baselines through the same flow so the area and timing comparisons are measured rather than cited. On the software side I maintain the Superfloat benchmark suite and the Clang/LLVM fork that adds `sf16` as a builtin type, and I wrote the study showing what the saturating datapath costs at the model level. All of it is public under MIT on GitHub. I previously presented this work at IndiaFOSS and GHCI, and I mentor hardware and AI builders at TinkerHub's TinkerSpace.

---

## Biography (≤150 words)

Alosh Denny is an AI engineer and researcher from Kerala, India, and the creator of Superfloat: an open-source numeric format, accelerator (Atreides) and compiler stack for fixed-point AI inference at the edge. He heads AI at Aleddo Technologies and runs LLMOps at Intensors, and previously built agentic security systems at Discern Security and Malayalam-language LLMs at Eduport. He has seven peer-reviewed publications across IEEE and Springer, alongside a self-directed line of interpretability work including SynthID watermark analysis and abliteration of MoE and multimodal brain-encoding models. A hobbyist and mentor at TinkerHub's TinkerSpace, he led the AI/ML and exploration effort of Team Horizon's Mars rover at the European Rover Challenge and has spoken at IndiaFOSS and GHCI. A Computer Science graduate of CUSAT, he writes Verilog when he isn't training models.
