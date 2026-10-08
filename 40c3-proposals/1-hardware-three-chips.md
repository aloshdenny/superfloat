# Form fields (do not paste this part)

**Session type:** Talk (40 minutes)
**Track:** Hardware
**Language:** English
**Title:** Three Chips, One Fair Fight: Taping Out An AI Accelerator With No Floating Point

**Resources to attach:**
- https://github.com/aloshdenny/superfloat.gpu
- https://github.com/aloshdenny/superfloat
- https://github.com/aloshdenny/superfloat.llvm

---
<!-- ABSTRACT -->
---

A floating point number spends most of its bits on dynamic range that a trained neural network never uses. So I deleted the exponent field, built an AI accelerator around what was left, and then did the thing almost nobody does: built the IEEE competition too, on the same process, through the same tools, at the same clock, so the comparison could not be rigged.

Three chips are signed off with zero violations: one using a fixed point format with no exponent, one using IEEE half precision, one using bfloat16. Same architecture, same flow, same everything except the arithmetic. The multiply accumulate unit without an exponent is 1.35 times smaller, uses 2.19 times less power, needs 2.77 times fewer flip flops and clocks 1.32 times faster.

Then the honest part: on the actual chips, running GPT-2, the IEEE versions are slightly faster in wall clock. The exponent free chip wins on energy by 45 percent and loses on speed. I will show you exactly why, and why that is the more useful result.

Everything was built with open source tools on an open process. No vendor licence anywhere in the flow.

---
<!-- DESCRIPTION -->
---

## Why this talk exists

The literature is full of clever number formats that beat floating point. Almost all of them are evaluated against a baseline the authors did not build, on a process the authors did not use, through a toolchain the authors did not run. The comparison is a citation, not a measurement.

I wanted to know what my format was actually worth, so I built its competition. Three complete chips, identical in every respect except the arithmetic unit inside them: one Superfloat (sign bit plus significand, no exponent field, which makes it plain fixed point), one IEEE FP16, one BFloat16. Same RTL above the arithmetic, same process, same cell library, same clock constraint, same open source place and route flow, signed off on the same day.

This talk is what that experiment says, including the parts that went against me.

## What is on the chips

An accelerator called Atreides: four cores, each with eight threads and an eight by eight systolic array, for 256 multiply accumulate units, running at 50 MHz on a 15 mm² die. A fourteen instruction, sixteen bit instruction set. A small on die scratchpad and a sixteen bit pin bus to the host. Nothing exotic. The point is not the microarchitecture, it is the controlled comparison.

Flow: Verilog, cocotb and Icarus for verification, Yosys, OpenROAD and LibreLane for synthesis and physical design, on the Sky130 open process. All three chips pass every check with zero violations: design rules, layout versus schematic, setup and hold across nine corners, antenna, slew, capacitance, fanout.

## What the arithmetic buys

Measured on one multiply accumulate unit, placed and routed on its own and signed off:

| | no exponent | IEEE FP16 | BFloat16 |
| --- | --- | --- | --- |
| Area | 16,611 µm² | 22,469 (1.35x) | 18,161 (1.09x) |
| Flip flops | 66 | 183 (2.77x) | 115 (1.74x) |
| Power | 0.782 mW | 1.710 (2.19x) | 1.180 (1.51x) |
| Max clock, slow corner | 92.2 MHz | 69.9 | 68.6 |

Removing the exponent removes the barrel shifter that aligns two exponents, the leading zero detector that renormalises the result, and the rounding and special case logic. That is most of a floating point unit by area and a large part of its critical path.

## The result that went against me

All three chips run every kernel in exactly the same number of cycles. So the comparison comes down to clock speed and energy, and on the fabricated layouts the exponent free chip does not have the faster clock. Running GPT-2 small with a 32 token prompt, time to first token is 140 seconds for Superfloat against 131 for FP16 and 135 for BF16. The IEEE chips win on speed.

They lose badly on energy. Chip power during the same work is 250 mW against 460 and 441. Per generated token that is 5.55 joules against 10.17 and 9.74. On YOLOv8n it is 53 joules per image against 98 and 94. Consistently about 45 percent less energy for the same answer in the same number of cycles.

I will spend real time on why the per unit clock advantage did not become a chip level clock advantage, because the answer is the most useful thing in the talk and it is not about arithmetic at all. At chip level the critical path leaves the multiplier entirely and lands in memory address generation. Cheaper arithmetic cannot raise your clock until the memory path is fixed. That is a finding about where time actually goes in a small accelerator, and it is the kind of thing you only learn by hardening the thing rather than modelling it.

## Timing of this talk

The next shuttle commitment is 4 November and tapeout is 7 December. Congress is three weeks after that. If the submission goes through, this talk happens with the design at the fab and the parts not yet in hand, which is a specific and slightly nerve racking place to be standing in front of a room.

## What you can take away

How to set up a fair comparison between numeric formats, and why almost all published ones are not fair. What an open source silicon flow can actually close today, with the signoff numbers. Why energy and speed can point in opposite directions on the same design. And a reproducible repository, MIT licensed, that you can run on a laptop.

No hardware background assumed. If you can read a table you can follow this talk.

---
<!-- NOTES, private to organisers -->
---

Everything is public and MIT licensed: github.com/aloshdenny/superfloat.gpu for the RTL, test suite and hardening configs, and github.com/aloshdenny/superfloat for the model side work. The IEEE FP16 and BF16 baseline RTL is in the same repository, so the comparison can be rerun by anyone with the open toolchain.

Every number in this proposal comes from a signoff report in a LibreLane run directory or from gate level simulation of a signed off netlist, with the report paths recorded alongside. Nothing here is projected or modelled except where I say so.

Scheduling context: the three Sky130 chips are signed off as of 1 October 2026. The next shuttle commitment is 4 November and tapeout is 7 December, both before Congress. Every measured result in the talk already exists, so the content holds regardless of the shuttle timing.

I can bring a demo: the full test suite runs from one pip install, one simulator package and one make command, and a failing assertion prints the hardware value next to the Python reference that disagreed with it.

On travel: I would be coming from Kerala, India, and I would need the limited travel support the CfP mentions. I will also need an official invitation letter for the German embassy, and I am aware of the six week processing warning, so an early decision would help a great deal.
