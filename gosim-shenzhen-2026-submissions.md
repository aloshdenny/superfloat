# GOSIM Shenzhen 2026, four proposals
Oct 16 to 17, 2026, Shenzhen. Form fields: Talk Title, Talk Abstract (public, max 3000 characters),
Talk Details (internal only, not published), Track.
Abstracts below run 1,000 to 1,500 characters, so there is room if you want to add to any of them.

---

# Proposal 1

**Track:** `Agentic AI Summit 智能体 AI 峰会`

**Talk Title:**
> Agents That Do Not Need Floating Point: Tool Calling at 8 Bits

**Talk Abstract:**

> An agent is only as deployable as the smallest machine it runs on. So here is a question worth answering before you design your stack: how much of an agent survives when you take the floating point away?
>
> Superfloat is an open source number format that deletes the exponent field and keeps only sign and significand, which makes it plain fixed point. I quantised Qwen3 dense models from 0.6B to 8B into 8 bit fixed point after training, with no fine tuning and no calibration set, and scored them on the Berkeley Function Calling Leaderboard.
>
> Tool calling survives intact. At 8B the quantised model scores 90.2 against 91.3 for the untouched one. At 1.7B it is 88.6 against 88.7. Push to 6 bits and the picture splits: 8B still holds at 88.0, while 0.6B collapses to 27.3. The cliff is a property of model size, not of the format.
>
> I will also show why a single accuracy number hides what actually breaks. A quantised agent can fail by refusing to emit a call at all, and a leaderboard score without a call rate cannot tell you which failure you are looking at.
>
> Everything is MIT licensed, with all sixteen evaluation arms archived and rerunnable on one consumer GPU.

**Talk Details** (internal, not published):

> Outline, roughly 25 minutes plus questions.
>
> 1. Why the exponent is dead weight in a trained network: measured weight distributions across several model families.
> 2. The one thing that must be right. Under 0.02 percent of weights in a trained checkpoint sit outside the representable range, but they are the largest weights in the model, and clipping them destroys it even at 16 bits where precision is plainly not the issue. Give every matmul a scale and the model comes back completely.
> 3. The BFCL results across Qwen3 0.6B, 1.7B, 4B and 8B at 8 bits and 6 bits, with the call rate reported next to every accuracy number.
> 4. What this means for agent deployment: 8 bit fixed point needs no training budget at all, 6 bit needs either a large model or a training run.
> 5. Where it is going: the same format is being carried into an open source accelerator now being hardened for fabrication.
>
> Evidence: sixteen archived BFCL arms plus eighteen loss runs, all public as JSONL at github.com/aloshdenny/superfloat with the runner script beside them. Stack is PyTorch and Hugging Face.
>
> I have submitted several proposals to different GOSIM tracks. They come from one project but share no material, and I am happy for you to take any subset, including only one.

---

# Proposal 2

**Track:** `Agentic AI on Edge 边缘智能体 AI`

**Talk Title:**
> Weights Want 4 Bits, Activations Want 6: Designing an Edge Datapath

**Talk Abstract:**

> Almost every quantisation story stops at the weights. The weights sit on a small grid, the activations stay in bf16, and nothing saturates the result of a matmul. That is mixed precision, and it is not what edge silicon actually does.
>
> This talk is about what happens when you take the format all the way down, saturating every register the way a real fixed point accelerator must. The answer is blunt. Quantise weights only and a 360M model is untouched. Saturate every register with no scale and the same model goes from 2.67 to 13.42 nats, which is dead. Add a scale per tensor, still dead. Per token, and it comes back to 2.95.
>
> That surviving rung is block floating point, and it is the first one that costs silicon the format exists to remove. Tensor and channel scales fold into a neighbouring weight or norm and are free at inference. Per token scales do not.
>
> Underneath is a second result with direct hardware consequences. Weights saturate at 3 to 4 bits. Activations need 6, and the cause is clipping rather than grid coarseness, because activations are one sided and heavy tailed after ReLU. Any datapath that gives both operands the same width is misallocating its transistors.
>
> All measured, all open source, all reproducible.

**Talk Details** (internal, not published):

> This is the talk I would want to hear before choosing an edge inference target, and it is aimed at people building or selecting hardware rather than at people training models.
>
> Outline, roughly 25 minutes plus questions.
>
> 1. What weights only quantisation quietly assumes, and why an accelerator cannot honour it. Measured activation magnitudes: entering matmuls they reach thousands, and the residual stream reaches tens of thousands, against a representable bound of one.
> 2. The saturating datapath ladder: no scale, per tensor, per channel, per token, per token plus residual. Six arms, general and tool loss for each.
> 3. The asymmetry between the two operands, from a sweep of weight width crossed with activation width, and what it implies for how you spend area.
> 4. What this costs in gates, using a real fixed point processing element hardened on an open process as the worked example.
> 5. Practical guidance: what to ask a vendor, and what to measure yourself before committing.
>
> Evidence is public at github.com/aloshdenny/superfloat, in PURE_SF.md for the datapath ladder and SCALING_LAWS.md for the operand asymmetry, with every run archived as JSONL.
>
> I have submitted several proposals to different GOSIM tracks. One project, no shared material, and any subset is fine.

---

# Proposal 3

**Track:** `Open Source Models & Infra 开源模型与基础设施`

**Talk Title:**
> 878 Training Runs on How Few Bits a Model Really Needs

**Talk Abstract:**

> Everyone reports a precision floor. Below 6 bits training falls over, below 8 bits post training quantisation falls over, below 4 bits everything dies. I reported those floors too. Then I ran 878 training runs to find out where they come from, and almost none of them are properties of the number format.
>
> They come from where you put the scale. Standard initialisation sets weight variance from fan in, so as layers widen the weights shrink under a fixed grid until every weight in the layer rounds to zero. Measured at initialisation, every layer with fan in of 144 or more is one hundred percent dead at 3 bits. The network is an exactly zero function and no gradient can revive it. Normalise per channel first, let the neighbouring norm absorb the scale so it never enters inference arithmetic, and the convolutional floor moves from 5 or 6 bits to 2.
>
> Also in the sweeps: width raises the precision requirement by 0.29 bits per doubling while depth does not move it at all, and post training damage is U shaped in training tokens, so the worst checkpoint to quantise is the over trained one you were about to ship.
>
> Plus two claims of mine that did not survive replication, and what killed them.

**Talk Details** (internal, not published):

> The point of this talk is methodology as much as result. It is about how to build a sweep that can falsify your own published claims, using quantisation as the worked example, and it is aimed at people who maintain models and infrastructure rather than at hardware specialists.
>
> Outline, roughly 25 minutes plus questions.
>
> 1. The floors as commonly reported, including in my own earlier write up.
> 2. The dead weight mechanism, with the per layer profile at initialisation, and the fix that costs nothing at inference.
> 3. Four scaling tiers and eight follow up experiments: critical precision against width, depth, parameter count and training tokens.
> 4. Two retired claims. I had published that low precision training needs a smaller learning rate. Sixty cells across a 640 times range of step sizes gave zero divergences at any precision. I had also published that trained weights always fit in range. One weight in ten thousand above the bound destroys a model even at 16 bits.
> 5. How the archive is built so that any of it can be rechecked: one JSONL per experiment, and three scripts that regenerate every figure on a laptop with no GPU.
>
> All public and MIT licensed at github.com/aloshdenny/superfloat. Stack is PyTorch, NumPy, Matplotlib, with Modal for the cloud sweeps.
>
> I have submitted several proposals to different GOSIM tracks, from one project but with no shared material.

---

# Proposal 4

**Track:** `Agentic Device 智能体设备`

**Talk Title:**
> Running GPT-2 on a Chip That Does Not Exist Yet

**Talk Abstract:**

> You cannot put a print statement inside silicon. So how do you find out what a processor does before anybody fabricates it?
>
> Atreides is an open source AI inference accelerator: a fixed point systolic array with no exponent hardware in the datapath, designed and hardened entirely with open tools. There are 4,811 lines of Verilog in it, and roughly 12,000 lines of Python. The Python is where the interesting work happens. cocotb turns a Verilog simulator into an ordinary Python object, so the reference model, the assembler, the memory images and the numerical error analysis are all code you already know how to write.
>
> I will run a GPT-2 layer stack instruction by instruction on a chip that exists only as a simulation, with a prefill phase and a decoding phase that uses a KV cache, and show what a cycle accurate model tells you that no estimate can.
>
> Including the optimisation that lost. Skipping multiplications by zero makes this machine slower at every sparsity level I measured, and pruning individual weights gives wrong answers, because lanes in the same warp disagree at the branch. When a multiply costs one cycle and carries no exponent hardware, control flow costs more than the arithmetic it avoids.
>
> Then where the current 256 multiplier version stands on its way to fabrication.

**Talk Details** (internal, not published):

> This is a hardware talk written for software people. No Verilog knowledge is assumed, and the through line is that open silicon tooling is now installable and testable by anyone who can write Python.
>
> Outline, roughly 25 minutes plus questions.
>
> 1. You cannot printf inside a chip. What cocotb changes: await a clock edge, read a register as an integer, compare against a NumPy model.
> 2. The design in one slide. Two cores, a systolic array, a fourteen instruction 16 bit ISA, a small on die scratchpad, hardened on Sky130 with zero DRC, LVS and antenna violations.
> 3. The Python that surrounds it: 126 cocotb tests, 308 archived simulation traces, an assembler that is just functions returning integers, and a numerical report scoring hardware output against IEEE fp32 in ULPs.
> 4. GPT-2 tiled to fit one mebibyte of data memory and executed as real instructions. One tile costs 70,327 cycles, and watching where they go is the entire value of the exercise.
> 5. The sparsity result, with cycle counts, plus the SIMT divergence that makes unstructured pruning incorrect rather than merely slow.
> 6. What the layout tool said: at signoff the worst path in the chip is memory address generation, not the multiplier. Cheaper arithmetic cannot raise the clock until the memory path is fixed.
> 7. Current work. A much larger version, four cores each with an 8 by 8 array for 256 multiply accumulate units, is being hardened now for a ChipFoundry OpenFrame shuttle.
>
> Public and MIT licensed at github.com/aloshdenny/superfloat.gpu. Toolchain is cocotb, pytest, Icarus Verilog, NumPy, then Yosys, OpenROAD and LibreLane. No vendor licence anywhere in the flow.
>
> Live demo possible: the test suite runs from one pip install, one simulator package and one make command, and a failing assertion prints the hardware value next to the Python reference that disagreed with it.
>
> I have submitted several proposals to different GOSIM tracks, from one project but with no shared material.
