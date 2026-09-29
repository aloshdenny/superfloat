# DevConf.IN 2027, two proposals
Feb 26 to 27, 2027, MIT WPU Pune. CfP closes Nov 8, 2026. In person, no expenses covered, free registration.
Form limits: title 100 characters, abstract 1000 characters, no names or job titles in the abstract.

---

# Proposal 1 (AI)

**Session type:** `Talk (45 minutes)`  (30 talk plus 15 Q and A)

**Track:** `Enterprise AI and MLOps`

**Proposal title:**
> Your LLM Does Not Need Floating Point: 8 Bit Inference With No Retraining

Alternates:
- Delete The Exponent: Serving Open Weight Models In Plain Fixed Point
- The Worst Checkpoint To Quantise Is The One You Were About To Ship

**Abstract:**

> Every open weight model you serve carries an exponent field it barely uses. Trained weights are bounded and cluster near zero, so most of a float's dynamic range is dead silicon.
>
> This talk shows what happens when you delete the exponent entirely and run inference in plain fixed point. Qwen3 models from 1.7B to 8B drop to 8 bits after training, with no fine tuning and no calibration set, and hold function calling accuracy within about a point of bf16 on the Berkeley Function Calling Leaderboard.
>
> Then the part that should change how you run a model pipeline. The worst checkpoint to quantise is the over trained one you were about to ship. Activations and weights want different widths, so a serving stack that gives them the same one is wasting bits. And a single weight in ten thousand can destroy a model even at 16 bits, where precision is plainly not the issue.
>
> Backed by 878 archived training runs. Open source, and reproducible on a laptop with no GPU.

**Notes** (private):

> All of it is MIT licensed and public at github.com/aloshdenny/superfloat: the format, the quantiser, the benchmark suite and every archived result. The stack is PyTorch, NumPy and Matplotlib, with Modal for the cloud sweeps.
>
> Every number quoted traces to a script and an archived JSONL file in that repository, and the figure scripts run without a GPU, so an attendee can regenerate anything I show on their own laptop during the talk.
>
> What is new here rather than retrospective: the tool calling results across Qwen3 0.6B to 8B are recent, and the same format is now being carried into silicon on an open process, which gives the talk a live second half rather than a settled story.
>
> I will attend both days in person at Pune. I have also submitted a hardware proposal to the Open Track. The two share a project but not a slide, and if you can only take one I am happy either way.

---

# Proposal 2 (hardware)

**Session type:** `Talk (45 minutes)`

**Track:** `Open Track`
(Second choice if Open Track is crowded: `Modern Infrastructure, Edge & Virtualization`.)

**Proposal title:**
> Running GPT-2 On A Chip That Does Not Exist Yet

Alternates:
- A Chip With More Python Than Verilog: Open Source Silicon, Verified In cocotb
- Taping Out An AI Accelerator With Nothing But Open Source Tools

**Abstract:**

> You cannot put a print statement inside silicon. So how do you find out what a processor does before anybody fabricates it?
>
> Atreides is an open source AI inference accelerator: a fixed point systolic array with no exponent hardware in the datapath, designed and hardened entirely with open tools. There are 4,811 lines of Verilog in it, and roughly 12,000 lines of Python. The Python is where the interesting work happens. cocotb turns a Verilog simulator into an ordinary Python object, so the reference model, the assembler, the memory images and the error analysis are all code you already know how to write.
>
> I will run a GPT-2 layer stack instruction by instruction on a chip that exists only as a simulation, show the optimisation that lost (skipping multiplications by zero makes this machine slower at every sparsity level, and pruning single weights gives wrong answers), and report where the current 256 multiplier version stands on its way to fabrication.

**Notes** (private):

> Public and MIT licensed at github.com/aloshdenny/superfloat.gpu, with the numerical side at github.com/aloshdenny/superfloat. Toolchain: cocotb, pytest, Icarus Verilog, NumPy, then Yosys, OpenROAD and LibreLane on the Sky130 process. No vendor licence anywhere in the flow.
>
> Current state, since the CfP asks for new work rather than retrospectives: the first version closed timing and was submitted through Tiny Tapeout with zero DRC, LVS and antenna violations. A much larger version, four cores each with an 8 by 8 systolic array for 256 multiply accumulate units, is being hardened right now for a ChipFoundry OpenFrame shuttle. That work will be either finished or very close by February, so the talk ends on a result rather than a plan.
>
> The measured claims in the abstract come from files in the repository: cycle counts from the archived JSON of the sparsity study, timing and area from LibreLane run directories with the report paths recorded beside them.
>
> Live demo: I can run the test suite on stage. It is one pip install, one package for the simulator and one make command, and a failing assertion prints a Q1.15 value next to the Python reference that disagreed with it.
>
> I will attend both days in person at Pune. I have also submitted a proposal to the Enterprise AI and MLOps track on the numerical side of the same project.
