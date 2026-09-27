# SciPy India 2026, second submission (HPC track)
Talk, 30 minutes. Deadline Oct 19, 2026 23:59 IST. Dec 19 to 20, IIT Madras, in person.

---

## Page 1, /info/

**Session type:** `Talk (30 minutes)`

**Proposal title:**
> Running GPT-2 on a Chip That Does Not Exist Yet

Alternates:
- A Chip With More Python Than Verilog
- Your Testbench Is a Coroutine: Verifying an AI Accelerator in Python
- 4,811 Lines of Verilog, 12,000 Lines of Python

**Track:** `High-performance computing (HPC)`

**Additional speakers:** leave empty

**Abstract** (public, appears in the programme):

> You cannot put a print statement inside silicon. So how do you find out what a processor does before anyone fabricates it?
>
> Atreides is a small open source accelerator for AI inference: two cores, a systolic array of eight multiply accumulate units, a fourteen instruction ISA, hardened on the Sky130 process. It has 4,811 lines of Verilog in it. It also has roughly 12,000 lines of Python, and the Python is where all the interesting work happens.
>
> This talk is about using cocotb to drive a Verilog simulator from Python, so that the reference model, the assembler, the memory images, the numerical error analysis and the performance study are all ordinary Python that you already know how to write. I will show a full GPT-2 layer stack, tiled and executed instruction by instruction on a chip that exists only as a simulation, and what a cycle accurate model tells you that an analytical estimate never will.
>
> Including the result I did not want: skipping multiplications by zero makes this machine slower, not faster, at every sparsity level I measured. Cheap arithmetic changes which optimisations are worth doing.

**Description** (full text, this is what reviewers read):

> ### What this talk is
>
> A practical tour of Python as a hardware verification language, using a real open source chip as the specimen.
>
> Atreides is an AI inference accelerator built around a fixed point number format with no exponent field. The silicon details matter less than the method: it is a processor small enough to understand completely, with a test suite of 126 cocotb tests and 308 archived simulation traces, and everything above the Verilog is Python that a SciPy audience already writes fluently.
>
> ### Outline, 25 minutes plus 5 for questions
>
> 1. **You cannot printf inside a chip** (3 min). How hardware verification normally works, and what changes when the testbench is a Python coroutine: `await ClockCycles(dut.clk, 1)`, then read a register as an integer and compare it against a model you wrote in NumPy.
>
> 2. **The design under test, in one slide** (3 min). Two cores, a 2x2 systolic array, a fourteen instruction 16 bit ISA, 128 bytes of on die scratchpad, a 50 MHz target on Sky130. Enough context to read the numbers that follow.
>
> 3. **The Python that surrounds it** (6 min). A reference implementation of the arithmetic in about 200 lines, an assembler that is just functions returning integers, memory image builders, and a numerical report that scores hardware output against IEEE fp32 in ULPs. On a 4x4 matrix multiply the hardware lands at 0.88 mean ULP error with a worst case of 3.01, and I can say that because the comparison is three lines of Python, not a waveform viewer and a lot of squinting.
>
> 4. **A real workload on imaginary hardware** (6 min). GPT-2 attention and feedforward layers, tiled by hand to fit one mebibyte of data memory, with a prefill phase and a decoding phase that uses a KV cache, executed as actual instructions in the simulator. One tile costs 70,327 cycles. Watching where they go is the entire value of the exercise.
>
> 5. **The optimisation that lost** (5 min). Dense 4x4 matrix multiply: 2,406 cycles. Branch around the zero weights: 2,730 cycles at zero sparsity, falling only to 2,610 at 75 percent sparsity. Never once a win. Worse, pruning individual elements gives wrong answers, because lanes in the same warp disagree at the branch and the machine diverges. Structured pruning by row is correct and still slower. When a multiply takes one cycle and carries no exponent hardware, control flow costs more than the arithmetic it avoids.
>
> 6. **What the layout tool said** (3 min). At signoff the worst timing path in the whole chip is memory address generation, not the multiplier. Making the arithmetic cheaper cannot raise the clock until the memory path is fixed. This is the most useful thing I learned all year, and I learned it from a report file.
>
> 7. **Running it yourself** (2 min). `pip install cocotb`, one apt package for the simulator, one make command. No licences, no vendor account, no hardware.
>
> ### Audience
>
> Python programmers who are curious about what happens below the software they optimise. People who write simulations and want to see a simulator used as an experimental instrument. Anyone who has wondered how open source silicon is actually tested. No Verilog knowledge is assumed and no hardware background is needed.
>
> **Level:** intermediate. Comfortable reading Python with async functions. Everything about digital design is explained as it appears.
>
> ### Takeaways
>
> - cocotb turns a hardware simulator into a normal Python object you can poke, so verification becomes a data analysis problem.
> - A cycle accurate model is a measurement instrument, and it answers questions that no spreadsheet estimate can.
> - Optimisations transfer badly across cost models. Skipping work is only worth it when the work is expensive.
> - Open source silicon tooling is now genuinely installable, and you can try all of this on a laptop this evening.

**Notes** (private, organisers only):

> The project is MIT licensed and public at github.com/aloshdenny/superfloat.gpu, with the number format and its model side benchmarks at github.com/aloshdenny/superfloat. The stack is cocotb, pytest, Icarus Verilog, NumPy, and LibreLane with OpenROAD and Yosys for the physical flow, all of it open source and installable without a vendor licence.
>
> Every figure quoted in the proposal comes from a file in the repository: the cycle counts from the archived JSON results of the sparsity study, the ULP numbers from the precision test logs, the timing and area figures from LibreLane run directories with the report paths written down next to them.
>
> I am based in Kerala and will attend both days in person. Happy to take a mock presentation slot.
>
> Disclosure: I have also submitted a talk on the numerical side of this work to the AI and machine learning track. The two share a project but not a single slide, and I am glad to present either one if you would rather not schedule both. If you can only take one, I would suggest this one for the HPC track since it is the more unusual material.

**Don't record this session:** leave unchecked

**Session image:** see the image note below

**Resources:**
- Atreides accelerator, RTL and test suite: https://github.com/aloshdenny/superfloat.gpu
- Superfloat number format and benchmarks: https://github.com/aloshdenny/superfloat
- cocotb: https://www.cocotb.org

---

## Page 2, /questions/

**Audience level:** `Intermediate`

**Setup requirements for attendees:**
> None. This is a talk, not a workshop, and nothing needs to be installed to follow it. For anyone who wants to reproduce the work afterwards, the full toolchain is free and open source: Python 3.8 or newer with cocotb and pytest, Icarus Verilog for simulation, and optionally GTKWave for waveforms. It runs on a laptop, with no GPU, no cloud account and no vendor licence. Rerunning the physical design flow additionally needs LibreLane and the Sky130 PDK, which is a larger install but still free.

**License agreement:** check it. Slides under CC BY 4.0. The code, the RTL and the archived results are already MIT.

**AI-generated content affirmation:** rewrite the text in your own voice first, then check it. See the note in my reply.

**Pronouns:** only you can answer this.

**First-time conference presenter:** `No` (IndiaFOSS).
