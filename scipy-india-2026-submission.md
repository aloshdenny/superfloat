# SciPy India 2026 — submission answers
Talk · Deadline Oct 19, 2026 23:59 IST · Dec 19–20, IIT Madras (in person, mandatory)

---

## Page 1 — /info/

**Session type:** `Talk (30 minutes)`

**Proposal title:**
> How Many Bits Does a Neural Network Actually Need? Lessons From 878 Training Runs

Alternates:
- Throw Away the Exponent: An Empirical Study of Precision in Neural Networks
- Four Bits Is Plenty (For Weights): What 878 Runs Say About Quantization

**Track:** `AI, machine learning, and data-driven discovery`
(Second choice if you'd rather reach the numerics crowd: *Numerical, computational, and visualisation tools*. I'd stay with AI/ML — bigger room, and the reproducibility material lands either way.)

**Additional speakers:** leave empty

**Abstract** (public, goes in the programme — 189 words):

> A float spends most of its bits on dynamic range. A trained neural network barely uses it: weights are bounded and clustered near zero. So what happens if you delete the exponent field entirely and spend every bit after the sign on the significand?
>
> Superfloat (SFx) is an open-source format that does exactly that, making it plain signed fixed-point. This talk is about the measurement campaign behind it: 878 archived training runs asking how few bits a network actually needs, and why almost every "precision floor" reported in the literature — including the ones I reported myself — turns out to be an artefact of where you put the scale rather than a property of the format.
>
> You'll see why weights survive at 3–4 bits while activations need 6, why width raises the precision requirement but depth does not, why the worst checkpoint to quantize is the over-trained one you were about to ship, and how Qwen3 models from 1.7B to 8B drop to 8-bit fixed point with no retraining and keep their function-calling accuracy. Plus two claims of mine that did not survive replication, and what killed them.

**Description** (min 500 chars — this is 573 words):

> ### What this talk is
>
> An empirical study, presented honestly, of a question that sounds simple and isn't: how many bits does a neural network need?
>
> The vehicle is Superfloat (SFx), an MIT-licensed numeric format that removes the exponent field from a float and spends every remaining bit on the significand — mathematically a signed fixed-point number, Q1.(x−1). The premise is measurable rather than theoretical: across YOLO11x, ConvNeXt, V-JEPA 2 and several LLM families, essentially all trained weights already fall inside [−1, 1], so the exponent is paying for range that inference never uses. The interesting part is everything that goes wrong when you act on that.
>
> ### Outline (25 min + 5 min Q&A)
>
> 1. **The premise, measured** (3 min) — what an exponent buys, and weight-distribution data from CNNs, ViTs and LLMs showing how little of it is used at inference.
> 2. **The quantizer, in about 40 lines of PyTorch** (4 min) — the grid, a bounded straight-through estimator, and module surgery. Then the bug that matters: with Kaiming initialisation, every layer with fan_in ≥ 144 is 100% dead at 3 bits. The network is an exactly-zero function and no gradient can revive it. One per-channel normalisation, absorbed by the neighbouring norm so it never enters inference arithmetic, moves the CNN floor from 5–6 bits to 2.
> 3. **What 878 runs say** (6 min) — four scaling tiers plus eight follow-up experiments. Critical precision rises +0.29 bits per width doubling but is flat from ResNet-20 to ResNet-56. Weights saturate at 3–4 bits; activations need 6, and the cause is clipping, not grid coarseness. Post-training damage is U-shaped in training tokens — over-training a 160M model to 300B tokens costs 2.5 bits of deployable precision.
> 4. **Two claims of mine that died** (5 min) — I had published that low-precision training needs a smaller learning rate. A 60-cell sweep across a 640× range of step sizes found zero divergences at any precision. I had also published that trained weights always fit in range; one weight in ten thousand sitting above 1.0 destroys a model even at 16 bits, where precision is plainly not the issue. What the checks looked like, and why both corrections live in the README next to the original claims.
> 5. **The payoff** (4 min) — Qwen3 dense models, 1.7B to 8B, quantized to 8-bit fixed point post-training with no retraining, scored on the Berkeley Function Calling Leaderboard: 90.2% at 8B against 91.3% for the untouched model.
> 6. **Packaged to be re-run** (3 min) — one JSONL per experiment, ~6,500 archived runs' worth of records, and three scripts that regenerate every figure on a laptop with no GPU.
>
> ### Audience
>
> Anyone who trains or deploys models and has watched a quantized checkpoint collapse without knowing why; people who enjoy numerics; people interested in how to structure a large sweep so its conclusions survive.
>
> **Level:** intermediate. You should have trained a network in PyTorch. No hardware or VLSI background needed — the accelerator this format was designed for gets one slide.
>
> ### Takeaways
>
> - Where a scale lives matters more than how many bits you have.
> - Weights and activations want different widths; a datapath giving both the same width is misallocating.
> - Which checkpoint you quantize changes the answer more than the method does.
> - A worked example of designing a sweep that can falsify your own published claims — and of publishing the retraction next to the claim.

**Notes** (private, for organisers):

> Everything in the talk is open source and MIT-licensed: github.com/aloshdenny/superfloat for the format, quantizer, benchmark suite and archived results, with the accelerator RTL and the LLVM fork in sibling repositories. The stack is PyTorch, NumPy, Matplotlib, Ultralytics and Modal for the cloud sweeps.
>
> Every number in the talk traces to a script and an archived JSONL file in the public repo, and both figure scripts run without a GPU, so anything I show can be regenerated by an attendee on a laptop.
>
> I've presented this work at IndiaFOSS. I'm based in Kerala and will attend IIT Madras in person on both days. Happy to take a mock-presentation slot.
>
> If the committee would prefer it, the same material works as a 3-hour workshop — build the quantizer, break it with Kaiming init, fix it with per-channel scales, and reproduce two figures from the archive. Say the word and I'll submit that separately.

**Don't record this session:** leave **unchecked** (recording is good for you)

**Session image:** optional — see the image section below

**Resources** (add as links):
- Superfloat repository — https://github.com/aloshdenny/superfloat
- The scaling study (878 runs) — https://github.com/aloshdenny/superfloat/blob/main/SCALING_LAWS.md
- Tool-calling PTQ/QAT + BFCL results — https://github.com/aloshdenny/superfloat/blob/main/TOOL_USE_QAT.md
- Atreides accelerator RTL — https://github.com/aloshdenny/superfloat.gpu

---

## Page 2 — /questions/

**Audience level:** `Intermediate`

**Setup requirements for attendees:**
> None — this is a talk, not a workshop, and nothing needs to be installed to follow it. For anyone who wants to reproduce the results afterwards: Python 3.10+, PyTorch, NumPy and Matplotlib are enough. Every figure in the talk regenerates from the archived JSONL files in the repository on a laptop CPU; no GPU, cloud account or dataset download is required. Training new runs needs a GPU, but nothing in the talk depends on that.

**License agreement:** **check it.** Slides under CC BY 4.0; the code and results are already MIT.

**AI-generated content affirmation:** **read the note below before checking this.**

**Pronouns:** *only you can answer this — I won't guess.*

**First-time conference presenter:** `No` (IndiaFOSS) — change to Yes if that talk hasn't happened yet.
