# Form fields (do not paste this part)

**Session type:** Talk (40 minutes)
**Track:** Science
**Language:** English
**Title:** How Few Bits Does A Neural Network Need? 878 Experiments, And Two Of My Own Claims Retracted

**Resources to attach:**
- https://github.com/aloshdenny/superfloat
- https://github.com/aloshdenny/superfloat/blob/main/SCALING_LAWS.md

---
<!-- ABSTRACT -->
---

Everyone who works on shrinking neural networks reports a precision floor. Below six bits training falls apart, below eight bits post training compression falls apart, below four bits everything dies. I reported those floors too, in my own paper.

Then I ran 878 training runs to find out where they come from, and almost none of them are properties of the number format. They are artefacts of where you put the scale.

The mechanism is embarrassing once you see it. Standard initialisation sets weight variance from the width of the layer, so as layers get wider the weights get smaller while the grid stays fixed, until every weight in the layer rounds to zero. Measured at initialisation, every layer above a certain width is one hundred percent dead at three bits. The network is an exactly zero function and no gradient can bring it back. Fix where the scale lives and the floor for convolutional networks drops from six bits to two.

Along the way the sweeps killed two claims I had published, including one I still cannot explain. This is a talk about measuring carefully enough to prove yourself wrong.

---
<!-- DESCRIPTION -->
---

## The question

A floating point number is built for general scientific computing. A trained neural network is not general: its weights are bounded and clustered tightly around zero, which means the exponent field, the part that buys enormous dynamic range, is mostly unused at inference time.

That observation is old and uncontroversial. What is not settled is how far you can push it. How few bits does a network actually need, and what determines the answer?

I spent a year on this with a format that takes the premise to its limit: one sign bit, every remaining bit on the significand, no exponent field at all, which makes it plain signed fixed point. Then 878 archived training runs across four scaling tiers and eight follow up experiments, on convolutional networks and transformers, under both training aware and post training compression.

## The central result

Precision floors are mostly not about precision.

Standard initialisation sets the standard deviation of a layer's weights from its fan in, so it shrinks as layers widen. The quantisation grid does not shrink with it. Past a certain width every weight in the layer sits inside the first grid bin and rounds to zero. Measured at initialisation under a naive scheme, every layer with fan in of 144 or more is one hundred percent dead at three bits. The forward pass returns exactly zero, the gradient is exactly zero, and the network can never recover. It looks like a precision failure. It is a scale placement failure.

Normalise per channel before quantising, and let the neighbouring normalisation layer absorb that scale so it never enters the inference arithmetic, and the dependence on width disappears entirely. The convolutional floor moves from five or six bits to two. Two bits is exactly ternary, the BitNet operating point, reached without any ternary specific training recipe.

## What else the sweeps found

- **Width raises the precision requirement, depth does not.** Critical precision rises 0.29 bits per doubling of width, against 0.50 predicted by the initialisation argument. It is flat from ResNet-20 to ResNet-56, because depth changes no layer's fan in. Deeper networks actually pay a smaller penalty.
- **The two operands are not symmetric.** Weights saturate at three to four bits. Activations need six, and the cause is clipping rather than grid coarseness, because activations are one sided and heavy tailed after ReLU. Any hardware datapath giving both operands the same width is misallocating its transistors.
- **Damage from post training compression is U shaped in training tokens, not monotone.** Both ends of a training run are fragile and the middle is not. The worst checkpoint to compress is the heavily trained one you were about to ship. Over training a 160 million parameter model to 300 billion tokens costs two and a half bits of deployable precision. I can explain the left half of that U and not the right half.
- **A precision inversion.** Under the corrected scheme, coarser formats sometimes beat finer ones, replicated across architecture, hardware and token budget. That is not supposed to happen and I will show you the data.

## Two claims I retracted

The part I find most worth talking about.

**One:** I had published that low precision training needs a smaller learning rate, because the grid is coarser and large steps overshoot. A sweep of sixty cells, six precisions crossed with ten learning rates spanning a 640 times range, found not one divergence at any precision, and put the optimum in the same place for every precision. The curves differ in height, not position. Claim retired. And I still cannot explain the divergence I originally observed, because the two recipes differ in optimiser, schedule, dataset and model all at once and nothing isolates which. That is an unresolved debt and I would rather say so than invent a story.

**Two:** I had published that trained weights always fall within the representable range. Zero of 53.6 million weights in a vision model fell outside. Then a language model whose largest weight is 7.47, with about one weight in ten thousand outside the bound. Those few hundred weights are the largest in the network, and clipping them destroys the model even at sixteen bits, where the grid step is three in a hundred thousand and precision is plainly not the issue. A claim that is 99.99 percent true and load bearing in the remaining 0.01 percent is worse than one that is plainly wrong, because no average warns you.

Both retractions now sit in the repository directly underneath the text they retract.

## Why this is a Science talk rather than an engineering one

The content is a measurement campaign and its epistemics: how to design a sweep that can falsify your own published claim, what a result that replicates across four axes is worth, and what to do when the data kills something you have already put your name to. The quantisation is the subject matter. The method is the point.

Everything is MIT licensed, archived as one structured file per experiment, and every figure regenerates on a laptop with no GPU.

---
<!-- NOTES, private to organisers -->
---

All public at github.com/aloshdenny/superfloat. Every claim in this proposal traces to a named script, an archived results file and a figure in that repository, and the mapping is written down in the README. Both figure scripts run on CPU, so anything I show can be reproduced by an attendee during the talk.

There is a manuscript of this work under preparation for a journal. The talk would cover the same results; nothing here is embargoed and the repository has been public throughout.

On travel: I would be coming from Kerala, India, and I would need the limited travel support the CfP mentions, plus an official invitation letter for the German embassy. The six week visa warning means an early decision would help a lot.
