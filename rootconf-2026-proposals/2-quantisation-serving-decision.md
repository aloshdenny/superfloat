**One-line summary.** Open weight models quantise to eight bit fixed point after training, with no retraining and no calibration set, and keep their tool calling intact, provided you get one detail right that silently destroys the model if you miss it.

## The problem

Teams serving open weight models reach for quantisation to cut memory and cost, then discover the outcome is unpredictable. Sometimes the model is fine. Sometimes it is subtly worse. Sometimes it is wrecked, and the failure does not look like a precision problem.

I spent a year measuring where the real constraint is, using a number format that removes the exponent field entirely and stores only sign and significand. That is the most aggressive version of the question: if trained weights are bounded and cluster near zero, how much of a float is a model actually using?

The answer for serving is good. Qwen3 dense models from 1.7B to 8B drop to eight bits post training, with no fine tuning and no calibration data, and hold function calling accuracy within about a point of the original. But the failure mode on the other side is sharp, and it is not the one people expect.

## Who this is for

Platform engineering, infrastructure, MLOps, developers running inference in production, and engineering leaders sizing hardware.

**Level:** intermediate.

## Takeaways

1. Roughly one weight in ten thousand in a trained checkpoint sits outside the representable range, and those are the largest weights in the model. Quantising them without a per matmul scale destroys the model even at sixteen bits, where precision is plainly not the problem. Check the checkpoint, do not assume.
2. Which checkpoint you quantise matters more than which method you use. Post training damage is not monotone in training tokens, and the worst checkpoint to quantise is often the heavily trained one you were about to ship.

## What I will share

Benchmarks and measurements across model families and sizes. Failure modes with the numbers attached. The design decision about where a scale lives and why that choice is free at inference. The trade-off between post training quantisation and training aware approaches. Open source tooling, MIT licensed and reproducible.

## My experience with this problem

Open source project. Research or investigation. Experiment or prototype.

## What failed, disappointed, or created unexpected problems

**The big one.** Two projection matrices in a transformer block are not fed by a normalisation layer, so an early version of my pipeline quantised them as they stood. General validation loss went from 2.53 to 7.73. At sixteen bits. The grid step at sixteen bits is three in a hundred thousand, so precision was never the issue: a few hundred outlier weights were being clipped, and they were the largest weights in the network. Giving every matmul its own scale recovered the model completely.

**My own published claim.** I had written that trained weights always fall inside the representable range. It held on a vision model where zero of 53.6 million weights fell outside. It failed on a language model whose largest weight is 7.47. The correction now sits in the repository next to the original claim.

**Six bit quantisation.** It looks like a small step down from eight. At 8B it costs about three points. At 0.6B the model stops working entirely.

## What I would do differently today

Measure the weight range of the actual checkpoint before choosing a format, rather than reasoning from what trained networks are supposed to look like. Give every matrix multiply a scale by default, since tensor and channel scales fold into a neighbouring weight or norm and cost nothing at inference. And treat a small model as a different problem from a large one, not a scaled down version of it.

## Trade-offs

**Post training quantisation against training aware quantisation.** At eight bits no training budget is needed at all, which removes the question entirely. At six bits you need either a large model or a training run, and that is a real cost decision.

**Weights only against a fully quantised datapath.** Weights only is what most serving stacks do and is nearly free. Saturating every intermediate value, which is what fixed point hardware actually requires, breaks the model unless you add a per token scale, and that scale costs silicon.

**Loss as a screen against running the full benchmark.** Loss costs about a hundredth of a benchmark sweep and correctly predicted both the safe configuration and the cliff in every case I checked.

## How this helps other practitioners

A way to evaluate competing quantisation options before committing hardware. A specific mistake to avoid that is invisible in the usual metrics. And a rule of thumb about checkpoint selection that applies whatever method you use.

**Current state:** open source.

**Tags:** #inference #mlops #quantization #modelserving #edge #benchmarks #opensource
