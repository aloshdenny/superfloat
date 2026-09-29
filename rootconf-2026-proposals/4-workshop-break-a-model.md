**One-line summary.** Participants quantise a small open weight model to eight bits, watch it break in the specific way that looks like a precision problem but is not, fix it, and then build the evaluation that would have caught the breakage.

## The problem

Most people meet quantisation as a flag in a serving config. It either works or it does not, and when it does not there is nothing to debug, because the model simply produces worse output. This workshop replaces that with hands on intuition. Participants implement the quantiser themselves in about forty lines, break their model deliberately, and diagnose it from the weights rather than from the output.

## Who this is for

Developers, MLOps, platform engineering. Anyone who has enabled a quantisation flag without knowing what it does.

**Level:** intermediate.

## Takeaways

1. The ability to inspect a checkpoint and predict whether a given quantisation will work, before running it.
2. An evaluation harness pattern that pairs a capability metric with a liveness metric, built during the session.

## What I will share

Code and implementation details. A debugging technique. A live build of both the quantiser and the evaluation. Open source tooling that participants leave with.

## Session plan

1. Implement the number format and a bounded straight through estimator. About forty lines.
2. Quantise a small open weight model and confirm it still works.
3. Break it on purpose, three ways: clip the outlier weights, misplace the scale, and quantise a layer that no normalisation feeds.
4. Diagnose each one from the weight distribution rather than from generated text.
5. Build an evaluation that pairs accuracy with a liveness metric, and watch it catch a failure that accuracy alone misses.

## Requirements for participants

A laptop with Python 3.10 or newer, PyTorch, NumPy and Matplotlib. No GPU, no cloud account and no dataset download. Everything runs on CPU with a small model. I will provide a setup script and a fallback notebook.

## My experience with this problem

Open source project. Experiment or prototype. Research or investigation.

## What failed, disappointed, or created unexpected problems

Every failure in the workshop is one I hit for real: the outlier weights that break a model at any bit width, the initialisation interaction that zeroes an entire layer at low precision, and the evaluation that rewards a model for going silent. The session is built around reproducing each of them deliberately in a few minutes.

## What I would do differently today

Teach range before precision. Everybody's intuition is that fewer bits means more error, and the failures that actually matter are about range and scale placement instead.

## Trade-offs

**Small model on a laptop against a realistic model on rented GPUs.** I chose small and local so that nobody spends the session fighting an environment, at the cost of some realism, which I cover with prepared results from the larger runs.

## How this helps other practitioners

A design approach for evaluating precision options. A debugging technique for a failure that is otherwise invisible. And a mistake to avoid that costs teams real production quality.

**Current state:** open source.

**Tags:** #workshop #quantization #inference #mlops #evaluation #handson
