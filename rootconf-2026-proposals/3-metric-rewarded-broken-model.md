**One-line summary.** A quantised model scored a perfect 100 on one category of a function calling benchmark while emitting tool calls on 1.2 percent of prompts, and the headline average made it look merely mediocre. This is a talk about validating a model change before it reaches production.

## The problem

Every platform team that serves models eventually has to approve a change to the model itself: a quantisation, a version bump, a different serving runtime. The change is evaluated with a benchmark, a number comes out, and somebody decides. The problem is that a benchmark number can be produced by a system that has stopped working, and the aggregate can hide it completely.

I hit this while measuring quantised models on a function calling benchmark. One category scores the model on correctly declining to call a function when no function applies. A model that has stopped emitting tool calls at all passes every case in that category for free. Mine scored 100 there, called on 1.2 percent of prompts overall, and would have been reported at about 28.7 percent if I had averaged the categories the way the benchmark invites you to.

That is not a quantisation problem. It is an evaluation design problem, and it generalises to any guardrail metric where the safe behaviour and the broken behaviour look identical from outside.

## Who this is for

Platform engineering, SRE, MLOps, developers shipping agents or tool using systems, and engineering leaders who sign off on model changes.

**Level:** intermediate.

## Takeaways

1. Every capability metric needs a liveness metric beside it. For tool calling that is the call rate. For your system it is whatever number distinguishes "correctly did nothing" from "did nothing".
2. A cheap screen can predict an expensive benchmark. Validation loss cost about one percent of a full benchmark run and located the failure cliff correctly every time, which changes how you budget evaluation.

## What I will share

Benchmarks and measurements. A failure mode with the full result table. The reasoning behind the screening strategy. Open source tooling, and a live walkthrough of the same trap in the raw results if the room wants it.

## My experience with this problem

Open source project. Research or investigation. Hard earned engineering lesson.

## What failed, disappointed, or created unexpected problems

**Reporting a single aggregate number.** It was the obvious thing to do and it would have published a broken model as a mediocre one.

**Trusting a category whose scoring rewards abstention.** I now treat any metric where doing nothing scores well as requiring a companion metric, without exception.

**Assuming a result from one model size transfers.** At 8B the six bit configuration holds at 88 percent. At 0.6B the same configuration collapses to 27. One number from one model would have produced confident and wrong guidance either way.

## What I would do differently today

Report every category separately with the call rate beside it. Never aggregate across categories with different failure semantics. Screen with a cheap proxy before paying for a benchmark sweep. And run the sweep across sizes, because the interesting behaviour is at the edges.

## Trade-offs

**Cheap proxy against real capability measurement.** Loss is fast and correlates well, but it cannot tell you the output is malformed JSON. The answer was to use it as a screen, not as evidence.

**Breadth against depth in evaluation.** I chose four model sizes crossed with three precisions over one model examined thoroughly. That caught the size dependence and cost me the deeper per category analysis I would otherwise have had.

## How this helps other practitioners

A specific class of mistake to avoid when approving model changes. A pattern for pairing capability metrics with liveness metrics. And an evaluation budget strategy that uses cheap signals to decide where to spend expensive ones.

**Current state:** open source.

**Tags:** #evaluation #observability #mlops #agents #inference #failurestory #modelserving
