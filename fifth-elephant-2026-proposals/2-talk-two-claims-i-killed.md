I published two empirical claims about training neural networks at low precision. Both were plausible, both matched what I had observed, and both are now wrong in my own repository, with the retraction sitting directly underneath the original text. This session is the story of how each one died, because the interesting part is not the claims themselves but the shape of the mistakes, which I think are common and mostly invisible.

The first claim was that low precision training needs a smaller learning rate, since the representable grid is coarser. I had seen a model diverge at one step size and train at another. A dedicated sweep of sixty cells, six precisions crossed with ten learning rates spanning a 640 times range, found not one divergence at any precision, and put the accuracy optimum in the same place for every precision. The curves differ in height, not in position. What the sweep could not do is explain the original observation, because the two recipes differ in optimiser, schedule, dataset and model, and nothing isolates which. So I retired a claim without being able to explain the evidence that produced it, which is an uncomfortable place to stop and an honest one.

The second claim was that trained weights always fall inside a bounded range. That held beautifully on a vision model where zero of 53.6 million weights fell outside. It failed on a language model whose largest weight is 7.47. About one weight in ten thousand exceeds the bound, and those few hundred weights are the largest in the network, so clipping them destroys the model even at sixteen bits of precision, where the grid step is three in a hundred thousand and precision is plainly not the issue. A claim that is 99.99 percent true and load bearing in the remaining 0.01 percent is worse than one that is simply wrong, because nothing in the aggregate warns you.

## Takeaways

1. A result that reproduces on one model family is a hypothesis, not a finding. Both of my failures were generalisations from a single well behaved case.
2. Publishing the retraction next to the claim, rather than quietly editing, is cheap and makes the whole body of work more usable. I will show what that looks like in practice.

## Which audiences will benefit most

Anyone who publishes empirical results, internally or externally: applied researchers, data scientists, and people who maintain a body of measurements that others rely on. The domain is numerical precision but no background in it is required. The methods are ordinary sweeps.

## Bio

Alosh Denny is an AI engineer and researcher from Kerala, and the creator of Superfloat, an open source number format, accelerator and compiler stack for running models without floating point. He heads AI at Aleddo Technologies and runs LLMOps at Intensors, has seven peer reviewed publications across IEEE and Springer, and has spoken at IndiaFOSS. He now designs experiments mainly to attack his own published claims.

## Draft slides

Not yet. I will share a draft with comments enabled if the session is selected.

## What I do not know yet

The first claim is genuinely unresolved. I retired it because the controlled sweep found nothing, but the original divergence was real and I still cannot say what caused it. The two recipes differ along four axes at once. I would like help designing the smallest experiment that isolates which one, ideally from someone who has untangled a confound like this before.

## Matchmaking tags

- I can help with: experience, technique, critique
- I need help with: ideas, technique, critique
- I would like to meet: researchers, practitioners, domain experts

## Topic tags

reproducibility, evaluation, deep learning
