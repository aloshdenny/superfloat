This poster presents a year of sweeps asking one question: how few bits does a neural network actually need, and what determines the answer? It covers 878 archived training runs across four scaling tiers and eight follow up experiments, on convolutional networks and transformers, under both training aware and post training compression.

The central result is that almost every precision floor in the literature, including the ones I reported myself, is an artefact of where the scale sits rather than a property of the number format. Standard initialisation sets weight variance from fan in, so as layers widen the weights shrink under a fixed grid until an entire layer rounds to zero. Measured at initialisation, every layer with fan in of 144 or more is one hundred percent dead at three bits: the network is an exactly zero function and no gradient can revive it. Normalise per channel first and the convolutional floor moves from five or six bits down to two. The poster format suits this material because the interesting parts are the tables, and the alleyway format suits it because most of the good questions I have had about this work came from someone pointing at one cell and asking why. Alongside the main result: critical precision rises 0.29 bits per doubling of width but is completely flat across depth, weights saturate at three to four bits while activations need six, and post training damage is U shaped in training tokens rather than monotone.

## Takeaways

1. Before concluding that a model needs more bits, check whether its weights are sitting below the grid step. That diagnosis takes minutes and changes the answer by three or four bits.
2. Width and depth are not interchangeable for precision. Widening a network raises what it needs, deepening it does not.

## Which audiences will benefit most

Practitioners who compress or deploy models, people who run large experiment sweeps and care how they are designed, and anyone who likes arguing with a table of numbers in a corridor.

## Bio

Alosh Denny is an AI engineer and researcher from Kerala, and the creator of Superfloat, an open source number format, accelerator and compiler stack for running models without floating point. He heads AI at Aleddo Technologies and runs LLMOps at Intensors, has seven peer reviewed publications across IEEE and Springer, and has spoken at IndiaFOSS. Every number on this poster traces to a script and an archived result file that anyone can rerun without a GPU.

## Draft slides

Not applicable. The poster draft can be shared ahead of time with comments enabled.

## What I do not know yet

The right hand branch of the U shape. Post training damage falls through early training, bottoms out in the middle, then rises sharply at the end, and I can explain the left branch but not the right. See my separate problem card on this if it is accepted, or come and argue with the poster.

## Matchmaking tags

- I can help with: experience, technique, dataset, critique
- I need help with: ideas, critique
- I would like to meet: researchers, practitioners, domain experts

## Topic tags

model compression, scaling laws, reproducibility
