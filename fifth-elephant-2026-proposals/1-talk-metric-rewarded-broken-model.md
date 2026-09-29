# Form fields (do not paste this part)

**Title:** It Scored 100 Percent Because It Had Stopped Answering

**I am submitting:** To speak

**I have a submission for:** A talk, 30-40 mins

**Submission type:** Talk/session proposal, 30-40 mins

---
<!-- PASTE EVERYTHING BELOW THIS LINE INTO THE MARKDOWN BODY -->
---

## Describe your session

A model I had compressed scored a perfect 100 on one category of a function calling benchmark. It looked like the best result in the table. It was produced by a model that had stopped emitting function calls almost entirely, on 1.2 percent of prompts, and the category in question scores a model on correctly declining to call a function when none applies. A system that has gone silent passes every one of those cases for free. Had I averaged the three categories the way the benchmark invites you to, I would have published that broken model as a mediocre one, at about 28.7 percent, and nobody reading the number could have told the difference.

This session is about that class of metric, where the safe behaviour and the broken behaviour are indistinguishable from outside the system, and about what it takes to notice. I will show the full result table, including the arm where the failure is obvious once the call rate sits beside the accuracy and invisible when it does not. Then the part I find more useful: a cheap proxy, validation loss, cost about one percent of a benchmark sweep and predicted both the safe configurations and the failure cliff correctly in every case I checked. That changes how you spend an evaluation budget, and it generalises well beyond compression. The same shape appears in any abstention metric, any guardrail that scores a refusal as a success, and any aggregate built from categories with different failure semantics.

## Takeaways

1. Every capability metric needs a liveness metric beside it. For tool calling that is the call rate. For your problem it is whatever number separates "correctly did nothing" from "did nothing".
2. A cheap proxy can decide where to spend expensive evaluation. Mine cost one percent of a benchmark run and located the cliff every time.

## Which audiences will benefit most

People who design or interpret evaluations: applied researchers, ML practitioners shipping models, and anyone who has to approve a model change on the strength of a benchmark number. No compression background is needed. The failure is about metric design, and the compression is only the vehicle that produced it.

## Bio

Alosh Denny is an AI engineer and researcher from Kerala, and the creator of Superfloat, an open source number format, accelerator and compiler stack for running models without floating point. He heads AI at Aleddo Technologies and runs LLMOps at Intensors, has seven peer reviewed publications across IEEE and Springer, and has spoken at IndiaFOSS. He spends most of his measurement time trying to falsify his own published results, with an uncomfortable success rate.

## Draft slides

Not yet. I will share a draft with comments enabled if the session is selected.

## What I do not know yet

Two things I would like critique on. First, whether the liveness metric idea has a cleaner general formulation than the one I use, which is essentially "instrument the denominator". Second, I only have this failure characterised on function calling. I suspect retrieval, classification with an abstain class, and moderation systems all have the same structure, and I would like to hear from people who have seen it in those settings before I claim it.

## Matchmaking tags

- I can help with: experience, technique, critique
- I need help with: experience, critique
- I would like to meet: practitioners, researchers, tool builders

## Topic tags

evaluation, benchmarking, model compression
