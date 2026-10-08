# Comment to post on the proposal

Change log, after the 7 Oct dry run.

**Proposal**

- Merged the two proposals into this one. The quantisation research and the infrastructure failure are now a single talk, and the research leads rather than trailing.
- New flow: why the research exists, then the platforms and the training setup, then the failure, then what I changed, then how you would know a compressed model still works, then who owns the decision.
- Retitled to match, since the old title only covered one half.

**Deck**

- Modal and RunPod introduced up front, with the model, the architecture and the stack.
- Debug journey expanded: the two wrong guesses, what killed each one, and the token arithmetic that actually found it. Including that none of it was automated, just container logs and a chat window, with no dashboard, trace or alert.
- Evaluation tooling added: Langfuse and LangGraph missing a missing tool call, then the OpenRouter and Promptfoo ensemble with explicit metrics pulled from the logs.
- Failures reframed as common platform defaults rather than personal mistakes.
- Text replaced with diagrams wherever the slide was describing a sequence or a breakdown: a timeline of the twenty four hours, the token arithmetic, the race condition, the fix, and an elimination ladder for the debugging. About a third of the deck is diagrams now.
- Analogies added for the general audience: quantisation as rounding, and the corpus arriving in boxes rather than all at once.
- All currency figures removed. Costs are expressed as ratios and GPU hours.
- Timed to 20 to 23 minutes with 5 kept for questions. The dry run ran 18.

Slides linked on the proposal.
