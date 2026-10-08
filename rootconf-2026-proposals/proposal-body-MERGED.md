**One-line summary.** I spent a year answering one question, how few bits a language model actually needs, and the infrastructure I answered it on failed in five different ways that had nothing to do with machine learning. This talk is both halves: the research, the day a container spent training on a tenth of its corpus while every dashboard stayed green, and the question underneath both, which is who owns a decision that crosses four teams.

## Why any of this exists

I build an open source numeric format and an accelerator to run it on. Before anyone commits silicon, somebody has to answer one question: how few bits does a model actually need?

Quantisation is the answer to it. Storing every number in a model with fewer digits, so it uses less memory, less power and generates tokens faster. It is rounding. The only interesting question is what the model forgets.

Four people have to agree before it happens. **Cost**, which GPU tier you are willing to pay for. **Platform**, RunPod or Modal or the cloud you already have. **Quality**, how far you compress before it stops being smart. **Hardware**, Nvidia or the faster new silicon with nobody to ask when it breaks. In most organisations you can settle two of those, maybe three. Never four. That is the thread that runs through the whole talk.

From roughly 878 archived training runs the answer is eight bits. Qwen3 8B scores 90.2 on the Berkeley Function Calling Leaderboard after post-training quantisation to eight bit fixed point, with no retraining and no calibration set, against 91.3 untouched. Drop to six bits and a 4B model falls to 63. At 0.6B it stops being a model at all.

## Answering it meant a lot of training runs, on other people's machines

878 runs across four platforms in a year: a GPU under my desk, a shared lab machine, RunPod and Modal. None of it is a production service. All of it has the operational character of one, because a job that dies at hour twenty is expensive and a job that succeeds while computing the wrong thing is worse.

The run that cost the most was Llama 3.2 1B, from scratch, quantisation aware, on one H100, with a corpus that arrives in shards rather than all at once. It finished its first leg with no errors.

## The debug journey

By hour six the loss was climbing. That is normal early on, so I left it. By hour twenty four it still had not come down. Train loss 0.95, validation loss 5.6.

Then I guessed, in order.

**Guess one: the eval split is noisy.** Killed by the fact that noise does not climb in a straight line for eighteen hours. This guess cost the most, because it let the run keep going.

**Guess two: it is overfitting.** Killed by the fact that a 1B model does not overfit a corpus that size in a single day.

**Guess three, which should have been first: count the tokens it actually saw.** Steps, times sequence length, times batch and accumulation. Four numbers already sitting in the log that I had never multiplied. Ninety six million tokens. Ten percent of the corpus, thirty times over. That is not pretraining, it is memorisation, which is exactly why it looked brilliant on training data and useless on anything real.

None of this was automated. I read container logs by hand and argued with a chat window. There was no dashboard, no trace and no alert, because nothing was watching for a job that succeeds while computing the wrong thing.

The cause was a cache that nobody calls a cache: a container only sees what another container wrote to a shared volume after it explicitly asks again. The data preparation job was still unpacking shards, both trainers looked once, and neither ever looked back. The whole fix is three lines and a guard clause: reload the volume, refuse to start on a partial corpus, and chain the legs so a six day run survives a platform that kills every container at twenty four hours.

Forty eight H100 hours, two containers, one full day each.

## Five failures, and none of them are exotic

The same run and its neighbours also produced a foreground launch whose remote jobs died with my laptop's DNS, a twenty four hour function cap discovered while planning a six day run, two workspaces holding volumes with the same name and different contents, and a zero sized persistent volume that let ten hours of work expire as though it never existed.

And the quiet one: results written to a remote archive and never folded back, so a document in the project said an experiment had never been run while a complete hundred and five cell grid sat in the archive. Recovering it cost one download. Re-running it would have cost a GPU day.

Every one of these is somebody's postmortem this quarter. They are not rookie mistakes, they are the default behaviour of the platforms, and they sort into three shapes: **shared state**, who guarantees the input is complete before a consumer reads it. **Lifetime**, what keeps the remote work alive and for how long. **Ownership**, whose job it was to notice.

## What I do differently now

Four checks, every run, no exceptions. A smoke test: one tiny pass end to end on a CPU before anything paid starts. Hash the dataset and refuse to start unless the hash matches. Checkpoint every couple of hours and push a copy to the Hub. And list the GPU tiers before choosing one, because the same mistake on a cheaper tier is a cheaper mistake.

Two minutes of smoke test decides the shape of six days of training.

## How would you even know the compressed model still works

This is where the second half lands, and it is the part I got most wrong.

I started with Langfuse and LangGraph, an LLM reading the logs as a judge. It never once flagged a missing tool call, because a judge grades the text it was given and silence reads as a perfectly reasonable answer. I now run an open ensemble through OpenRouter and Promptfoo, three cheap models from different families so they do not share a blind spot, with explicit metrics pulled out of the logs rather than out of an opinion.

The failure that made the point: a quantised model scored a perfect 100 on one benchmark category, because that category scores a model on correctly declining to call a function and the model had stopped calling functions at all. It answered 1.2 percent of prompts. Averaged across categories it would have been published as mediocre rather than dead.

**Every capability metric needs a liveness metric beside it.** For tool calling that is the call rate. For your system it is whatever separates correctly did nothing from did nothing.

## The part I cannot answer

Precision changes the served model numerically without changing its version number. It is a cost decision, a platform decision, a quality decision and a hardware decision at once, and in most teams nobody owns all four, so it either never happens or it happens with no validation story. Same shape as the result that sat in an archive nobody owned.

So I will finish on three questions for the room. Who signs off when the model changes numerically but not by version? What evidence does your team require? And has anyone ever actually rejected a change on it?

## Who this is for

Platform engineering, SRE, infrastructure, MLOps, and engineering leaders who approve model changes. Anyone whose team runs training or batch jobs on somebody else's hardware. No quantisation background needed; the format is explained in one slide and the failures are platform failures.

**Level:** intermediate.

## Takeaways

1. A producer and a consumer sharing a network volume is a distributed systems problem, not a filesystem operation. Make the consumer assert on completeness rather than trust what it can see.
2. A cheap screen beats careful review. A CPU smoke test and a throughput probe before every paid run has caught more of my expensive mistakes than reading the code ever did.
3. Every capability metric needs a liveness metric beside it, or a system that has stopped working will score well.

## Format

Twenty to twenty three minutes, with five minutes kept for questions. Roughly a third of the deck is diagrams: the platform model, the training loop, a timeline of the twenty four hours, the token arithmetic, the race condition, and the fix.

**Current state:** open source. Every number above traces to a script and an archived result file at github.com/aloshdenny/superfloat, and the figure scripts run on a laptop with no GPU.

**Tags:** #mlops #infrastructure #failurestory #gpu #serverless #inference #quantization #evaluation #observability #casestudy
