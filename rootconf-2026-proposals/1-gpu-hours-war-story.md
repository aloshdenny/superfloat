**One-line summary.** A pretraining run quietly trained on one shard of its corpus for a full day because two containers never reloaded a shared volume, and the warning sign was visible the whole time and read as noise. This is what running a thousand cell experiment grid on rented GPUs actually teaches you.

## The problem

I run a quantisation research project whose results come from roughly a thousand training runs, spread across four compute platforms over a year: a desk GPU, a shared lab machine, RunPod, and Modal. None of it is a production service. All of it has the operational character of one, because a job that dies at hour twenty is expensive, and a job that succeeds while computing the wrong thing is worse.

What makes it hard is that almost none of the expensive failures were modelling bugs. They were plumbing. A network volume with read caching that nobody told me was a cache. A CLI whose foreground mode ties remote job lifetime to my laptop's network. A platform function timeout shorter than the run. Two volumes with the same name in two workspaces. Results that existed, but not where the repository said they were.

The scale is small enough that every failure is legible and large enough that each one costs real money. That makes it a good teaching corpus.

## Who this is for

Platform engineering, SRE, infrastructure, and engineering leaders. Anyone whose team runs training or batch jobs on somebody else's hardware.

**Level:** intermediate.

## Takeaways

1. A producer and a consumer sharing a network volume is a distributed systems problem, not a filesystem operation. Your job should refuse to start on a partial input rather than trust what it can see.
2. Cheap screens beat careful review. A CPU smoke test and a throughput probe before every paid run has caught more of my expensive mistakes than reading the code ever did.

## What I will share

Production experience, where the production is a research pipeline. Failure modes and debugging, with the actual numbers. Benchmarks and measurements. The operational practice that came out of each incident. The trade-off between building a general job runner and just writing the next script.

## My experience with this problem

Open source project. Experiment or prototype. Research or investigation. Hard earned engineering lesson.

## What failed, disappointed, or created unexpected problems

**The headline one.** On a serverless GPU platform, a container only sees files another container has committed after it explicitly reloads the volume. I started two training containers while the data preparation job was still writing shards. Neither reloaded. Both trained on shard zero alone, 96 million tokens, for twenty four hours, which is about thirty epochs of the same data. Train loss 0.95, validation loss 5.6. Textbook memorisation. Two times twenty four H100 hours, gone. The validation curve had been climbing for hours and I read it as evaluation noise.

**Four more that cost real time.**

- Launching without a detach flag. The platform's foreground mode ties the remote job's life to the local client connection. A DNS blip on my machine tore down four concurrently running jobs. They survived only because the scripts checkpoint.
- A hard twenty four hour cap on function runtime, discovered while planning a six day run. The fix is a resume chain: run under a timeout, checkpoint on the termination signal, and have the job respawn itself on the timeout exit code.
- Two workspaces each holding a volume with the identical name and different contents. A listing from the wrong one looked exactly like catastrophic data loss.
- A pod created with a zero sized persistent volume on an earlier platform. Ten hours of work, nothing on disk.

**And the quiet one.** Results were written to a remote archive and never folded back into the repository, so a document in the project said an experiment "was never run" while a complete hundred and five cell grid sat in the archive. Recovering it cost one download. Rerunning it would have cost a GPU day.

## What I would do differently today

Never start a consumer while a producer is still writing, and make the consumer assert on input completeness rather than trust its own view of the filesystem. Detach every launch by default. Assume nothing about paths, working directories or concurrency safety carries across a platform move, even for scripts with a long track record. Before any paid run: compile it, run one tiny end to end pass on CPU, measure real throughput, then choose a budget. And list the results archive before running anything, because the cheapest experiment is the one you already ran.

## Trade-offs

**Build a general job runner, or keep writing one off scripts.** I built a generic runner that passes paths through environment variables, which made the scripts portable across four hosts without edits, at the cost of a layer of indirection that hides real errors behind a subprocess boundary.

**Checkpoint frequently, or run fast.** Frequent checkpoints cost throughput and have saved every run that was interrupted, which by now is most of them.

**Fix the scripts, or fix the runner.** Several failures came from scripts making assumptions about their own directory. I fixed those at the runner level rather than editing scripts that have an archived result history, knowingly trading correctness at the source for not invalidating comparability.

## How this helps other practitioners

A set of operational practices for anyone running batch or training work on rented infrastructure. A debugging technique for the class of bug where the job succeeds and the output is wrong. And a way of thinking about cheap screens as the highest return activity in the whole pipeline.

**Current state:** in progress.

**Tags:** #mlops #infrastructure #failurestory #gpu #serverless #inference #observability #casestudy
