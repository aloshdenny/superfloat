# Rootconf 2026, five proposals
Nov 13 to 14, 2026, Bangalore and hybrid. CfP closes Oct 9, 2026.
Theme: platforms for AI, and AI for platforms.

Ranked by how well they fit the audience. If you only submit some, submit 1 and 2.

---

# 1. The infrastructure war story (talk)

**I am submitting:** To speak
**Submission type:** A talk, 30 to 40 mins

**Session title:**
> We Burned 48 GPU Hours Training On 96 Million Tokens, And The Loss Curve Told Us For Hours

**One-line summary:**
> A pretraining run quietly trained on one shard of its corpus for a full day because two containers never reloaded a shared volume, and the warning sign was visible the whole time and read as noise. This is what running a thousand cell experiment grid on rented GPUs actually teaches you.

**What problem are you addressing?**
> I run a quantisation research project whose results come from roughly a thousand training runs, spread across four different compute platforms over a year: a desk GPU, a shared lab machine, RunPod, and Modal. None of it is a production service. All of it has the operational character of one, because a job that dies at hour twenty is expensive, and a job that succeeds while computing the wrong thing is worse.
>
> What makes it hard is that almost none of the expensive failures were modelling bugs. They were plumbing. A distributed filesystem with read caching that nobody told me was a cache. A CLI whose foreground mode ties remote job lifetime to my laptop's network. A platform function timeout shorter than the run. Two volumes with the same name in two workspaces. Results that existed but were not where the repository said they were.
>
> The scale is small enough that every failure is legible and large enough that each one costs real money. That makes it a good teaching corpus.

**Intended audience:** Platform engineering, SRE, infrastructure, engineering leaders, and anyone whose team runs training or batch jobs on someone else's hardware

**Level:** Intermediate

**Practical takeaways:**
> 1. A producer and consumer sharing a network volume is a distributed systems problem, not a filesystem operation, and your job should refuse to start on a partial input rather than trust what it can see.
> 2. Cheap screens beat careful review. A CPU smoke test and a throughput probe before every paid run has caught more of my expensive mistakes than reading the code ever did.

**What will you share?**
> Production experience, though the production is a research pipeline. Failure modes and debugging, with the actual numbers. Benchmarks and measurements. Operational practices that came out of each incident. Trade-offs between building a general job runner and just writing the next script.

**Experience with this problem:** Open source project, experiment or prototype, research or investigation, hard earned engineering lesson

**What approaches failed, disappointed, or created unexpected problems?**
> The headline one. On a serverless GPU platform, a container only sees files another container has committed after it explicitly reloads the volume. I started two training containers while the data preparation job was still writing shards. Neither reloaded. Both trained on shard zero alone, 96 million tokens, for twenty four hours, which is about thirty epochs of the same data. Train loss 0.95, validation loss 5.6. Textbook memorisation. Two times twenty four H100 hours, gone. The validation curve had been climbing for hours and I read it as evaluation noise.
>
> Four more that cost real time:
> - Launching without a detach flag. The platform's foreground mode ties the remote job's life to the local client connection. A DNS blip on my machine tore down four concurrently running jobs. They only survived because the scripts checkpoint.
> - A hard twenty four hour cap on function runtime, discovered while planning a six day run. The fix is a resume chain: run under a timeout, checkpoint on the termination signal, and have the job respawn itself on the timeout exit code.
> - Two workspaces each containing a volume with the identical name and different contents. A listing from the wrong one looked exactly like catastrophic data loss.
> - A pod created with a zero sized persistent volume on an earlier platform. Ten hours of work, nothing on disk.
>
> And the quiet one. Results were written to a remote archive and never folded back into the repository, so a document in the project said an experiment "was never run" while a complete hundred and five cell grid sat in the archive. Recovering it cost one download. Rerunning it would have cost a GPU day.

**What will you do differently today?**
> Never start a consumer while a producer is still writing, and make the consumer assert on input completeness rather than trust its own view of the filesystem. Detach every launch by default. Assume nothing about paths, working directories or concurrency safety carries across a platform move, even for scripts with a long track record. Before any paid run: compile it, run one tiny end to end pass on CPU, measure real throughput, then choose a budget. And list the results archive before running anything, because the cheapest experiment is the one you already ran.

**What trade-offs did you consider?**
> Build a general job runner or keep writing one off scripts. I built a generic runner that passes paths through environment variables, which made the scripts portable across four hosts without edits, at the cost of a layer of indirection that hides real errors behind a subprocess boundary.
>
> Checkpoint frequently or run fast. Frequent checkpoints cost throughput and have saved every single run that was interrupted, which by now is most of them.
>
> Fix the scripts or fix the runner. Several failures came from scripts making assumptions about their own directory. I fixed those at the runner level rather than editing scripts with an archived result history, knowingly trading correctness at the source for not invalidating comparability.

**How can this help other practitioners?**
> A set of operational practices for anyone running batch or training work on rented infrastructure, a debugging technique for the class of bug where the job succeeds and the output is wrong, and a way of thinking about cheap screens as the highest return activity in the whole pipeline.

**Current state:** In progress

**Tags:** #mlops #infrastructure #failurestory #gpu #serverless #inference #observability #casestudy

---

# 2. Quantisation as a serving decision (talk)

**I am submitting:** To speak
**Submission type:** A talk, 30 to 40 mins

**Session title:**
> Your Model Does Not Need Floating Point, But It Does Need One Thing You Probably Skipped

**One-line summary:**
> Open weight models quantise to eight bit fixed point after training with no retraining and no calibration set, and keep their tool calling intact, provided you get one detail right that silently destroys the model if you miss it.

**What problem are you addressing?**
> Teams serving open weight models reach for quantisation to cut memory and cost, then discover that the outcome is unpredictable. Sometimes the model is fine. Sometimes it is subtly worse. Sometimes it is wrecked, and the failure does not look like a precision problem.
>
> I spent a year measuring where the real constraint is, using a number format that removes the exponent field entirely and stores only sign and significand. That is the most aggressive version of the question: if trained weights are bounded and cluster near zero, how much of a float is a model actually using?
>
> The answer for serving is good. Qwen3 dense models from 1.7B to 8B drop to eight bits post training, with no fine tuning and no calibration data, and hold function calling accuracy within about a point of the original. But the failure mode on the other side is sharp, and it is not the one people expect.

**Intended audience:** Platform engineering, infrastructure, MLOps, developers running inference in production, engineering leaders sizing hardware

**Level:** Intermediate

**Practical takeaways:**
> 1. Roughly one weight in ten thousand in a trained checkpoint sits outside the representable range, and those are the largest weights in the model. Quantising them without a per matmul scale destroys the model even at sixteen bits, where precision is plainly not the problem. Check the checkpoint, do not assume.
> 2. Which checkpoint you quantise matters more than which method you use. Post training damage is not monotone in training tokens, and the worst checkpoint to quantise is often the heavily trained one you were about to ship.

**What will you share?**
> Benchmarks and measurements across model families and sizes. Failure modes with the numbers attached. Design decisions about where a scale lives and why that choice is free at inference. Trade-offs between post training quantisation and training aware approaches. Open source tooling, all MIT licensed and reproducible.

**Experience with this problem:** Open source project, research or investigation, experiment or prototype

**What approaches failed, disappointed, or created unexpected problems?**
> The big one. Two projection matrices in a transformer block are not fed by a normalisation layer, so an early version of my pipeline quantised them as they stood. General validation loss went from 2.53 to 7.73. At sixteen bits. The grid step at sixteen bits is three in a hundred thousand, so precision was never the issue: a few hundred outlier weights were being clipped, and they were the largest weights in the network. Giving every matmul its own scale recovered the model completely.
>
> Second, my own published claim that trained weights always fall inside the representable range. It held on a vision model where zero of 53.6 million weights fell outside. It failed on a language model whose largest weight is 7.47. The correction now sits in the repository next to the original claim.
>
> Third, six bit quantisation. It looks like a small step down from eight. At 8B it costs about three points. At 0.6B the model stops working entirely.

**What will you do differently today?**
> Measure the weight range of the actual checkpoint before choosing a format, rather than reasoning from what trained networks are supposed to look like. Give every matrix multiply a scale by default, since tensor and channel scales fold into a neighbouring weight or norm and cost nothing at inference. And treat a small model as a different problem from a large one rather than a scaled down version of it.

**What trade-offs did you consider?**
> Post training quantisation against training aware quantisation. At eight bits there is no training budget needed at all, which removes the entire question. At six bits you need either a large model or a training run, and that is a real cost decision.
>
> Weights only against a fully quantised datapath. Weights only is what most serving stacks do and is nearly free. Saturating every intermediate value, which is what fixed point hardware actually requires, breaks the model unless you add a per token scale, and that scale costs silicon.
>
> Loss as a screen against running the full benchmark. Loss costs about a hundredth of a benchmark sweep and correctly predicted both the safe configuration and the cliff in every case I checked.

**How can this help other practitioners?**
> A way to evaluate competing quantisation options before committing hardware, a specific mistake to avoid that is invisible in the usual metrics, and a rule of thumb about checkpoint selection that applies whatever method you use.

**Current state:** Open source

**Tags:** #inference #mlops #quantization #modelserving #edge #benchmarks #opensource

---

# 3. The metric that rewarded a broken model (talk)

**I am submitting:** To speak
**Submission type:** A talk, 30 to 40 mins

**Session title:**
> It Scored 100 Percent Because It Had Stopped Answering

**One-line summary:**
> A quantised model scored a perfect 100 on one category of a function calling benchmark while emitting tool calls on 1.2 percent of prompts, and the headline average made it look merely mediocre. This is a talk about validating a model change before it reaches production.

**What problem are you addressing?**
> Every platform team that serves models eventually has to approve a change to the model itself: a quantisation, a version bump, a different serving runtime. The change is evaluated with a benchmark, a number comes out, and somebody decides. The problem is that a benchmark number can be produced by a system that has stopped working, and the aggregate can hide it completely.
>
> I hit this while measuring quantised models on a function calling benchmark. One category scores the model on correctly declining to call a function when no function applies. A model that has stopped emitting tool calls at all passes every case in that category for free. Mine scored 100 there, called on 1.2 percent of prompts overall, and would have been reported at about 28.7 percent if I had averaged the categories the way the benchmark invites you to.
>
> That is not a quantisation problem. It is an evaluation design problem, and it generalises to any guardrail metric where the safe behaviour and the broken behaviour look identical from outside.

**Intended audience:** Platform engineering, SRE, MLOps, developers shipping agents or tool using systems, engineering leaders who sign off on model changes

**Level:** Intermediate

**Practical takeaways:**
> 1. Every capability metric needs a liveness metric beside it. For tool calling that is the call rate. For your system it is whatever number distinguishes "correctly did nothing" from "did nothing".
> 2. A cheap screen can predict an expensive benchmark. Validation loss cost about one percent of a full benchmark run and located the failure cliff correctly every time, which changes how you budget evaluation.

**What will you share?**
> Benchmarks and measurements, a failure mode with the full result table, the reasoning behind the screening strategy, and open source tooling. A live walkthrough of the same trap in the raw results if the room wants it.

**Experience with this problem:** Open source project, research or investigation, hard earned engineering lesson

**What approaches failed, disappointed, or created unexpected problems?**
> Reporting a single aggregate number. It was the obvious thing to do and it would have published a broken model as a mediocre one.
>
> Trusting a category whose scoring rewards abstention. I now treat any metric where doing nothing scores well as requiring a companion metric, without exception.
>
> Assuming a result from one model size transfers. At 8B the six bit configuration holds at 88 percent. At 0.6B the same configuration collapses to 27. One number from one model would have produced confident and wrong guidance either way.

**What will you do differently today?**
> Report every category separately with the call rate beside it, never aggregate across categories with different failure semantics, and screen with a cheap proxy before paying for a benchmark sweep. And run the sweep across sizes, since the interesting behaviour is at the edges.

**What trade-offs did you consider?**
> Cheap proxy against real capability measurement. Loss is fast and correlates well but cannot tell you that the output is malformed JSON. The answer was to use it as a screen, not as evidence.
>
> Breadth against depth in evaluation. I chose four model sizes crossed with three precisions over one model examined thoroughly, which caught the size dependence and cost the deeper per category analysis I would otherwise have.

**How can this help other practitioners?**
> A specific class of mistake to avoid when approving model changes, a pattern for pairing capability metrics with liveness metrics, and an evaluation budget strategy that uses cheap signals to decide where to spend expensive ones.

**Current state:** Open source

**Tags:** #evaluation #observability #mlops #agents #inference #failurestory #modelserving

---

# 4. Hands-on: quantise a model and prove it still works (workshop)

**I am submitting:** To teach a workshop
**Submission type:** Hands-on workshop, 2 to 4 hours

**Session title:**
> Break A Model, Then Fix It: A Hands-On Session On Quantisation And Proving It Worked

**One-line summary:**
> Participants quantise a small open weight model to eight bits, watch it break in the specific way that looks like a precision problem but is not, fix it, and then build the evaluation that would have caught the breakage.

**What problem are you addressing?**
> Most people meet quantisation as a flag in a serving config. It either works or it does not, and when it does not there is nothing to debug because the model simply produces worse output. This workshop replaces that with hands on intuition: participants implement the quantiser themselves in about forty lines, break their model deliberately, and diagnose it from the weights rather than from the output.

**Intended audience:** Developers, MLOps, platform engineering, anyone who has enabled a quantisation flag without knowing what it does

**Level:** Intermediate

**Practical takeaways:**
> 1. The ability to inspect a checkpoint and predict whether a given quantisation will work, before running it.
> 2. An evaluation harness pattern that pairs a capability metric with a liveness metric, built during the session.

**What will you share?**
> Code and implementation details, a debugging technique, a live build of both the quantiser and the evaluation, and open source tooling that participants leave with.

**Experience with this problem:** Open source project, experiment or prototype, research or investigation

**What approaches failed, disappointed, or created unexpected problems?**
> Every failure in the workshop is one I hit for real: the outlier weights that break a model at any bit width, the initialisation interaction that zeroes an entire layer, and the evaluation that rewards a model for going silent. The session is built around reproducing them deliberately in a few minutes each.

**What will you do differently today?**
> Teach range before precision. Everybody's intuition is that fewer bits means more error, and the failures that actually matter are about range and scale placement instead.

**What trade-offs did you consider?**
> Small model on a laptop against a realistic model on rented GPUs. I chose small and local so that nobody spends the session fighting an environment, at the cost of some realism, which I cover with prepared results from the larger runs.

**How can this help other practitioners?**
> A design approach for evaluating precision options, a debugging technique for a failure that is otherwise invisible, and a mistake to avoid that costs teams real production quality.

**Current state:** Open source

**Tags:** #workshop #quantization #inference #mlops #evaluation #handson

---

# 5. Who owns the precision decision? (Birds of Feather)

**I am submitting:** To run a Birds of Feather (BOF) session
**Submission type:** Birds of Feather

**Session title:**
> Who Owns The Precision Decision, And How Does Your Team Prove It Was Safe?

**One-line summary:**
> Quantisation changes the model, but the decision usually sits with whoever owns the serving cost. I want to compare notes on who actually makes that call across teams, and what evidence they require before it ships.

**What problem are you addressing?**
> Precision is one of the few changes that crosses every team boundary at once. It is a cost decision, a platform decision, a model quality decision and a hardware decision, and in most organisations nobody owns all four. The result is that it either never happens or happens without a validation story.
>
> I have measured the technical side extensively and I have very little visibility into how other teams organise the decision. That asymmetry is what makes this a discussion rather than a talk.

**Intended audience:** Platform engineering, MLOps, SRE, engineering leaders

**Level:** Intermediate

**Practical takeaways:**
> 1. A comparison of how different teams structure the approval, what evidence each requires, and where it breaks down.
> 2. A shared list of validation practices worth stealing.

**What will you share?**
> I will open with about ten minutes of measurements to give the room a common factual footing, including the failure modes that are not obvious, then facilitate rather than present.

**Experience with this problem:** Open source project, research or investigation

**What approaches failed, disappointed, or created unexpected problems?**
> On my own side, treating precision as a purely numerical question. Almost every real difficulty turned out to be about which artefact you are allowed to change and who has to sign off, which is not something I can answer from my own work.

**What will you do differently today?**
> Ask the people who ship this in production rather than infer it from papers.

**What trade-offs did you consider?**
> Running this as a talk against a BOF. I have enough material for a talk, but the interesting half of the question is the half I do not have data on, and a room full of platform engineers does.

**How can this help other practitioners?**
> A set of operational practices, and a way of framing a decision that currently falls between teams.

**Current state:** In progress

**Tags:** #bof #mlops #platformengineering #inference #governance #modelserving
