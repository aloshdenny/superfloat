# Both talks in plain English
For explaining to the editorial team, or anyone who asks what you are talking about.

First, the one piece of background both talks sit on, in one sentence you can say first:

> I work on shrinking AI models so they run on cheap hardware, which means storing
> every number inside the model using fewer digits. My job for the last year has
> been finding out when that works, when it breaks, and when it looks like it
> worked but did not.

---

# Talk 1: "It Scored 100 Percent Because It Had Stopped Answering"

## In one sentence

I shrank a model, tested it, and it got a perfect score on part of the test precisely because it had stopped working.

## The analogy

Imagine you are testing an assistant who can use tools: a calculator, a calendar, a search box. The test has three parts. Can it pick the right tool? Can it choose between several tools? And can it recognise when no tool is needed and just answer directly?

Now imagine the assistant has gone mute and stopped using tools altogether.

Part three gives it full marks. Every single time a question did not need a tool, it correctly did not use one. It looks careful. It is actually broken.

## What actually happened

I compressed a model and ran it on a standard test for tool use. It used a tool on 1.2 percent of questions, meaning it had essentially stopped. On the "knows when not to use a tool" part of the test it scored 100 percent, the best number in my whole results table.

The test invites you to average the three parts into one headline number. If I had done that, this dead model would have been published at around 28.7 percent. That reads as mediocre. Nobody looking at it could tell the model was not working at all.

The fix is one extra column: how often did it even try? Once "tried on 1.2 percent of questions" sits next to the score, the problem is obvious in a second.

## Why anyone should care

This is not really about model compression. It is about any score where doing nothing counts as doing well. Spam filters that block everything. Fraud systems that flag nothing. Safety filters that refuse everything. A content moderator that rejects every post has a perfect record on letting bad content through.

Anyone who approves a change based on a benchmark number can be fooled this way, and the aggregate is what hides it.

## The second half of the talk

There was also a cheap trick that worked. A rough internal health check, costing about one percent of what the full test costs to run, correctly predicted which configurations were fine and which were broken, every single time. So you can use the cheap check to decide where to spend the expensive one. That matters to anyone with a limited evaluation budget, which is everyone.

## Takeaway in one line

Every score that measures capability needs a second number beside it that proves the system is still alive.

---

# Talk 2: "Two Things I Published, And Then Killed"

## In one sentence

I published two findings about shrinking AI models, then ran the experiments that proved both of them wrong, and this is the story of how each one died.

## Why it is interesting

Not because the two findings matter much on their own. Because both mistakes had the same shape, and it is a shape most people make: I saw something happen once, in one setup, and I wrote it down as a general rule.

## The first dead claim

**What I said:** when you shrink a model's numbers, you have to train it more gently, because the numbers are coarser and a big training step overshoots.

**The analogy:** I thought you had to drive slower on a bumpy road.

**What killed it:** I ran 60 experiments. Six different levels of coarseness, ten different training speeds, spanning a 640 times range from very gentle to very aggressive. Not one of them crashed. And the best speed turned out to be the same one regardless of how coarse the numbers were. The rule was wrong.

**The honest part, and the reason this one is worth a talk:** I still cannot explain what I originally saw. My original crash was real. But the old setup and the new one differ in four ways at once, and nothing in my experiment isolates which one mattered. So I retired a claim while still owing an explanation for the evidence that produced it. That is an uncomfortable place to stop, and I think stopping there honestly is better than inventing a story.

## The second dead claim

**What I said:** the numbers inside a trained model are always small, so a format that can only hold small numbers is safe.

**The analogy:** a ruler with very fine millimetre markings, but only one metre long. It measures almost everything in the room beautifully. Then you try to measure a seven metre pole with it, and no amount of fine markings helps. The problem is not how precisely you can measure. It is how big a thing you can hold.

**What killed it:** On a vision model this was perfectly true. Zero out of 53.6 million numbers were too big. Then I checked a language model. Its largest number was 7.47, and about one in ten thousand numbers was too big to fit.

That one in ten thousand is the whole story. Those few hundred numbers happen to be the largest and most important ones in the model, and squashing them to fit destroys it. It destroys it even when you use a very fine format, where the markings are three hundred thousandths apart, so precision was never the problem.

**Why this one is worse than a plainly wrong claim:** it was 99.99 percent true. Every average I could compute said it was fine. The failure lived entirely in the fraction that averages erase.

## What the talk is really about

Two things.

One: a result that holds on one model family is a hypothesis, not a finding. Both of my failures were a confident generalisation from a single well behaved example.

Two: when you find out you were wrong, put the correction next to the original claim instead of quietly editing it. My repository now has both, with the retraction written underneath the text it retracts. It costs nothing and it makes the whole body of work more trustworthy, because a reader can see what I believed, what changed, and why.

## Takeaway in one line

Test your own claims the way a rival would, and when you are wrong, leave the evidence of being wrong where people can see it.

---

# If someone asks how the two talks relate

Both are about being wrong in a way that an average will not show you.

Talk 1 is a broken system that an average makes look fine. Talk 2 is a claim that is true on average and false exactly where it matters. Same lesson from two directions: the number you report is usually hiding the thing you most need to know.
