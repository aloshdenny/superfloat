# Form fields (do not paste this part)

**Title:** Why Is The Checkpoint You Were About To Ship The Worst One To Compress?

**I am submitting:** Have a problem I want to discuss

**I have a submission for:** Birds of Feather, I have a problem/idea I want like-minded folks to flock around

**Submission type:** Problem/reverse CfP, potential Birds of Feather

---
<!-- PASTE EVERYTHING BELOW THIS LINE INTO THE MARKDOWN BODY -->
---

## The problem

Take one model, compress it after training, and measure the damage. Now do that at seven points along its training run, from 2.1 billion tokens to 300 billion. The damage is not monotone. It falls steeply, bottoms out in the middle of training, and then rises sharply at the end. For a 160 million parameter model at seven bits, the penalty is 0.60 nats early, 0.13 nats at the minimum, and 3.16 nats at the final checkpoint. The most heavily trained checkpoint, the one you would actually ship, is roughly twenty five times worse to compress than one from the middle of its own training run.

I can explain half of this. The early fragility is a scale placement artefact: young weights have not settled into the scale the trained network eventually uses, and putting the scale in the right place removes the effect entirely. At 410 million parameters and 2.1 billion tokens the eight bit penalty goes from +0.399 to +0.003, turning the most fragile checkpoint in the ladder into the least damaged one.

The right hand branch does not yield to the same fix. At the final checkpoint the same correction moves the 160 million parameter model from +0.655 to +0.654, which is nothing. There are two obvious candidate causes, over-training and learning rate decay, and my experiment cannot separate them because they are perfectly confounded along a training run. I have one piece of evidence that it is not purely learning rate decay, since the same U appears in a different architecture at a different size under a different compression regime. Beyond that I am stuck, and the practical advice I am currently giving people ("do not compress your final checkpoint") is a rule of thumb resting on an unexplained effect, which I dislike.

What I want from a room: help designing the smallest experiment that separates over-training from schedule effects. Ideas about what actually changes in a weight distribution late in training that a per channel scale cannot absorb. And anyone who has seen a similar late training fragility in a different setting, since I suspect this is not specific to compression.

## What kind of help would be useful

- idea
- experience
- technique
- critique
- domain context

## Topic tags

model compression, scaling laws, evaluation

## Willing to be contacted before the event

Yes. Happy to be contacted for matching or clarification, and happy to share the raw result files in advance with anyone who wants to look at the curves before we talk.

## Bio

Alosh Denny, creator of Superfloat, an open source number format and accelerator for running models without floating point, and currently Head of AI at Aleddo Technologies and LLMOps lead at Intensors. Kerala based, seven peer reviewed publications, IndiaFOSS speaker. He has spent a year running 878 training runs to find out how few bits a model needs, and this is the result he can measure cleanly and cannot explain at all.
