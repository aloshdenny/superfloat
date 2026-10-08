# Form fields (do not paste this part)

**Session type:** Talk (40 minutes)
**Track:** Security and Hacking
**Language:** English
**Title:** Erasing A Face From A Model Of A Brain

**Resources to attach:**
- https://medium.com/@aloshdenny/uncensoring-the-brain-1cd2cfde1de5
- https://medium.com/@aloshdenny/erasing-a-face-9e82c2332952

---
<!-- ABSTRACT -->
---

Abliteration is the technique people use to strip refusal out of a language model: find the direction in the weights that encodes "say no", subtract it, and the model stops refusing. It has exactly one proven use case, on one kind of model, for one kind of behaviour.

So I pointed it at something else entirely. TRIBE v2 is Meta's brain encoding model: feed it video, audio and text, and it predicts fMRI activity across roughly 20,000 points on the cortex. It is a digital twin of a brain's response, and it is a public download. I tried to remove one specific person's face from it, so the model would stop representing that individual in the regions that handle face identity, while still processing every other face normally.

Most of what I tried did not work, and the interesting part is how convincingly it pretended to. One direction separated the target from everyone else with a perfect score and turned out to be tracking which website the photos came from. This is a talk about editing a model of a brain, about six different ways I caught myself being wrong, and about what it means that this is now something a person can do on rented GPUs.

---
<!-- DESCRIPTION -->
---

## Where this starts

If you have poked at open weight language models you know abliteration. You collect activations on harmful and harmless prompts, find the direction that separates them, project it out of the weights, and the model loses its ability to refuse. It is a weights level edit, not a prompt trick, and it is the best known way to uncensor a model.

What nobody had tested, as far as I could find, is whether the idea is about language models at all. So I took it somewhere structurally different in two directions at once.

**A different kind of model.** TRIBE v2 is a brain encoding model from Meta. It fuses a video encoder, an audio encoder and a language model, and predicts human fMRI activity across about 20,484 points on the cortical surface. There is no language modelling objective anywhere in it. It was trained on scans from around 720 people.

**A different kind of target.** Not a behaviour like refusal, which is binary and shows up in the output you can read. A specific person's identity, represented inside the face processing regions, which is not binary and which you cannot read off the output at all.

## The validation problem, which is the actual content

Here is why this is harder than it sounds, and why most of the talk is about checking rather than editing.

If I edit the model so it responds less to one person's face, there is a trivial way to pass: just damage face processing in general. The model now responds less to that person, and also to everyone, and every obvious test I could think of at first would score that as a success. So the entire exercise becomes an exercise in falsifying your own result.

Six independent checks, and most of them said my result was fake:

- A contrastive direction built the standard way separated held out photos of the target from general faces perfectly, a clean 1.0 on the separation score. I believed it for a while. Then I scored the same photos with a face recognition model that had never seen the brain model, and found no correlation at all between how strongly a face loaded on my "identity" direction and how much that face actually resembled the target. The direction had learned where the photos came from, not who was in them.
- Ablating that direction moved the model's real predictions by essentially nothing.
- A controlled photo swap experiment moved motor cortex more than any face region.

What eventually did work was structurally different: isolating the model's own final readout, a single matrix mapping its internal representation to the cortical surface, and training a small low rank correction on top of it, with a loss that explicitly pins every non target output back to the frozen model's own baseline rather than merely discouraging it from moving. That produced a suppression effect on held out photos of the target that did not appear on random faces, and did not appear in three non face control regions I kept as a selectivity boundary. Then it replicated on a second, unrelated person, with fresh weights and nothing reused.

And then it broke. Growing the first person's photo set by a factor of three made the effect vanish. I now think that is a problem with how I defined the held out set rather than a failure of the method, but I ran out of time to prove it, and I will show you the evidence and let you decide.

## Why this belongs in Security and Hacking

Three reasons.

**The technique generalises further than its reputation.** Abliteration is discussed as an LLM jailbreak trick. If a weights level edit transfers to a fused multimodal model with no language objective and a completely different output space, then it is a general tool for removing a learned association from a model's weights, and the threat model for every published model gets wider.

**Brain encoding models are a new class of target.** There is a growing pile of public models that map stimuli to neural responses. They are downloadable and they are editable. Before anyone builds anything downstream of them, it is worth knowing what an attacker with a consumer GPU and a scraped photo album can do to one.

**The honest version of the ethics.** Building this required assembling photo galleries of real, named, non consenting people from web search. I will talk about that directly rather than skipping it, including why I ended up using people whose faces are already a cultural baseline, what I think the legitimate use is, and where I think the line sits. I am not going to pretend this is a neutral capability demo. The motivating application I care about is trigger suppression for abuse survivors, where current therapies take years, and that same capability is obviously dual use.

## What this talk is not

It is not an attack on a human brain. TRIBE v2 is a model that predicts brain responses, trained on scans. Editing it edits a model's representation, not anybody's cortex. I will be precise about that distinction early, because the headline version of this work is much more exciting than the truthful version and I would rather you leave with the truthful one.

## What you will take away

A working mental model of what abliteration actually does to weights. A concrete, worked example of how a result that looks perfect can be measuring your data collection instead of your phenomenon, with the specific checks that caught it. And a clear picture of what is currently possible against this class of model, by one person, on rented hardware.

---
<!-- NOTES, private to organisers -->
---

Two long public write ups of this work are linked in the resources. They are written for a general audience; the talk goes deeper on the validation methodology and the negative results.

Content note for scheduling: the first phase of this work used an adult content category as its target concept, and the identity phase used two adult film performers as the subjects, for the specific reason that they produce unusually strong and consistent responses in the model compared to other celebrities I tested. I can present the whole talk without any explicit imagery, and I intend to. Flagging it so nobody is surprised.

Ethics: the photo galleries are of real, publicly known, non consenting individuals, assembled from web image search. The repository is private for that reason, and I do not intend to publish the galleries. I would rather address this on stage than have it raised only in questions, so it has a slot in the talk.

Technical detail available on request: the suppression works through a rank 16 residual, about 66,000 parameters, added to the model's frozen 2048 by 20,484 readout. Face selective regions defined anatomically. Five fold cross validation split by photo rather than by trial. Three non face control regions as a selectivity boundary throughout.

On travel: I would be coming from Kerala, India, and I would need the limited travel support the CfP mentions, plus an official invitation letter for the German embassy. I am aware of the six week visa warning, so an early decision would help.
