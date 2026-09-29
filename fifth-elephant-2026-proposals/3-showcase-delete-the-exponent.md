# Form fields (do not paste this part)

**Title:** Delete The Exponent: What A Model Actually Does With Its Bits

**I am submitting:** To speak

**I have a submission for:** Showcase, a cool idea I have built, 20 mins

**Submission type:** Showcase, new ideas, 20 mins

---
<!-- PASTE EVERYTHING BELOW THIS LINE INTO THE MARKDOWN BODY -->
---

## Describe your session

A floating point number spends most of its bits buying dynamic range. A trained neural network barely uses that range: weights are bounded and clustered near zero. So I built the format that takes the premise literally. Superfloat keeps the sign bit, spends every remaining bit on the significand, and has no exponent field at all, which makes it plain signed fixed point. Twenty minutes is enough to show what that buys and what it costs.

What it buys, measured rather than argued: open weight language models from 1.7B to 8B parameters drop to eight bits after training, with no fine tuning and no calibration set, and hold their function calling accuracy within about a point of the original. On some vision and video models the compressed version scores higher than the full precision control trained on the same schedule. In silicon, which is where this ends up, an arithmetic unit with no exponent needs no barrel shifter, no leading zero detector and no rounding logic, and comes out around 1.4 times smaller than an IEEE half precision unit and 5.3 times smaller than single precision through an identical open source hardening flow. What it costs is one detail that has to be right, and the showcase ends there: a few hundred outlier weights, one in ten thousand, will destroy the model at any bit width if you quantise them without a scale.

## Takeaways

1. The dynamic range of a float is mostly unused at inference, and you can check whether that holds for your own checkpoint in about five lines.
2. Where you put the scale matters more than how many bits you keep. Most reported precision floors are artefacts of scale placement.

## Which audiences will benefit most

Practitioners curious about what is underneath the models they use, people who care about running models on small or cheap devices, and anyone who enjoys watching a simple premise get tested to destruction. No hardware background needed.

## Bio

Alosh Denny is an AI engineer and researcher from Kerala, and the creator of Superfloat, an open source number format, accelerator and compiler stack for running models without floating point. He heads AI at Aleddo Technologies and runs LLMOps at Intensors, has seven peer reviewed publications across IEEE and Springer, and has spoken at IndiaFOSS. He is currently trying to get a chip built around this idea onto an actual wafer.

## Draft slides

Not yet. I will share a draft with comments enabled if the session is selected.

## What I do not know yet

The format works well on everything I have measured, which is language, vision, video and a brain encoding model. I do not know where it fails, and I would like to. If your domain has models with genuinely wide weight distributions, or where small absolute errors compound in ways language modelling loss would hide, I want to hear about it and ideally test on it.

## Matchmaking tags

- I can help with: ideas, experience, technique, critique
- I need help with: dataset, collaborator, critique
- I would like to meet: practitioners, domain experts, tool builders

## Topic tags

model compression, edge AI, open source
