**One-line summary.** Quantisation changes the model, but the decision usually sits with whoever owns the serving cost. I want to compare notes on who actually makes that call across teams, and what evidence they require before it ships.

## The problem

Precision is one of the few changes that crosses every team boundary at once. It is a cost decision, a platform decision, a model quality decision and a hardware decision, and in most organisations nobody owns all four. The result is that it either never happens, or it happens without a validation story.

I have measured the technical side extensively and I have very little visibility into how other teams organise the decision. That asymmetry is exactly why this should be a discussion rather than a talk.

## Questions for the room

- Who signs off when the served model changes numerically but not by version?
- What evidence does your team require, and has anyone ever rejected a change on that evidence?
- Does your serving platform expose precision as a knob, and should it?
- When a quantised model gets worse in a way your metrics do not show, how would you find out?

## Who this is for

Platform engineering, MLOps, SRE, engineering leaders.

**Level:** intermediate.

## Takeaways

1. A comparison of how different teams structure the approval, what evidence each requires, and where it breaks down.
2. A shared list of validation practices worth stealing.

## What I will share

About ten minutes of measurements to give the room a common factual footing, including the failure modes that are not obvious, and then I facilitate rather than present.

## My experience with this problem

Open source project. Research or investigation.

## What failed, disappointed, or created unexpected problems

On my own side, treating precision as a purely numerical question. Almost every real difficulty turned out to be about which artefact you are allowed to change and who has to sign off, which is not something I can answer from my own work.

## What I would do differently today

Ask the people who ship this in production rather than infer it from papers.

## Trade-offs

**Running this as a talk against a BOF.** I have enough material for a talk, but the interesting half of the question is the half I have no data on, and a room full of platform engineers does.

## How this helps other practitioners

A set of operational practices, and a way of framing a decision that currently falls between teams.

**Current state:** in progress.

**Tags:** #bof #mlops #platformengineering #inference #governance #modelserving
