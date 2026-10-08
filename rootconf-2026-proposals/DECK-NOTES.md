# Rootconf decks, in Alosh's style

Rebuilt 2026-10-07 with the `alosh-ppt-style` kit: 20 x 11.25 canvas, black on white,
lowercase, no borders or footers, one idea per slide, picture-led. Speaker notes on
every slide, photo credits in the notes rather than on the slides.

| file | slides | for |
| --- | --- | --- |
| `rootconf_talk_deck.pptx` | 62 | the talk: 48 gpu hours + who owns the precision decision |
| `rootconf_workshop_deck.pptx` | 19 | the workshop delivery plan |

Sources in `deck-src/` (`talk.js`, `workshop.js`, `img/`). Rebuild with:

```bash
cd deck-src && NODE_PATH=<scratch>/node_modules node talk.js
```

The earlier corporate-styled versions are in `old-style/` if you want them back.



## Round 4: text replaced with diagrams (talk 68 to 62, workshop 21 to 19)

Four new renderers live in `deck-src/diagrams.js`, in the same style as the kit:
black on white, flat grey blocks, one thin black line, nothing coloured.

| renderer | what it is for | used on |
| --- | --- | --- |
| `timeline` | progress over time: a line, dots, a time above, a caption below, one dot filled black for the beat that matters | talk: the 24 hours of the run. workshop: the whole three hour run sheet |
| `grid` | a breakup of entities into flat labelled boxes | talk: the run config in four numbers. workshop: what people leave with |
| `flow` | a chain of named steps joined by arrows, at a size that fits words (the kit's `equation` sets labels at 44pt, which only suits symbols) | talk: the platform model, the resume chain, the eval before and after |
| `ladder` | an elimination sequence: things tried, struck through, and the one that worked | talk: the two wrong guesses and the arithmetic |

What became a diagram:

- **three slides of hour-by-hour prose → one timeline.** Hour 1, 2, 6, 24, with hour 24
  as the filled dot. Hour 6 is where the chance was missed and the timeline makes that
  visible rather than narrated.
- **four slides of guessing → one ladder.** Guess one struck out with its reason on the
  right, guess two struck out, then "count the tokens it actually saw" in black.
- **two icon rows about eval tooling → one flow.** langfuse → missed it ⇒ promptfoo +
  log metrics.
- **the config list → a four box grid.** 2048, 16, one h100, about six days.
- **the resume chain → a flow.** leg 1 → ckpt → leg 2 → …
- **the platform model** moved from the kit's `equation` to `flow`, because "control
  plane" at 44pt was cramped against its neighbours.
- **workshop: three run sheet slides → one timeline** across the full three hours.
- **workshop: what they leave with → a three box grid.**

Net effect: six fewer slides in the talk, two fewer in the workshop, and every place
where the deck was describing a sequence or a breakdown now shows it instead.

## Reviewer round 3 (dry run call, 7 Oct) is the current deck: 68 slides

See `CHANGELOG-TO-POST.md` for the comment to paste on the proposal. The round 2
notes below are superseded but kept for the trail.

### Still only you can fill

- **The portrait**, slide 1.
- **Guess one and guess two** in the debug journey, if your actual order differed.
- **The ensemble model names.** The transcript has Gemini flash, Nemo and Qwen. The
  deck says "three cheap models from different families" and leaves the names to the
  speaker note, because the transcribed version numbers looked wrong.
- **The 10% figure.** You said 10% of the corpus on the call; the repo's archived run
  works out to 96M tokens. The deck uses 10% and 96M together, which is consistent only
  if the prepared corpus was about 960M tokens at that moment. Worth a glance.

---

## Reviewer round 2, what changed in the talk deck (was 50 to 63)

| they said | what i did |
| --- | --- |
| introduce the platforms | a diagram of laptop / control plane / gpu box / volume, then one row defining modal and runpod by **what they own** (modal's container is disposable, runpod's disk is optional). Both definitions are set up so the later bugs land |
| discontinuous flow | the real fix: the opening now plants "how few bits does a model actually need?" as the reason the 878 runs exist. The second half is a callback to it, not a swerve |
| smoke test | expanded into its own row: compile, one tiny cpu pass, a probe that reports real tokens per second, then pick a budget. Followed by 2 minutes vs 6 days |
| how to train the model, typical architecture | llama 3.2 1b from scratch, plus a shards to loader to model to checkpoint loop, plus the real config: 20b budget, seq 2048, batch 2, accum 8, one h100, six days |
| include the tech stack | pytorch 2.8 / cuda 12.8, modal h100 + volume + gated-tokenizer secret, hf datasets, one training script that checkpoints on sigterm |
| context of model (llm/vision) | one line, then moved on: decoder-only lm, and the failure has nothing to do with language |
| more diagrams, less text | five diagrams now: the platform model, the training loop, the token arithmetic, prepare/volume/train, and the fix. Several text slides were merged into them |
| the debug journey | ten new slides: the symptom, guess one and why it died, guess two and why it died, the one useful question, the arithmetic that cracked it, 96m vs 20b, shard zero, the cache, the race, the fix |

### promptfoo / openrouter / langfuse

None of them appear anywhere in this project, so the deck does not claim otherwise.
The eval slide says what is actually true: a real benchmark scored by an **AST matcher**
over function calls (deterministic, no judge), with validation loss as a cheap screen at
about one percent of the cost. The speaker note tells you to say that plainly and then
open the judge-versus-matcher tradeoff to the room, since most of them do run a judge.
**If you have used promptfoo or langfuse elsewhere, that story belongs on that slide and
only you can write it.**

### Two things only you can fill

- **The portrait**, slide 1.
- **Guess one and guess two** in the debug journey. I reconstructed them from the
  artefacts: "the eval split is noisy", then "it is overfitting", then the token
  arithmetic. The arithmetic and the fix are from `modal_1b.py` and are solid. The two
  guesses are my reconstruction of your reasoning. If you actually thought something
  else first, swap it in, because that part of the talk only works if it is true.

### Length

63 slides. Fast, one idea each, so roughly 35 minutes. If you need 45, the first cuts
are the four habit slides after the smoke-test row, and the "no error, no alert, no
stack trace" slide.

## Both decks still need from you

- **A portrait.** Slide 1 of each has the grey portrait slot the kit leaves for you.
- **The Google Slides upload**, same as before: Drive, open with Google Slides, Share,
  Anyone with the link, role **Commenter**. Talk goes on the gpu-hours submission,
  workshop on the break-a-model one.

## Photos used, and their licences

Credits are already in the speaker notes of the slide each one sits on.

| slide subject | file | licence | credit |
| --- | --- | --- | --- |
| squirrel, "a cache" | cache.jpg | CC BY-SA 3.0 | Grendelkhan, Wikimedia Commons |
| unplugged plug | plug.jpg | CC BY 2.0 | Shixart1985 |
| expired parking meter | meter.jpg | CC0 | MarkBuckawicki |
| twin doors, Lisbon | twins.jpg | CC BY 2.0 | Guldem Ustun |
| empty cardboard box | emptybox.jpg | CC BY-SA 2.0 | Meathead Movers |
| blank notebook | blankexam.jpg | CC0 | Jan Kahanek |
| vital signs monitor | flatline.jpg | public domain | US Navy, PO1 James Stenberg |
| giraffe above the treeline | giraffe.jpg | CC BY-SA 4.0 | Chrisruvy |
| empty lectern | lectern.jpg | CC BY-SA 3.0 DE | Michael Lucan |

Four are share-alike (cache, emptybox, giraffe, lectern). That only matters if the deck
is published rather than presented. The empty box has a small printed mover's logo on it;
swap it if that bothers you.

---

# Deck: We burned 48 GPU hours training on 96 million tokens

18 slides, 16:9, built for a 30 to 40 minute slot with the "Who owns the precision
decision" proposal folded in as the second half. Speaker notes are on every slide.

Files: `rootconf-48-gpu-hours.pptx` (the deck), `rootconf-48-gpu-hours.pdf` (quick read),
`deck-build.js` (the generator, if a rebuild is easier than hand editing).

## Getting the commentable link the editors asked for

1. Go to Google Drive, New, File upload, pick `rootconf-48-gpu-hours.pptx`
2. Right click it, Open with, Google Slides, then File, Save as Google Slides
3. Share, General access, Anyone with the link, and set the role to **Commenter**
4. Copy that link into the Rootconf submission at
   https://hasgeek.com/rootconf/2026-cfp/sub/we-burned-48-gpu-hours-training-on-96-million-toke-SEAc821eD3rNzg7TPfVMm4

Commenter, not Viewer. Viewer looks the same and silently blocks the thing they asked for.

## Structure

| # | Slide | Beat |
| --- | --- | --- |
| 1 | Title | |
| 2 | Not a production service. It behaves like one | setup: 878 runs, 4 platforms, 0 modelling bugs |
| 3 | 01 The run that looked fine | divider |
| 4 | Twenty four hours, in four beats | timeline, 48 H100 hours callout |
| 5 | What actually happened | shard zero, the reload, "nobody called it a cache" |
| 6 | 02 It was never one bug | divider |
| 7 | What each one cost, in GPU hours | chart: 48 lost, 10 lost, 24 avoided |
| 8 | Four more, and none of them were the model | detach, 24h cap, volume name, zero sized volume |
| 9 | The common shape | shared state, lifetime, ownership of results |
| 10 | 03 What I do differently now | divider |
| 11 | Four practices | cheap screens, assert, detach, list the archive |
| 12 | And what each one costs | the tradeoffs the reviewers asked for |
| 13 | 04 The failure with no stack trace | divider |
| 14 | The decision my whole project turns on | four owners, nobody owns all four |
| 15 | What the evidence actually looks like | chart: BFCL across Qwen3 0.6B to 8B |
| 16 | Two failures that look like success | the 100 percent score, the 16 bit break |
| 17 | The part I cannot answer, and you can | the four questions, folded in from the BOF |
| 18 | What to take away | close |

The fold happens at slide 13. Everything before it is the war story, everything after
is the ownership question, and slide 9 is the hinge that makes the join feel intended
rather than bolted on.

## For the pitch call

**Pick a two hour slot, not a 45 minute one.** Wednesday 7 to 9 pm or Saturday 4 to 6 pm.
The editors said they want to hear the pitch and see a demo; 45 minutes with other
people in the queue will not leave room for both.

**Demo option.** `python make_lab_figures.py --results-dir results` regenerates a figure
from the archived JSONL on a laptop with no GPU. It takes seconds and it proves the claim
on slide 18 that every number in the talk traces to an archived result file. That is a
better demo than anything live-training.

**Numbers they may push on.** 48 H100 hours, train 0.95 against validation 5.6, 96 million
tokens, roughly thirty epochs, the 105 cell grid recovered from the archive, and the Qwen3
BFCL figures on slide 15. All of these are in the repository.

## The workshop question

They asked whether "Break a model, then fix it" runs in November or in the January to
March quarter. November is five weeks out and already sits between GOSIM (16 to 17 Oct),
Fifth Elephant (4 to 5 Dec), SciPy India (19 to 20 Dec) and 40C3 (27 to 30 Dec).

Recommend January to March. It is the honest answer about capacity, and the workshop gets
better by then: the OpenFrame silicon will have taped out, so the "and here is what this
costs in hardware" section stops being a projection.

---

# Workshop deck: Break a model, then fix it

10 slides. This is a **delivery plan**, not the workshop content: how the three hours
run, what breaks, and what I do when the room falls apart. Speaker notes on every slide.

Files: `rootconf-workshop-plan.pptx`, `rootconf-workshop-plan.pdf`,
`workshop-deck-build.js`.

Same upload route as the talk deck: Drive, open with Google Slides, Share, Anyone with
the link, role **Commenter**. Paste into
https://hasgeek.com/rootconf/2026-cfp/sub/break-a-model-then-fix-it-a-hands-on-session-on-qu-Qk1AbPfTpgMBL1amQcrNKo

| # | Slide | Purpose |
| --- | --- | --- |
| 1 | Title | frames it as a delivery plan |
| 2 | The arc, in four moves | build it, break it, diagnose, evaluate |
| 3 | The run sheet | minute by minute across three hours |
| 4 | The three things we break on purpose | each one a real bug, with what it looks like and what it is |
| 5 | Three numbers, three diagnoses | 100% dead weights, max 7.47, 1.2% call rate |
| 6 | What it takes to run on a laptop | no GPU, no cloud, no dataset, Colab fallback |
| 7 | Keeping thirty people in the same place | stage branches, tests, pairs, one helper past twenty |
| 8 | What goes wrong, and what I do | four failure modes with the fallback for each |
| 9 | What people leave with | quantiser, diagnosis routine, evaluation pattern |
| 10 | What I would ask for | three hours, cap of thirty, setup script in the joining email, Q1 |

## Things to decide before the pitch

- **Which checkpoint.** The deck says "one small open weight checkpoint, around 150 MB".
  Pythia-70M and SmolLM2-135M both work. Pick one and pin the revision before you build
  the setup script, because the outlier clip demo depends on the checkpoint actually
  having outliers.
- **Whether the three breaks are each really a one line change.** The deck claims that.
  It is true of the dead layer and the outlier clip. Check the third before you say it.
- **Stage branches do not exist yet.** Slide 7 promises a branch per stage. That is the
  single biggest build item and the reason Q1 is the honest answer on timing.

## Build order if they say yes

1. Pin the checkpoint and write `verify.py`
2. Write the reference solution end to end, then cut it back into stages
3. Tag a branch per stage
4. Mirror it into a Colab notebook, cell for cell
5. Dry run with two people who have not seen it, and time every stage
