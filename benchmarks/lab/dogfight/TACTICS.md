# Air combat scenario and manoeuvre coverage

A specification for what a dogfight policy must be exposed to. The current
harness (`expert.py`, 9 manoeuvres, 1v1, guns only) covers maybe a tenth of
this. Written as the target, with the gap stated at the end.

Nothing here is classified or hard to find; it is the content of unclassified
fighter-weapons-school syllabi, NATOPS-style manuals, and the open ACM
literature. The hard part is not knowing the manoeuvres, it is enumerating the
*situations* that select between them.

---

## 0. First: the data problem

**Real military telemetry is not obtainable.** Fighter aircraft flight-test and
ACMI data from air forces is classified and ITAR-controlled. There is no legal
route to F-22 or Typhoon dogfight telemetry, and anything claiming to be it
should be treated as either fabricated or a leak you do not want in a
repository.

**What is obtainable, and is genuinely good:**

| source | format | what it is |
| --- | --- | --- |
| DCS World PvP servers and leagues | Tacview ACMI 2.x | thousands of logged human-vs-human engagements, many by pilots with 1000+ hours in type |
| Falcon 4.0 BMS | ACMI / internal logs | long-running community, strong BFM culture |
| Community BFM leagues and ladders | Tacview | structured 1v1 and 2v2 with known entry conditions |
| Air-racing and aerobatic telemetry | IGC, ACMI | useful for high-alpha and energy modelling, not for tactics |

Tacview ACMI is a documented text format: per-object, per-timestamp position,
attitude, velocity, plus events (weapon release, hit, destroyed). It parses
easily and carries exactly the state this harness already computes.

**What human data adds that a scripted expert never will:**

- *Deception.* Feints, baiting an overshoot, pretending to be out of energy.
- *Reaction-time distributions.* Humans have 200-500 ms of decision lag that
  varies with workload. A scripted policy reacts in one frame, always.
- *Situational-awareness failure.* Losing sight, checking the wrong direction,
  target fixation into the ground. These are the dominant real-world losses
  and a scripted opponent never commits them.
- *Suboptimal-but-effective play.* Humans win from positions BFM doctrine says
  are lost, and lose from positions doctrine says are won.
- *Genuine adaptation.* A human changes strategy mid-fight in response to what
  you did three moves ago.

**The catch, and it is a real one.** To behaviour-clone from human telemetry
into a *typed* decision model you must map continuous trajectories onto a
discrete manoeuvre vocabulary, and nobody logs "I am now performing a high
yo-yo". Three options:

1. **Rule-based segmentation.** Classify windows from the geometry the harness
   already computes: a break turn is sustained >6G with aspect opening; a high
   yo-yo is out-of-plane displacement with closure falling and ATA held. Cheap,
   auditable, and wrong at the boundaries.
2. **Hand-label a seed set, train a segmenter.** A few hours of expert labels
   over ACMI windows, then a classifier to label the rest. Best accuracy,
   needs a person who knows BFM.
3. **Abandon discrete labels.** Clone continuous stick and throttle directly.
   Loses the typed-decision structure that makes this a System One study at
   all, and loses interpretability of *why* a quantized policy failed.

Option 1 is the honest starting point and option 2 is where it should end up.
Worth being explicit that a rule-based segmenter labelling human data will
inherit the rule author's idea of what the human was doing, which caps how much
the human data can teach beyond the rules.

---

## 1. The state that selects the manoeuvre

Every decision below keys on some subset of these. A policy that cannot observe
a variable cannot learn the manoeuvre that depends on it.

**Relative geometry**
- **Range** and **range rate (closure, Vc)**
- **ATA** (antenna train angle) — my nose off the line of sight to him
- **AA** (aspect angle) — where I am relative to his tail; 0 is his six, 180 head-on
- **HCA** (heading crossing angle) — angle between our two velocity vectors
- **Turn circle** — his turn radius and whether I am inside or outside it
- **Control zone** — the cone behind him where I can stay without overshooting
- **3/9 line** — his wingline; crossing it forward is a positional overshoot

**Energy**
- **Specific energy** Es = h + V²/2g, and **specific excess power** Ps
- **Corner velocity** — speed giving max instantaneous turn rate at structural G
- **Sustained vs instantaneous turn rate**
- **Turn radius vs turn rate** — the one-circle/two-circle decision
- The **E-M (doghouse) diagram** for both aircraft: this is what decides which
  fight you want

**Self state**
- AOA, G available, altitude above the hard deck, fuel, weapons remaining
- Gun rounds, missile count and type, seeker status, chaff/flare remaining

**Sensor and awareness**
- Tally (visual on the bandit), sorted (which bandit is which), spiked (being
  locked), mud (SAM), naked (no warning)
- Radar mode, gimbal limits, Doppler notch, lock status
- **Loss of tally is its own state** and drives an entire branch of behaviour

---

## 2. Defensive — you are the prey

### 2.1 The situations

| # | situation | key discriminator |
| --- | --- | --- |
| D1 | Undetected attacker, no tally | you do not know yet |
| D2 | Spiked at range, no tally | RWR only |
| D3 | Missile launch detected, BVR | kinematic defeat available |
| D4 | Attacker entering visual range, high aspect | merge is coming |
| D5 | Attacker in your control zone, out of gun range | he has position, not a shot |
| D6 | Attacker in the saddle, gun range | immediate |
| D7 | Attacker overshooting (flight path) | opportunity |
| D8 | Attacker overshooting (3/9 line) | major opportunity, roles may reverse |
| D9 | Attacker high six | he has energy to trade |
| D10 | Attacker low six | he is trading energy to climb |
| D11 | Attacker in lag | he is managing, not shooting |
| D12 | Attacker in lead | he is shooting or about to |
| D13 | Low energy, attacker fast | worst case |
| D14 | Low altitude, no room below | the deck is the second enemy |
| D15 | Multiple attackers, mutual support intact | you cannot defeat both |
| D16 | Multiple attackers, one engaged one supporting | defeat the engaged one, watch the other |
| D17 | Lost tally while defensive | the highest-risk state in air combat |
| D18 | Out of weapons / out of fuel | disengagement is the only win |

### 2.2 The manoeuvres

**Denying the gun solution**

- **Break turn.** Maximum instantaneous G into the attacker. Purpose: generate
  angle-off faster than he can track, force flight-path overshoot. Cost:
  enormous energy bleed; you are slow afterwards. The canonical answer to D6.
- **Jink.** Short, hard, *aperiodic* displacements. Purpose: defeat a tracking
  gun solution by making the required lead unpredictable. Not a turn — the aim
  is unpredictability, not angles.
- **Guns jink / out-of-plane jink.** The same, but deliberately across the
  attacker's plane of motion, which is harder to track than in-plane.
- **Last-ditch manoeuvre.** Inside ~2000 ft with the pipper tracking: a violent
  out-of-plane displacement accepting total energy loss. You are betting the
  fight on making him miss now.

**Forcing the overshoot**

- **Hard turn.** Sustained, below max G, holding corner velocity. Preserves
  energy while still generating angles. The answer to D5 where a break would
  be wasteful.
- **Defensive spiral.** Descending, turning, at high AOA. Trades altitude for
  sustained turn rate, forces the attacker to either overshoot or follow into a
  rate fight at low altitude. Requires altitude (D14 forbids it).
- **Barrel roll defence.** High-G barrel roll across his flight path, killing
  your forward velocity and forcing him out in front. The classic answer to D7.
- **Break into the vertical / pitchback.** Trading speed for a fast heading
  change when you have energy in hand.

**Reversing the roles**

- **Flat scissors.** Repeated horizontal reversals after an overshoot, each one
  trying to end up behind. Won by the aircraft with the lower speed and smaller
  turn radius. Deliberately slow — you are fighting to be *slower* than him.
- **Rolling scissors (vertical).** After an overshoot in the vertical: a series
  of barrel rolls, each pilot trying to stay behind. Won by the better nose
  authority at low speed.
- **Lag reversal.** As he goes to lag, reverse into him to deny the reposition.
- **Split-S.** Half roll inverted, then pull through. 180 degrees of heading for
  a large altitude cost. Fast, committing, and fatal near the deck.
- **Pitchback / Immelmann.** Up and over: half loop then roll upright. 180
  degrees of heading, trading speed for altitude. The opposite trade to split-S.

**Separating**

- **Extend / drag.** Unload to near 0G, full power, run. Purpose: rebuild energy
  or leave. Works only if you have a speed advantage or he is committed
  elsewhere.
- **Unload and accelerate.** The general principle: at 0G the aircraft
  accelerates fastest. Almost every energy-rebuilding manoeuvre is this plus a
  direction.
- **Notch / beam.** Put the threat on your 3/9 line to enter the Doppler notch
  of his radar. Defeats pulse-Doppler tracking and the missiles that depend on
  it. Pairs with chaff.
- **Terrain masking.** Put ground between you and his sensor. Requires terrain
  in the simulation, which most harnesses (including this one) lack.
- **Chaff / flare with manoeuvre.** Countermeasures without a simultaneous
  manoeuvre are close to useless; the manoeuvre is what makes the decoy work.

**Post-stall (thrust-vectoring aircraft only)**

- **Herbst manoeuvre / J-turn.** Pitch to very high AOA, yaw the nose around
  using thrust vectoring, recover pointing the other way. Extremely fast
  heading reversal at the cost of essentially all energy.
- **Pugachev's Cobra.** Rapid pitch past vertical while maintaining flight
  path, then recover. A deceleration device; forces an overshoot from a
  pursuer with no other option. Tactically narrow but real.
- **Kulbit.** Near-zero-radius backflip. Same family, same cost.

The honest framing: these buy one shot opportunity in exchange for becoming a
low-energy target. Against a single opponent with no support they can win.
Against a wingman they are suicide, which is exactly the kind of context
dependence a policy has to learn rather than be told.

---

## 3. Offensive — you are the predator

The three problems, which must be solved *simultaneously*: **closure control,
angle-off control, range control**. Every offensive manoeuvre below is a way of
trading one against the others.

### 3.1 The prey you will meet

| # | prey type | what it does | what it demands of you |
| --- | --- | --- | --- |
| P1 | Unaware, non-manoeuvring | nothing | a clean conversion; do not overshoot from greed |
| P2 | Unaware, then breaks late | sudden max-G break | you must already be in lag |
| P3 | Predictable defender | single-direction break, holds it | trivially exploitable with a yo-yo |
| P4 | Energy fighter | extends, uses vertical, refuses angles | deny separation; two-circle rate fight |
| P5 | Angles fighter | one-circle, forces scissors, fights slow | do NOT go slow with him; stay fast, use vertical |
| P6 | Superior-turning aircraft | wants a radius fight | force a rate fight, or leave |
| P7 | Superior-energy aircraft | wants a rate fight | force a radius fight, get slow and nose-on |
| P8 | Thrust-vectoring / post-stall | will reverse violently at low speed | never arrive slow and close |
| P9 | Defender with a wingman | drags you into his support | check six before committing; this is how you die |
| P10 | Bait | deliberately defensive to draw you in | the hardest to detect and the reason for mutual support |

### 3.2 The manoeuvres

**Pursuit curves — the fundamental three**

- **Pure pursuit.** Nose on him. Increases closure, generates no angles. Used
  for a gun snapshot or when closing from behind a non-manoeuvring target.
- **Lead pursuit.** Nose ahead of him. Closes range *and* angles fastest, which
  is what you need for a tracking gun solution, and which overshoots fastest if
  you misjudge.
- **Lag pursuit.** Nose behind him. Controls closure, preserves turning room,
  prevents overshoot. The manoeuvre that wins fights and looks like doing
  nothing.

The skill is the *transition* between them as range and angle-off change. A
policy that picks one and holds it is the naive-pursuit baseline in
`tournament.py`, and it loses.

**Repositioning out of plane**

- **High yo-yo.** Climb out of the turn plane, trading speed for position,
  reducing closure and preserving turning room, then come back down behind him.
  The answer to "I am overtaking and about to overshoot".
- **Low yo-yo.** Drop below the turn plane to cut the corner and accelerate,
  closing range. The answer to "he is extending and I am falling behind".
- **High-speed yo-yo.** A larger high yo-yo when badly overtaking.
- **Barrel roll attack / lag displacement roll.** A roll around his flight path
  to bleed closure and drop back into the control zone without crossing his
  3/9. The precise tool for an imminent flight-path overshoot.
- **Vertical reposition / pitchback.** Use the vertical to change your turn
  circle relative to his.

**Choosing the fight**

- **Two-circle (rate) fight.** Nose-to-tail at the merge; both aircraft turn
  the same direction around separate circles. Won by the higher *sustained turn
  rate*. Pick this if you out-rate him.
- **One-circle (radius) fight.** Nose-to-nose; both turn into each other around
  a shared circle. Won by the smaller *turn radius*, which usually means the
  slower aircraft. Pick this if you out-radius him, and never pick it against a
  post-stall aircraft.
- **Lead turn.** Begin the turn *before* the merge to arrive with angles
  already gained. The single highest-value skill at the merge.

**Converting and killing**

- **Conversion turn.** From a perch or bounce, the turn that puts you in the
  control zone with acceptable closure.
- **Gun tracking shot** vs **gun snapshot**: tracking requires holding the
  pipper through his manoeuvre; a snapshot accepts a brief crossing solution.
- **Missile employment.** Rmax, Rmin, Rne (no-escape), seeker field of view,
  off-boresight capability, and the target's energy state. A shot outside the
  no-escape zone is a shot that trades your position for nothing.
- **Extend and re-attack.** Disengage deliberately, rebuild energy, come back.
  Requires knowing you can get out, which requires knowing where his friends
  are.

**Managing the overshoot you caused**

- **Flight-path overshoot.** You cross his flight path but stay behind his 3/9.
  Recoverable; go to lag, reposition.
- **3/9 line overshoot.** You end up in front of his wingline. Roles reverse;
  you are now defensive and should be flying section 2.
- Distinguishing these two in the state is essential and most naive policies
  cannot, which is why they die after a good attack.

---

## 4. Neutral — the merge

The merge is where most fights are decided, and it is a *decision*, not a
manoeuvre: which fight do you want, given the two E-M diagrams?

| entry | choice | wins if |
| --- | --- | --- |
| Head-on, level | one-circle vs two-circle | one-circle: smaller radius. two-circle: higher rate |
| Head-on, offset | lead turn early or late | early gains angles, late preserves energy |
| Vertical merge | go up or go down | up if you have energy and thrust-to-weight; down if you need speed |
| Offset with altitude split | high man drops, low man climbs | the high man holds the advantage |
| Blow-through | accept no fight, separate | you are outnumbered or low on fuel/weapons |

Manoeuvres at the merge: lead turn, slice-back, oblique turn, vertical pitch,
nose-to-nose or nose-to-tail selection, and the deliberate decision *not* to
engage.

---

## 5. Multi-ship — where real air combat lives

1v1 is a training construct. Everything below is the actual problem, and none
of it is in the current harness.

### 5.1 Formations and doctrines

- **Welded wing.** Wingman glued to lead. Simple, poor offensively.
- **Fighting wing.** Wingman in a cone behind lead. Better, still one shooter.
- **Double attack.** Both aircraft shoot; roles swap dynamically.
- **Loose deuce.** Engaged/supporting roles swap by opportunity. The most
  capable and the most demanding of situational awareness.
- **Combat spread / line abreast.** Lateral separation 1-1.5 nm for mutual
  visual and radar coverage.
- **Wall, box, wedge, ladder.** Multi-ship presentations at the BVR merge.

### 5.2 Offensive multi-ship tactics

- **Bracket.** Split laterally to force the enemy to commit to one; whoever he
  does not commit to gets the kill.
- **Pincer.** Converging attacks from two axes, timed so his defensive turn
  against one exposes him to the other.
- **Sandwich.** Let him commit to one, then trap him between two.
- **Drag and bag / hook.** One aircraft drags him into a chase; the other,
  unseen, converts.
- **Grinder / wheel.** Sequential attacks with each attacker disengaging to
  reposition while the next engages. Never gives the defender a free moment.
- **Shooter-cover.** One engages, one watches for the third party.

### 5.3 Defensive multi-ship

- **Defensive split.** Two defenders turn away from each other, forcing each
  attacker to pick one and breaking their mutual support.
- **Cross turn / in-place turn / tactical turn / check turn.** Formation
  manoeuvres to change heading while preserving mutual support.
- **Beam one, drag the other.** Against a pincer: defeat one geometrically
  while extending from the second.
- **Sandwich escape.** Vertical, usually, because the horizontal is covered.

### 5.4 The principle underneath all of it

**Mutual support.** Every multi-ship tactic exists to preserve it or to destroy
the enemy's. The characteristic failure mode is a pilot who wins his 1v1 and is
shot by the wingman he forgot. A policy trained only on 1v1 will learn exactly
this failure and will look excellent in 1v1 evaluation while doing it.

### 5.5 Deconfliction

Altitude blocks, 3/9 awareness between friendlies, weapons-release constraints
with a friendly in the line of fire. A multi-ship policy that is not scored on
fratricide will commit it.

---

## 6. Beyond visual range — the part before the dogfight

Included because a policy that arrives at the merge already defensive has
usually lost, and BVR is where that is decided.

- **Intercept geometry.** Cut-off, pure pursuit, lead collision.
- **Crank.** Turn to the edge of radar gimbal limits after launch to reduce
  closure while keeping the lock.
- **Notch.** Beam the threat to enter his Doppler notch.
- **Drag.** Turn cold and run, extending the missile's flight.
- **Pump.** Deliberate in-and-out to drag the enemy into a trap.
- **F-pole / A-pole.** Separation at missile impact, and range at which you can
  turn away. These set the entire BVR timeline.
- **Skate, short skate, banzai.** Launch-and-leave doctrines at different
  aggression levels.
- **Grinder / wall** as BVR presentations.

---

## 7. Scenario generation matrix

The useful artefact. Coverage is the product of these axes, sampled rather than
enumerated (the full cross product is ~10^6 cells).

| axis | values |
| --- | --- |
| Numbers | 1v1, 1v2, 2v1, 2v2, 2v4, 4v4 |
| Entry | BVR intercept, head-on merge, offset merge, offensive perch, defensive perch, line abreast neutral, bounce (no tally) |
| Entry range | 500 ft, 1 nm, 3 nm, 10 nm, 20 nm, 40 nm |
| Entry aspect | 0, 45, 90, 135, 180 deg |
| Energy state | both fast, both slow, attacker fast/defender slow, attacker slow/defender fast |
| Altitude band | deck (500 ft AGL), low (5k), medium (15k), high (35k), split |
| Aircraft matchup | symmetric; rate-superior vs radius-superior; thrust-vectoring vs conventional; 4th vs 5th gen |
| Weapons | guns only; guns + IR; guns + IR + radar; bingo weapons |
| Sensors | full; no radar; no RWR; no tally at entry; degraded |
| Terrain | none; rolling; mountainous; canyon |
| Hard deck | 3000 ft; 1000 ft; none (allows deck-level fighting) |
| Fuel | full; bingo; emergency |

**Stratify the sampling.** Uniform sampling over this space spends most of its
budget on boring cells. Weight toward: the merge, the first 20 seconds after a
bounce, overshoot resolution, and anything within 1500 ft — that is where
decisions are dense and where quantization error compounds fastest.

---

## 8. What the current harness actually covers

Stated plainly so nobody mistakes the taxonomy above for the implementation.

| | covered | missing |
| --- | --- | --- |
| Numbers | 1v1 | everything multi-ship, which is most of section 5 |
| Manoeuvres | 9: pursue, lag, lead, break L/R, extend, high/low yo-yo, recover | scissors (both), split-S, Immelmann, barrel-roll defence, jink, defensive spiral, notch, all post-stall |
| Weapons | guns, fixed cone, fixed envelope | missiles, seeker FOV, no-escape zones, countermeasures |
| Sensors | perfect omniscient state | radar, gimbal limits, Doppler notch, tally/loss of tally |
| Entry | 4 setups, symmetric, medium altitude | BVR, bounce, energy asymmetry, altitude splits |
| Terrain | none, flat hard deck | masking, canyons, the entire low-altitude fight |
| Aircraft | F-16 vs F-16 | asymmetric matchups, thrust vectoring (JSBSim ships an f22 with pitch vectoring) |
| Opponent | one scripted rule-based expert | human data, self-play, opponent diversity |

The gap that most affects the *quantization* result specifically is manoeuvre
vocabulary and opponent diversity. With 9 manoeuvres and one opponent, 71% of
expert decisions are a single class, which compresses the dynamic range the
precision study can resolve. A richer vocabulary and a league of opponents
would spread the decision distribution and make the SF8-to-SF4 band, currently
unresolved on win rate, separable.

---

## 9. Suggested build order

1. **Manoeuvre vocabulary to ~20**, adding scissors, split-S, Immelmann,
   barrel-roll defence, jink, defensive spiral. Cheap, and directly improves
   the precision study by flattening the class distribution.
2. **Opponent league** — the five scripted policies plus frozen past
   checkpoints. Removes the single-opponent overfit.
3. **Tacview ingest + rule-based segmenter.** Human data as *opponents* first,
   which needs no labels at all, and only then as cloning targets.
4. **2v1 and 2v2** with an explicit mutual-support term in scoring. This is the
   biggest single jump in realism and the one where a quantized policy is most
   likely to fail differently from fp32, because it adds a memory and
   attention-allocation problem rather than just a control problem.
5. **Missiles and sensors**, which turn a pure control problem into a partially
   observed one.
6. **Terrain.**

Steps 1-2 are days. Step 3 is weeks and depends on a BFM-literate labeller.
Steps 4-6 are a different project.
