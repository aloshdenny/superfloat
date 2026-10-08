const { Deck, fa6 } = require("/Users/aoxo/.claude/skills/alosh-ppt-style/scripts/kit.js");
const IMG = "/private/tmp/claude-501/-Users-aoxo-vscode-superfloat/06089d0f-33ff-4bb7-8ab6-f005f506b6d4/scratchpad/alosh/img/";
const { timeline, grid, flow, ladder } = require("/private/tmp/claude-501/-Users-aoxo-vscode-superfloat/06089d0f-33ff-4bb7-8ab6-f005f506b6d4/scratchpad/alosh/diagrams.js");
const OUT = "/private/tmp/claude-501/-Users-aoxo-vscode-superfloat/06089d0f-33ff-4bb7-8ab6-f005f506b6d4/scratchpad/alosh/rootconf_talk_deck.pptx";

(async () => {
  const d = new Deck();

  /* ================================================== A. open */
  await d.title({
    title: "we burned 48 gpu hours training on 96 million tokens",
    name: "alosh denny",
    role: "superfloat · rootconf 2026",
    note: "Portrait slot is yours. 20 to 23 minutes of talking, 5 minutes of questions. The dry run was 18, so this version has room.",
  });

  d.statement([{ t: "this is a talk about failure.", b: true }, " mostly mine."]);

  d.big(["but it starts somewhere else entirely."]);

  /* ================================================== B. quantisation first */
  d.statement(["i build a number format, ", { t: "and a chip to run it on.", b: true },
    " before anyone commits silicon, somebody has to answer one question."]);

  d.big(["how few bits does a model actually need?"],
    { note: "This is the origin. The whole devops disaster happens downstream of trying to answer this. Leading here was the reviewers' main structural ask." });

  await d.iconDef({
    icon: fa6.FaCompress,
    term: "quantisation",
    def: ["compressing a model so it uses less memory, less power, and generates tokens faster."],
  });

  d.statement([{ t: "it is rounding.", b: true },
    " every number in the model, written with fewer digits. the model gets smaller and cheaper. the question is what it forgets."],
    { note: "The layman's analogy prakhar asked for. Rounding is the right one because it is literally true, not a metaphor." });

  d.big(["and four people have to agree to do it."]);

  await d.iconRows([
    { icon: fa6.FaMoneyBill, label: [{ t: "cost. ", b: true }, "which gpu tier you are willing to pay for."] },
    { icon: fa6.FaLayerGroup, label: [{ t: "platform. ", b: true }, "runpod, modal, or the cloud you already have."] },
    { icon: fa6.FaChartLine, label: [{ t: "quality. ", b: true }, "how far you compress before it stops being smart."] },
    { icon: fa6.FaMicrochip, label: [{ t: "hardware. ", b: true }, "nvidia, or the faster new silicon with no community yet."] },
  ], { note: "Your four decisions, in your words from the dry run. Groq and AMD are the 'faster, newer, nobody to ask when it breaks' option." });

  d.statement(["in most organisations you can settle two of those. maybe three. ",
    { t: "never four.", b: true }]);

  d.stat([{ value: "8 bits", label: "the floor. compress past it and the model stops being intelligent", size: 96 }],
    { note: "From roughly 800 archived runs." });

  d.statement(["at six bits, a 4b model drops to ", { t: "63", b: true }, ". at eight it is level with sixteen."]);

  d.big(["to find that out, you train a lot of models."]);

  /* ================================================== C. the infrastructure */
  d.statement([{ t: "878", b: true }, " runs. four platforms in a year. none of them on hardware ", { t: "i own.", b: true }]);

  flow(d, [
    { box: { label: "laptop", sub: "where i type" } },
    { op: "→" },
    { box: { label: "control plane", sub: "schedules the work" } },
    { op: "→" },
    { box: { label: "gpu box", sub: "exists for one job" } },
    { op: "+" },
    { box: { label: "volume", sub: "outlives the job" } },
  ], { foot: ["four things. ", { t: "three of them are somebody else's.", b: true }],
       note: "Front-loaded architecture diagram, which Rolland asked for. Two of today's failures come from getting the laptop wrong and the volume wrong." });

  await d.iconRows([
    { icon: fa6.FaBolt, label: [{ t: "modal. ", b: true }, "a python function that runs on an h100. pay by the second. the container dies at 24 hours, always."] },
    { icon: fa6.FaServer, label: [{ t: "runpod. ", b: true }, "a machine you rent by the hour. yours until you stop paying. keeps nothing unless you attach a disk."] },
    { icon: fa6.FaDesktop, label: [{ t: "before those, ", b: true }, "a gpu under my desk and a shared lab machine."] },
  ], { note: "Bharadwaj and Sohham both asked for this: introduce the platforms before the story that depends on them." });

  d.statement(["llama 3.2, 1b parameters, ", { t: "from scratch", b: true },
    ". a decoder-only language model, and none of what follows is about language."]);

  await d.equation([
    { box: { w: 3.0, h: 1.8, label: "shards", caption: "the corpus, in pieces" } },
    { op: "→" },
    { box: { w: 3.0, h: 1.8, label: "loader", caption: "reads what it can see" } },
    { op: "→" },
    { box: { w: 3.0, h: 1.8, label: "1b", caption: "quantisation-aware" } },
    { op: "→" },
    { box: { w: 3.2, h: 1.8, label: "ckpt", caption: "back onto the volume" } },
  ], { foot: ["one loop. ", { t: "the first box is the one that matters today.", b: true }] });

  d.statement(["the corpus does not arrive all at once. ", { t: "it arrives in boxes.", b: true },
    " ten percent, twenty, thirty."],
    { note: "Plant this. It is the entire bug, two minutes before the bug." });

  grid(d, [
    { label: "2048", sub: "sequence length" },
    { label: "16", sub: "batch × accumulation" },
    { label: "1 × h100", sub: "one gpu, rented by the second" },
    { label: "~6 days", sub: "if nothing goes wrong" },
  ], { foot: ["the run, in four numbers"],
       note: "Entity breakup rather than a list. The last box is the one that collides with the 24 hour ceiling on the next slide." });

  await d.iconRows([
    { icon: fa6.FaFileCode, label: [{ t: "pytorch 2.8. ", b: true }, "cuda 12.8."] },
    { icon: fa6.FaBolt, label: [{ t: "modal. ", b: true }, "h100 container, shared volume, a secret for the gated tokenizer."] },
    { icon: fa6.FaCubes, label: [{ t: "huggingface. ", b: true }, "datasets and the tokenizer."] },
    { icon: fa6.FaTerminal, label: [{ t: "one script. ", b: true }, "checkpoints on sigterm, resumes from latest."] },
  ], { note: "The stack. That last line is the only reason any of the later failures were survivable." });

  flow(d, [
    { box: { label: "leg 1", sub: "runs 23 hours" } },
    { op: "→" },
    { box: { label: "ckpt", sub: "saved on sigterm" } },
    { op: "→" },
    { box: { label: "leg 2", sub: "resumes, spawns leg 3" } },
    { op: "→" },
    { box: { label: "…", sub: "until the budget is spent" } },
  ], { foot: ["six days does not fit in a 24 hour container. ", { t: "so it runs in legs.", b: true }],
       note: "The chain is the thing that makes a six day run possible on a platform that kills every container at 24 hours." });

  /* ================================================== D. the run, and the debug */
  d.big(["so here's the run that cost the most."]);

  timeline(d, [
    { t: "hour 1", label: "prepare starts unpacking the corpus" },
    { t: "hour 2", label: "training starts. no errors anywhere" },
    { t: "hour 6", label: ["the loss is climbing. ", { t: "normal, early on", b: true }, ". i leave it"] },
    { t: "hour 24", label: ["it still has not come down. ", { t: "not normal", b: true }], mark: true },
  ], { foot: ["one day, and the only thing that went wrong ", { t: "never raised anything", b: true }],
       note: "Progress over time, as a timeline rather than three sentences. Hour 6 is where I had the chance and missed it; hour 24 is the filled dot." });

  d.stat([
    { value: "0.95", label: "train loss" },
    { value: "5.6", label: "validation loss" },
  ], { arrow: "vs.", note: "Let it sit. Ask the room what that gap means before you say it." });

  d.big(["so i started guessing. in order."]);

  ladder(d, [
    { tag: "guess one", text: "the eval split is noisy", dead: true,
      why: "noise does not climb in a straight line for eighteen hours" },
    { tag: "guess two", text: "it is overfitting", dead: true,
      why: "a 1b model does not overfit a corpus this size in a day" },
    { tag: "guess three", text: "count the tokens it actually saw",
      why: "took thirty seconds. should have been first" },
  ], { foot: ["two hypotheses, then ", { t: "arithmetic", b: true }],
       note: "The elimination ladder replaces four text slides. Walk the two dead rungs quickly; the third is the point and it leads straight into the next slide. The honest line is that guess three cost nothing and should have come first." });

  await d.iconRows([
    { icon: fa6.FaTerminal, label: ["read the container logs by hand"] },
    { icon: fa6.FaMagnifyingGlass, label: ["pasted them into a chat window and argued with it"] },
    { icon: fa6.FaBug, label: ["no dashboard, no trace, ", { t: "no alert.", b: true }, " nothing was watching for this"] },
  ], { note: "Rolland asked how much of the debugging was automated. The honest answer is none of it. Say that plainly; it is the most relatable slide in the deck." });

  d.big(["and then i did the arithmetic."]);

  await d.equation([
    { box: { w: 2.6, h: 1.6, label: "steps", caption: "from the log" } },
    { op: "×" },
    { box: { w: 2.6, h: 1.6, label: "2048", caption: "sequence" } },
    { op: "×" },
    { box: { w: 2.6, h: 1.6, label: "16", caption: "batch × accum" } },
    { op: "=" },
    { box: { w: 3.4, h: 1.6, label: "96m", caption: "tokens. that is all it saw" } },
  ], { foot: ["four numbers already in the log. ", { t: "i had never multiplied them.", b: true }],
       note: "The pivot. Everything before was hypothesis. This was arithmetic and it took thirty seconds." });

  d.stat([
    { value: "10%", label: "of the corpus it trained on" },
    { value: "90%", label: "it never saw at all" },
  ], { arrow: "vs." });

  d.statement(["the first shard, thirty times over. ",
    { t: "that is not pretraining, that is memorisation.", b: true }]);

  d.statement(["which is why it looked ", { t: "brilliant on training data", b: true },
    " and useless on anything real."]);

  d.big(["so why did it only ever see the first box?"]);

  await d.photo({
    cap: [{ t: "a cache.", b: true }, " nobody called it one."],
    img: IMG + "cache.jpg",
    note: "Photo credit: Grendelkhan, Wikimedia Commons, CC BY-SA 3.0. Share-alike applies if this deck is published.",
  });

  await d.equation([
    { box: { w: 3.2, h: 2.0, label: "prepare", caption: "still unpacking boxes" } },
    { op: "→" },
    { box: { w: 3.2, h: 2.0, label: "volume", caption: "one box visible, so far" } },
    { op: "→" },
    { box: { w: 3.2, h: 2.0, label: "train ×2", caption: "looked once, never again" } },
  ], { foot: ["and ", { t: "both", b: true }, " trainers were fighting over the same box"] });

  d.statement(["a container only sees what another container wrote ", { t: "after it asks again.", b: true }]);

  d.statement(["i had never told the trainer ", { t: "the data had finished loading.", b: true }]);

  await d.equation([
    { box: { w: 3.4, h: 1.8, label: "reload", caption: "ask the volume again" } },
    { op: "+" },
    { box: { w: 4.4, h: 1.8, label: "assert", caption: "refuse to start on a partial corpus" } },
    { op: "+" },
    { box: { w: 3.4, h: 1.8, label: "chain", caption: "survive the 24h ceiling" } },
  ], { foot: ["the whole fix is ", { t: "three lines and a guard clause", b: true }] });

  d.stat([{ value: "48", label: "h100 hours. two containers, one full day each.", size: 120 }]);

  /* ================================================== E. common failures */
  d.big(["and none of this is special to me."]);

  await d.photo({
    cap: [{ t: "the foreground launch.", b: true }, " my laptop's dns killed four remote jobs."],
    img: IMG + "plug.jpg",
    note: "Photo credit: Shixart1985, Wikimedia Commons, CC BY 2.0. The platform's default mode ties the remote job's lifetime to the local client.",
  });

  await d.photo({
    cap: [{ t: "a twenty four hour cap.", b: true }, " the run needed six days."],
    img: IMG + "meter.jpg",
    note: "Photo: MarkBuckawicki, Wikimedia Commons, CC0.",
  });

  d.statement(["that cap is also ", { t: "the only reason this cost 48 hours", b: true },
    " and not a fortnight of credits."],
    { note: "Your line from the dry run, and it lands. The constraint that annoyed you is the one that stopped the bleeding." });

  await d.photo({
    cap: [{ t: "two volumes, one name.", b: true }, " ten very bad minutes."],
    img: IMG + "twins.jpg",
    note: "Photo credit: Guldem Ustun, Wikimedia Commons, CC BY 2.0.",
  });

  await d.photo({
    cap: [{ t: "a zero sized volume.", b: true }, " the models expired like they never existed."],
    img: IMG + "emptybox.jpg",
    note: "Photo credit: Meathead Movers, Wikimedia Commons, CC BY-SA 2.0. This one was runpod. No checkpoint, no disk, nothing at the end of it.",
  });

  d.statement(["every one of these is ", { t: "somebody's postmortem this quarter.", b: true },
    " they are not exotic. they are the default."],
    { note: "Sohham's note: frame these as common industry failures, not as rookie mistakes you happened to make." });

  await d.iconRows([
    { icon: fa6.FaDatabase, label: [{ t: "shared state. ", b: true }, "who guarantees the input is complete before a consumer reads it?"] },
    { icon: fa6.FaPlug, label: [{ t: "lifetime. ", b: true }, "what keeps the remote work alive, and for how long?"] },
    { icon: fa6.FaBoxArchive, label: [{ t: "ownership. ", b: true }, "whose job was it to notice?"] },
  ], { note: "Three shapes. Every failure in the talk is one of them. Scale this up: a month-long run at a frontier lab fails the same way, with four more zeros on it." });

  /* ================================================== F. the checklist */
  d.big(["hence, four checks. every run, no exceptions."]);

  await d.iconRows([
    { icon: fa6.FaFlask, label: [{ t: "smoke test. ", b: true }, "one tiny pass end to end, on a cpu, before anything paid starts."] },
    { icon: fa6.FaShieldHalved, label: [{ t: "hash the data. ", b: true }, "if the hash matches, all of it loaded. if it does not, do not start."] },
    { icon: fa6.FaArrowsRotate, label: [{ t: "checkpoint. ", b: true }, "every couple of hours, and push a copy to the hub."] },
    { icon: fa6.FaMoneyBill, label: [{ t: "list the tiers. ", b: true }, "take the cheaper gpu. the same mistake on it is a cheaper mistake."] },
  ], { note: "Your four, in your order. Note that the fourth is about choosing the tier, not about the results archive." });

  d.stat([
    { value: "2 minutes", label: "what the smoke test costs" },
    { value: "6 days", label: "what it is deciding the shape of" },
  ], { arrow: "vs.", note: "Deliberately no rupee or dollar figures anywhere in this deck. Prakhar's note: cloud pricing moves, ratios do not." });

  d.statement(["the consumer asserts. ", { t: "it does not trust.", b: true }]);

  /* ================================================== G. back to the question */
  d.big(["now. back to the question i started with."]);

  d.stat([
    { value: "91.3", label: "bf16, untouched" },
    { value: "90.2", label: "eight bit, no retraining, no calibration set" },
  ], { arrow: "→", note: "Qwen3 8B on the Berkeley Function Calling Leaderboard, scored by AST match." });

  d.statement(["eight bits is free. six bits, at the small end, ", { t: "is not a model any more.", b: true }]);

  d.big(["but how would you even know?"]);

  flow(d, [
    { box: { label: "langfuse", sub: "one llm judging the logs" } },
    { op: "→" },
    { box: { label: "missed it", sub: "never flagged a missing tool call" } },
    { op: "⇒" },
    { box: { label: "promptfoo", sub: "three models, different families" } },
    { op: "+" },
    { box: { label: "log metrics", sub: "counted, not judged" } },
  ], { foot: ["a judge grades the text it was given. ", { t: "silence reads as a perfectly reasonable answer.", b: true }],
       note: "What I tried first was langfuse and langgraph with an llm as a judge over the logs. What I run now is an open ensemble through openrouter and promptfoo, three cheap models from different families so they do not share a blind spot, plus explicit metrics pulled out of the logs. Name the models out loud; check the versions before you put them on a slide." });

  await d.photo({
    cap: [{ t: "three judges gave it a ten.", b: true }, " it had stopped answering."],
    img: IMG + "blankexam.jpg",
    note: "Photo: Jan Kahanek via Unsplash, Wikimedia Commons, CC0. One benchmark category scores a model on correctly declining to call a tool, so a dead model passes every case for free.",
  });

  d.stat([{ value: "1.2%", label: "of prompts it still answered, while scoring a perfect 100 on one category", size: 110 }]);

  d.big(["every capability metric needs a liveness metric."]);

  d.lines([
    ["who signs off when the model changes numerically but not by version?"],
    ["what evidence does your team require?"],
    ["has anyone ever rejected a change on it?"],
  ], { size: 24, note: "Leave real time here. Five minutes of questions after this." });

  d.closing(["so who signs it off at your place?"]);

  await d.write(OUT);
  console.log("wrote talk deck");
})();
