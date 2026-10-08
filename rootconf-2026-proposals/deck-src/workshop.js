const { Deck, fa6 } = require("/Users/aoxo/.claude/skills/alosh-ppt-style/scripts/kit.js");
const { timeline, grid } = require("/private/tmp/claude-501/-Users-aoxo-vscode-superfloat/06089d0f-33ff-4bb7-8ab6-f005f506b6d4/scratchpad/alosh/diagrams.js");

(async () => {
  const d = new Deck();

  await d.title({
    title: "break a model, then fix it",
    name: "alosh denny",
    role: "workshop plan · rootconf 2026",
    note: "For the editorial review. Say in the first ten seconds that this deck is the delivery plan, not the workshop content.",
  });

  d.statement(["three hours. everyone leaves having ", { t: "broken a model on purpose.", b: true }]);

  d.statement(["this is not the workshop. ", { t: "this is how the room spends the time.", b: true }]);

  await d.iconRows([
    { icon: fa6.FaWrench, label: [{ t: "they build it. ", b: true }, "forty lines, one test that has to go green."] },
    { icon: fa6.FaBug, label: [{ t: "they break it. ", b: true }, "three ways, each a one line change."] },
    { icon: fa6.FaChartColumn, label: [{ t: "they diagnose it. ", b: true }, "from the weights, not from the output."] },
    { icon: fa6.FaListCheck, label: [{ t: "they measure it. ", b: true }, "an eval that catches what accuracy misses."] },
  ], { note: "The order matters. They break things before they learn to diagnose. Diagnosis is boring until you have a corpse." });

  timeline(d, [
    { t: "0:00", label: "every setup prints PASS" },
    { t: "0:15", label: "they write the quantiser" },
    { t: "0:40", label: "they quantise a real model" },
    { t: "1:10", label: "they break it, three ways", mark: true },
    { t: "1:50", label: "they diagnose it from the weights" },
    { t: "2:20", label: "they build the evaluation" },
  ], { foot: ["three hours. ", { t: "the last forty minutes are the overflow", b: true },
              ", if i lose time earlier i demo the eval instead of building it"],
       note: "The whole run sheet on one slide. 0:00 is non-negotiable: it is the difference between starting at 0:00 and starting at 0:25. Ten minute break at 1:00." });

  d.big(["the three breaks are all bugs i shipped."]);

  await d.photo({
    cap: [{ t: "the dead layer.", b: true }, " every weight rounds to zero."],
    img: "/private/tmp/claude-501/-Users-aoxo-vscode-superfloat/06089d0f-33ff-4bb7-8ab6-f005f506b6d4/scratchpad/alosh/img/flatline.jpg",
    note: "Quantise a freshly initialised wide layer at three bits. The network becomes an exactly zero function and no gradient can revive it. Looks like a precision limit, is actually scale placement. Photo: US Navy, Petty Officer 1st Class James Stenberg. Public domain.",
  });

  await d.photo({
    cap: [{ t: "the outlier clip.", b: true }, " one weight in ten thousand did not fit."],
    img: "/private/tmp/claude-501/-Users-aoxo-vscode-superfloat/06089d0f-33ff-4bb7-8ab6-f005f506b6d4/scratchpad/alosh/img/giraffe.jpg",
    note: "Quantise the two projection matrices that no normalisation feeds, with no scale. Validation loss 2.53 to 7.73, at sixteen bits, where the grid step is three in a hundred thousand. Photo credit: Chrisruvy, Wikimedia Commons, CC BY-SA 4.0. Share-alike applies if published.",
  });

  await d.photo({
    cap: [{ t: "the silent one.", b: true }, " it stopped answering and the score went up."],
    img: "/private/tmp/claude-501/-Users-aoxo-vscode-superfloat/06089d0f-33ff-4bb7-8ab6-f005f506b6d4/scratchpad/alosh/img/lectern.jpg",
    note: "Quantise too far, then score it the obvious way. One category rewards declining to answer, so a dead model scores a perfect 100 there. Photo credit: Michael Lucan, Wikimedia Commons, CC BY-SA 3.0 DE. Share-alike applies if published.",
  });

  d.stat([
    { value: "100%", label: "of weights in a wide layer round to zero at three bits" },
    { value: "7.47", label: "the largest weight in a checkpoint everyone assumes is bounded by one" },
    { value: "1.2%", label: "of prompts a broken model still answered, while scoring 100" },
  ], { note: "Three numbers, three diagnoses. Each one is the signal for one of the three breaks." });

  d.statement(["every diagnosis is ", { t: "a histogram or a counter.", b: true }]);

  d.statement(["nobody has to read generated text. ", { t: "that's the part that goes wrong in a room.", b: true }],
    { note: "The pedagogical argument. Reading sample output is subjective and slow. Reading a weight histogram is neither, and it transfers to their own checkpoint on Monday." });

  await d.iconRows([
    { icon: fa6.FaLaptop, label: ["runs on a laptop. python, pytorch, numpy, matplotlib."] },
    { icon: fa6.FaMicrochip, label: [{ t: "no gpu. ", b: true }, "no cloud account. no dataset download."] },
    { icon: fa6.FaStopwatch, label: ["under ten minutes of actual compute, all of it on cpu."] },
    { icon: fa6.FaCodeBranch, label: ["an identical colab notebook, cell for cell, as the fallback."] },
  ], { note: "This removes the single biggest reason hands-on ML workshops fall apart: half the room waiting on a download or a driver." });

  d.statement(["a branch per stage. ", { t: "nobody gets stranded.", b: true }],
    { note: "The repo carries the finished state of each stage. Anyone who falls behind checks out the next one and rejoins with working code they can read." });

  d.statement(["each stage ends on a test. ", { t: "green means move on.", b: true },
    " that replaces me asking thirty people whether they're done."]);

  d.statement(["if i lose time early, the last hour is ", { t: "the thing that gives.", b: true },
    " i demo the eval instead of building it."]);

  grid(d, [
    { label: "a quantiser", sub: "forty lines they wrote and understand" },
    { label: "three checks", sub: "runnable against any checkpoint they have on monday" },
    { label: "an eval pattern", sub: "capability paired with liveness, useful well past quantisation" },
  ], { foot: ["what they walk out with"],
       note: "The third generalises to anything that can fail by going quiet." });

  d.lines([
    [{ t: "three hours", b: true }, ", thirty people, one helper past that"],
    ["the setup script ", { t: "in the joining email", b: true }, ", a week ahead"],
    [{ t: "january", b: true }, ", not november — but november if you need it"],
  ], { size: 24, note: "The asks. November sits between four other commitments, and Q1 puts the workshop after the silicon tapes out. Say I will do November if the schedule needs it." });

  d.closing(["will my checkpoint survive this, and how would i know if it hadn't?"],
    { note: "The test of the three hours is whether they can answer that about their own model without running the quantiser." });

  await d.write("/private/tmp/claude-501/-Users-aoxo-vscode-superfloat/06089d0f-33ff-4bb7-8ab6-f005f506b6d4/scratchpad/alosh/rootconf_workshop_deck.pptx");
  console.log("wrote workshop deck");
})();
