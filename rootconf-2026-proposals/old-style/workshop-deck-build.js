const pptxgen = require("pptxgenjs");
const SKILL = "/Users/aoxo/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/919ca43e-a87e-41d2-a9f4-2727b275519c/0c34452b-0463-4566-92ad-d8d044da88ae/skills/pptx";
const { applyTheme } = require(SKILL + "/scripts/apply_theme.js");

const THEME = {
  name: "Break A Model",
  headFontFace: "Arial",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1A1A1E", lt1: "FFFFFF",
    dk2: "232733", lt2: "F4F5F7",
    accent1: "E8633A", accent2: "3FA796", accent3: "F2B134",
    accent4: "8A93A6", accent5: "C0392B", accent6: "5B7CFA",
    hlink: "5B7CFA", folHlink: "8A93A6",
  },
};
const HEX = THEME.colors;
const MUTED = "6B7280";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Alosh Denny";
pres.title = "Break a model, then fix it";
const C = pres.SchemeColor;

const W = 13.3, H = 7.5, M = 0.65;
const CW = W - 2 * M;

/* ---------------------------------------------------------------- layouts */
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.0, w: CW, h: 2.0,
        fontSize: 44, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: " " } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 4.15, w: 10.6, h: 1.9,
        fontSize: 15, color: HEX.accent4, valign: "top" }, text: " " } },
    { text: { text: "Workshop delivery plan  ·  Rootconf 2026  ·  Alosh Denny", options: { x: M, y: 6.72,
        w: 8, h: 0.35, fontSize: 11, color: HEX.accent4, isTextBox: true, margin: 0 } } },
  ],
});

pres.defineSlideMaster({
  title: "CONTENT_LIGHT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.5, w: CW, h: 1.0,
        fontSize: 32, bold: true, color: C.text1, valign: "middle", align: "left" }, text: " " } },
    { text: { text: "Break a model, then fix it", options: { x: M, y: 6.92, w: 6, h: 0.3,
        fontSize: 10, color: MUTED, isTextBox: true, margin: 0 } } },
  ],
  slideNumber: { x: 12.5, y: 6.92, w: 0.5, h: 0.3, fontSize: 10, color: MUTED, align: "right" },
});

pres.defineSlideMaster({
  title: "CLOSE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.5, w: CW, h: 1.0,
        fontSize: 32, bold: true, color: C.background1, valign: "middle", align: "left" }, text: " " } },
  ],
});

/* ---------------------------------------------------------------- helpers */
let objSeq = 0;
const nm = (s) => `${s}-${++objSeq}`;

function card(slide, { x, y, w, h, fill = C.background2, name = "card" }) {
  const opts = { x, y, w, h, fill: { color: fill }, rectRadius: 0.08, line: { color: fill },
    objectName: nm(name) };
  if (fill !== C.text2) {
    opts.shadow = { type: "outer", color: "9AA0AC", blur: 8, offset: 1, angle: 90, opacity: 0.22 };
  }
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, opts);
}

function textCard(slide, { x, y, w, h, head, body, headColor = C.text1 }) {
  card(slide, { x, y, w, h });
  slide.addText(head, { x: x + 0.34, y: y + 0.24, w: w - 0.68, h: 0.56, fontSize: 17, bold: true,
    color: headColor, isTextBox: true, margin: 0, valign: "top", objectName: nm("tchead") });
  slide.addText(body, { x: x + 0.34, y: y + 0.86, w: w - 0.68, h: h - 1.12, fontSize: 13.5,
    color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: nm("tcbody") });
}

function numCard(slide, { x, y, w, h, n, head, body, accent = C.accent1 }) {
  card(slide, { x, y, w, h });
  slide.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.28, w: 0.46, h: 0.46,
    fill: { color: accent }, line: { color: accent }, objectName: nm("dot") });
  slide.addText(String(n), { x: x + 0.3, y: y + 0.28, w: 0.46, h: 0.46, fontSize: 15, bold: true,
    color: C.background1, align: "center", valign: "middle", isTextBox: true, margin: 0,
    objectName: nm("dotnum") });
  slide.addText(head, { x: x + 0.92, y: y + 0.24, w: w - 1.25, h: 0.56, fontSize: 17, bold: true,
    color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: nm("cardhead") });
  slide.addText(body, { x: x + 0.92, y: y + 0.86, w: w - 1.25, h: h - 1.12, fontSize: 13.5,
    color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: nm("cardbody") });
}

function stat(slide, { x, y, w, value, label, color = C.accent1, size = 46 }) {
  slide.addText(value, { x, y, w, h: 0.8, fontSize: size, bold: true, color,
    isTextBox: true, margin: 0, align: "left", objectName: nm("stat") });
  slide.addText(label, { x, y: y + 0.8, w, h: 0.95, fontSize: 13, color: C.text1,
    isTextBox: true, margin: 0, align: "left", valign: "top", objectName: nm("statlbl") });
}

/* ================================================================== slides */
pres.addSection({ title: "Workshop plan" });
const SEC = { sectionTitle: "Workshop plan" };
let s;

/* 1 ----------------------------------------------------------------------- */
s = pres.addSlide({ masterName: "TITLE_DARK", ...SEC });
s.addText("Break a model, then fix it", { placeholder: "title" });
s.addText("A three hour hands-on session on quantisation, and on proving the thing you quantised still works. This is the delivery plan: how the room spends the time, what runs on a laptop, and what I do when a third of it does not.",
  { placeholder: "body" });
s.addNotes("Opening frame for the editors: this is not the workshop content, it is how I intend to run it. Lead with the fact that everything is CPU only and that there are pre-baked checkpoints at every stage so nobody gets stranded.");

/* 2 ----------------------------------------------------------------------- */
s = pres.addSlide({ masterName: "CONTENT_LIGHT", ...SEC });
s.addText("The arc, in four moves", { placeholder: "title" });
numCard(s, { x: M, y: 1.8, w: 5.85, h: 2.2, n: 1, head: "Build it",
  body: "Participants write the quantiser themselves. The grid, rounding, and a bounded straight through estimator. About forty lines, with a unit test that has to go green before anyone moves on." });
numCard(s, { x: M + 6.15, y: 1.8, w: 5.85, h: 2.2, n: 2, head: "Break it, on purpose",
  body: "Three deliberate failures, one at a time, each a one line change. All three are bugs I actually shipped, and each one fails in a way that does not look like what it is." });
numCard(s, { x: M, y: 4.25, w: 5.85, h: 2.2, n: 3, accent: C.accent2, head: "Diagnose from the weights",
  body: "Not from generated text. Participants learn to read the weight distribution and predict which break they are looking at before the answer is revealed." });
numCard(s, { x: M + 6.15, y: 4.25, w: 5.85, h: 2.2, n: 4, accent: C.accent2, head: "Build the evaluation",
  body: "The last hour builds the harness that catches the failure accuracy alone misses, and runs it against all four models from the session: one good, three broken." });
s.addNotes("The point of the order is that they break things before they learn to diagnose. Diagnosis is boring until you have a corpse.");

/* 3 ----------------------------------------------------------------------- */
s = pres.addSlide({ masterName: "CONTENT_LIGHT", ...SEC });
s.addText("The run sheet", { placeholder: "title" });
const rows = [
  ["0:00", "15 min", "Setup check and the premise. Everyone runs verify.py and sees PASS before anything starts.", MUTED],
  ["0:15", "25 min", "Build the quantiser. One provided test must go green. Nobody proceeds on a red test.", C.text1],
  ["0:40", "20 min", "Quantise a real checkpoint to eight bits. Generate text before and after. It still works.", C.text1],
  ["1:00", "10 min", "Break.", MUTED],
  ["1:10", "40 min", "Break it three ways, roughly twelve minutes each. Every break is a one line change.", HEX.accent5],
  ["1:50", "30 min", "Diagnose from the weight distribution. Predict the break, then reveal.", C.text1],
  ["2:20", "40 min", "Build the evaluation. Run it on all four models. Watch accuracy alone pass a broken one.", HEX.accent2],
];
rows.forEach(([t, dur, body, col], i) => {
  const y = 1.72 + i * 0.68;
  card(s, { x: M, y, w: CW, h: 0.58 });
  s.addText(t, { x: M + 0.3, y: y + 0.07, w: 0.72, h: 0.44, fontSize: 15, bold: true, color: col,
    isTextBox: true, margin: 0, valign: "middle", objectName: nm("rowt") });
  s.addText(dur, { x: M + 1.15, y: y + 0.07, w: 0.85, h: 0.44, fontSize: 12, color: MUTED,
    isTextBox: true, margin: 0, valign: "middle", objectName: nm("rowd") });
  s.addText(body, { x: M + 2.15, y: y + 0.07, w: CW - 2.5, h: 0.44, fontSize: 13.5, color: C.text1,
    isTextBox: true, margin: 0, valign: "middle", objectName: nm("rowb") });
});
s.addNotes("Three hours is the number I would ask for. It compresses to two by dropping the evaluation build and demoing it instead, and that is the version I would run if the slot is shorter. It does not compress below two.");

/* 4 ----------------------------------------------------------------------- */
s = pres.addSlide({ masterName: "CONTENT_LIGHT", ...SEC });
s.addText("The three things we break on purpose", { placeholder: "title" });
textCard(s, { x: M, y: 2.15, w: 3.83, h: 3.45, headColor: C.accent5,
  head: "The dead layer",
  body: "Quantise a freshly initialised wide layer at three bits. Every weight rounds to zero. The network is an exactly zero function and no gradient can revive it.\n\nLooks like: a model that refuses to train.\n\nActually is: the scale sitting in the wrong place, not a precision limit." });
textCard(s, { x: M + 4.08, y: 2.15, w: 3.83, h: 3.45, headColor: C.accent5,
  head: "The outlier clip",
  body: "Quantise the two projection matrices that no normalisation layer feeds, without giving them a scale. Validation loss goes from 2.53 to 7.73.\n\nLooks like: not enough bits.\n\nActually is: a few hundred clipped weights, at sixteen bits, where the grid step is three in a hundred thousand." });
textCard(s, { x: M + 8.17, y: 2.15, w: 3.83, h: 3.45, headColor: C.accent5,
  head: "The silent one",
  body: "Quantise too far, then score it the obvious way. The model stops producing output and the benchmark rewards it for being cautious.\n\nLooks like: a mediocre result.\n\nActually is: a dead model, and a metric that cannot tell the difference." });
s.addNotes("All three are failures I shipped. Say so. The third one is the bridge into the evaluation hour, and it is the one people remember.");

/* 5 ----------------------------------------------------------------------- */
s = pres.addSlide({ masterName: "CONTENT_LIGHT", ...SEC });
s.addText("Three numbers, three diagnoses", { placeholder: "title" });
stat(s, { x: M, y: 1.95, w: 3.6, value: "100%", label: "of weights in a wide layer round to zero at three bits. Count the zeros and the dead layer announces itself in one line." });
stat(s, { x: M + 4.1, y: 1.95, w: 3.6, value: "7.47", color: C.accent5,
  label: "the largest weight in a checkpoint everyone assumes is bounded by one. Take a per channel max and the outlier clip is obvious." });
stat(s, { x: M + 8.2, y: 1.95, w: 3.6, value: "1.2%", color: C.accent6,
  label: "of prompts a broken model still answered, while scoring a perfect 100 on one benchmark category. Count attempts, not just successes." });
card(s, { x: M, y: 4.55, w: CW, h: 1.6, fill: C.text2 });
s.addText("Every diagnosis in this workshop is a histogram or a counter.", { x: M + 0.45, y: 4.78,
  w: CW - 0.9, h: 0.5, fontSize: 22, bold: true, color: C.background1, isTextBox: true, margin: 0,
  objectName: nm("q1") });
s.addText("Nobody needs to interpret generated text, which is the part that normally goes wrong in a room with mixed experience. The signal is a number that is obviously right or obviously not.",
  { x: M + 0.45, y: 5.28, w: CW - 0.9, h: 0.65, fontSize: 14, color: HEX.accent4, isTextBox: true,
    margin: 0, objectName: nm("q1s") });
s.addNotes("This is the pedagogical argument for the workshop. Reading sample output is subjective and slow. Reading a weight histogram is neither, and it transfers to whatever checkpoint they work on at their own job.");

/* 6 ----------------------------------------------------------------------- */
s = pres.addSlide({ masterName: "CONTENT_LIGHT", ...SEC });
s.addText("What it takes to run on a laptop", { placeholder: "title" });
textCard(s, { x: M, y: 1.8, w: 5.85, h: 2.1,
  head: "What participants need",
  body: "Python 3.10 or newer, PyTorch, NumPy, Matplotlib. A setup script pins all of it. Total install is under five minutes on a normal connection." });
textCard(s, { x: M + 6.15, y: 1.8, w: 5.85, h: 2.1, headColor: C.accent2,
  head: "What they do not need",
  body: "No GPU. No cloud account. No vendor licence. No dataset download. The whole session is under ten minutes of actual compute, all of it on CPU." });
textCard(s, { x: M, y: 4.15, w: 5.85, h: 2.1,
  head: "The model",
  body: "One small open weight checkpoint, around 150 MB, pinned by revision and cached by the setup script. Small enough to quantise in seconds, real enough that the failures are the real failures." });
textCard(s, { x: M + 6.15, y: 4.15, w: 5.85, h: 2.1, headColor: C.accent6,
  head: "The fallback",
  body: "An identical Colab notebook, cell for cell. Anyone whose local Python fights them moves there in under a minute and loses nothing." });
s.addNotes("Emphasise the no GPU point to the editors. It removes the single biggest reason hands-on ML workshops fall apart, which is half the room waiting on a download or a driver.");

/* 7 ----------------------------------------------------------------------- */
s = pres.addSlide({ masterName: "CONTENT_LIGHT", ...SEC });
s.addText("Keeping thirty people in the same place", { placeholder: "title" });
numCard(s, { x: M, y: 1.8, w: 5.85, h: 2.2, n: 1, accent: C.accent6, head: "A checkpoint after every stage",
  body: "The repository carries the finished state of each stage as its own branch. Anyone who falls behind checks out the next one and rejoins the room immediately, with working code they can read." });
numCard(s, { x: M + 6.15, y: 1.8, w: 5.85, h: 2.2, n: 2, accent: C.accent6, head: "Tests, not my judgement",
  body: "Each stage ends on a provided test. Green means move on. That replaces me walking round asking thirty people whether they are done." });
numCard(s, { x: M, y: 4.25, w: 5.85, h: 2.2, n: 3, accent: C.accent6, head: "Pairs, by choice",
  body: "I ask people to sit in twos at the start. It halves the number of broken environments and it means the faster half of the room is occupied teaching rather than waiting." });
numCard(s, { x: M + 6.15, y: 4.25, w: 5.85, h: 2.2, n: 4, accent: C.accent6, head: "One helper past twenty",
  body: "Below twenty I can run the floor alone. Above that I would ask for one volunteer who has done the setup in advance, purely for environment triage in the first fifteen minutes." });
s.addNotes("This slide exists because the editors' real question about any workshop is whether it will collapse. Answer it before they ask.");

/* 8 ----------------------------------------------------------------------- */
s = pres.addSlide({ masterName: "CONTENT_LIGHT", ...SEC });
s.addText("What goes wrong, and what I do", { placeholder: "title" });
const risks = [
  ["Venue wifi is slow or absent", "The setup script runs before arrival and caches everything. I bring the environment and the checkpoint on USB sticks as well."],
  ["Someone's Python is unfixable", "Colab notebook, identical cells. Decided in a minute, instead of debugged for half an hour."],
  ["The room runs at two different speeds", "Stage branches. The fast half goes on to an optional fourth break; nobody waits and nobody is stranded."],
  ["I lose time early and the clock slips", "The evaluation hour is the designed overflow. I demo it instead of building it and the session still lands."],
];
risks.forEach(([head, body], i) => {
  const y = 1.78 + i * 1.22;
  card(s, { x: M, y, w: CW, h: 1.05 });
  s.addText(head, { x: M + 0.38, y: y + 0.16, w: 4.3, h: 0.7, fontSize: 15.5, bold: true,
    color: C.accent5, isTextBox: true, margin: 0, valign: "middle", objectName: nm("rhead") });
  s.addText(body, { x: M + 4.85, y: y + 0.16, w: CW - 5.25, h: 0.7, fontSize: 13.5, color: C.text1,
    isTextBox: true, margin: 0, valign: "middle", objectName: nm("rbody") });
});
s.addNotes("Every one of these has happened to me in a room. Say that. The fourth one is the important one: the session is designed so the last hour is the thing that gives, not the middle.");

/* 9 ----------------------------------------------------------------------- */
s = pres.addSlide({ masterName: "CONTENT_LIGHT", ...SEC });
s.addText("What people leave with", { placeholder: "title" });
textCard(s, { x: M, y: 1.85, w: 3.83, h: 2.25, headColor: C.accent2,
  head: "A working quantiser",
  body: "Forty lines they wrote, that they understand, and that runs on any checkpoint they have access to on Monday morning." });
textCard(s, { x: M + 4.08, y: 1.85, w: 3.83, h: 2.25, headColor: C.accent2,
  head: "A diagnosis routine",
  body: "Three checks against a checkpoint: dead weight fraction, per channel maximum against the representable bound, and a liveness counter." });
textCard(s, { x: M + 8.17, y: 1.85, w: 3.83, h: 2.25, headColor: C.accent2,
  head: "An evaluation pattern",
  body: "A capability metric with a liveness metric beside it, which generalises well past quantisation to anything that can fail by going quiet." });
card(s, { x: M, y: 4.45, w: CW, h: 1.5, fill: C.text2 });
s.addText("The test is whether they can answer one question about their own model.", { x: M + 0.45,
  y: 4.67, w: CW - 0.9, h: 0.5, fontSize: 21, bold: true, color: C.background1, isTextBox: true,
  margin: 0, objectName: nm("q2") });
s.addText("Will this checkpoint survive being quantised, and how would I know if it had not? If they can answer that without running the quantiser, the three hours worked.",
  { x: M + 0.45, y: 5.17, w: CW - 0.9, h: 0.6, fontSize: 14, color: HEX.accent4, isTextBox: true,
    margin: 0, objectName: nm("q2s") });
s.addNotes("Close the content section on the outcome, not the syllabus.");

/* 10 ---------------------------------------------------------------------- */
s = pres.addSlide({ masterName: "CLOSE_DARK", ...SEC });
s.addText("What I would ask for", { placeholder: "title" });
const asks = [
  ["Three hours, and a cap of thirty", "Two hours works with the evaluation demoed rather than built. Past thirty I would want one volunteer for setup triage."],
  ["The setup script in the joining email", "A week ahead. This single thing is the difference between starting at 0:00 and starting at 0:25."],
  ["The January to March quarter, not November", "November sits between four other commitments. Q1 also puts it after the silicon tapes out, so the hardware section is measured rather than projected."],
];
asks.forEach(([head, tail], i) => {
  const y = 2.0 + i * 1.42;
  s.addShape(pres.shapes.OVAL, { x: M, y: y + 0.08, w: 0.5, h: 0.5, fill: { color: HEX.accent1 },
    line: { color: HEX.accent1 }, objectName: nm("adot") });
  s.addText(String(i + 1), { x: M, y: y + 0.08, w: 0.5, h: 0.5, fontSize: 16, bold: true,
    color: C.background1, align: "center", valign: "middle", isTextBox: true, margin: 0,
    objectName: nm("anum") });
  s.addText(head, { x: M + 0.8, y, w: 11.2, h: 0.5, fontSize: 20, bold: true, color: C.background1,
    isTextBox: true, margin: 0, valign: "middle", objectName: nm("ahead") });
  s.addText(tail, { x: M + 0.8, y: y + 0.52, w: 11.2, h: 0.62, fontSize: 14, color: HEX.accent4,
    isTextBox: true, margin: 0, valign: "top", objectName: nm("atail") });
});
s.addText("Happy to run it in November if that is what the schedule needs. The answer above is about doing it well, not about whether I will do it.",
  { x: M, y: 6.45, w: CW, h: 0.5, fontSize: 13, italic: true, color: HEX.accent2, isTextBox: true,
    margin: 0, objectName: nm("aclose") });
s.addNotes("End on willingness, not on conditions. The Q1 preference is a recommendation and they should hear it as one.");

/* ---------------------------------------------------------------- write */
(async () => {
  const out = "rootconf-workshop-plan.pptx";
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote " + out);
})();
