const pptxgen = require("pptxgenjs");
const SKILL = "/Users/aoxo/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/919ca43e-a87e-41d2-a9f4-2727b275519c/0c34452b-0463-4566-92ad-d8d044da88ae/skills/pptx";
const { applyTheme } = require(SKILL + "/scripts/apply_theme.js");

const THEME = {
  name: "Burned GPU Hours",
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
const MUTED = "6B7280";   // small text on light backgrounds (contrast)

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13.3 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Alosh Denny";
pres.title = "We Burned 48 GPU Hours Training On 96 Million Tokens";
const C = pres.SchemeColor;

const W = 13.3, H = 7.5, M = 0.65;
const CW = W - 2 * M;                   // content width 12.0

/* ---------------------------------------------------------------- layouts */
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.0, w: CW, h: 2.0,
        fontSize: 44, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: " " } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 4.15, w: 10.6, h: 1.9,
        fontSize: 15, color: HEX.accent4, valign: "top" }, text: " " } },
    { text: { text: "Rootconf 2026  ·  Alosh Denny", options: { x: M, y: 6.72, w: 8, h: 0.35,
        fontSize: 11, color: HEX.accent4, isTextBox: true, margin: 0 } } },
  ],
});

pres.defineSlideMaster({
  title: "SECTION_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "body", type: "body", x: M, y: 2.5, w: 3, h: 1.0,
        fontSize: 64, bold: true, color: HEX.accent1, valign: "bottom", align: "left" }, text: " " } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 3.6, w: CW, h: 1.2,
        fontSize: 38, bold: true, color: C.background1, valign: "top", align: "left" }, text: " " } },
  ],
});

pres.defineSlideMaster({
  title: "CONTENT_LIGHT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.52, w: CW, h: 0.95,
        fontSize: 32, bold: true, color: C.text1, valign: "middle", align: "left" }, text: " " } },
    { text: { text: "github.com/aloshdenny/superfloat", options: { x: M, y: 6.92, w: 6, h: 0.3,
        fontSize: 10, color: MUTED, isTextBox: true, margin: 0 } } },
  ],
  slideNumber: { x: 12.5, y: 6.92, w: 0.5, h: 0.3, fontSize: 10, color: MUTED, align: "right" },
});

pres.defineSlideMaster({
  title: "CHART_LIGHT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.52, w: CW, h: 0.95,
        fontSize: 32, bold: true, color: C.text1, valign: "middle", align: "left" }, text: " " } },
    { placeholder: { options: { name: "chart", type: "chart", x: M, y: 1.65, w: 8.1, h: 4.75,
        color: C.text1 }, text: " " } },
    { text: { text: "github.com/aloshdenny/superfloat", options: { x: M, y: 6.92, w: 6, h: 0.3,
        fontSize: 10, color: MUTED, isTextBox: true, margin: 0 } } },
  ],
  slideNumber: { x: 12.5, y: 6.92, w: 0.5, h: 0.3, fontSize: 10, color: MUTED, align: "right" },
});

pres.defineSlideMaster({
  title: "CLOSE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.62, w: CW, h: 0.95,
        fontSize: 32, bold: true, color: C.background1, valign: "middle", align: "left" }, text: " " } },
  ],
});

/* ---------------------------------------------------------------- helpers */
let objSeq = 0;
const nm = (s) => `${s}-${++objSeq}`;

function card(slide, { x, y, w, h, fill = C.background2, name = "card" }) {
  const opts = {
    x, y, w, h, fill: { color: fill }, rectRadius: 0.08, line: { color: fill },
    objectName: nm(name),
  };
  if (fill !== C.text2) {
    opts.shadow = { type: "outer", color: "9AA0AC", blur: 8, offset: 1, angle: 90, opacity: 0.22 };
  }
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, opts);
}

// big number + label block
function stat(slide, { x, y, w, value, label, color = C.accent1, size = 50 }) {
  slide.addText(value, { x, y, w, h: 0.85, fontSize: size, bold: true, color,
    isTextBox: true, margin: 0, align: "left", objectName: nm("stat") });
  slide.addText(label, { x, y: y + 0.85, w, h: 0.8, fontSize: 13, color: C.text1,
    isTextBox: true, margin: 0, align: "left", valign: "top", objectName: nm("statlbl") });
}

// numbered card with heading + body
function numCard(slide, { x, y, w, h, n, head, body, accent = C.accent1 }) {
  card(slide, { x, y, w, h });
  slide.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.28, w: 0.46, h: 0.46,
    fill: { color: accent }, line: { color: accent }, objectName: nm("dot") });
  slide.addText(String(n), { x: x + 0.3, y: y + 0.28, w: 0.46, h: 0.46, fontSize: 15, bold: true,
    color: C.background1, align: "center", valign: "middle", isTextBox: true, margin: 0,
    objectName: nm("dotnum") });
  slide.addText(head, { x: x + 0.92, y: y + 0.26, w: w - 1.25, h: 0.5, fontSize: 17, bold: true,
    color: C.text1, isTextBox: true, margin: 0, valign: "middle", objectName: nm("cardhead") });
  slide.addText(body, { x: x + 0.92, y: y + 0.76, w: w - 1.25, h: h - 1.0, fontSize: 13.5,
    color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: nm("cardbody") });
}

// plain card with heading + body, no number
function textCard(slide, { x, y, w, h, head, body, headColor = C.text1 }) {
  card(slide, { x, y, w, h });
  slide.addText(head, { x: x + 0.34, y: y + 0.24, w: w - 0.68, h: 0.56, fontSize: 17, bold: true,
    color: headColor, isTextBox: true, margin: 0, valign: "top", objectName: nm("tchead") });
  slide.addText(body, { x: x + 0.34, y: y + 0.86, w: w - 0.68, h: h - 1.12, fontSize: 13.5,
    color: C.text1, isTextBox: true, margin: 0, valign: "top", objectName: nm("tcbody") });
}

const CHART_BASE = {
  showLegend: false, showTitle: false,
  catAxisLabelColor: HEX.dk1, valAxisLabelColor: HEX.dk1,
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  valGridLine: { color: "DCDFE4", size: 1 },
  catGridLine: { style: "none" },
  showValue: true, dataLabelPosition: "outEnd",
  dataLabelColor: HEX.dk1, dataLabelFontSize: 12, dataLabelFontFace: "+mn-lt",
  dataLabelFontBold: true,
};

/* ================================================================ SECTIONS */
/* -- 1: title ------------------------------------------------------------ */
pres.addSection({ title: "Opening" });

let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Opening" });
s.addText("We burned 48 GPU hours training on 96 million tokens", { placeholder: "title" });
s.addText(
  "And the loss curve told us for hours. What a year of running a thousand-run experiment grid on other people's GPUs teaches you about failure, and about the question underneath every one of them: who owns the decision.",
  { placeholder: "body" });
s.addNotes("Hook: the number in the title is real and it is small enough to be embarrassing rather than abstract. Say up front that none of the expensive failures were modelling bugs. The talk has two halves: what broke, and who was supposed to be responsible for it.");

/* -- setup --------------------------------------------------------------- */
s = pres.addSlide({ masterName: "CONTENT_LIGHT", sectionTitle: "Opening" });
s.addText("Not a production service. It behaves like one", { placeholder: "title" });
stat(s, { x: M, y: 2.1, w: 3.6, value: "878", label: "archived training runs behind one research project" });
stat(s, { x: M + 4.1, y: 2.1, w: 3.6, value: "4", label: "compute platforms in a year: a desk GPU, a lab machine, RunPod, Modal", color: C.accent6 });
stat(s, { x: M + 8.2, y: 2.1, w: 3.6, value: "0", label: "of the expensive failures were modelling bugs", color: C.accent2 });
textCard(s, { x: M, y: 4.3, w: CW, h: 1.7,
  head: "Why the operational character is the same",
  body: "A job that dies at hour twenty is expensive. A job that succeeds while computing the wrong thing is worse, because nothing pages you. Small enough that every failure is legible, large enough that each one costs real money." });
s.addNotes("Set the frame. I am not claiming production scale. I am claiming that batch work on rented hardware has the same failure surface, and that the research setting makes each failure unusually easy to see end to end.");

/* -- 2: the incident ----------------------------------------------------- */
pres.addSection({ title: "The incident" });

s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "The incident" });
s.addText("01", { placeholder: "body" });
s.addText("The run that looked fine", { placeholder: "title" });
s.addNotes("Transition into the single most expensive failure.");

s = pres.addSlide({ masterName: "CONTENT_LIGHT", sectionTitle: "The incident" });
s.addText("Twenty four hours, in four beats", { placeholder: "title" });
const beats = [
  ["Hour 0", "The data preparation job starts writing shards to a shared network volume.", MUTED],
  ["Hour 0", "Two training containers launch against the same volume. Neither waits.", MUTED],
  ["Hour 6", "Validation loss starts climbing. I read it as evaluation noise and leave it.", C.accent3],
  ["Hour 24", "Run ends. Train loss 0.95. Validation loss 5.6.", C.accent5],
];
beats.forEach(([t, b, col], i) => {
  const y = 1.9 + i * 1.12;
  card(s, { x: M, y, w: 8.5, h: 0.95 });
  s.addText(t, { x: M + 0.3, y: y + 0.18, w: 1.1, h: 0.6, fontSize: 15, bold: true, color: col,
    isTextBox: true, margin: 0, valign: "middle", objectName: nm("beatt") });
  s.addText(b, { x: M + 1.5, y: y + 0.18, w: 6.8, h: 0.6, fontSize: 13.5, color: C.text1,
    isTextBox: true, margin: 0, valign: "middle", objectName: nm("beatb") });
});
card(s, { x: M + 8.95, y: 1.9, w: 3.05, h: 4.31, fill: C.text2 });
s.addText("48", { x: M + 9.25, y: 2.85, w: 2.45, h: 1.1, fontSize: 64, bold: true, color: C.accent1,
  isTextBox: true, margin: 0, align: "center", objectName: nm("big48") });
s.addText("H100 hours, gone", { x: M + 9.25, y: 4.0, w: 2.45, h: 0.5, fontSize: 15, bold: true,
  color: C.background1, isTextBox: true, margin: 0, align: "center", objectName: nm("big48l") });
s.addText("Two containers, one full day each", { x: M + 9.15, y: 4.52, w: 2.65, h: 0.8,
  fontSize: 12, color: HEX.accent4, isTextBox: true, margin: 0, align: "center", objectName: nm("big48s") });
s.addNotes("Walk the timeline slowly. The point of hour 6 is that the diagnostic signal was present and I dismissed it. Do not skip past that, it is the most relatable part of the story.");

s = pres.addSlide({ masterName: "CONTENT_LIGHT", sectionTitle: "The incident" });
s.addText("What actually happened", { placeholder: "title" });
textCard(s, { x: M, y: 1.85, w: 5.85, h: 2.3, headColor: C.accent5,
  head: "Both containers trained on shard zero alone",
  body: "96 million tokens instead of the full corpus. About thirty epochs of the same data in twenty four hours. Train loss 0.95 against validation loss 5.6 is not a model that learned. It is a model that memorised." });
textCard(s, { x: M + 6.15, y: 1.85, w: 5.85, h: 2.3,
  head: "The cause, in one sentence",
  body: "A container only sees files another container has committed after it explicitly reloads the volume. Neither training container ever reloaded, so both saw the corpus exactly as it looked the instant they started." });
card(s, { x: M, y: 4.45, w: CW, h: 1.65, fill: C.text2 });
s.addText("Nobody called it a cache. It is a cache.", { x: M + 0.45, y: 4.68, w: CW - 0.9, h: 0.55,
  fontSize: 22, bold: true, color: C.background1, isTextBox: true, margin: 0, objectName: nm("q1") });
s.addText("A producer and a consumer sharing a network volume is a distributed systems problem, not a filesystem operation. I had been treating it as a directory.",
  { x: M + 0.45, y: 5.25, w: CW - 0.9, h: 0.7, fontSize: 14, color: HEX.accent4, isTextBox: true,
    margin: 0, objectName: nm("q1s") });
s.addNotes("The reframe is the lesson. Everyone in the room has written code that assumed a shared mount behaves like a local directory.");

/* -- 3: the pattern ------------------------------------------------------ */
pres.addSection({ title: "The pattern" });

s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "The pattern" });
s.addText("02", { placeholder: "body" });
s.addText("It was never one bug", { placeholder: "title" });
s.addNotes("Pivot from the single incident to the class of failures.");

s = pres.addSlide({ masterName: "CHART_LIGHT", sectionTitle: "The pattern" });
s.addText("What each one cost, in GPU hours", { placeholder: "title" });
s.addChart(pres.charts.BAR, [{
  name: "GPU hours",
  labels: ["Shard zero, no\nvolume reload", "Zero sized volume\non an older host", "Re-run avoided by\nreading the archive"],
  values: [48, 10, 24],
}], {
  placeholder: "chart",
  barDir: "col",
  chartColors: [HEX.accent5, HEX.accent1, HEX.accent2],
  varyColors: true,
  valAxisMaxVal: 60,
  barGapWidthPct: 110,
  dataLabelFormatCode: "0",
  ...CHART_BASE,
});
textCard(s, { x: M + 8.45, y: 1.9, w: 3.55, h: 2.2, headColor: C.accent2,
  head: "The green one is a saving",
  body: "An experiment that already existed, sitting in a results archive while the repository said it had never been run. One download instead of a GPU day." });
textCard(s, { x: M + 8.45, y: 4.3, w: 3.55, h: 2.1,
  head: "The cheapest to fix",
  body: "Was also the one with no error, no alert and no stack trace. It was a question of who owned the result, not of what the code did." });
s.addNotes("Keep this chart on screen while making the point that the third bar is categorically different: it is a failure of ownership rather than of code, and it is the bridge into the second half of the talk.");

s = pres.addSlide({ masterName: "CONTENT_LIGHT", sectionTitle: "The pattern" });
s.addText("Four more, and none of them were the model", { placeholder: "title" });
numCard(s, { x: M, y: 1.85, w: 5.85, h: 2.1, n: 1, head: "The foreground launch",
  body: "The platform's default mode ties the remote job's life to my laptop's connection. A local DNS blip tore down four concurrently running jobs. They survived only because the scripts checkpoint." });
numCard(s, { x: M + 6.15, y: 1.85, w: 5.85, h: 2.1, n: 2, head: "A twenty four hour function cap",
  body: "Discovered while planning a six day run. Fix is a resume chain: run under a timeout, checkpoint on the termination signal, respawn on the timeout exit code." });
numCard(s, { x: M, y: 4.2, w: 5.85, h: 2.1, n: 3, head: "Two workspaces, one volume name",
  body: "The same volume name existed in two workspaces with different contents. A listing from the wrong one looked exactly like catastrophic data loss, for about ten very bad minutes." });
numCard(s, { x: M + 6.15, y: 4.2, w: 5.85, h: 2.1, n: 4, head: "A zero sized persistent volume",
  body: "A pod created with no persistent storage attached on an earlier platform. Ten hours of work, nothing on disk at the end of it." });
s.addNotes("Move briskly. One sentence of colour each. The payload is the next slide.");

s = pres.addSlide({ masterName: "CONTENT_LIGHT", sectionTitle: "The pattern" });
s.addText("The common shape", { placeholder: "title" });
s.addText("Every expensive failure was a question about what was responsible for something, answered wrong.",
  { x: M, y: 1.75, w: CW, h: 0.7, fontSize: 21, bold: true, color: C.text1, isTextBox: true,
    margin: 0, objectName: nm("shape") });
textCard(s, { x: M, y: 2.75, w: 3.83, h: 2.3, headColor: C.accent1,
  head: "Shared state",
  body: "Who guarantees the input is complete before a consumer reads it? I assumed the filesystem did. It does not." });
textCard(s, { x: M + 4.08, y: 2.75, w: 3.83, h: 2.3, headColor: C.accent1,
  head: "Lifetime",
  body: "What keeps the remote work alive, and for how long? I assumed my terminal was incidental. It was load bearing." });
textCard(s, { x: M + 8.17, y: 2.75, w: 3.83, h: 2.3, headColor: C.accent1,
  head: "Ownership of results",
  body: "Who is responsible for a finished result reaching the place people read? Nobody was, so it did not." });
s.addText("Not one of them needed a better model. All three needed somebody to have decided who was responsible.",
  { x: M, y: 5.45, w: CW, h: 0.6, fontSize: 15, italic: true, color: MUTED, isTextBox: true,
    margin: 0, objectName: nm("shapesub") });
s.addNotes("This is the hinge of the talk. Land it clearly, then move into practices.");

/* -- 4: what changed ----------------------------------------------------- */
pres.addSection({ title: "What changed" });

s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "What changed" });
s.addText("03", { placeholder: "body" });
s.addText("What I do differently now", { placeholder: "title" });
s.addNotes("Short practices section. Keep it concrete and fast.");

s = pres.addSlide({ masterName: "CONTENT_LIGHT", sectionTitle: "What changed" });
s.addText("Four practices", { placeholder: "title" });
numCard(s, { x: M, y: 1.85, w: 5.85, h: 2.1, n: 1, accent: C.accent2, head: "Cheap screens before spend",
  body: "Compile it, run one tiny pass end to end on CPU, measure real throughput, then choose a budget. This has caught more of my expensive mistakes than reading the code ever did." });
numCard(s, { x: M + 6.15, y: 1.85, w: 5.85, h: 2.1, n: 2, accent: C.accent2, head: "Consumers assert, they do not trust",
  body: "A training job now refuses to start unless the token count it can see matches the token count it was promised. The check is three lines and would have saved 48 hours." });
numCard(s, { x: M, y: 4.2, w: 5.85, h: 2.1, n: 3, accent: C.accent2, head: "Detach by default, checkpoint always",
  body: "Assume the control plane will disconnect, because it will. Every failure in this talk was survivable in proportion to how recently the job had checkpointed." });
numCard(s, { x: M + 6.15, y: 4.2, w: 5.85, h: 2.1, n: 4, accent: C.accent2, head: "List the archive before you run",
  body: "The cheapest experiment is the one you already ran. Checking costs one API call and has twice now returned a result I was about to pay to recompute." });
s.addNotes("If time is short this is the slide to compress. The audience can read it.");

s = pres.addSlide({ masterName: "CONTENT_LIGHT", sectionTitle: "What changed" });
s.addText("And what each one costs", { placeholder: "title" });
textCard(s, { x: M, y: 1.85, w: 3.83, h: 3.75, headColor: C.accent6,
  head: "A runner, or one off scripts",
  body: "I built a generic runner that passes paths through environment variables. It made the scripts portable across four hosts with no edits.\n\nThe cost is a subprocess boundary that hides real errors, and two of the failures in this talk were harder to diagnose because of it." });
textCard(s, { x: M + 4.08, y: 1.85, w: 3.83, h: 3.75, headColor: C.accent6,
  head: "Checkpoint often, or run fast",
  body: "Frequent checkpointing costs throughput on every single run, forever.\n\nIt has also saved every run that was interrupted, which by now is most of them. I stopped treating this as a tuning parameter." });
textCard(s, { x: M + 8.17, y: 1.85, w: 3.83, h: 3.75, headColor: C.accent6,
  head: "Fix the script, or fix the runner",
  body: "Several failures came from scripts making assumptions about their own working directory.\n\nI fixed them at the runner level rather than editing scripts with an archived result history, knowingly leaving the root cause in place to keep old results comparable." });
s.addNotes("Tradeoffs slide, which the reviewers specifically asked for. Be honest that the third one is a compromise I am not fully happy with.");

/* -- 5: who decides ------------------------------------------------------ */
pres.addSection({ title: "Who decides" });

s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Who decides" });
s.addText("04", { placeholder: "body" });
s.addText("The failure with no stack trace", { placeholder: "title" });
s.addNotes("Second half. The folded in material starts here.");

s = pres.addSlide({ masterName: "CONTENT_LIGHT", sectionTitle: "Who decides" });
s.addText("The decision my whole project turns on", { placeholder: "title" });
s.addText("Quantisation: storing every number in the model with fewer digits so it runs on cheaper hardware.",
  { x: M, y: 1.72, w: CW, h: 0.55, fontSize: 17, color: HEX.accent4, isTextBox: true, margin: 0,
    objectName: nm("qdef") });
textCard(s, { x: M, y: 2.5, w: 2.81, h: 2.0, headColor: C.accent1, head: "A cost decision",
  body: "Memory, throughput, the hardware you have to buy." });
textCard(s, { x: M + 3.0625, y: 2.5, w: 2.81, h: 2.0, headColor: C.accent1, head: "A platform decision",
  body: "The serving stack has to expose it, and own the rollback." });
textCard(s, { x: M + 6.125, y: 2.5, w: 2.81, h: 2.0, headColor: C.accent1, head: "A quality decision",
  body: "The served model changes numerically, with no version bump." });
textCard(s, { x: M + 9.1875, y: 2.5, w: 2.81, h: 2.0, headColor: C.accent1, head: "A hardware decision",
  body: "Weights and activations want different widths. Silicon has to agree." });
card(s, { x: M, y: 4.8, w: CW, h: 1.5, fill: C.text2 });
s.addText("In most organisations, nobody owns all four.", { x: M + 0.45, y: 5.02, w: CW - 0.9, h: 0.5,
  fontSize: 22, bold: true, color: C.background1, isTextBox: true, margin: 0, objectName: nm("q2") });
s.addText("So it either never happens, or it happens without a validation story. Same failure mode as the result that sat in an archive nobody owned.",
  { x: M + 0.45, y: 5.52, w: CW - 0.9, h: 0.6, fontSize: 14, color: HEX.accent4, isTextBox: true,
    margin: 0, objectName: nm("q2s") });
s.addNotes("Tie it explicitly back to the archive failure. Same shape: four plausible owners, no actual owner, so the thing does not get done or gets done unverified.");

s = pres.addSlide({ masterName: "CHART_LIGHT", sectionTitle: "Who decides" });
s.addText("What the evidence actually looks like", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "bf16 (untouched)", labels: ["Qwen3 0.6B", "Qwen3 1.7B", "Qwen3 4B", "Qwen3 8B"], values: [81.9, 88.7, 89.6, 91.3] },
  { name: "8 bit fixed point", labels: ["Qwen3 0.6B", "Qwen3 1.7B", "Qwen3 4B", "Qwen3 8B"], values: [78.7, 88.6, 90.4, 90.2] },
  { name: "6 bit fixed point", labels: ["Qwen3 0.6B", "Qwen3 1.7B", "Qwen3 4B", "Qwen3 8B"], values: [27.3, 78.9, 62.9, 88.0] },
], {
  placeholder: "chart",
  barDir: "col",
  chartColors: [HEX.accent4, HEX.accent2, HEX.accent5],
  valAxisMaxVal: 100,
  barGapWidthPct: 60,
  dataLabelFormatCode: "0.0",
  ...CHART_BASE,
  showLegend: true, legendPos: "b", legendColor: HEX.dk1, legendFontSize: 12, legendFontFace: "+mn-lt",
  dataLabelFontSize: 9,
});
textCard(s, { x: M + 8.45, y: 1.9, w: 3.55, h: 2.2, headColor: C.accent2,
  head: "Eight bits is free",
  body: "Function calling accuracy holds within about a point of the original at every size, with no retraining and no calibration set." });
textCard(s, { x: M + 8.45, y: 4.3, w: 3.55, h: 1.75, headColor: C.accent5,
  head: "Six bits is not one answer",
  body: "Free at 8B. Fatal at 0.6B. One model would have produced confident, wrong guidance in either direction." });
s.addNotes("Berkeley Function Calling Leaderboard, scored by AST match. The reason to show this is not the result, it is that this is the shape of evidence somebody should be required to produce before a precision change ships.");

s = pres.addSlide({ masterName: "CONTENT_LIGHT", sectionTitle: "Who decides" });
s.addText("Two failures that look like success", { placeholder: "title" });
textCard(s, { x: M, y: 1.85, w: 5.85, h: 2.45, headColor: C.accent5,
  head: "It scored 100 percent because it stopped answering",
  body: "One benchmark category scores a model on correctly declining to call a tool. A model that had stopped calling tools entirely, on 98.8 percent of prompts, scored a perfect 100 there. Averaged across categories it would have been published at 28.7 percent: mediocre, not dead." });
textCard(s, { x: M + 6.15, y: 1.85, w: 5.85, h: 2.45, headColor: C.accent5,
  head: "Broken at sixteen bits",
  body: "Two projection matrices in each block are fed by no normalisation layer. Quantised without a scale, validation loss went from 2.53 to 7.73, at sixteen bits, where the grid step is three in a hundred thousand. A few hundred outlier weights, clipped." });
card(s, { x: M, y: 4.6, w: CW, h: 1.6, fill: C.text2 });
s.addText("Every capability metric needs a liveness metric beside it.", { x: M + 0.45, y: 4.82, w: CW - 0.9,
  h: 0.5, fontSize: 22, bold: true, color: C.background1, isTextBox: true, margin: 0, objectName: nm("q3") });
s.addText("For tool calling that is the call rate. For your system it is whatever number separates correctly did nothing from did nothing.",
  { x: M + 0.45, y: 5.32, w: CW - 0.9, h: 0.6, fontSize: 14, color: HEX.accent4, isTextBox: true,
    margin: 0, objectName: nm("q3s") });
s.addNotes("These two are the concrete answer to what evidence a reviewer should demand. Neither is visible in the headline number a vendor or a team would report.");

s = pres.addSlide({ masterName: "CONTENT_LIGHT", sectionTitle: "Who decides" });
s.addText("The part I cannot answer, and you can", { placeholder: "title" });
const qs = [
  ["Who signs off", "when the served model changes numerically but not by version number?"],
  ["What evidence", "does your team require, and has anyone ever actually rejected a change on it?"],
  ["Should the platform", "expose precision as a knob at all, or is that a decision that belongs upstream?"],
  ["How would you find out", "if a quantised model got worse in a way your current metrics do not show?"],
];
qs.forEach(([head, tail], i) => {
  const y = 1.72 + i * 1.12;
  card(s, { x: M, y, w: CW, h: 0.95 });
  s.addText([
    { text: head + "  ", options: { bold: true, color: C.accent1 } },
    { text: tail, options: { color: C.text1 } },
  ], { x: M + 0.4, y: y + 0.2, w: CW - 0.8, h: 0.6, fontSize: 16, isTextBox: true, margin: 0,
    valign: "middle", objectName: nm("q" + i) });
});
s.addText("I have measured the technical side to death. How other teams organise this decision, I have almost no visibility into.",
  { x: M, y: 6.32, w: CW, h: 0.45, fontSize: 14, italic: true, color: MUTED, isTextBox: true,
    margin: 0, objectName: nm("qclose") });
s.addNotes("Leave real time here. This is the folded in Birds of Feather material and the room is the expert, not me. If discussion runs long, cut the practices slide earlier rather than this one.");

/* -- 6: close ------------------------------------------------------------ */
pres.addSection({ title: "Close" });

s = pres.addSlide({ masterName: "CLOSE_DARK", sectionTitle: "Close" });
s.addText("What to take away", { placeholder: "title" });
const takeaways = [
  ["Shared state between jobs is a distributed systems problem", "not a filesystem operation. Make the consumer assert, not trust."],
  ["A cheap screen catches more than careful review", "a CPU smoke test and a throughput probe, every time, before anything paid starts."],
  ["The failures that cost the most had no owner", "not no alert. Ask who owns the decision before you ask what the code did."],
];
takeaways.forEach(([head, tail], i) => {
  const y = 2.05 + i * 1.42;
  s.addShape(pres.shapes.OVAL, { x: M, y: y + 0.08, w: 0.5, h: 0.5, fill: { color: HEX.accent1 },
    line: { color: HEX.accent1 }, objectName: nm("tdot") });
  s.addText(String(i + 1), { x: M, y: y + 0.08, w: 0.5, h: 0.5, fontSize: 16, bold: true,
    color: C.background1, align: "center", valign: "middle", isTextBox: true, margin: 0,
    objectName: nm("tnum") });
  s.addText(head, { x: M + 0.8, y, w: 11.2, h: 0.5, fontSize: 20, bold: true, color: C.background1,
    isTextBox: true, margin: 0, valign: "middle", objectName: nm("thead") });
  s.addText(tail, { x: M + 0.8, y: y + 0.52, w: 11.2, h: 0.5, fontSize: 14, color: HEX.accent4,
    isTextBox: true, margin: 0, valign: "top", objectName: nm("ttail") });
});
s.addText("github.com/aloshdenny/superfloat   ·   every number in this talk traces to an archived result file",
  { x: M, y: 6.5, w: CW, h: 0.5, fontSize: 13, color: HEX.accent2, isTextBox: true, margin: 0,
    objectName: nm("close") });
s.addNotes("Close on the ownership line, then open questions. Mention that the repository has the raw results and that both figure scripts run on a laptop with no GPU.");

/* ---------------------------------------------------------------- write */
(async () => {
  const out = "rootconf-48-gpu-hours.pptx";
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote " + out);
})();
