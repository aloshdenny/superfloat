// Extra slide renderers in Alosh's style, for things that were text and should not be:
// progress over time, a breakup of entities, a chain of steps, an elimination ladder.
// Same rules as the kit: black on white, no borders, flat grey/black blocks,
// thin black connector lines only. Calibri. Nothing coloured.

const K = require("/Users/aoxo/.claude/skills/alosh-ppt-style/scripts/kit.js");
const { W, H, BLACK, WHITE, MUTE, PH, FONT, SZ_MED, runs } = K;

function blank(d, note) {
  const s = d.pres.addSlide();
  s.background = { color: WHITE };
  if (note) s.addNotes(note);
  return s;
}

// progress over time. items: [{ t:"hour 6", label:"the loss starts climbing", mark?:true }]
// mark = the beat that matters, drawn as a filled black dot instead of a hollow one.
function timeline(d, items, { foot, note } = {}) {
  const s = blank(d, note);
  const x0 = 2.6, x1 = 17.4, y = 5.5;
  const n = items.length;
  const step = n > 1 ? (x1 - x0) / (n - 1) : 0;
  const colW = Math.min(4.6, step * 0.95 || 4.6);

  s.addShape("line", { x: x0, y, w: x1 - x0, h: 0, line: { color: BLACK, width: 1.25 } });

  items.forEach((it, i) => {
    const cx = x0 + i * step;
    const r = it.mark ? 0.21 : 0.15;
    s.addShape("ellipse", {
      x: cx - r, y: y - r, w: r * 2, h: r * 2,
      fill: { color: it.mark ? BLACK : WHITE },
      line: { color: it.mark ? BLACK : MUTE, width: it.mark ? 1 : 1.5 },
      objectName: `tl-dot-${i}`,
    });
    s.addText(it.t, {
      x: cx - colW / 2, y: y - 1.35, w: colW, h: 0.7,
      align: "center", valign: "bottom", fontFace: FONT, fontSize: 24,
      bold: true, color: BLACK, isTextBox: true, margin: 0,
    });
    s.addText(runs(Array.isArray(it.label) ? it.label : [it.label]), {
      x: cx - colW / 2, y: y + 0.5, w: colW, h: 2.1,
      align: "center", valign: "top", fontSize: 15, isTextBox: true,
      lineSpacingMultiple: 1.25, margin: 0,
    });
  });

  if (foot) s.addText(runs(foot), {
    x: 2.0, y: 9.4, w: 16.0, h: 0.7, align: "center", fontSize: 17, isTextBox: true, margin: 0,
  });
  return s;
}

// a breakup of entities. items: [{ label:"2048", sub:"sequence length" }]
// cols defaults to the item count, so 4 items make one row of four, 6 make two rows of three.
function grid(d, items, { cols, foot, note } = {}) {
  const s = blank(d, note);
  const c = cols || items.length;
  const rows = Math.ceil(items.length / c);
  const gap = 0.7;
  const bw = (15.6 - gap * (c - 1)) / c;
  const bh = rows > 1 ? 2.5 : 2.9;
  const totalH = rows * bh + (rows - 1) * gap;
  const y0 = (H - totalH) / 2 - (foot ? 0.5 : 0);
  const x0 = (W - (c * bw + (c - 1) * gap)) / 2;

  items.forEach((it, i) => {
    const r = Math.floor(i / c), cc = i % c;
    const x = x0 + cc * (bw + gap), y = y0 + r * (bh + gap);
    s.addShape("rect", { x, y, w: bw, h: bh, fill: { color: PH }, line: { type: "none" },
      objectName: `grid-${i}` });
    s.addText(runs(Array.isArray(it.label) ? it.label : [{ t: it.label, b: true }]), {
      x: x + 0.35, y: y + 0.45, w: bw - 0.7, h: 0.95,
      align: "center", valign: "middle", fontSize: 27, isTextBox: true, margin: 0,
    });
    if (it.sub) s.addText(it.sub, {
      x: x + 0.35, y: y + 1.45, w: bw - 0.7, h: bh - 1.8,
      align: "center", valign: "top", fontFace: FONT, fontSize: 15, color: MUTE,
      isTextBox: true, lineSpacingMultiple: 1.2, margin: 0,
    });
  });

  if (foot) s.addText(runs(foot), {
    x: 2.0, y: y0 + totalH + 0.55, w: 16.0, h: 0.7,
    align: "center", fontSize: 17, isTextBox: true, margin: 0,
  });
  return s;
}

// a chain of named steps, for word-y flows the kit's equation() would set far too large.
// items: [{ box:{ label, sub } } | { op:"→" }]
function flow(d, items, { foot, note } = {}) {
  const s = blank(d, note);
  const opW = 0.9, gap = 0.35, bh = 1.9, cy = 5.1;
  const boxes = items.filter((i) => i.box).length;
  const avail = 17.0 - (items.length - boxes) * opW - gap * (items.length - 1);
  const bw = Math.min(4.2, avail / boxes);
  const total = items.reduce((a, it) => a + (it.op ? opW : bw), 0) + gap * (items.length - 1);
  let x = (W - total) / 2;

  items.forEach((it, i) => {
    if (it.op) {
      s.addText(it.op, { x, y: cy - 0.5, w: opW, h: 1.0, align: "center", valign: "middle",
        fontFace: FONT, fontSize: 34, color: BLACK, isTextBox: true, margin: 0 });
      x += opW + gap;
      return;
    }
    s.addShape("rect", { x, y: cy - bh / 2, w: bw, h: bh, fill: { color: PH },
      line: { type: "none" }, objectName: `flow-${i}` });
    s.addText(runs(Array.isArray(it.box.label) ? it.box.label : [{ t: it.box.label, b: true }]), {
      x: x + 0.22, y: cy - bh / 2 + 0.3, w: bw - 0.44, h: 1.3,
      align: "center", valign: "middle", fontSize: 22, isTextBox: true,
      lineSpacingMultiple: 1.1, margin: 0 });
    if (it.box.sub) s.addText(it.box.sub, {
      x: x - 0.3, y: cy + bh / 2 + 0.2, w: bw + 0.6, h: 1.1,
      align: "center", valign: "top", fontFace: FONT, fontSize: 14, color: MUTE,
      isTextBox: true, lineSpacingMultiple: 1.2, margin: 0 });
    x += bw + gap;
  });

  if (foot) s.addText(runs(foot), {
    x: 2.0, y: 9.3, w: 16.0, h: 0.7, align: "center", fontSize: 17, isTextBox: true, margin: 0 });
  return s;
}

// an elimination ladder: things tried, struck out, and the one that worked.
// items: [{ tag:"guess one", text:"the eval split is noisy", why:"noise does not climb...", dead:true }]
function ladder(d, items, { foot, note } = {}) {
  const s = blank(d, note);
  const bh = 1.75, gap = 0.45;
  const totalH = items.length * bh + (items.length - 1) * gap;
  const y0 = (H - totalH) / 2 - (foot ? 0.45 : 0);
  const x = 2.4, bw = 15.2;

  items.forEach((it, i) => {
    const y = y0 + i * (bh + gap);
    s.addShape("rect", { x, y, w: bw, h: bh, fill: { color: PH }, line: { type: "none" },
      objectName: `rung-${i}` });
    s.addText(it.tag, { x: x + 0.45, y: y + 0.25, w: 3.2, h: 0.55, fontFace: FONT, fontSize: 15,
      color: MUTE, isTextBox: true, margin: 0 });
    s.addText([{ text: it.text, options: { fontFace: FONT, fontSize: 26, bold: !it.dead,
      color: it.dead ? MUTE : BLACK, strike: !!it.dead } }], {
      x: x + 0.45, y: y + 0.72, w: bw - 6.4, h: 0.8, valign: "middle", isTextBox: true, margin: 0 });
    if (it.why) s.addText(it.why, { x: x + bw - 5.7, y: y + 0.72, w: 5.2, h: 0.8,
      align: "right", valign: "middle", fontFace: FONT, fontSize: 15, color: MUTE,
      isTextBox: true, lineSpacingMultiple: 1.2, margin: 0 });
  });

  if (foot) s.addText(runs(foot), {
    x: 2.0, y: y0 + totalH + 0.5, w: 16.0, h: 0.7, align: "center", fontSize: 17,
    isTextBox: true, margin: 0 });
  return s;
}

module.exports = { timeline, grid, flow, ladder };
