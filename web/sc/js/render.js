/*! Single-Crystal Diffraction · © 2026 Miloš Dubajić · MIT · https://github.com/dubajicmilos/xrays-on-detector */

/**
 * Canvas drawing: the spot pattern and the powder trace.
 *
 * Everything is drawn in one pass over the reflections, with the view
 * transform kept on the object so a hover or a click can be turned back into
 * an hkl without searching in screen space.
 */

import {
  formatHkl,
  niceStep,
  sampleLut,
  spotRadius,
  stretch,
} from "./display.js";
import { pairUp, rotateInPlane } from "./overlay.js";
import { powderProfile } from "./powder.js";

const DIM = "#8794b0";
const LINE = "#2a3145";
const ACCENT = "#6f9ee0";
const GOOD = "#8fe3c0";
const BG = "#090b12";
const LABEL = "#c3cce0";
// the second crystal, in the overlay and on the powder plot alike
const SECOND = "#e2a06a";

/**
 * The smallest stretched value whose colour stands out from the background:
 * the first entry of the map with a channel at 48 or more.
 */
function visibleFrom(lut) {
  for (let i = 0; i < 256; i++)
    if (Math.max(lut[3 * i], lut[3 * i + 1], lut[3 * i + 2]) >= 48)
      return i / 255;
  return 1;
}

/** "rgb(r,g,b)" from sampleLut, at the given alpha. */
const withAlpha = (rgb, a) => rgb.replace("rgb(", "rgba(").replace(")", `,${a})`);

/**
 * The shortest step between neighbouring reflections in the plane, in the
 * units of result.x and result.y: the shorter of the two basis vectors and
 * their sum and difference, projected the way _drawKey projects them.
 * Infinity when the result carries no basis.
 */
function latticeStep(result, structure) {
  if (!structure || !result.g1 || !result.g2) return Infinity;
  const B = structure.B;
  const onPlane = (g) => {
    const v = [0, 1, 2].map(
      (r) => B[r][0] * g[0] + B[r][1] * g[1] + B[r][2] * g[2],
    );
    const along = (axis) => v[0] * axis[0] + v[1] * axis[1] + v[2] * axis[2];
    return [along(result.xAxis), along(result.yAxis)];
  };
  const a = onPlane(result.g1);
  const b = onPlane(result.g2);
  return Math.min(
    Math.hypot(a[0], a[1]),
    Math.hypot(b[0], b[1]),
    Math.hypot(a[0] + b[0], a[1] + b[1]),
    Math.hypot(a[0] - b[0], a[1] - b[1]),
  );
}

export class PatternView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.scale = 1; // pixels per 1/Angstrom
    this.cx = 0;
    this.cy = 0;
    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;
    this.spots = null; // {x, y, r, i, set} in device pixels, for hit testing
    // In the overlay, which reflection of the other crystal each one is paired
    // with: [A index -> {b, dq}, B index -> {a, dq}].
    this.pairs = null;
    this.powder = null; // the last powder plot's geometry, for hit testing
    this._traces = new WeakMap(); // powder result -> {n, trace}
  }

  /** Fit the canvas backing store to its box, honouring devicePixelRatio. */
  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const box = this.canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(box.width * dpr));
    const h = Math.max(1, Math.round(box.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.dpr = dpr;
    return { w, h };
  }

  clear() {
    const { w, h } = this.resize();
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, w, h);
    return { w, h };
  }

  message(text, colour = DIM) {
    const { w, h } = this.clear();
    const ctx = this.ctx;
    ctx.fillStyle = colour;
    ctx.font = `${13 * this.dpr}px "Segoe UI", system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const [i, line] of String(text).split("\n").entries())
      ctx.fillText(line, w / 2, h / 2 + (i - 0.5) * 20 * this.dpr);
  }

  /**
   * Reciprocal-space coordinates -> device pixels.
   *
   * `scale` is the fit, recomputed from the extent of the data on every draw;
   * `zoom` multiplies it and `pan` is a straight offset in device pixels. Pan
   * deliberately sits outside the zoom so a drag moves the picture by exactly
   * the distance the pointer moved, whatever the magnification.
   */
  toScreen(x, y) {
    return [
      this.cx + x * this.scale * this.zoom + this.panX,
      // y up on screen, as reciprocal space is drawn everywhere else
      this.cy - y * this.scale * this.zoom + this.panY,
    ];
  }

  /** Magnify about a point in client coordinates, keeping it under the cursor. */
  zoomAt(clientX, clientY, factor, lo = 0.25, hi = 60) {
    const before = this.zoom;
    const next = Math.min(hi, Math.max(lo, before * factor));
    if (next === before) return false;
    const box = this.canvas.getBoundingClientRect();
    const px = (clientX - box.left) * this.dpr;
    const py = (clientY - box.top) * this.dpr;
    // Solve for the pan that leaves the world point under the cursor fixed.
    const k = next / before;
    this.panX = px - this.cx - (px - this.cx - this.panX) * k;
    this.panY = py - this.cy - (py - this.cy - this.panY) * k;
    this.zoom = next;
    return true;
  }

  /** Magnify about the middle of the pane, for the +/- buttons. */
  zoomCentre(factor) {
    const box = this.canvas.getBoundingClientRect();
    return this.zoomAt(
      box.left + box.width / 2,
      box.top + box.height / 2,
      factor,
    );
  }

  panBy(dx, dy) {
    this.panX += dx * this.dpr;
    this.panY += dy * this.dpr;
  }

  /** Back to the fit that the data extent implies. */
  resetView() {
    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;
  }

  /**
   * Draw a section or a SAED pattern.
   *
   * `structure` is needed only to turn the two zone basis vectors into screen
   * directions for the axes key. `rotation` turns the whole pattern by that
   * many degrees anticlockwise about the origin, which is the crystal turning
   * about the zone axis; the axes key turns with it.
   */
  drawSpots(result, structure, opts) {
    const {
      gain = 1,
      log = true,
      lut,
      spotScale = 9,
      labels = true,
      labelThreshold = 0.35,
      showRings = false,
      rotation = 0,
    } = opts;
    const { w, h } = this.clear();
    const ctx = this.ctx;
    this.spots = null;
    this.pairs = null;
    this.powder = null;

    if (!result || result.count === 0) {
      this.message(
        "no reflections here\ntry a lower d min" +
          (result && result.zoneRadii ? "" : ", or a different layer"),
        "#f0a05a",
      );
      return;
    }

    const value = stretch(result.intensity, { gain, log });
    // Where each reflection is drawn: its plane coordinates, turned by the
    // rotation when there is one. The fit below is made on the unturned
    // pattern, so the pattern keeps its size as it is rotated; a d min cuts
    // it to a disc, which fits the same way at any angle.
    const xy = rotation ? rotateInPlane(result, rotation) : result;
    // Frame the spots that can be seen. A dark-ended colour map draws the
    // weakest ones near-black, and fitting to every spot above 1e-3 left a
    // large cell's visible pattern as a small cluster mid-pane.
    const seen = visibleFrom(lut);
    let lim = 0,
      limSeen = 0;
    for (let i = 0; i < result.count; i++) {
      if (value[i] <= 1e-3) continue;
      const r = Math.max(Math.abs(result.x[i]), Math.abs(result.y[i]));
      lim = Math.max(lim, r);
      if (value[i] >= seen) limSeen = Math.max(limSeen, r);
    }
    if (lim <= 0) {
      this.message(
        "every reflection in this section is extinct\nraise the contrast to check",
        "#f0a05a",
      );
      return;
    }
    if (limSeen > 0) lim = limSeen;

    const dpr = this.dpr;
    this.cx = w / 2;
    this.cy = h / 2;
    this.scale = this._fitScale(
      [{ ...result, value }],
      lim,
      spotScale,
      seen,
      opts.avoid || [],
    );
    // Spots no wider than about half the lattice step on screen, so a dense
    // lattice stays a lattice instead of merging into one bright disc.
    const cap = Math.max(
      1.1 * dpr,
      0.42 * latticeStep(result, structure) * this.scale * this.zoom,
    );
    const spots = [];

    if (showRings && result.zoneRadii) {
      ctx.strokeStyle = "#243049";
      ctx.lineWidth = 1 * this.dpr;
      for (const r of result.zoneRadii) {
        if (!r) continue;
        const [px, py] = this.toScreen(0, 0);
        ctx.beginPath();
        ctx.arc(px, py, r * this.scale * this.zoom, 0, 2 * Math.PI);
        ctx.stroke();
      }
    }

    // Additive blending so overlapping spots build up rather than punch holes,
    // which is how the Game of Diffraction paints its detector too. Each spot
    // is its colour-map colour with a soft edge and a core that whitens with
    // intensity, the way a spot looks on an exposure.
    ctx.globalCompositeOperation = "lighter";
    for (let i = 0; i < result.count; i++) {
      const v = value[i];
      if (v <= 1e-3) continue;
      const [px, py] = this.toScreen(xy.x[i], xy.y[i]);
      const r = Math.min(spotRadius(v, spotScale) * dpr, cap);
      const R = 1.6 * r;
      if (px < -R || py < -R || px > w + R || py > h + R) continue;
      const colour = sampleLut(lut, v);
      const glow = ctx.createRadialGradient(px, py, 0, px, py, R);
      glow.addColorStop(0, `rgba(255,255,255,${(0.1 + 0.8 * v * v).toFixed(3)})`);
      glow.addColorStop(0.25, colour);
      glow.addColorStop(0.6, withAlpha(colour, 0.55));
      glow.addColorStop(1, withAlpha(colour, 0));
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(px, py, R, 0, 2 * Math.PI);
      ctx.fill();
      spots.push({ x: px, y: py, r, i, set: 0 });
    }
    ctx.globalCompositeOperation = "source-over";

    this._drawBeam();

    if (labels) {
      const items = [];
      for (const s of spots) {
        if (value[s.i] < labelThreshold) continue;
        items.push({
          text: formatHkl([
            result.hkl[3 * s.i],
            result.hkl[3 * s.i + 1],
            result.hkl[3 * s.i + 2],
          ]),
          x: s.x,
          y: s.y - s.r - 2 * this.dpr,
          rank: value[s.i],
        });
      }
      this._drawLabels(items);
    }

    this.spots = spots;
    this._drawKey(result, structure, lim, rotation);
    return spots.length;
  }

  /**
   * The scale the view opens at: the largest that keeps every visible spot,
   * with its glow and its label, inside the canvas and off everything laid
   * over it.
   *
   * `sets` are {count, x, y, value} spot lists, `lim` their extent in
   * reciprocal units, `seen` the value from which a spot counts as visible,
   * and `avoid` the page's own overlays as [x, y, w, h] in CSS pixels over
   * the canvas. The canvas's axes key and scale bar are added here. The fit
   * starts from the whole pane and steps down 5% at a time until nothing
   * collides: a tenth of the extent as a margin was either too little (the
   * outer row under the scale bar, the top labels on the edge) or, on a
   * round pattern, more than the corners needed.
   */
  _fitScale(sets, lim, spotScale, seen, avoid) {
    const dpr = this.dpr;
    const { width: w, height: h } = this.canvas;
    const W = w / dpr,
      H = h / dpr;
    const glow = 1.6 * spotRadius(1, spotScale) * dpr;
    const label = 15 * dpr;
    const edge = 6 * dpr;
    const boxes = [
      ...avoid,
      [10, H - 106, 104, 66], // axes key
      [W - 34 - W / 3, H - 52, W / 3 + 20, 32], // scale bar, at its longest
    ].map(([x, y, bw, bh]) => [x * dpr, y * dpr, (x + bw) * dpr, (y + bh) * dpr]);

    let s = Math.max(1, Math.min(w, h) - 2 * (glow + edge)) / (2 * lim);
    for (let step = 0; step < 16; step++) {
      let clear = true;
      for (const set of sets) {
        for (let i = 0; i < set.count && clear; i++) {
          if (set.value[i] < seen) continue;
          const px = w / 2 + set.x[i] * s;
          const py = h / 2 - set.y[i] * s;
          const x0 = px - glow,
            y0 = py - glow - label,
            x1 = px + glow,
            y1 = py + glow;
          if (x0 < edge || y0 < edge || x1 > w - edge || y1 > h - edge)
            clear = false;
          else
            for (const b of boxes)
              if (x0 < b[2] && x1 > b[0] && y0 < b[3] && y1 > b[1]) {
                clear = false;
                break;
              }
        }
      }
      if (clear) break;
      s *= 0.95;
    }
    return s;
  }

  /**
   * The direct beam: not a reflection, but it locates the origin. A faint
   * halo and a small stop, as the beam and its stop show on a pattern.
   */
  _drawBeam() {
    const ctx = this.ctx;
    const dpr = this.dpr;
    const [ox, oy] = this.toScreen(0, 0);
    const halo = ctx.createRadialGradient(ox, oy, 0, ox, oy, 28 * dpr);
    halo.addColorStop(0, "rgba(150,185,255,0.25)");
    halo.addColorStop(1, "rgba(150,185,255,0)");
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(ox, oy, 28 * dpr, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = "#161b28";
    ctx.strokeStyle = "#5a6684";
    ctx.lineWidth = 1 * dpr;
    ctx.beginPath();
    ctx.arc(ox, oy, 5.5 * dpr, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  }

  /**
   * Draw hkl labels strongest first, skipping any whose box would overlap a
   * label already placed. A dense pattern then thins its labels out instead
   * of printing them over each other until none can be read.
   *
   * Each item is {text, x, y, rank}: (x, y) is the bottom centre of the text
   * in device pixels, and a higher rank is placed first. The placed boxes are
   * bucketed on a coarse grid, so each test looks at a few neighbours rather
   * than at every label drawn so far.
   */
  _drawLabels(items, max = 500) {
    const ctx = this.ctx;
    const dpr = this.dpr;
    const size = 11 * dpr;
    const gap = 2 * dpr;
    const cell = 64 * dpr;
    ctx.font = `${size}px Consolas, ui-monospace, monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.lineJoin = "round";
    ctx.lineWidth = 3 * dpr;
    ctx.strokeStyle = "rgba(9,11,18,0.9)";
    ctx.fillStyle = LABEL;

    const grid = new Map();
    const cellsOf = ([x0, y0, x1, y1], visit) => {
      for (let cx = Math.floor(x0 / cell); cx <= Math.floor(x1 / cell); cx++)
        for (let cy = Math.floor(y0 / cell); cy <= Math.floor(y1 / cell); cy++)
          if (visit(`${cx},${cy}`)) return true;
      return false;
    };
    const hits = (b) =>
      cellsOf(b, (key) =>
        (grid.get(key) || []).some(
          (p) => b[0] < p[2] && b[2] > p[0] && b[1] < p[3] && b[3] > p[1],
        ),
      );

    let shown = 0;
    for (const it of [...items].sort((a, b) => b.rank - a.rank)) {
      if (shown >= max) break;
      const half = ctx.measureText(it.text).width / 2;
      const box = [it.x - half - gap, it.y - size - gap, it.x + half + gap, it.y + gap];
      if (hits(box)) continue;
      cellsOf(box, (key) => {
        if (!grid.has(key)) grid.set(key, []);
        grid.get(key).push(box);
        return false;
      });
      ctx.strokeText(it.text, it.x, it.y);
      ctx.fillText(it.text, it.x, it.y);
      shown++;
    }
    return shown;
  }

  /**
   * The two in-plane reciprocal directions and a scale bar. The directions
   * are turned by `rotation` degrees, as the pattern is.
   */
  _drawKey(result, structure, lim, rotation = 0) {
    const ctx = this.ctx;
    const { width: w, height: h } = this.canvas;
    const dpr = this.dpr;
    const ox = 54 * dpr;
    const oy = h - 54 * dpr;
    const len = 30 * dpr;
    const cr = Math.cos((rotation * Math.PI) / 180);
    const sr = Math.sin((rotation * Math.PI) / 180);

    ctx.fillStyle = "rgba(9,11,18,0.72)";
    ctx.fillRect(ox - 44 * dpr, oy - 52 * dpr, 104 * dpr, 66 * dpr);

    for (const [g, colour] of [
      [result.g1, ACCENT],
      [result.g2, GOOD],
    ]) {
      if (!g) continue;
      const v = [
        structure.B[0][0] * g[0] +
          structure.B[0][1] * g[1] +
          structure.B[0][2] * g[2],
        structure.B[1][0] * g[0] +
          structure.B[1][1] * g[1] +
          structure.B[1][2] * g[2],
        structure.B[2][0] * g[0] +
          structure.B[2][1] * g[1] +
          structure.B[2][2] * g[2],
      ];
      const ux =
        v[0] * result.xAxis[0] +
        v[1] * result.xAxis[1] +
        v[2] * result.xAxis[2];
      const uy =
        v[0] * result.yAxis[0] +
        v[1] * result.yAxis[1] +
        v[2] * result.yAxis[2];
      const gx = cr * ux - sr * uy;
      const gy = sr * ux + cr * uy;
      const n = Math.hypot(gx, gy);
      if (n < 1e-9) continue;
      const dx = (gx / n) * len;
      const dy = -(gy / n) * len;
      ctx.strokeStyle = colour;
      ctx.fillStyle = colour;
      ctx.lineWidth = 1.3 * dpr;
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.lineTo(ox + dx, oy + dy);
      ctx.stroke();
      // arrow head
      const a = Math.atan2(dy, dx);
      ctx.beginPath();
      ctx.moveTo(ox + dx, oy + dy);
      ctx.lineTo(
        ox + dx - 5 * dpr * Math.cos(a - 0.4),
        oy + dy - 5 * dpr * Math.sin(a - 0.4),
      );
      ctx.lineTo(
        ox + dx - 5 * dpr * Math.cos(a + 0.4),
        oy + dy - 5 * dpr * Math.sin(a + 0.4),
      );
      ctx.fill();
      ctx.font = `${9.5 * dpr}px Consolas, ui-monospace, monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(formatHkl(g), ox + dx * 1.42, oy + dy * 1.42);
    }

    // Sized from what is on screen, not from the extent of the data, so the
    // bar stays about a sixth of the pane instead of growing with the zoom
    // until it runs the whole width.
    const step = niceStep(w / (this.scale * this.zoom));
    const px = step * this.scale * this.zoom;
    const x0 = w - 24 * dpr - px;
    const y0 = h - 30 * dpr;
    // Both overlays sit on top of the pattern, so they get a backing panel;
    // at high zoom the spots run underneath them and the bar was unreadable.
    ctx.fillStyle = "rgba(9,11,18,0.72)";
    ctx.fillRect(x0 - 10 * dpr, y0 - 22 * dpr, px + 20 * dpr, 32 * dpr);
    ctx.strokeStyle = DIM;
    ctx.fillStyle = DIM;
    ctx.lineWidth = 1.4 * dpr;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x0 + px, y0);
    ctx.stroke();
    ctx.font = `${10 * dpr}px Consolas, ui-monospace, monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(`${step} Å⁻¹`, x0 + px / 2, y0 - 4 * dpr);
  }

  /**
   * Two patterns in one frame, sections or SAED patterns: crystal A, and
   * crystal B rotated by `twist` degrees about the zone axis.
   *
   * The rotation is applied to the in-plane coordinates only, which is exact
   * for both: each is drawn in the plane normal to the zone axis, so a
   * rotation of the crystal about that axis is a rotation of the picture
   * about its centre. `rotation` turns the two together, on top of the twist.
   * Matched reflections are ringed in green when `showMatch` is set, so the
   * coincidences and the misfits can be told apart at a glance.
   */
  drawOverlay(resultA, resultB, opts = {}) {
    const {
      twist = 0,
      rotation = 0,
      gain = 1,
      log = false,
      lut,
      spotScale = 9,
      labels = false,
      labelThreshold = 0.35,
      showMatch = true,
      showRings = false,
      tol = 0.03,
      nameA = "crystal 1",
      nameB = "crystal 2",
      structure = null,
      // "overlay" draws both crystals equally; "coincidences" keeps what the two
      // share and fades the rest; "differences" does the opposite, which is how
      // you look for the reflections that fingerprint one phase in the other.
      mode = "overlay",
    } = opts;
    const { w, h } = this.clear();
    const ctx = this.ctx;
    this.spots = null;
    this.pairs = null;
    this.powder = null;

    if (!resultA || !resultA.count) {
      this.message("no reflections in the first crystal\nlower d min", "#f0a05a");
      return;
    }
    const hasB = !!(resultB && resultB.count);
    // B against A, the twist alone, is what the pairing is about; both are
    // drawn with the rotation on top.
    const relB = hasB ? rotateInPlane(resultB, twist) : null;
    const xyA = rotation ? rotateInPlane(resultA, rotation) : resultA;
    const rotB = hasB
      ? rotation
        ? rotateInPlane(resultB, twist + rotation)
        : relB
      : null;
    const valueA = stretch(resultA.intensity, { gain, log });
    const valueB = rotB ? stretch(resultB.intensity, { gain, log }) : null;

    // Fitted as the two sit against each other, before the common rotation,
    // so the pair keeps its size as it is rotated (as in drawSpots).
    let lim = 0;
    for (let i = 0; i < resultA.count; i++)
      if (valueA[i] > 1e-3)
        lim = Math.max(lim, Math.abs(resultA.x[i]), Math.abs(resultA.y[i]));
    if (rotB)
      for (let i = 0; i < resultB.count; i++)
        if (valueB[i] > 1e-3)
          lim = Math.max(lim, Math.abs(relB.x[i]), Math.abs(relB.y[i]));
    if (lim <= 0) {
      this.message("every reflection in view is extinct\nraise the contrast", "#f0a05a");
      return;
    }

    this.cx = w / 2;
    this.cy = h / 2;
    // Both crystals are flat colours, so every spot drawn is visible; the
    // legend stacked on the axes key is one more thing to keep clear of.
    const sets = [{ ...resultA, value: valueA }];
    if (rotB) sets.push({ count: resultB.count, x: relB.x, y: relB.y, value: valueB });
    this.scale = this._fitScale(sets, lim, spotScale, 1e-3, [
      ...(opts.avoid || []),
      [10, h / this.dpr - 174, 300, 60],
    ]);

    const colourB = SECOND;
    const hits = rotB
      ? pairUp(resultA, { count: resultB.count, x: relB.x, y: relB.y }, { tol })
      : null;
    const hitSet = hits ? new Set(hits.pairs.map((p) => p.a)) : null;
    if (hits) {
      const ab = new Map();
      const ba = new Map();
      for (const p of hits.pairs) {
        ab.set(p.a, { b: p.b, dq: p.dq });
        ba.set(p.b, { a: p.a, dq: p.dq });
      }
      this.pairs = [ab, ba];
    }

    if (showRings && resultA.zoneRadii) {
      ctx.strokeStyle = "#243049";
      ctx.lineWidth = 1 * this.dpr;
      const [ox, oy] = this.toScreen(0, 0);
      for (const r of resultA.zoneRadii) {
        if (!r) continue;
        ctx.beginPath();
        ctx.arc(ox, oy, r * this.scale * this.zoom, 0, 2 * Math.PI);
        ctx.stroke();
      }
    }

    // Per-spot opacity, so the two modes can pick out what coincides and what
    // does not without a second pass over the reflections.
    const matched = (i) => !!(hitSet && hitSet.has(i));
    const alphaA =
      mode === "coincidences"
        ? (i) => (matched(i) ? 0.95 : 0.1)
        : mode === "differences"
          ? (i) => (matched(i) ? 0.1 : 0.95)
          : () => 0.85;
    const alphaB = () => (mode === "differences" ? 0.45 : 0.72);

    // Every spot drawn, of either crystal, so a hover can name either one.
    const spots = [];
    const paint = (xs, ys, value, colour, alpha, count, set) => {
      ctx.fillStyle = colour;
      for (let i = 0; i < count; i++) {
        if (value[i] <= 1e-3) continue;
        const a = alpha(i);
        if (a <= 0) continue;
        ctx.globalAlpha = a;
        const [px, py] = this.toScreen(xs[i], ys[i]);
        const r = spotRadius(value[i], spotScale) * this.dpr;
        if (px < -r || py < -r || px > w + r || py > h + r) continue;
        ctx.beginPath();
        ctx.arc(px, py, r, 0, 2 * Math.PI);
        ctx.fill();
        spots.push({ x: px, y: py, r, i, set });
      }
      ctx.globalAlpha = 1;
    };

    // Crystal 2 sits underneath so crystal 1 stays readable where they overlap.
    if (rotB) paint(rotB.x, rotB.y, valueB, colourB, alphaB, resultB.count, 1);
    paint(xyA.x, xyA.y, valueA, ACCENT, alphaA, resultA.count, 0);

    if (showMatch && hits && hits.pairs.length) {
      ctx.strokeStyle = GOOD;
      ctx.lineWidth = 1.6 * this.dpr;
      for (const p of hits.pairs) {
        const [px, py] = this.toScreen(xyA.x[p.a], xyA.y[p.a]);
        const r = spotRadius(valueA[p.a], spotScale) * this.dpr + 5 * this.dpr;
        ctx.beginPath();
        ctx.arc(px, py, r, 0, 2 * Math.PI);
        ctx.stroke();
      }
    }

    this._drawBeam();

    if (labels) {
      const items = [];
      for (const s of spots) {
        if (s.set !== 0 || valueA[s.i] < labelThreshold) continue;
        items.push({
          text: formatHkl([
            resultA.hkl[3 * s.i],
            resultA.hkl[3 * s.i + 1],
            resultA.hkl[3 * s.i + 2],
          ]),
          x: s.x,
          y: s.y - s.r - 2 * this.dpr,
          rank: valueA[s.i],
        });
      }
      this._drawLabels(items, 400);
    }

    // Legend: which colour is which crystal, and the twist actually applied.
    // Bottom left, stacked on the axes key: the top-left corner belongs to the
    // page's title overlay, and the two used to print over each other. It stays
    // on the canvas so an exported PNG still says which crystal is which.
    const dpr = this.dpr;
    const rows = [[ACCENT, nameA]];
    if (rotB) rows.push([colourB, `${nameB}   twist ${twist.toFixed(1)}°`]);
    if (rotB && showMatch && hits)
      rows.push([GOOD, `${hits.pairs.length} coincident`]);
    ctx.font = `${10.5 * dpr}px Consolas, ui-monospace, monospace`;
    const boxW =
      Math.max(...rows.map(([, text]) => ctx.measureText(text).width)) +
      36 * dpr;
    const boxH = (12 + rows.length * 16) * dpr;
    const bx = 10 * dpr;
    const by = h - 114 * dpr - boxH;
    ctx.fillStyle = "rgba(9,11,18,0.78)";
    ctx.fillRect(bx, by, boxW, boxH);
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    rows.forEach(([colour, text], row) => {
      const ly = by + (14 + row * 16) * dpr;
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.arc(bx + 12 * dpr, ly, 4 * dpr, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = DIM;
      ctx.fillText(text, bx + 24 * dpr, ly);
    });

    this.spots = spots;
    this._drawKey(resultA, structure, lim, rotation);
    return spots.length;
  }

  /** Nearest reflection of the first crystal to a client-space point, or null. */
  pick(clientX, clientY) {
    const hit = this.pickSpot(clientX, clientY, 0);
    return hit ? hit.i : null;
  }

  /**
   * Nearest spot to a client-space point as {set, i}, or null: set 0 is the
   * first crystal and 1 the second, in the overlay. `only` keeps one set.
   */
  pickSpot(clientX, clientY, only = null) {
    if (!this.spots) return null;
    const box = this.canvas.getBoundingClientRect();
    const x = (clientX - box.left) * this.dpr;
    const y = (clientY - box.top) * this.dpr;
    let best = null;
    let bestD = 14 * this.dpr;
    for (const s of this.spots) {
      if (only !== null && s.set !== only) continue;
      const d = Math.hypot(s.x - x, s.y - y);
      if (d < Math.max(s.r, bestD)) {
        if (d < bestD || best === null) {
          bestD = Math.max(d, 4);
          best = s;
        }
      }
    }
    return best ? { set: best.set, i: best.i } : null;
  }

  /**
   * The powder trace, with axes.
   *
   * `avoid` holds the page's overlays as [x, y, w, h] in CSS pixels, as for
   * the spot views; the plot starts under whichever of them sit in its top
   * half (the title card), so the card no longer covers the axis.
   *
   * `second` is a second crystal's powder result over the same 2θ range. Its
   * trace is drawn under the first in its own colour, with its own row of
   * reflection ticks, as the phases of a multi-phase Rietveld plot are.
   * `hover` ({set, i}) picks out one peak: its own profile, a line down to its
   * tick, and the tick itself.
   */
  drawPowder(p, opts) {
    const {
      labels = true,
      labelThreshold = 5,
      avoid = [],
      second = null,
      nameA = "",
      nameB = "",
      hover = null,
    } = opts;
    const { w, h } = this.clear();
    const ctx = this.ctx;
    const dpr = this.dpr;
    this.spots = null;
    this.pairs = null;

    let top = 22;
    for (const [, y, , bh] of avoid)
      if (y < h / dpr / 2) top = Math.max(top, y + bh + 12);
    const padL = 52 * dpr;
    const padR = 18 * dpr;
    const padT = top * dpr;
    // tick labels, the axis title, and the status strip under both
    const padB = 70 * dpr;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;
    const ttMax = p.x[p.x.length - 1];
    // The axis runs a little below zero, to hold a row of reflection ticks
    // under the trace's baseline for each crystal.
    const yMin = second ? -16 : -9;
    const yMax = 108;
    const X = (tt) => padL + (tt / ttMax) * plotW;
    const Y = (v) => padT + plotH * (1 - (v - yMin) / (yMax - yMin));

    ctx.strokeStyle = "#1c2334";
    ctx.lineWidth = 1 * dpr;
    ctx.font = `${10 * dpr}px Consolas, ui-monospace, monospace`;
    ctx.fillStyle = DIM;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    // A "nice" step (1, 2, 2.5 or 5 times a power of ten) for about one tick
    // every 72 px, printed with as many decimals as it needs, so the labels
    // read what the ticks are at (2.5 is printed as 2.5, not rounded to 3).
    const tick = niceStep((6 * ttMax) / Math.max(3, plotW / (72 * dpr)));
    const mant = tick / Math.pow(10, Math.floor(Math.log10(tick)));
    const decimals =
      Math.max(0, -Math.floor(Math.log10(tick))) +
      (Math.abs(mant - 2.5) < 1e-9 ? 1 : 0);
    for (let tt = 0; tt <= ttMax + 1e-9; tt += tick) {
      ctx.beginPath();
      ctx.moveTo(X(tt), padT);
      ctx.lineTo(X(tt), padT + plotH);
      ctx.stroke();
      ctx.fillText(tt.toFixed(decimals), X(tt), padT + plotH + 6 * dpr);
    }
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const v of [0, 25, 50, 75, 100]) {
      ctx.beginPath();
      ctx.moveTo(padL, Y(v));
      ctx.lineTo(padL + plotW, Y(v));
      ctx.stroke();
      ctx.fillText(String(v), padL - 6 * dpr, Y(v));
    }

    ctx.strokeStyle = LINE;
    ctx.strokeRect(padL, padT, plotW, plotH);

    // Sampled at twelve points a peak width, so every apex is on the trace:
    // at the result's own 2000 points a narrow peak's top fell between two
    // samples and was drawn up to a tenth low, under its own label. Kept per
    // result, so a hover redraws without summing every peak again.
    const traceOf = (r) => {
      let got = this._traces.get(r);
      if (!got) {
        const n = Math.min(
          40000,
          Math.max(r.x.length, Math.ceil((12 * ttMax) / r.fwhm) + 1),
        );
        got = {
          n,
          trace: powderProfile(r.twoTheta, r.intensity, ttMax, r.fwhm, n),
        };
        this._traces.set(r, got);
      }
      return got;
    };
    const drawTrace = ({ n, trace }, stroke, fillTop, fillBottom) => {
      ctx.beginPath();
      ctx.moveTo(X(trace.x[0]), Y(trace.y[0]));
      for (let i = 1; i < n; i++) ctx.lineTo(X(trace.x[i]), Y(trace.y[i]));
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 1.3 * dpr;
      ctx.stroke();
      ctx.lineTo(X(ttMax), Y(0));
      ctx.lineTo(X(0), Y(0));
      ctx.closePath();
      const fill = ctx.createLinearGradient(0, Y(100), 0, Y(0));
      fill.addColorStop(0, fillTop);
      fill.addColorStop(1, fillBottom);
      ctx.fillStyle = fill;
      ctx.fill();
    };
    const sets = second ? [p, second] : [p];
    const traces = sets.map(traceOf);
    // The second crystal underneath, so the first stays readable where the
    // two traces cross.
    if (second)
      drawTrace(
        traces[1],
        SECOND,
        "rgba(226,160,106,0.22)",
        "rgba(226,160,106,0.03)",
      );
    drawTrace(
      traces[0],
      ACCENT,
      "rgba(111,158,224,0.32)",
      "rgba(111,158,224,0.04)",
    );

    // The drawn apex near a 2θ, which a close neighbour can lift above the
    // peak's own height.
    const apex = ({ n, trace }, tt) => {
      const j = Math.round(tt / (ttMax / (n - 1)));
      let v = 0;
      for (let k = Math.max(0, j - 2); k <= Math.min(n - 1, j + 2); k++)
        v = Math.max(v, trace.y[k]);
      return v;
    };

    // Where every reflection falls, as the row of ticks under a Rietveld plot:
    // one row per crystal, in its trace's colour when there are two. A peak
    // made only of extinct reflections is not a reflection and gets no tick.
    const rows = second
      ? [
          [-2, -7, "rgba(111,158,224,0.9)"],
          [-9, -14, "rgba(226,160,106,0.9)"],
        ]
      : [[-2, -7, "rgba(143,227,192,0.75)"]];
    sets.forEach((r, s) => {
      const [a, b, colour] = rows[s];
      ctx.strokeStyle = colour;
      ctx.lineWidth = 1 * dpr;
      ctx.beginPath();
      for (let i = 0; i < r.count; i++) {
        if (r.absent && r.absent[i]) continue;
        const x = X(r.twoTheta[i]);
        ctx.moveTo(x, Y(a));
        ctx.lineTo(x, Y(b));
      }
      ctx.stroke();
    });

    if (hover && sets[hover.set] && hover.i < sets[hover.set].count) {
      const r = sets[hover.set];
      const tt = r.twoTheta[hover.i];
      const I = r.intensity[hover.i];
      const sigma = r.fwhm / (2 * Math.sqrt(2 * Math.LN2));
      ctx.save();
      ctx.beginPath();
      ctx.rect(padL, padT, plotW, plotH);
      ctx.clip();
      // the peak's own profile, under the summed trace
      const m = 64;
      ctx.beginPath();
      ctx.moveTo(X(tt - 4 * sigma), Y(0));
      for (let k = 0; k <= m; k++) {
        const t = tt - 4 * sigma + (8 * sigma * k) / m;
        ctx.lineTo(X(t), Y(I * Math.exp(-0.5 * ((t - tt) / sigma) ** 2)));
      }
      ctx.lineTo(X(tt + 4 * sigma), Y(0));
      ctx.closePath();
      ctx.fillStyle = "rgba(143,227,192,0.28)";
      ctx.fill();
      ctx.strokeStyle = GOOD;
      ctx.lineWidth = 1.2 * dpr;
      ctx.stroke();
      // a line from its tick up to the top of the trace there
      const [a, b] = rows[hover.set];
      ctx.setLineDash([3 * dpr, 3 * dpr]);
      ctx.beginPath();
      ctx.moveTo(X(tt), Y(b));
      ctx.lineTo(X(tt), Y(Math.max(I, apex(traces[hover.set], tt))));
      ctx.stroke();
      ctx.setLineDash([]);
      // and the tick itself, brighter and wider than its neighbours
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2.2 * dpr;
      ctx.beginPath();
      ctx.moveTo(X(tt), Y(a));
      ctx.lineTo(X(tt), Y(b));
      ctx.stroke();
      ctx.restore();
    }

    if (labels) {
      const items = [];
      for (let i = 0; i < p.count; i++) {
        if (p.intensity[i] < labelThreshold) continue;
        items.push({
          text: formatHkl(p.hkl[i]),
          x: X(p.twoTheta[i]),
          y: Y(apex(traces[0], p.twoTheta[i])) - 3 * dpr,
          rank: p.intensity[i],
        });
      }
      this._drawLabels(items);
    }

    // Under the tick labels, clear of the status line below it.
    ctx.fillStyle = DIM;
    ctx.font = `${11 * dpr}px "Segoe UI", system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText("2θ  (degrees)", padL + plotW / 2, padT + plotH + 24 * dpr);

    // Which trace is which crystal, on the canvas so an exported PNG still
    // says so: in the margin above the plot, right-aligned, unless the title
    // card reaches across to it, and then just inside the plot's corner.
    if (second) {
      const key = [
        [ACCENT, nameA],
        [SECOND, nameB],
      ];
      ctx.font = `${10.5 * dpr}px Consolas, ui-monospace, monospace`;
      const boxW =
        Math.max(...key.map(([, text]) => ctx.measureText(text).width)) +
        40 * dpr;
      const boxH = (12 + key.length * 16) * dpr;
      let bx = padL + plotW - boxW;
      let by = 10 * dpr;
      const covered = avoid.some(
        ([ax, ay, aw, ah]) =>
          ax * dpr < bx + boxW &&
          (ax + aw) * dpr > bx &&
          ay * dpr < by + boxH &&
          (ay + ah) * dpr > by,
      );
      if (covered || by + boxH > padT - 4 * dpr) {
        bx -= 8 * dpr;
        by = padT + 8 * dpr;
      }
      ctx.fillStyle = "rgba(9,11,18,0.78)";
      ctx.fillRect(bx, by, boxW, boxH);
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      key.forEach(([colour, text], row) => {
        const ly = by + (14 + row * 16) * dpr;
        ctx.strokeStyle = colour;
        ctx.lineWidth = 2 * dpr;
        ctx.beginPath();
        ctx.moveTo(bx + 8 * dpr, ly);
        ctx.lineTo(bx + 24 * dpr, ly);
        ctx.stroke();
        ctx.fillStyle = DIM;
        ctx.fillText(text, bx + 30 * dpr, ly);
      });
    }

    this.powder = {
      padL,
      padT,
      plotW,
      plotH,
      ttMax,
      sets,
      rows: rows.map(([a, b]) => [Y(a), Y(b)]),
    };
  }

  /**
   * The powder peak under a client-space point: {set, i} for a peak, {tt}
   * with the 2θ under the cursor when no peak is near, or null off the plot.
   *
   * Over a row of reflection ticks the nearest tick in that row is taken, so
   * every reflection can be reached, however weak. Over the trace it is the
   * peak whose own profile is highest at the cursor, which is the peak the
   * cursor is on rather than merely the nearest centre: on the flank of a
   * strong peak that is the strong one, and a weak peak standing apart is
   * still found where it is drawn.
   */
  pickPowder(clientX, clientY) {
    const g = this.powder;
    if (!g) return null;
    const box = this.canvas.getBoundingClientRect();
    const x = (clientX - box.left) * this.dpr;
    const y = (clientY - box.top) * this.dpr;
    if (
      x < g.padL ||
      x > g.padL + g.plotW ||
      y < g.padT ||
      y > g.padT + g.plotH
    )
      return null;
    const perDeg = g.plotW / g.ttMax;
    const tt = (x - g.padL) / perDeg;
    const slack = 3 * this.dpr;
    for (let s = 0; s < g.sets.length; s++) {
      const [y0, y1] = g.rows[s];
      if (y < y0 - slack || y > y1 + slack) continue;
      const r = g.sets[s];
      let best = -1;
      let bestD = 6 * this.dpr;
      for (let i = 0; i < r.count; i++) {
        if (r.absent && r.absent[i]) continue;
        const d = Math.abs(r.twoTheta[i] - tt) * perDeg;
        if (d < bestD) {
          bestD = d;
          best = i;
        }
      }
      return best >= 0 ? { set: s, i: best } : { tt };
    }
    let best = null;
    let bestV = 0;
    for (let s = 0; s < g.sets.length; s++) {
      const r = g.sets[s];
      const sigma = r.fwhm / (2 * Math.sqrt(2 * Math.LN2));
      const reach = Math.max(2 * sigma, (6 * this.dpr) / perDeg);
      for (let i = 0; i < r.count; i++) {
        if (r.absent && r.absent[i]) continue;
        const dt = r.twoTheta[i] - tt;
        if (Math.abs(dt) > reach) continue;
        const v = r.intensity[i] * Math.exp(-0.5 * (dt / sigma) ** 2);
        if (best === null || v > bestV) {
          best = { set: s, i };
          bestV = v;
        }
      }
    }
    return best || { tt };
  }
}
