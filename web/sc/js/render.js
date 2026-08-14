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

const DIM = "#8794b0";
const LINE = "#2a3145";
const ACCENT = "#6f9ee0";
const GOOD = "#8fe3c0";
const BG = "#090b12";

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
    this.spots = null; // {x, y, r, i} in device pixels, for hit testing
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
   * directions for the axes key.
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
    } = opts;
    const { w, h } = this.clear();
    const ctx = this.ctx;
    this.spots = null;

    if (!result || result.count === 0) {
      this.message(
        "no reflections here\ntry a lower d min, or a different layer",
        "#f0a05a",
      );
      return;
    }

    const value = stretch(result.intensity, { gain, log });
    let lim = 0;
    for (let i = 0; i < result.count; i++) {
      if (value[i] <= 1e-3) continue;
      lim = Math.max(lim, Math.abs(result.x[i]), Math.abs(result.y[i]));
    }
    if (lim <= 0) {
      this.message(
        "every reflection in this section is extinct\nraise the contrast to check",
        "#f0a05a",
      );
      return;
    }
    lim *= 1.1;

    this.cx = w / 2;
    this.cy = h / 2;
    this.scale = Math.min(w, h) / (2 * lim);
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
    // which is how the Game of Diffraction paints its detector too.
    ctx.globalCompositeOperation = "lighter";
    for (let i = 0; i < result.count; i++) {
      const v = value[i];
      if (v <= 1e-3) continue;
      const [px, py] = this.toScreen(result.x[i], result.y[i]);
      const r = spotRadius(v, spotScale) * this.dpr;
      if (px < -r || py < -r || px > w + r || py > h + r) continue;
      ctx.fillStyle = sampleLut(lut, v);
      ctx.beginPath();
      ctx.arc(px, py, r, 0, 2 * Math.PI);
      ctx.fill();
      spots.push({ x: px, y: py, r, i });
    }
    ctx.globalCompositeOperation = "source-over";

    // The direct beam: not a reflection, but it locates the origin.
    const [ox, oy] = this.toScreen(0, 0);
    ctx.strokeStyle = DIM;
    ctx.lineWidth = 1 * this.dpr;
    ctx.beginPath();
    ctx.arc(ox, oy, 4 * this.dpr, 0, 2 * Math.PI);
    ctx.stroke();

    if (labels) {
      ctx.fillStyle = DIM;
      ctx.font = `${10 * this.dpr}px Consolas, ui-monospace, monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      let shown = 0;
      for (const s of spots) {
        if (value[s.i] < labelThreshold || shown > 500) continue;
        const hkl = [
          result.hkl[3 * s.i],
          result.hkl[3 * s.i + 1],
          result.hkl[3 * s.i + 2],
        ];
        ctx.fillText(formatHkl(hkl), s.x, s.y - s.r - 2 * this.dpr);
        shown++;
      }
    }

    this.spots = spots;
    this._drawKey(result, structure, lim);
    return spots.length;
  }

  /** The two in-plane reciprocal directions and a scale bar. */
  _drawKey(result, structure, lim) {
    const ctx = this.ctx;
    const { width: w, height: h } = this.canvas;
    const dpr = this.dpr;
    const ox = 54 * dpr;
    const oy = h - 54 * dpr;
    const len = 30 * dpr;

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
      const gx =
        v[0] * result.xAxis[0] +
        v[1] * result.xAxis[1] +
        v[2] * result.xAxis[2];
      const gy =
        v[0] * result.yAxis[0] +
        v[1] * result.yAxis[1] +
        v[2] * result.yAxis[2];
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

  /** Nearest reflection to a client-space point, or null. */
  pick(clientX, clientY) {
    if (!this.spots) return null;
    const box = this.canvas.getBoundingClientRect();
    const x = (clientX - box.left) * this.dpr;
    const y = (clientY - box.top) * this.dpr;
    let best = null;
    let bestD = 14 * this.dpr;
    for (const s of this.spots) {
      const d = Math.hypot(s.x - x, s.y - y);
      if (d < Math.max(s.r, bestD)) {
        if (d < bestD || best === null) {
          bestD = Math.max(d, 4);
          best = s.i;
        }
      }
    }
    return best;
  }

  /** The powder trace, with axes. */
  drawPowder(p, opts) {
    const { labels = true, labelThreshold = 5 } = opts;
    const { w, h } = this.clear();
    const ctx = this.ctx;
    const dpr = this.dpr;
    this.spots = null;

    const padL = 52 * dpr;
    const padR = 18 * dpr;
    const padT = 22 * dpr;
    const padB = 42 * dpr;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;
    const ttMax = p.x[p.x.length - 1];
    const yMax = 108;
    const X = (tt) => padL + (tt / ttMax) * plotW;
    const Y = (v) => padT + plotH * (1 - v / yMax);

    ctx.strokeStyle = "#1c2334";
    ctx.lineWidth = 1 * dpr;
    ctx.font = `${10 * dpr}px Consolas, ui-monospace, monospace`;
    ctx.fillStyle = DIM;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const tick = niceStep(ttMax) * 1.5;
    for (let tt = 0; tt <= ttMax + 1e-9; tt += tick) {
      ctx.beginPath();
      ctx.moveTo(X(tt), padT);
      ctx.lineTo(X(tt), padT + plotH);
      ctx.stroke();
      ctx.fillText(tt.toFixed(tick < 1 ? 1 : 0), X(tt), padT + plotH + 6 * dpr);
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

    ctx.beginPath();
    ctx.moveTo(X(p.x[0]), Y(p.y[0]));
    for (let i = 1; i < p.x.length; i++) ctx.lineTo(X(p.x[i]), Y(p.y[i]));
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1.3 * dpr;
    ctx.stroke();
    ctx.lineTo(X(ttMax), Y(0));
    ctx.lineTo(X(0), Y(0));
    ctx.closePath();
    ctx.fillStyle = "rgba(111,158,224,0.16)";
    ctx.fill();

    if (labels) {
      ctx.fillStyle = DIM;
      ctx.font = `${10 * dpr}px Consolas, ui-monospace, monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      for (let i = 0; i < p.count; i++) {
        if (p.intensity[i] < labelThreshold) continue;
        ctx.fillText(
          formatHkl(p.hkl[i]),
          X(p.twoTheta[i]),
          Y(p.intensity[i]) - 3 * dpr,
        );
      }
    }

    // Just under the tick labels, not at the foot of the canvas: the status
    // line lives down there and the two collided.
    ctx.fillStyle = DIM;
    ctx.font = `${11 * dpr}px "Segoe UI", system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText("2θ  (degrees)", padL + plotW / 2, padT + plotH + 22 * dpr);
  }
}
