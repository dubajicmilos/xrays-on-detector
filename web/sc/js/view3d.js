/*! Single-Crystal Diffraction · © 2026 Miloš Dubajić · MIT · https://github.com/dubajicmilos/xrays-on-detector */

/**
 * The reciprocal lattice in 3D, drawn on the pattern's own 2D canvas by an
 * orthographic projection, back to front.
 *
 * Every reflection is a spot at its Q, coloured and sized from |F|^2 by the
 * stretch the flat views use, and dimmed with depth so the near side reads as
 * near. The Ewald sphere, the incident beam and the rotation axis can be drawn
 * with it, and the reflections the sphere passes through at the current omega
 * are ringed: the rotation photograph, seen from inside reciprocal space. The
 * plane of the current zone and layer can be shown too, its reflections kept
 * bright and the rest faded, which is where a flat section sits in the whole
 * lattice.
 *
 * The camera is a rotation from the crystal frame to the screen, with rows for
 * the screen's right, its up and the direction towards the viewer. A drag
 * turns it about the screen's own axes, so it never locks the way a pair of
 * Euler angles does.
 */

import { matMul, rotX, rotY } from "../../js/physics.js";
import { formatHkl, niceStep, sampleLut, spotRadius, stretch } from "./display.js";
import { ewaldSphere } from "./lattice.js";

const DIM = "#8794b0";
const HOT = "#f2c14e";
const PLANE = "143,227,192";
const SPHERE = "111,158,224";
const AXES = [
  ["a*", "#6f9ee0"],
  ["b*", "#8fe3c0"],
  ["c*", "#e2a06a"],
];

/** Rows back to orthonormal, so many small turns do not shear the view. */
function orthonormal(R) {
  const unit = (v) => {
    const m = Math.hypot(v[0], v[1], v[2]);
    return v.map((c) => c / m);
  };
  const x = unit(R[0]);
  const d = x[0] * R[1][0] + x[1] * R[1][1] + x[2] * R[1][2];
  const y = unit(R[1].map((c, j) => c - d * x[j]));
  const z = [
    x[1] * y[2] - x[2] * y[1],
    x[2] * y[0] - x[0] * y[2],
    x[0] * y[1] - x[1] * y[0],
  ];
  return [x, y, z];
}

export class LatticeView {
  /** `view` is the PatternView whose canvas, context and labeller it shares. */
  constructor(view) {
    this.view = view;
    this.R = null;
    this.zoom = 1;
    // what the camera was last set up for, so a new zone or structure starts
    // looking down the zone axis again while a new layer or d min does not
    this.key = null;
    this.drawn = null; // the projected spots, for picking
    this.scale = 1;
  }

  /** Down the zone axis, as the section view looks at the zero layer. */
  reset(lat) {
    this.R = [lat.xAxis.slice(), lat.yAxis.slice(), lat.nHat.slice()];
    this.zoom = 1;
  }

  /** Turn the view by a drag of (dx, dy) CSS pixels: the near side follows. */
  turn(dx, dy) {
    if (!this.R) return;
    this.R = orthonormal(
      matMul(rotY(dx * 0.008), matMul(rotX(dy * 0.008), this.R)),
    );
  }

  zoomBy(f, lo = 0.3, hi = 30) {
    const next = Math.min(hi, Math.max(lo, this.zoom * f));
    if (next === this.zoom) return false;
    this.zoom = next;
    return true;
  }

  /**
   * Draw the lattice `lat` (computeLattice). `hover` is the index of a
   * reflection to pick out; `tol` how near the sphere, in 1/Angstrom, a
   * reflection must be to count as on it. Returns the indices on the sphere.
   */
  draw(lat, opts) {
    const {
      gain = 1,
      log = false,
      lut,
      spotScale = 9,
      labels = true,
      labelThreshold = 0.35,
      ewald = true,
      wavelength = 1.5406,
      omega = 0,
      tol = 0.08,
      plane = true,
      nodes = false,
      hover = null,
    } = opts;
    const v = this.view;
    const { w, h } = v.clear();
    const ctx = v.ctx;
    const dpr = v.dpr;
    // the flat views' hit data does not describe this picture
    v.spots = null;
    v.pairs = null;
    v.powder = null;
    if (!this.R) this.reset(lat);
    const R = this.R;
    const cx = w / 2;
    const cy = h / 2;
    // the lattice out to d min is a ball of radius qMax: fit it in the pane
    const s = ((0.44 * Math.min(w, h)) / lat.qMax) * this.zoom;
    this.scale = s;
    const project = (p) => [
      cx + s * (R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2]),
      cy - s * (R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2]),
      R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2],
    ];
    const [ox, oy] = project([0, 0, 0]);

    // the rotation axis, through the origin
    const top = lat.yAxis.map((c) => c * lat.qMax * 1.08);
    const [ax0, ay0] = project(top.map((c) => -c));
    const [ax1, ay1] = project(top);
    ctx.strokeStyle = "rgba(135,148,176,0.55)";
    ctx.lineWidth = 1 * dpr;
    ctx.setLineDash([4 * dpr, 4 * dpr]);
    ctx.beginPath();
    ctx.moveTo(ax0, ay0);
    ctx.lineTo(ax1, ay1);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = DIM;
    ctx.font = `${10 * dpr}px Consolas, ui-monospace, monospace`;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText("ω", ax1 + 5 * dpr, ay1);

    // the section plane of the current layer, a disc out to d min
    if (plane && Math.abs(lat.height) < lat.qMax) {
      const r0 = Math.sqrt(lat.qMax ** 2 - lat.height ** 2);
      const c0 = lat.nHat.map((c) => c * lat.height);
      ctx.beginPath();
      for (let k = 0; k <= 96; k++) {
        const t = (2 * Math.PI * k) / 96;
        const p = [0, 1, 2].map(
          (j) =>
            c0[j] + r0 * (Math.cos(t) * lat.xAxis[j] + Math.sin(t) * lat.yAxis[j]),
        );
        const [px, py] = project(p);
        if (k === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.fillStyle = `rgba(${PLANE},0.06)`;
      ctx.fill();
      ctx.strokeStyle = `rgba(${PLANE},0.45)`;
      ctx.lineWidth = 1 * dpr;
      ctx.stroke();
    }

    // the Ewald sphere: its outline, and the incident beam into the origin
    const sphere = ewald ? ewaldSphere(lat, wavelength, omega) : null;
    if (sphere) {
      const [px, py] = project(sphere.centre);
      ctx.beginPath();
      ctx.arc(px, py, sphere.k * s, 0, 2 * Math.PI);
      ctx.fillStyle = `rgba(${SPHERE},0.05)`;
      ctx.fill();
      ctx.strokeStyle = `rgba(${SPHERE},0.6)`;
      ctx.lineWidth = 1.2 * dpr;
      ctx.stroke();
      ctx.strokeStyle = HOT;
      ctx.globalAlpha = 0.75;
      ctx.setLineDash([6 * dpr, 4 * dpr]);
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(ox, oy);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }

    // the reflections, back to front
    const value = stretch(lat.intensity, { gain, log });
    const n = lat.count;
    const sx = new Float32Array(n);
    const sy = new Float32Array(n);
    const sz = new Float32Array(n);
    const order = [];
    for (let i = 0; i < n; i++) {
      if (value[i] <= 1e-3 && !nodes) continue;
      const p = [lat.Q[3 * i], lat.Q[3 * i + 1], lat.Q[3 * i + 2]];
      [sx[i], sy[i], sz[i]] = project(p);
      if (sx[i] < -30 || sy[i] < -30 || sx[i] > w + 30 || sy[i] > h + 30)
        continue;
      order.push(i);
    }
    order.sort((a, b) => sz[a] - sz[b]);

    const onSphere = (i) => {
      if (!sphere) return false;
      const dx = lat.Q[3 * i] - sphere.centre[0];
      const dy = lat.Q[3 * i + 1] - sphere.centre[1];
      const dz = lat.Q[3 * i + 2] - sphere.centre[2];
      return Math.abs(Math.hypot(dx, dy, dz) - sphere.k) < tol;
    };

    const drawn = [];
    const hot = [];
    for (const i of order) {
      // 0 at the far edge of the ball, 1 at the near one
      const depth = 0.5 + 0.5 * Math.max(-1, Math.min(1, sz[i] / lat.qMax));
      let alpha = 0.3 + 0.7 * depth;
      if (plane && lat.zoneIndex[i] !== lat.layer) alpha *= 0.3;
      const x = sx[i];
      const y = sy[i];
      if (value[i] <= 1e-3) {
        // a lattice node that scatters too weakly to show as a spot
        ctx.globalAlpha = alpha * 0.7;
        ctx.fillStyle = DIM;
        ctx.fillRect(x - 0.75 * dpr, y - 0.75 * dpr, 1.5 * dpr, 1.5 * dpr);
        drawn.push({ x, y, r: 1.5 * dpr, i });
        continue;
      }
      const r =
        spotRadius(value[i], spotScale * 0.6, 1) * dpr * (0.75 + 0.5 * depth);
      ctx.globalAlpha = alpha;
      ctx.fillStyle = sampleLut(lut, value[i]);
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 2 * Math.PI);
      ctx.fill();
      const spot = { x, y, r, i };
      drawn.push(spot);
      // ringed only when there is a spot to see; the readout says it of any
      if (value[i] >= 0.05 && onSphere(i)) hot.push(spot);
    }
    ctx.globalAlpha = 1;

    // the ones in diffracting position, ringed over everything
    ctx.strokeStyle = HOT;
    ctx.lineWidth = 1.6 * dpr;
    for (const p of hot) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r + 3 * dpr, 0, 2 * Math.PI);
      ctx.stroke();
    }
    const picked = hover !== null ? drawn.find((p) => p.i === hover) : null;
    if (picked) {
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.4 * dpr;
      ctx.beginPath();
      ctx.arc(picked.x, picked.y, picked.r + 5 * dpr, 0, 2 * Math.PI);
      ctx.stroke();
    }

    // the origin, where the direct beam leaves reciprocal space
    ctx.strokeStyle = "#5a6684";
    ctx.fillStyle = "#161b28";
    ctx.lineWidth = 1 * dpr;
    ctx.beginPath();
    ctx.arc(ox, oy, 4 * dpr, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();

    // Labels: the hovered reflection, those on the sphere, then the strong
    // ones of the section plane, or with no plane the strongest of all.
    // Labelling every strong reflection of the whole lattice only buries it.
    if (labels || hot.length || picked) {
      const items = [];
      const label = (p, rank) =>
        items.push({
          text: formatHkl([
            lat.hkl[3 * p.i],
            lat.hkl[3 * p.i + 1],
            lat.hkl[3 * p.i + 2],
          ]),
          x: p.x,
          y: p.y - p.r - 2 * dpr,
          rank,
        });
      if (picked) label(picked, 10);
      // the 40 strongest of those on the sphere
      for (const p of [...hot]
        .sort((a, b) => value[b.i] - value[a.i])
        .slice(0, 40))
        label(p, 2 + value[p.i]);
      if (labels)
        for (const p of drawn)
          if (
            value[p.i] >= labelThreshold &&
            (!plane || lat.zoneIndex[p.i] === lat.layer)
          )
            label(p, value[p.i]);
      v._drawLabels(items, plane ? 200 : 100);
    }

    this._drawKey(lat, w, h, s);
    this.drawn = drawn;
    return hot.map((p) => p.i);
  }

  /**
   * The three reciprocal axes as they point on screen, foreshortened, and a
   * scale bar for lengths across the screen.
   */
  _drawKey(lat, w, h, s) {
    const ctx = this.view.ctx;
    const dpr = this.view.dpr;
    const R = this.R;
    const ox = 54 * dpr;
    const oy = h - 54 * dpr;
    const len = 30 * dpr;
    ctx.fillStyle = "rgba(9,11,18,0.72)";
    ctx.fillRect(ox - 44 * dpr, oy - 52 * dpr, 104 * dpr, 66 * dpr);
    ctx.font = `${9.5 * dpr}px Consolas, ui-monospace, monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    lat.axes.forEach((a, j) => {
      const m = Math.hypot(a[0], a[1], a[2]);
      const gx = (R[0][0] * a[0] + R[0][1] * a[1] + R[0][2] * a[2]) / m;
      const gy = (R[1][0] * a[0] + R[1][1] * a[1] + R[1][2] * a[2]) / m;
      const [name, colour] = AXES[j];
      const dx = gx * len;
      const dy = -gy * len;
      ctx.strokeStyle = colour;
      ctx.fillStyle = colour;
      ctx.lineWidth = 1.3 * dpr;
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.lineTo(ox + dx, oy + dy);
      ctx.stroke();
      if (Math.hypot(dx, dy) > 4 * dpr) {
        const t = Math.atan2(dy, dx);
        ctx.beginPath();
        ctx.moveTo(ox + dx, oy + dy);
        ctx.lineTo(ox + dx - 5 * dpr * Math.cos(t - 0.4), oy + dy - 5 * dpr * Math.sin(t - 0.4));
        ctx.lineTo(ox + dx - 5 * dpr * Math.cos(t + 0.4), oy + dy - 5 * dpr * Math.sin(t + 0.4));
        ctx.fill();
      }
      ctx.fillText(name, ox + dx * 1.35 + (Math.abs(dx) < 3 ? 9 * dpr : 0), oy + dy * 1.35);
    });

    const step = niceStep(w / s);
    const px = step * s;
    const x0 = w - 24 * dpr - px;
    const y0 = h - 30 * dpr;
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

  /** The reflection nearest a client-space point, front-most on a tie, or null. */
  pick(clientX, clientY) {
    if (!this.drawn) return null;
    const box = this.view.canvas.getBoundingClientRect();
    const x = (clientX - box.left) * this.view.dpr;
    const y = (clientY - box.top) * this.view.dpr;
    let best = null;
    let bestD = Infinity;
    // drawn is back to front, so on a tie the later, nearer spot wins
    for (const p of this.drawn) {
      const d = Math.hypot(p.x - x, p.y - y);
      if (d <= Math.max(p.r, 7 * this.view.dpr) && d <= bestD) {
        bestD = d;
        best = p.i;
      }
    }
    return best;
  }
}
