/*! Single-Crystal Diffraction · © 2026 Miloš Dubajić · MIT · https://github.com/dubajicmilos/xrays-on-detector */

/**
 * Plane sections of the weighted reciprocal lattice.
 *
 * A port of single_crystal/section.py. A section is named by a zone axis
 * [uvw] and a layer n, and holds every reflection obeying the zone law
 *
 *     h u + k v + l w = n
 *
 * which is the plane of reciprocal space normal to the *direct* lattice vector
 * u a + v b + w c. That is what a precession camera records, and it is the only
 * reading of "the [uvw] cut" with lattice rows in it for a general cell: the
 * plane perpendicular to the *reciprocal* vector (uvw) usually contains no
 * lattice points at all unless the cell is orthogonal.
 *
 *     [001] n=0  ->  the hk0 plane
 *     [100] n=3  ->  the 3kl plane
 *     [110] n=0  ->  a diagonal cut, h + k = 0
 *
 * Plot coordinates are Cartesian, in 1/Angstrom, through Q = B . hkl.
 * Projecting the raw (h,k,l) triple instead is only correct for a cubic cell
 * and shears every other one.
 */

import { cross, dot, matVec, norm, TWO_PI, unit } from "../../js/physics.js";

const gcd = (a, b) => {
  a = Math.abs(a);
  b = Math.abs(b);
  while (b) [a, b] = [b, a % b];
  return a;
};

/** Primitive direction and the common factor divided out. */
export function reduceZone(uvw) {
  const t = uvw.map((v) => Math.round(v));
  if (!t.some((v) => v !== 0)) throw new Error("the zone axis cannot be 0 0 0");
  const g = gcd(gcd(t[0], t[1]), t[2]);
  return { uvw: t.map((v) => v / g), factor: g };
}

/** [g, s, t] with a s + b t = g = gcd(a, b). */
function bezout(a, b) {
  let [oldR, r] = [a, b];
  let [oldS, s] = [1, 0];
  let [oldT, t] = [0, 1];
  while (r !== 0) {
    const q = Math.floor(oldR / r);
    [oldR, r] = [r, oldR - q * r];
    [oldS, s] = [s, oldS - q * s];
    [oldT, t] = [t, oldT - q * t];
  }
  return [oldR, oldS, oldT];
}

/** An integer p with p . [uvw] = 1, so layer n starts at n p. */
export function layerStep(uvw) {
  const [u, v, w] = uvw;
  const [g, s, t] = bezout(u, v);
  const [g2, m, n] = bezout(g, w);
  if (Math.abs(g2) !== 1)
    throw new Error(`zone axis ${uvw.join(" ")} is not primitive`);
  const sign = g2 === 1 ? 1 : -1;
  const p = [s * m * sign, t * m * sign, n * sign];
  if (p[0] * u + p[1] * v + p[2] * w !== 1)
    throw new Error(`failed to solve the zone law for ${uvw.join(" ")}`);
  return p;
}

/**
 * Two integer hkl spanning the reflections of the zero layer.
 *
 * The pair achieving the two shortest |Q| is returned, which for a rank-2
 * lattice is guaranteed to be a basis and gives the axes a crystallographer
 * would draw. The search box always contains them: for a primitive [uvw] the
 * kernel has vectors with entries no larger than max|u,v,w|.
 */
export function zoneBasis(B, uvw) {
  const R = Math.max(...uvw.map(Math.abs)) + 1;
  const cand = [];
  for (let h = -R; h <= R; h++)
    for (let k = -R; k <= R; k++)
      for (let l = -R; l <= R; l++) {
        if (h === 0 && k === 0 && l === 0) continue;
        if (h * uvw[0] + k * uvw[1] + l * uvw[2] !== 0) continue;
        cand.push([h, k, l]);
      }
  if (cand.length < 2) throw new Error(`no reflections lie in the zone ${uvw}`);
  cand.sort((p, q) => norm(matVec(B, p)) - norm(matVec(B, q)));

  const g1 = cand[0];
  let g2 = null;
  for (let i = 1; i < cand.length; i++) {
    const c = cross(g1, cand[i]);
    if (c[0] || c[1] || c[2]) {
      g2 = cand[i];
      break;
    }
  }
  if (!g2) throw new Error(`the zone ${uvw} spans only one direction`);

  // A primitive zone axis has a kernel of covolume |uvw| in index space, so
  // this is an exact check that the pair generates every reflection of the
  // zone rather than a sublattice of them.
  const area = norm(cross(g1, g2));
  const expect = norm(uvw);
  if (Math.abs(area - expect) > 1e-6 * Math.max(1, expect))
    throw new Error(`zone basis spans a sublattice of zone ${uvw}`);

  // Both signs are equally valid and the sort returns whichever it reached
  // first. Fix them so an axis reads [100] and not [-100], then orient the
  // pair so g1 x g2 runs along the zone axis; without this the picture
  // mirrors itself as the zone changes.
  const canonical = (g) => {
    for (const v of g) if (v) return v < 0 ? g.map((x) => -x) : g;
    return g;
  };
  const a = canonical(g1);
  let b = canonical(g2);
  if (dot(cross(a, b), uvw) < 0) b = b.map((x) => -x);
  return [a, b];
}

/**
 * Every reflection of the [uvw] zone at the given layer, out to dMin.
 * The 000 beam is excluded; it is not a reflection.
 */
export function computeSection(
  structure,
  { uvw = [0, 0, 1], layer = 0, dMin = 0.7, radiation = "xray" } = {},
) {
  const red = reduceZone(uvw);
  const zone = red.uvw;
  layer = Math.round(layer);
  const B = structure.B;
  const [g1, g2] = zoneBasis(B, zone);
  const p = layerStep(zone);
  if (!(dMin > 0)) throw new Error("d min must be positive");
  const qMax = TWO_PI / dMin;

  // Plane normal: the direct lattice vector u a + v b + w c. Q . normal is
  // 2*pi*n on layer n, which is what fixes the layer spacing.
  const normal = matVec(structure.A, zone);
  const nHat = unit(normal);
  const height = (TWO_PI * layer) / norm(normal);

  const empty = {
    count: 0,
    hkl: new Int32Array(0),
    x: new Float64Array(0),
    y: new Float64Array(0),
    q: new Float64Array(0),
    d: new Float64Array(0),
    intensity: new Float64Array(0),
    uvw: zone,
    layer,
    zoneFactor: red.factor,
    g1,
    g2,
    xAxis: [1, 0, 0],
    yAxis: [0, 1, 0],
    height,
    radiation,
  };
  if (Math.abs(height) > qMax) return empty;
  const rMax = Math.sqrt(Math.max(qMax * qMax - height * height, 0));

  const e1 = matVec(B, g1);
  const e2 = matVec(B, g2);
  const origin = matVec(
    B,
    p.map((v) => v * layer),
  );
  const oIn = origin.map((v, i) => v - dot(origin, nHat) * nHat[i]);

  // Solving the Gram system bounds the two in-plane indices exactly, rather
  // than guessing a box and hoping it was big enough.
  const g11 = dot(e1, e1),
    g12 = dot(e1, e2),
    g22 = dot(e2, e2);
  const det = g11 * g22 - g12 * g12;
  const gi = [
    [g22 / det, -g12 / det],
    [-g12 / det, g11 / det],
  ];
  const R = rMax + norm(oIn);
  const n1 = norm(e1),
    n2 = norm(e2);
  const aMax =
    Math.ceil(R * (Math.abs(gi[0][0]) * n1 + Math.abs(gi[0][1]) * n2)) + 1;
  const bMax =
    Math.ceil(R * (Math.abs(gi[1][0]) * n1 + Math.abs(gi[1][1]) * n2)) + 1;

  const hkl = [];
  for (let a = -aMax; a <= aMax; a++)
    for (let b = -bMax; b <= bMax; b++) {
      const h = layer * p[0] + a * g1[0] + b * g2[0];
      const k = layer * p[1] + a * g1[1] + b * g2[1];
      const l = layer * p[2] + a * g1[2] + b * g2[2];
      if (h === 0 && k === 0 && l === 0) continue;
      const qx = B[0][0] * h + B[0][1] * k + B[0][2] * l;
      const qy = B[1][0] * h + B[1][1] * k + B[1][2] * l;
      const qz = B[2][0] * h + B[2][1] * k + B[2][2] * l;
      if (Math.hypot(qx, qy, qz) <= qMax) hkl.push(h, k, l);
    }
  if (!hkl.length) return empty;

  const idx = Int32Array.from(hkl);
  const count = idx.length / 3;
  const q = new Float64Array(count);
  const intensity = structure.intensity(idx, radiation, q);

  // Plot axes: x along the shortest reciprocal row of the zone, y completing
  // a right-handed set with the plane normal.
  const xAxis = unit(e1);
  const yAxis = unit(cross(nHat, xAxis));

  const Q = structure.q(idx);
  const x = new Float64Array(count);
  const y = new Float64Array(count);
  const d = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    const v = [Q[3 * i], Q[3 * i + 1], Q[3 * i + 2]];
    x[i] = dot(v, xAxis);
    y[i] = dot(v, yAxis);
    d[i] = TWO_PI / q[i];
  }

  return {
    count,
    hkl: idx,
    x,
    y,
    q,
    d,
    intensity,
    uvw: zone,
    layer,
    zoneFactor: red.factor,
    g1,
    g2,
    xAxis,
    yAxis,
    height,
    radiation,
  };
}

/** "h + k = 3", the zone law currently in force, written out. */
export function zoneLaw(uvw, layer) {
  const parts = [];
  const names = ["h", "k", "l"];
  for (let i = 0; i < 3; i++) {
    const c = uvw[i];
    if (!c) continue;
    if (c === 1) parts.push(`+ ${names[i]}`);
    else if (c === -1) parts.push(`- ${names[i]}`);
    else parts.push(`${c > 0 ? "+" : "-"} ${Math.abs(c)}${names[i]}`);
  }
  return `${parts.join(" ").replace(/^\+\s*/, "")} = ${layer}`;
}
