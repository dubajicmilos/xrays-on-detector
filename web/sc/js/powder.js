/*! Single-Crystal Diffraction · © 2026 Miloš Dubajić · MIT · https://github.com/dubajicmilos/xrays-on-detector */

/**
 * Powder pattern, ported from single_crystal/powder.py.
 *
 * Every reflection out to d min is enumerated and those sharing a d-spacing
 * are summed, so multiplicity is counted by construction rather than looked up
 * from the Laue class. Accidental overlaps -- two unrelated families at the
 * same d -- merge too, which is what a real diffractometer does as well.
 *
 * The Lorentz factor 1 / (sin^2(theta) cos(theta)) applies to all three
 * radiations. The polarisation factor (1 + cos^2(2 theta)) / 2 is x-rays only:
 * neutrons scatter off nuclei with no polarisation dependence, and it is not
 * meaningful for the electron case here either.
 *
 * Two merging rules are available. The default keeps reflections apart when
 * their d-spacings differ by more than dTol relatively; passing twoThetaTol > 0
 * merges on a 2theta window instead, which is the instrument's view of what is
 * resolvable (a relative d rule separates near-degenerate reflections of a
 * pseudocubic cell at every angle, so the peak count follows the cell size
 * rather than the resolution).
 *
 * When comparing against pymatgen's XRDCalculator, note that it applies no
 * Debye-Waller damping (its get_pattern takes debye_waller_factors only in some
 * builds). Either compare with B = 0 throughout, or pass the factors through,
 * otherwise the comparison is of two different models and the thermal term
 * shows up as an intensity difference that has nothing to do with the physics
 * under test.
 */

import { TWO_PI } from "../../js/physics.js";

export function computePowder(
  structure,
  {
    wavelength = 1.5406,
    radiation = "xray",
    twoThetaMax = 120,
    fwhm = 0.15,
    nPoints = 2000,
    dTol = 1e-5,
    // Merge on a 2theta window instead of a relative d-spacing when this is > 0.
    // Two reflections are then one peak when their 2theta differ by less than
    // this many degrees, which is how a diffractometer resolves them: a
    // relative d rule keeps near-degenerate reflections of a pseudocubic cell
    // apart at every angle, so the peak list grows with the cell size rather
    // than with what an instrument could actually separate.
    twoThetaTol = 0,
  } = {},
) {
  if (!(wavelength > 0)) throw new Error("wavelength must be positive");
  const ttMax = Math.min(twoThetaMax, 179.9);
  // Bragg: lambda = 2 d sin(theta), so the largest reachable 2*theta fixes the
  // smallest d worth enumerating.
  const dMin = wavelength / (2 * Math.sin((ttMax / 2) * (Math.PI / 180)));

  const hkl = structure.hklWithin(dMin);
  if (!hkl.length)
    throw new Error(
      `no reflections with d ≥ ${dMin.toFixed(3)} Å; try a shorter wavelength ` +
        "or a larger 2θ limit",
    );

  const q = new Float64Array(hkl.length / 3);
  const raw = structure.intensity(hkl, radiation, q);

  const rows = [];
  for (let i = 0; i < q.length; i++) {
    const d = TWO_PI / q[i];
    const sinTheta = wavelength / (2 * d);
    if (sinTheta > 1) continue;
    const theta = Math.asin(sinTheta);
    const tt = (2 * theta * 180) / Math.PI;
    if (tt > ttMax) continue;
    let lp = 1 / (Math.sin(theta) ** 2 * Math.cos(theta));
    if (radiation === "xray") lp *= (1 + Math.cos(2 * theta) ** 2) / 2;
    rows.push({
      d,
      tt,
      I: raw[i] * lp,
      F2: raw[i],
      hkl: [hkl[3 * i], hkl[3 * i + 1], hkl[3 * i + 2]],
    });
  }
  if (!rows.length) throw new Error("no reflections fall inside the 2θ range");

  // Merge families sharing a d-spacing: sort by d and cut where the relative
  // gap exceeds the tolerance.
  rows.sort((a, b) => b.d - a.d);
  const peaks = [];
  let group = [rows[0]];
  for (let i = 1; i < rows.length; i++) {
    const split =
      twoThetaTol > 0
        ? Math.abs(rows[i].tt - rows[i - 1].tt) > twoThetaTol
        : Math.abs(rows[i].d - rows[i - 1].d) > dTol * rows[i - 1].d;
    if (split) {
      peaks.push(group);
      group = [];
    }
    group.push(rows[i]);
  }
  peaks.push(group);

  const twoTheta = new Float64Array(peaks.length);
  const intensity = new Float64Array(peaks.length);
  const dOut = new Float64Array(peaks.length);
  const multiplicity = new Int32Array(peaks.length);
  const hklOut = [];
  for (let i = 0; i < peaks.length; i++) {
    const g = peaks[i];
    twoTheta[i] = g.reduce((s, r) => s + r.tt, 0) / g.length;
    intensity[i] = g.reduce((s, r) => s + r.I, 0);
    dOut[i] = g.reduce((s, r) => s + r.d, 0) / g.length;
    multiplicity[i] = g.length;
    // Representative index: the one a crystallographer would write, i.e. the
    // most positive of the family.
    hklOut.push(mostPositive(g.map((r) => r.hkl)));
  }

  let top = 0;
  for (const v of intensity) if (v > top) top = v;
  if (top > 0)
    for (let i = 0; i < intensity.length; i++) intensity[i] *= 100 / top;

  // Every reflection of every peak, in peak order, for peakFamilies to split a
  // peak into its families when it is asked about. Kept flat: a large cell
  // has tens of thousands of reflections, and the readout needs one peak's.
  const n = rows.length;
  const members = {
    start: new Int32Array(peaks.length + 1),
    hkl: new Int32Array(3 * n),
    F2: new Float64Array(n),
    d: new Float64Array(n),
    tt: new Float64Array(n),
    I: new Float64Array(n),
    maxF2: 0,
    f2Floor: 0,
  };
  let at = 0;
  let maxF2 = 0;
  for (let i = 0; i < peaks.length; i++) {
    members.start[i] = at;
    for (const r of peaks[i]) {
      members.hkl[3 * at] = r.hkl[0];
      members.hkl[3 * at + 1] = r.hkl[1];
      members.hkl[3 * at + 2] = r.hkl[2];
      members.F2[at] = r.F2;
      members.d[at] = r.d;
      members.tt[at] = r.tt;
      members.I[at] = top > 0 ? (r.I * 100) / top : r.I;
      if (r.F2 > maxF2) maxF2 = r.F2;
      at++;
    }
  }
  members.start[peaks.length] = at;
  members.maxF2 = maxF2;
  // An extinct reflection's |F|^2 is rounding noise, some 1e-25 of the
  // strongest; anything under this floor is taken as zero.
  members.f2Floor = 1e-9 * maxF2;

  // Peaks made only of extinct reflections: the symmetry of the structure
  // forbids them, so they get no tick and no readout, as a reflection list
  // generated from the space group would not have them at all.
  const absent = new Uint8Array(peaks.length);
  for (let i = 0; i < peaks.length; i++) {
    let none = true;
    for (let r = members.start[i]; r < members.start[i + 1] && none; r++)
      if (members.F2[r] > members.f2Floor) none = false;
    absent[i] = none ? 1 : 0;
  }

  const { x, y } = powderProfile(twoTheta, intensity, ttMax, fwhm, nPoints);

  return {
    count: peaks.length,
    twoTheta,
    intensity,
    d: dOut,
    multiplicity,
    hkl: hklOut,
    absent,
    members,
    // the cell's own point symmetry, which peakFamilies sorts reflections by
    latticeOps: latticeOps(structure),
    x,
    y,
    fwhm,
    wavelength,
    radiation,
  };
}

/** Of a list of [h, k, l], the most positive: largest h, then k, then l. */
function mostPositive(list) {
  let best = list[0];
  for (const h of list)
    if (
      h[0] > best[0] ||
      (h[0] === best[0] && (h[1] > best[1] || (h[1] === best[1] && h[2] > best[2])))
    )
      best = h;
  return best;
}

const opsCache = new WeakMap();

/**
 * The point symmetry of the lattice itself: every integer matrix M with
 * entries -1, 0 or 1 and determinant +-1 that keeps |B M h| = |B h| for all
 * h, i.e. M^T G* M = G* for the reciprocal metric G* = B^T B. For a cubic cell
 * that is the 48 operations of m-3m, for a hexagonal one the 24 of 6/mmm, in
 * that cell's own axes. Matrices act on (h, k, l) as a column.
 *
 * Equal within 2e-5 of the largest diagonal term: about the precision a CIF
 * gives a cell to, and the scale of the d tolerance the peaks are merged by.
 */
export function latticeOps(structure) {
  let ops = opsCache.get(structure);
  if (ops) return ops;
  const B = structure.B;
  const G = [0, 1, 2].map((i) =>
    [0, 1, 2].map((j) => B[0][i] * B[0][j] + B[1][i] * B[1][j] + B[2][i] * B[2][j]),
  );
  const tol = 2e-5 * Math.max(G[0][0], G[1][1], G[2][2]);
  const form = (u, v) => {
    let s = 0;
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) s += u[i] * G[i][j] * v[j];
    return s;
  };
  const vectors = [];
  for (let a = -1; a <= 1; a++)
    for (let b = -1; b <= 1; b++)
      for (let c = -1; c <= 1; c++) if (a || b || c) vectors.push([a, b, c]);
  // Column j of M is the image of the j-th basis vector, so it must be a
  // vector of the same length; the angles between the columns are checked
  // after.
  const unitCols = [0, 1, 2].map((j) =>
    vectors.filter((v) => Math.abs(form(v, v) - G[j][j]) <= tol),
  );
  ops = [];
  for (const c0 of unitCols[0])
    for (const c1 of unitCols[1]) {
      if (Math.abs(form(c0, c1) - G[0][1]) > tol) continue;
      for (const c2 of unitCols[2]) {
        if (Math.abs(form(c0, c2) - G[0][2]) > tol) continue;
        if (Math.abs(form(c1, c2) - G[1][2]) > tol) continue;
        const det =
          c0[0] * (c1[1] * c2[2] - c2[1] * c1[2]) -
          c1[0] * (c0[1] * c2[2] - c2[1] * c0[2]) +
          c2[0] * (c0[1] * c1[2] - c1[1] * c0[2]);
        if (Math.abs(det) !== 1) continue;
        // row-major, M[r][j] = column j's r-th entry
        ops.push([c0[0], c1[0], c2[0], c0[1], c1[1], c2[1], c0[2], c1[2], c2[2]]);
      }
    }
  opsCache.set(structure, ops);
  return ops;
}

const mul3 = (A, B) => {
  const C = new Array(9);
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++)
      C[3 * r + c] =
        A[3 * r] * B[c] + A[3 * r + 1] * B[3 + c] + A[3 * r + 2] * B[6 + c];
  return C;
};

const laueCache = new WeakMap();

/**
 * The crystal's Laue group, as the operations of the lattice's symmetry that
 * the diffraction pattern keeps: those M with |F(M h)|^2 = |F(h)|^2 over the
 * reflection list. It is read off the intensities because a P1 file, or a
 * bundled structure, carries no space group to read it from.
 *
 * "Equal" allows for coordinates the file rounded: a hexagonal 1/3 written as
 * 0.3333 breaks the symmetry at about 1e-4, which a strict test would take as
 * a lower symmetry and split every family. A pair counts as equal within 2 %
 * of its sum, or 2e-3 of the strongest |F|^2 for weak ones; nearer than that,
 * two reflections at one d cannot be told apart in a powder pattern anyway.
 * The operations that pass are closed under products, so the families are
 * the orbits of a group. Large lists are tested on an even sample of 20 000.
 */
export function diffractionOps(p) {
  let ops = laueCache.get(p);
  if (ops) return ops;
  const m = p.members;
  const n = m.F2.length;
  let span = 0;
  for (let r = 0; r < 3 * n; r++) span = Math.max(span, Math.abs(m.hkl[r]));
  const w = 2 * span + 1;
  // one number per hkl while it stays exact in a double, a string past that
  const key =
    w ** 3 < Number.MAX_SAFE_INTEGER
      ? (h, k, l) => ((h + span) * w + (k + span)) * w + (l + span)
      : (h, k, l) => `${h},${k},${l}`;
  const F2of = new Map();
  for (let r = 0; r < n; r++)
    F2of.set(key(m.hkl[3 * r], m.hkl[3 * r + 1], m.hkl[3 * r + 2]), m.F2[r]);
  const floor = 2e-3 * m.maxF2;
  const stride = Math.max(1, Math.floor(n / 20000));
  const keeps = (M) => {
    for (let r = 0; r < n; r += stride) {
      const h = m.hkl[3 * r];
      const k = m.hkl[3 * r + 1];
      const l = m.hkl[3 * r + 2];
      const a = M[0] * h + M[1] * k + M[2] * l;
      const b = M[3] * h + M[4] * k + M[5] * l;
      const c = M[6] * h + M[7] * k + M[8] * l;
      // an image just past the 2θ limit has no partner to compare with
      if (Math.abs(a) > span || Math.abs(b) > span || Math.abs(c) > span)
        continue;
      const g = F2of.get(key(a, b, c));
      if (g === undefined) continue;
      const f = m.F2[r];
      if (Math.abs(g - f) > 0.02 * (f + g) + floor) return false;
    }
    return true;
  };
  ops = p.latticeOps.filter(keeps);
  const seen = new Set(ops.map((M) => M.join(",")));
  for (let i = 0; i < ops.length; i++)
    for (let j = 0; j < ops.length; j++) {
      const P = mul3(ops[i], ops[j]);
      const name = P.join(",");
      if (!seen.has(name)) {
        seen.add(name);
        ops.push(P);
      }
    }
  laueCache.set(p, ops);
  return ops;
}

/**
 * The families one powder peak is made of, as a reflection list gives them:
 * [{hkl, m, d, tt, F2, I, absent}], strongest first, with I on the plot's
 * 0-100 scale, hkl the most positive member, and `absent` set for a family
 * whose |F| is zero (an extinction).
 *
 * A family is an orbit of the crystal's Laue group (diffractionOps), so 330
 * and 411 of a cubic cell stay apart although they share a d, and in m-3 so
 * do hkl and khl, which share a d but not an |F|.
 */
export function peakFamilies(p, i) {
  const m = p.members;
  const ops = diffractionOps(p);
  const fams = new Map();
  for (let r = m.start[i]; r < m.start[i + 1]; r++) {
    const h = m.hkl[3 * r];
    const k = m.hkl[3 * r + 1];
    const l = m.hkl[3 * r + 2];
    // the orbit's most positive member names it
    let best = null;
    for (const M of ops) {
      const img = [
        M[0] * h + M[1] * k + M[2] * l,
        M[3] * h + M[4] * k + M[5] * l,
        M[6] * h + M[7] * k + M[8] * l,
      ];
      if (best === null || mostPositive([best, img]) === img) best = img;
    }
    const name = best ? best.join(",") : `${h},${k},${l}`;
    let f = fams.get(name);
    if (!f) fams.set(name, (f = { list: [], d: 0, tt: 0, I: 0, F2: 0 }));
    f.list.push([h, k, l]);
    f.d += m.d[r];
    f.tt += m.tt[r];
    f.I += m.I[r];
    f.F2 += m.F2[r];
  }
  return [...fams.values()]
    .map((f) => {
      const count = f.list.length;
      const F2 = f.F2 / count;
      return {
        hkl: mostPositive(f.list),
        m: count,
        d: f.d / count,
        tt: f.tt / count,
        F2,
        I: f.I,
        absent: F2 <= m.f2Floor,
      };
    })
    .sort((a, b) => b.I - a.I);
}

/**
 * The trace: a Gaussian of `fwhm` degrees for every peak, sampled at
 * `nPoints` from 2theta = 0 to `ttMax`.
 *
 * computePowder samples it at its nPoints for the result; the plot samples
 * it again finely enough to catch every apex, since a peak narrower than the
 * sample spacing has its top fall between two samples and is drawn low.
 */
export function powderProfile(twoTheta, intensity, ttMax, fwhm, nPoints) {
  const x = new Float64Array(nPoints);
  const y = new Float64Array(nPoints);
  const sigma = fwhm / (2 * Math.sqrt(2 * Math.LN2));
  for (let i = 0; i < nPoints; i++) x[i] = (ttMax * i) / (nPoints - 1);
  // A Gaussian is dead beyond four sigma, so each peak only touches the points
  // near it; painting the whole trace per peak is what makes this slow.
  const step = ttMax / (nPoints - 1);
  for (let i = 0; i < twoTheta.length; i++) {
    const I = intensity[i];
    if (I < 1e-6) continue;
    const lo = Math.max(0, Math.floor((twoTheta[i] - 4 * sigma) / step));
    const hi = Math.min(
      nPoints - 1,
      Math.ceil((twoTheta[i] + 4 * sigma) / step),
    );
    for (let j = lo; j <= hi; j++)
      y[j] += I * Math.exp(-0.5 * ((x[j] - twoTheta[i]) / sigma) ** 2);
  }
  return { x, y };
}
