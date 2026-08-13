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
      `no reflections with d ≥ ${dMin.toFixed(3)} Å; try a longer wavelength ` +
        "or a smaller 2θ limit",
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
    if (Math.abs(rows[i].d - rows[i - 1].d) > dTol * rows[i - 1].d) {
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
    const best = g
      .map((r) => r.hkl)
      .sort((p, r) => p[0] - r[0] || p[1] - r[1] || p[2] - r[2])
      .pop();
    hklOut.push(best);
  }

  let top = 0;
  for (const v of intensity) if (v > top) top = v;
  if (top > 0)
    for (let i = 0; i < intensity.length; i++) intensity[i] *= 100 / top;

  const x = new Float64Array(nPoints);
  const y = new Float64Array(nPoints);
  const sigma = fwhm / (2 * Math.sqrt(2 * Math.LN2));
  for (let i = 0; i < nPoints; i++) x[i] = (ttMax * i) / (nPoints - 1);
  // A Gaussian is dead beyond four sigma, so each peak only touches the points
  // near it; painting the whole trace per peak is what makes this slow.
  const step = ttMax / (nPoints - 1);
  for (let i = 0; i < peaks.length; i++) {
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

  return {
    count: peaks.length,
    twoTheta,
    intensity,
    d: dOut,
    multiplicity,
    hkl: hklOut,
    x,
    y,
    wavelength,
    radiation,
  };
}
