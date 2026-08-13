/*! Single-Crystal Diffraction · © 2026 Miloš Dubajić · MIT · https://github.com/dubajicmilos/xrays-on-detector */

/**
 * Selected-area electron diffraction down a zone axis.
 * A port of single_crystal/tem.py.
 *
 * A flat section shows one layer of the reciprocal lattice exactly. A real
 * electron pattern does not: the Ewald sphere is very large but not flat, so
 * it cuts the zero layer over a disc and clips the layers above it into rings.
 * Those rings are the higher-order Laue zones.
 *
 * At 200 kV the wavelength is 0.0251 Angstrom, so k = 2*pi/lambda is 250
 * 1/Angstrom against reciprocal spacings of order 1. The sphere is nearly flat
 * over the zero layer, which is why a section is a good picture of it near the
 * middle and a poor one at the edge.
 *
 * Thickness enters through the relrod: a slab of thickness t turns each
 * reciprocal lattice point into a rod of length ~2/t along the beam, so a
 * reflection with excitation error s_g contributes I = |F|^2 sinc^2(pi t s_g).
 * That is the kinematic result. Dynamical scattering, which really governs
 * electron intensities in a thick crystal, is out of scope and the app says so
 * rather than quietly implying otherwise.
 */

import { cross, dot, matVec, norm, TWO_PI, unit } from "../../js/physics.js";
import { electronWavelength } from "../../js/scatter.js";
import { layerStep, reduceZone, zoneBasis } from "./section.js";

const sinc = (x) => (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x));

/** Smallest d that still contains Laue ring `zone`, in Angstrom. */
export function dMinForZone(structure, uvw, zone, kv = 200) {
  const { uvw: z } = reduceZone(uvw);
  if (zone <= 0) return Infinity;
  const k = TWO_PI / electronWavelength(kv);
  const H = TWO_PI / norm(matVec(structure.A, z));
  const r2 = 2 * k * H * zone - (H * zone) ** 2;
  if (r2 <= 0) return Infinity;
  return TWO_PI / (Math.sqrt(r2) * 1.06);
}

export function computeTem(
  structure,
  {
    uvw = [0, 0, 1],
    kv = 200,
    thickness = 50,
    dMin = 0.5,
    maxZone = 0,
    sMax = null,
  } = {},
) {
  const { uvw: zone } = reduceZone(uvw);
  const lam = electronWavelength(kv);
  const k = TWO_PI / lam;
  const sCut = sMax === null ? 1 / Math.max(thickness, 1e-6) : sMax;

  const B = structure.B;
  const [g1, g2] = zoneBasis(B, zone);
  const p = layerStep(zone);
  const normal = matVec(structure.A, zone);
  const nHat = unit(normal);
  const qMax = TWO_PI / dMin;

  // Where each Laue ring falls, so the app can say "the first ring is at 23
  // 1/Angstrom" instead of just showing nothing.
  const H = TWO_PI / norm(normal);
  const zoneRadii = [];
  for (let n = 0; n <= maxZone; n++) {
    const r2 = 2 * k * H * n - (H * n) ** 2;
    zoneRadii.push(r2 >= 0 ? Math.sqrt(r2) : null);
  }

  const e1 = matVec(B, g1);
  const e2 = matVec(B, g2);
  const g11 = dot(e1, e1),
    g12 = dot(e1, e2),
    g22 = dot(e2, e2);
  const det = g11 * g22 - g12 * g12;
  const gi = [
    [g22 / det, -g12 / det],
    [-g12 / det, g11 / det],
  ];
  const n1 = norm(e1),
    n2 = norm(e2);

  const seen = new Set();
  const hkl = [];
  for (let layer = 0; layer <= maxZone; layer++) {
    const origin = matVec(
      B,
      p.map((v) => v * layer),
    );
    const oIn = origin.map((v, i) => v - dot(origin, nHat) * nHat[i]);
    const R = qMax + norm(oIn);
    const aMax =
      Math.ceil(R * (Math.abs(gi[0][0]) * n1 + Math.abs(gi[0][1]) * n2)) + 1;
    const bMax =
      Math.ceil(R * (Math.abs(gi[1][0]) * n1 + Math.abs(gi[1][1]) * n2)) + 1;
    for (let a = -aMax; a <= aMax; a++)
      for (let b = -bMax; b <= bMax; b++) {
        const h = layer * p[0] + a * g1[0] + b * g2[0];
        const kk = layer * p[1] + a * g1[1] + b * g2[1];
        const l = layer * p[2] + a * g1[2] + b * g2[2];
        if (h === 0 && kk === 0 && l === 0) continue;
        const key = `${h},${kk},${l}`;
        if (seen.has(key)) continue;
        const qx = B[0][0] * h + B[0][1] * kk + B[0][2] * l;
        const qy = B[1][0] * h + B[1][1] * kk + B[1][2] * l;
        const qz = B[2][0] * h + B[2][1] * kk + B[2][2] * l;
        if (Math.hypot(qx, qy, qz) > qMax) continue;

        // Ewald sphere of radius k through the origin, centred at -k_i. The
        // beam runs antiparallel to the zone axis -- [uvw] points from the
        // specimen towards the gun, the usual convention -- so the centre is
        // at +k nHat and a point is on the sphere when |Q - k nHat| = k.
        //
        // The sign is not cosmetic. The sphere curves away from the beam, so
        // the layers it clips are on the far side: solving for Q = (r, 0, z)
        // gives z = (r^2 + z^2) / 2k > 0. Reverse it and every higher-order
        // Laue zone silently disappears.
        const cx = qx - k * nHat[0];
        const cy = qy - k * nHat[1];
        const cz = qz - k * nHat[2];
        const sg = k - Math.hypot(cx, cy, cz);
        if (Math.abs(sg) > sCut) continue;
        seen.add(key);
        hkl.push(h, kk, l);
      }
  }

  const xAxis = unit(e1);
  const yAxis = unit(cross(nHat, xAxis));
  const idx = Int32Array.from(hkl);
  const count = idx.length / 3;
  const q = new Float64Array(count);
  const base = count
    ? structure.intensity(idx, "electron", q)
    : new Float64Array(0);

  const x = new Float64Array(count);
  const y = new Float64Array(count);
  const d = new Float64Array(count);
  const sg = new Float64Array(count);
  const laueZone = new Int32Array(count);
  const intensity = new Float64Array(count);
  const Q = structure.q(idx);
  for (let i = 0; i < count; i++) {
    const v = [Q[3 * i], Q[3 * i + 1], Q[3 * i + 2]];
    x[i] = dot(v, xAxis);
    y[i] = dot(v, yAxis);
    d[i] = TWO_PI / q[i];
    sg[i] = k - Math.hypot(...v.map((c, j) => c - k * nHat[j]));
    laueZone[i] =
      idx[3 * i] * zone[0] +
      idx[3 * i + 1] * zone[1] +
      idx[3 * i + 2] * zone[2];
    intensity[i] = base[i] * sinc(thickness * sg[i]) ** 2;
  }

  const zonesVisible = zoneRadii.filter((r) => r !== null && r <= qMax).length;
  return {
    count,
    hkl: idx,
    x,
    y,
    q,
    d,
    intensity,
    sg,
    laueZone,
    uvw: zone,
    g1,
    g2,
    xAxis,
    yAxis,
    kv,
    wavelength: lam,
    thickness,
    qMax,
    zoneRadii,
    zonesVisible,
    radiation: "electron",
  };
}
