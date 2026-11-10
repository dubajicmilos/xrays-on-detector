/*! Single-Crystal Diffraction · © 2026 Miloš Dubajić · MIT · https://github.com/dubajicmilos/xrays-on-detector */

/**
 * The whole weighted reciprocal lattice out to d min: what the 3D view draws
 * and what the full reflection list exports.
 *
 * Every reflection with d >= d min is listed with its Cartesian Q in the
 * crystal frame (Q = B . hkl, 2*pi convention), |Q|, d and |F|^2, and with its
 * index on the current zone axis, h u + k v + l w, which is the layer of the
 * section view it belongs to. The zone also fixes the frame the 3D view opens
 * in: x and y as the section draws them, and the zone axis towards the viewer,
 * so the opening picture is the zero-layer section seen face on.
 */

import { cross, matVec, norm, TWO_PI, unit } from "../../js/physics.js";
import { reduceZone, zoneBasis } from "./section.js";

/** More points than this cannot be drawn at an interactive rate. */
export const MAX_LATTICE = 200000;

export function computeLattice(
  structure,
  { dMin = 0.8, radiation = "xray", uvw = [0, 0, 1], layer = 0 } = {},
) {
  if (!(dMin > 0)) throw new Error("d min must be positive");
  const hkl = structure.hklWithin(dMin);
  const count = hkl.length / 3;
  if (!count)
    throw new Error(`no reflections with d ≥ ${dMin.toFixed(3)} Å; lower d min`);
  if (count > MAX_LATTICE) {
    // the count grows as 1/d^3, which sizes the suggestion
    const need = dMin * Math.cbrt(count / MAX_LATTICE);
    throw new Error(
      `${count} reflections with d ≥ ${dMin.toFixed(3)} Å are more than the ` +
        `3D view can turn smoothly. Raise d min above about ${need.toFixed(3)} Å.`,
    );
  }

  const q = new Float64Array(count);
  const intensity = structure.intensity(hkl, radiation, q);
  const Q = structure.q(hkl);
  const d = new Float64Array(count);
  for (let i = 0; i < count; i++) d[i] = TWO_PI / q[i];

  // The frame of the section view for this zone axis.
  const red = reduceZone(uvw);
  const zone = red.uvw;
  const B = structure.B;
  const [g1] = zoneBasis(B, zone);
  const normal = matVec(structure.A, zone);
  const nHat = unit(normal);
  const xAxis = unit(matVec(B, g1));
  const yAxis = unit(cross(nHat, xAxis));
  const zoneIndex = new Int32Array(count);
  for (let i = 0; i < count; i++)
    zoneIndex[i] =
      hkl[3 * i] * zone[0] + hkl[3 * i + 1] * zone[1] + hkl[3 * i + 2] * zone[2];
  layer = Math.round(layer);

  return {
    count,
    hkl,
    Q,
    q,
    d,
    intensity,
    zoneIndex,
    uvw: zone,
    zoneFactor: red.factor,
    layer,
    // the spacing of the layers along the zone axis, and where layer n sits
    layerSpacing: TWO_PI / norm(normal),
    height: (TWO_PI * layer) / norm(normal),
    nHat,
    xAxis,
    yAxis,
    // the reciprocal axes, for the key
    axes: [
      matVec(B, [1, 0, 0]),
      matVec(B, [0, 1, 0]),
      matVec(B, [0, 0, 1]),
    ],
    dMin,
    qMax: TWO_PI / dMin,
    radiation,
  };
}

/**
 * Where the Ewald sphere is when the crystal has turned by omega degrees about
 * the vertical axis of the section view, in the crystal frame.
 *
 * At omega = 0 the beam runs antiparallel to the zone axis, as in the SAED
 * view ([uvw] points from the specimen to the source), so the sphere of radius
 * k = 2 pi / lambda has its centre at +k along the zone axis and touches the
 * zero layer at the origin. Turning the crystal by +omega turns the beam, seen
 * from the crystal, by -omega. Returns {centre, k, beam} with `beam` the unit
 * incident direction.
 */
export function ewaldSphere(lat, wavelength, omegaDeg) {
  const k = TWO_PI / wavelength;
  const t = (-omegaDeg * Math.PI) / 180;
  const a = lat.yAxis;
  const b0 = lat.nHat.map((v) => -v);
  // Rodrigues: rotate b0 by t about the unit axis a
  const c = Math.cos(t);
  const s = Math.sin(t);
  const dot = a[0] * b0[0] + a[1] * b0[1] + a[2] * b0[2];
  const ax = cross(a, b0);
  const beam = [0, 1, 2].map(
    (j) => b0[j] * c + ax[j] * s + a[j] * dot * (1 - c),
  );
  return { centre: beam.map((v) => -k * v), k, beam };
}
