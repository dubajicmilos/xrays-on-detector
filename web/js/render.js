/**
 * Detector image rendering, ported from xrays_on_detector/render.py.
 *
 * Each reflection carries I = |F|^2 * excitation * polarization and is spread
 * over the panel as a 2D Gaussian normalised to that total, so grazing spots
 * are broader and fainter per pixel but keep the same integrated counts.
 */
import { BEAM, TWO_PI } from "./physics.js";

/** Polarization factor P(2theta). */
export function polarization(twoTheta, mode, khatX) {
  if (mode === "none") return 1;
  if (mode === "unpolarized") return 0.5 * (1 + Math.cos(twoTheta) ** 2);
  if (mode === "horizontal") return 1 - khatX * khatX;
  throw new Error(`unknown polarization mode ${mode}`);
}

/**
 * Accumulate excited reflections into a float image.
 *
 * Returns {image: Float32Array (nSlow x nFast, row 0 = top), table: [...]}.
 * The table holds one record per reflection that landed on the panel.
 */
export function renderFrame(
  det,
  refl,
  { wavelength, sigma, polarizationMode = "horizontal", minSigmaPx = 0.6 } = {},
) {
  const k = TWO_PI / wavelength;
  const { nFast, nSlow, pixelSize, distance } = det;
  const image = new Float32Array(nFast * nSlow);
  const table = [];

  for (let i = 0; i < refl.count; i++) {
    const khat = [refl.khat[3 * i], refl.khat[3 * i + 1], refl.khat[3 * i + 2]];
    const p = det.projectOne(khat);
    if (!p.inside) continue;

    const P = polarization(refl.twoTheta[i], polarizationMode, khat[0]);
    const total = refl.F2[i] * refl.excitation[i] * P;
    const ci = Math.max(p.cosInc, 1e-3);
    const sPx = Math.max(
      minSigmaPx,
      ((sigma / k) * distance) / (pixelSize * ci),
    );

    addGaussian(image, nFast, nSlow, p.fast, p.slow, sPx, total);
    table.push({
      h: refl.hkl[3 * i],
      k: refl.hkl[3 * i + 1],
      l: refl.hkl[3 * i + 2],
      fast: p.fast,
      slow: p.slow,
      eps: refl.eps[i],
      twoThetaDeg: (refl.twoTheta[i] * 180) / Math.PI,
      intensity: total,
    });
  }
  return { image, table };
}

/**
 * Add a normalised 2D Gaussian of integral `total` at (cx, cy).
 * cx indexes the fast (column) axis; the slow axis is written top-down, so
 * +vertical maps to a decreasing row index, as in render.py.
 */
function addGaussian(image, nFast, nSlow, cx, cy, sPx, total, nSigma = 4) {
  const rad = Math.ceil(nSigma * sPx);
  const col0 = Math.max(0, Math.floor(cx) - rad);
  const col1 = Math.min(nFast, Math.floor(cx) + rad + 1);
  const rowCentre = nSlow - 1 - cy;
  const row0 = Math.max(0, Math.floor(rowCentre) - rad);
  const row1 = Math.min(nSlow, Math.floor(rowCentre) + rad + 1);
  if (col0 >= col1 || row0 >= row1) return;

  const norm = total / (TWO_PI * sPx * sPx);
  const inv = 1 / (2 * sPx * sPx);
  for (let r = row0; r < row1; r++) {
    const gy = Math.exp(-((r - rowCentre) ** 2) * inv);
    const base = r * nFast;
    for (let c = col0; c < col1; c++) {
      image[base + c] += norm * gy * Math.exp(-((c - cx) ** 2) * inv);
    }
  }
}

/**
 * Paint a float image into a canvas through a colour-map LUT.
 * `lut` is a flat 768-entry RGB array (256 x 3).
 */
export function paint(
  canvas,
  image,
  nFast,
  nSlow,
  lut,
  { log = true, gain = 1 } = {},
) {
  if (canvas.width !== nFast || canvas.height !== nSlow) {
    canvas.width = nFast;
    canvas.height = nSlow;
  }
  const ctx = canvas.getContext("2d", { willReadFrequently: false });
  const img = ctx.createImageData(nFast, nSlow);
  const px = img.data;

  let vmax = 0;
  for (let i = 0; i < image.length; i++) if (image[i] > vmax) vmax = image[i];

  const denom = log ? Math.log1p(gain * 500) : 1;
  const kk = vmax > 0 ? (gain * 500) / vmax : 0;

  for (let i = 0; i < image.length; i++) {
    let v;
    if (vmax <= 0) v = 0;
    else if (log) v = Math.log1p(image[i] * kk) / denom;
    else v = Math.min(1, (image[i] * gain) / vmax);
    let idx = (v * 255) | 0;
    if (idx < 0) idx = 0;
    else if (idx > 255) idx = 255;
    const j = idx * 3;
    const o = i * 4;
    px[o] = lut[j];
    px[o + 1] = lut[j + 1];
    px[o + 2] = lut[j + 2];
    px[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

/** Ray endpoints for the 3D view, as flat pairs suitable for LineSegments. */
export function rayGeometry(det, refl, table, missLength) {
  const { centre, eFast, eSlow } = det.frame();
  const hit = new Float32Array(table.length * 6);
  table.forEach((r, n) => {
    const u = (r.fast - det.beamCenterFast) * det.pixelSize;
    const v = (r.slow - det.beamCenterSlow) * det.pixelSize;
    const o = n * 6;
    hit[o] = 0;
    hit[o + 1] = 0;
    hit[o + 2] = 0;
    hit[o + 3] = centre[0] + u * eFast[0] + v * eSlow[0];
    hit[o + 4] = centre[1] + u * eFast[1] + v * eSlow[1];
    hit[o + 5] = centre[2] + u * eFast[2] + v * eSlow[2];
  });

  const missPts = [];
  for (let i = 0; i < refl.count; i++) {
    const khat = [refl.khat[3 * i], refl.khat[3 * i + 1], refl.khat[3 * i + 2]];
    if (det.projectOne(khat).inside) continue;
    missPts.push(
      0,
      0,
      0,
      khat[0] * missLength,
      khat[1] * missLength,
      khat[2] * missLength,
    );
  }
  return { hit, miss: Float32Array.from(missPts) };
}

/** Blocked-ray endpoints (reflection geometry), shorter so they read as stopped. */
export function blockedGeometry(khatFlat, count, length) {
  const out = new Float32Array(count * 6);
  for (let i = 0; i < count; i++) {
    const o = i * 6;
    out[o + 3] = khatFlat[3 * i] * length;
    out[o + 4] = khatFlat[3 * i + 1] * length;
    out[o + 5] = khatFlat[3 * i + 2] * length;
  }
  return out;
}

export { BEAM };
