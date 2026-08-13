/*! Single-Crystal Diffraction · © 2026 Miloš Dubajić · MIT · https://github.com/dubajicmilos/xrays-on-detector */

/**
 * How an intensity becomes something you can see.
 * A port of single_crystal/display.py, so the desktop app and this one show
 * the same picture for the same numbers.
 *
 * A diffraction pattern spans many decades. Plotted linearly, one reflection
 * is white and the rest are black; that is true to the data and useless to
 * look at. The stretch below maps intensity onto a fixed number of decades
 * below the strongest reflection, with a gain that slides the window.
 */

/** How many decades below the brightest reflection stay visible. */
export const DECADES = 6;

/**
 * Map intensity to 0..1 for display.
 *
 * `iMax` pins the normalisation to a chosen reference instead of the strongest
 * reflection present. Pass the strongest of the whole structure when stepping
 * through layers, or an upper layer is renormalised to its own weak maximum
 * and looks as bright as the zero layer.
 */
export function stretch(intensity, { gain = 1, log = true, iMax = null } = {}) {
  const n = intensity.length;
  const out = new Float64Array(n);
  let top = iMax;
  if (top === null) {
    top = 0;
    for (let i = 0; i < n; i++) if (intensity[i] > top) top = intensity[i];
  }
  if (!(top > 0)) return out;

  const floor = Math.pow(10, -DECADES);
  for (let i = 0; i < n; i++) {
    let v = (intensity[i] / top) * gain;
    v = v < 0 ? 0 : v > 1 ? 1 : v;
    if (log) {
      if (v < floor) v = floor;
      v = (Math.log10(v) + DECADES) / DECADES;
    }
    out[i] = v;
  }
  return out;
}

/** Marker radius in pixels from the stretched value. */
export const spotRadius = (value, scale = 9, floor = 1.1) =>
  floor + scale * value;

/**
 * [1, -1, 0] -> "11̅0", with an overbar on negative indices.
 *
 * U+0305 attaches to the character before it, so a two-digit index needs one
 * after every digit: -11 written as "11̅" bars only the second digit and reads
 * as the pair (1, -1). Barring both gives an unbroken line over "11".
 * Indices of ten and above are also spaced apart, or "1 12 0" reads as four.
 */
export function formatHkl(hkl) {
  const wide = hkl.some((v) => Math.abs(v) > 9);
  return hkl
    .map((v) => {
      const digits = String(Math.abs(v));
      return v < 0 ? [...digits].map((d) => d + "̅").join("") : digits;
    })
    .join(wide ? " " : "");
}

/** A round number near span/6, for a scale bar or a grid. */
export function niceStep(span) {
  if (!(span > 0)) return 1;
  const raw = span / 6;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 2.5, 5, 10]) if (raw <= m * mag) return m * mag;
  return 10 * mag;
}

/** Sample a flat 256*3 colormap, returning "rgb(r,g,b)". */
export function sampleLut(lut, value) {
  let i = Math.round(value * 255);
  i = i < 0 ? 0 : i > 255 ? 255 : i;
  return `rgb(${lut[3 * i]},${lut[3 * i + 1]},${lut[3 * i + 2]})`;
}
