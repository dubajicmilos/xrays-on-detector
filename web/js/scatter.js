/*! Single-Crystal Diffraction · © 2026 Miloš Dubajić · MIT · https://github.com/dubajicmilos/xrays-on-detector */

/**
 * Atomic scattering factors for x-rays, neutrons and electrons.
 *
 * A port of single_crystal/scatter.py, reading the same generated tables under
 * data/. tools/export_scattering.py writes them once and copies them here, so
 * the desktop app and this one cannot drift apart.
 *
 *   x-ray     f(s), Cromer-Mann four-Gaussian plus a constant, in electrons
 *   electron  f(s), four-Gaussian, in Angstrom
 *   neutron   b_c, one number per nuclide, in fm
 *
 * s = sin(theta)/lambda = |Q| / 4 pi throughout, in 1/Angstrom.
 *
 * The three are in different units, so an intensity is comparable within a
 * radiation and not across one. Nothing here normalises them; the app displays
 * intensity relative to the strongest reflection in view and labels the unit.
 *
 * Neutron scattering lengths are signed, and several are negative (H, Ti, V,
 * Mn). That is physical, and it is what makes contrast variation and
 * deuteration work, so the sign is carried into the structure factor sum and
 * never taken as a magnitude.
 */

export const RADIATIONS = ["xray", "neutron", "electron"];

export const RADIATION_LABEL = {
  xray: "X-rays",
  neutron: "Neutrons",
  electron: "Electrons",
};

/** What |F|^2 is measured in, so a number copied out carries its unit. */
export const UNITS = { xray: "e", neutron: "fm", electron: "Å" };

const tables = { xray: null, neutron: null, electron: null };

/** Hand in the three parsed JSON tables once, at boot. */
export function setTables({ xray, neutron, electron }) {
  tables.xray = xray;
  tables.neutron = neutron;
  tables.electron = electron;
}

export const tableFor = (radiation) => tables[radiation];

/** Every symbol the x-ray table covers; what the CIF reader validates against. */
export const knownElements = () => Object.keys(tables.xray || {});

/** Symbols this radiation has no entry for, in the order first seen. */
export function missingFor(radiation, symbols) {
  const table = tables[radiation];
  if (!table) throw new Error(`the ${radiation} table has not been loaded`);
  const seen = new Set();
  const out = [];
  for (const s of symbols) {
    if (!(s in table) && !seen.has(s)) {
      seen.add(s);
      out.push(s);
    }
  }
  return out;
}

/**
 * Scattering factor of one symbol at one s.
 *
 * Neutrons ignore s: a nucleus is a point on this length scale, so b_c is
 * constant in Q. That is exactly why a neutron pattern keeps its high-angle
 * reflections strong while an x-ray one fades.
 */
export function factor(radiation, symbol, s) {
  const table = tables[radiation];
  const e = table && table[symbol];
  if (e === undefined) {
    throw new Error(
      `no ${RADIATION_LABEL[radiation].toLowerCase()} scattering data for ${symbol}`,
    );
  }
  if (radiation === "neutron") return e;
  // The number of Gaussians differs by table: four for the Cromer-Mann x-ray
  // fit, five for the Peng electron fit. Read the length rather than assume.
  const [a, b, c] = e;
  const s2 = s * s;
  let f = c;
  for (let i = 0; i < a.length; i++) f += a[i] * Math.exp(-b[i] * s2);
  return f;
}

/**
 * f for a list of symbols at one s, as a Float64Array in the same order.
 * Used per reflection, so the table lookup is hoisted out of the atom loop.
 */
export function factorsAt(radiation, symbols, s) {
  const out = new Float64Array(symbols.length);
  for (let i = 0; i < symbols.length; i++)
    out[i] = factor(radiation, symbols[i], s);
  return out;
}

/**
 * Relativistic electron wavelength in Angstrom for an accelerating voltage in
 * kV. 200 kV gives 0.02508 Angstrom.
 */
export function electronWavelength(kv) {
  const V = kv * 1e3;
  const h = 6.62607015e-34;
  const m = 9.1093837015e-31;
  const e = 1.602176634e-19;
  const c = 2.99792458e8;
  return (
    (h / Math.sqrt(2 * m * e * V * (1 + (e * V) / (2 * m * c * c)))) * 1e10
  );
}
