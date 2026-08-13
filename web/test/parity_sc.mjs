/**
 * Parity harness for the single-crystal app: the JavaScript against fixtures
 * exported from the validated Python package.
 *
 *   python tools/export_sc_fixture.py     # regenerate fixture_sc.json
 *   node web/test/parity_sc.mjs
 *
 * The Python is the reference because it is the side checked against pymatgen
 * in tests/test_single_crystal.py, on an independent CIF reader, an
 * independent symmetry expansion and an independent form factor table. This
 * harness carries that verification across to the browser build.
 *
 * Nothing here is allowed to "look close": each group asserts an explicit
 * tolerance and prints the worst deviation, so a regression shows up as a
 * number rather than a pass/fail flip.
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { setTables } from "../js/scatter.js";
import * as scatter from "../js/scatter.js";
import * as display from "../sc/js/display.js";
import { computePowder } from "../sc/js/powder.js";
import { computeSection } from "../sc/js/section.js";
import { Structure } from "../sc/js/structure.js";
import { computeTem } from "../sc/js/tem.js";

const here = dirname(fileURLToPath(import.meta.url));
const read = (...p) => JSON.parse(readFileSync(join(here, ...p), "utf8"));

const fx = read("fixture_sc.json");
setTables({
  xray: read("..", "data", "scattering_factors.json"),
  neutron: read("..", "data", "neutron_lengths.json"),
  electron: read("..", "data", "electron_factors.json"),
});

let failures = 0;
function report(name, worst, tol, extra = "") {
  const ok = Number.isFinite(worst) && worst <= tol;
  if (!ok) failures++;
  const w = worst.toExponential(2).padStart(9);
  console.log(
    `${ok ? "PASS" : "FAIL"}  ${name.padEnd(40)} worst ${w}  tol ${tol.toExponential(0)}  ${extra}`,
  );
}

/** Relative difference, falling back to absolute for values near zero. */
const rel = (a, b) => Math.abs(a - b) / Math.max(1e-12, Math.abs(b));

/**
 * Compare two intensity lists after matching them by hkl.
 *
 * Both sides enumerate the same lattice but need not walk it in the same
 * order, so comparing position by position would report a shuffle as a physics
 * error. Keying on hkl also catches a reflection present on one side only,
 * which is the failure that matters most.
 */
function byHkl(hkl, values) {
  const m = new Map();
  for (let i = 0; i < values.length; i++)
    m.set(`${hkl[3 * i]},${hkl[3 * i + 1]},${hkl[3 * i + 2]}`, i);
  return m;
}

const structures = {};
for (const [name, doc] of Object.entries(fx.structures))
  structures[name] = new Structure(doc);

// --- 1. atomic scattering factors ------------------------------------------
{
  let worst = 0;
  for (const c of fx.factors)
    worst = Math.max(worst, rel(scatter.factor(c.radiation, c.symbol, c.s), c.f));
  report("atomic factors, 3 radiations", worst, 1e-13, `${fx.factors.length} values`);

  // The sign of a negative b is the whole point of the neutron table.
  const bH = scatter.factor("neutron", "H", 0);
  const bD = scatter.factor("neutron", "D", 0);
  report("b(H) < 0 < b(D)", bH < 0 && bD > 0 ? 0 : 1, 0, `${bH} vs ${bD} fm`);
}

// --- 2. lattice -------------------------------------------------------------
{
  let worst = 0;
  for (const [name, s] of Object.entries(structures)) {
    const ref = fx.sections.find((c) => c.structure === name);
    if (!ref) continue;
    // |Q| from the fixture must come back out of B for the same hkl.
    const q = new Float64Array(ref.count);
    s.intensity(Int32Array.from(ref.hkl), ref.radiation, q);
    for (let i = 0; i < ref.count; i++) worst = Math.max(worst, rel(q[i], ref.q[i]));
  }
  report("|Q| = |B . hkl|", worst, 1e-12, `${Object.keys(structures).length} cells`);
}

// --- 3. sections ------------------------------------------------------------
{
  let worstI = 0,
    worstXY = 0,
    worstGeom = 0,
    countMismatch = 0,
    missing = 0,
    total = 0;
  for (const c of fx.sections) {
    const s = structures[c.structure];
    const got = computeSection(s, {
      uvw: c.uvw,
      layer: c.layer,
      dMin: c.d_min,
      radiation: c.radiation,
    });
    if (got.count !== c.count) countMismatch++;
    worstGeom = Math.max(
      worstGeom,
      Math.abs(got.height - c.height),
      got.g1.reduce((m, v, i) => Math.max(m, Math.abs(v - c.g1[i])), 0),
      got.g2.reduce((m, v, i) => Math.max(m, Math.abs(v - c.g2[i])), 0),
      Math.abs(got.zoneFactor - c.zone_factor),
    );

    const want = byHkl(c.hkl, c.intensity);
    const top = Math.max(...c.intensity);
    for (let i = 0; i < got.count; i++) {
      const key = `${got.hkl[3 * i]},${got.hkl[3 * i + 1]},${got.hkl[3 * i + 2]}`;
      const j = want.get(key);
      if (j === undefined) {
        missing++;
        continue;
      }
      total++;
      // Intensity spans decades, so judge it against the strongest reflection
      // of the section: an absolute error there is what shows on screen.
      worstI = Math.max(worstI, Math.abs(got.intensity[i] - c.intensity[j]) / top);
      worstXY = Math.max(
        worstXY,
        Math.abs(Math.abs(got.x[i]) - Math.abs(c.x[j])),
        Math.abs(Math.abs(got.y[i]) - Math.abs(c.y[j])),
      );
    }
  }
  report("section reflection count", countMismatch, 0, `${fx.sections.length} sections`);
  report("section hkl identity", missing, 0, `${total} reflections matched`);
  report("section zone basis, height, factor", worstGeom, 1e-12);
  report("section |F|^2 (relative to the peak)", worstI, 1e-12, `${total} reflections`);
  report("section plot coordinates", worstXY, 1e-10);
}

// --- 4. electron diffraction ------------------------------------------------
{
  let worstI = 0,
    worstS = 0,
    worstR = 0,
    countMismatch = 0,
    missing = 0,
    total = 0;
  for (const c of fx.tem) {
    const got = computeTem(structures[c.structure], {
      uvw: c.uvw,
      kv: c.kv,
      thickness: c.thickness,
      dMin: c.d_min,
      maxZone: c.max_zone,
    });
    if (got.count !== c.count) countMismatch++;
    worstR = Math.max(worstR, Math.abs(got.wavelength - c.wavelength));
    for (let z = 0; z < c.zone_radii.length; z++)
      if (c.zone_radii[z] !== null)
        worstR = Math.max(worstR, rel(got.zoneRadii[z], c.zone_radii[z]));

    const want = byHkl(c.hkl, c.intensity);
    const top = Math.max(...c.intensity, 1e-300);
    for (let i = 0; i < got.count; i++) {
      const key = `${got.hkl[3 * i]},${got.hkl[3 * i + 1]},${got.hkl[3 * i + 2]}`;
      const j = want.get(key);
      if (j === undefined) {
        missing++;
        continue;
      }
      total++;
      worstI = Math.max(worstI, Math.abs(got.intensity[i] - c.intensity[j]) / top);
      worstS = Math.max(worstS, Math.abs(got.sg[i] - c.s_g[j]));
      if (got.laueZone[i] !== c.laue_zone[j]) missing++;
    }
  }
  report("SAED reflection count", countMismatch, 0, `${fx.tem.length} patterns`);
  report("SAED hkl and Laue zone", missing, 0, `${total} reflections matched`);
  report("SAED wavelength and ring radii", worstR, 1e-11);
  report("SAED excitation error s_g", worstS, 1e-9);
  report("SAED intensity with relrod", worstI, 1e-11, `${total} reflections`);
}

// --- 5. powder --------------------------------------------------------------
{
  let worstTT = 0,
    worstI = 0,
    worstM = 0,
    countMismatch = 0,
    total = 0;
  for (const c of fx.powder) {
    const got = computePowder(structures[c.structure], {
      wavelength: c.wavelength,
      radiation: c.radiation,
      twoThetaMax: c.two_theta_max,
    });
    if (got.count !== c.count) {
      countMismatch++;
      continue;
    }
    for (let i = 0; i < c.count; i++) {
      total++;
      worstTT = Math.max(worstTT, Math.abs(got.twoTheta[i] - c.two_theta[i]));
      worstI = Math.max(worstI, Math.abs(got.intensity[i] - c.intensity[i]));
      worstM = Math.max(worstM, Math.abs(got.multiplicity[i] - c.multiplicity[i]));
    }
  }
  report("powder peak count", countMismatch, 0, `${fx.powder.length} patterns`);
  report("powder 2theta (degrees)", worstTT, 1e-10, `${total} peaks`);
  report("powder intensity (0-100)", worstI, 1e-10);
  report("powder multiplicity", worstM, 0);
}

// --- 6. display -------------------------------------------------------------
{
  let worst = 0;
  for (const c of fx.stretch) {
    const got = display.stretch(Float64Array.from(c.intensity), {
      gain: c.gain,
      log: c.log,
    });
    for (let i = 0; i < got.length; i++)
      worst = Math.max(worst, Math.abs(got[i] - c.value[i]));
  }
  report("display stretch", worst, 1e-14, `${fx.stretch.length} settings`);

  let bad = 0;
  for (const c of fx.hkl_labels)
    if (display.formatHkl(c.hkl) !== c.text) {
      bad++;
      console.log(`      ${c.hkl} -> ${JSON.stringify(display.formatHkl(c.hkl))} ` +
                  `expected ${JSON.stringify(c.text)}`);
    }
  report("hkl labels with overbars", bad, 0, `${fx.hkl_labels.length} cases`);
}

console.log();
if (failures) {
  console.log(`${failures} GROUP${failures > 1 ? "S" : ""} FAILED`);
  process.exit(1);
}
console.log("ALL PASS");
