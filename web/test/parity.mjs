/**
 * Parity harness: the JavaScript physics against fixtures exported from the
 * validated Python package.
 *
 *   python tools/export_web_data.py     # regenerate fixture.json
 *   node web/test/parity.mjs
 *
 * Every ported function is covered. Nothing here is allowed to "look close":
 * each group asserts an explicit tolerance and the worst deviation is printed
 * so a regression shows up as a number, not a pass/fail flip.
 */
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import * as P from '../js/physics.js';

const here = dirname(fileURLToPath(import.meta.url));
const fx = JSON.parse(readFileSync(join(here, 'fixture.json'), 'utf8'));
const table = JSON.parse(readFileSync(join(here, '..', 'data', 'scattering_factors.json'), 'utf8'));
const cspbbr3 = JSON.parse(readFileSync(join(here, '..', 'data', 'cspbbr3.json'), 'utf8'));

let failures = 0;
const results = [];

function report(name, worst, tol, extra = '') {
  const ok = Number.isFinite(worst) && worst <= tol;
  if (!ok) failures++;
  results.push({ name, worst, tol, ok, extra });
  const w = worst.toExponential(2).padStart(9);
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(38)} worst ${w}  tol ${tol.toExponential(0)}  ${extra}`);
}

const maxAbs = (xs) => xs.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
const matDiff = (A, Bm) => maxAbs(A.flat().map((v, i) => v - Bm.flat()[i]));
const relDiff = (a, b) => Math.abs(a - b) / Math.max(1, Math.abs(b));

// --- 1. circle matrices ----------------------------------------------------
{
  let worst = 0;
  for (const c of fx.rotations) {
    const [mu, eta, chi, phi] = c.angles;
    worst = Math.max(worst, matDiff(P.sampleMatrix(mu, eta, chi, phi), c.Z));
  }
  report('sample matrix Z = MU.ETA.CHI.PHI', worst, 1e-14,
         `${fx.rotations.length} angle sets`);

  worst = 0;
  for (const c of fx.detector_matrices) {
    const [nu, delta] = c.nu_delta;
    worst = Math.max(worst, matDiff(P.detectorMatrix(nu, delta), c.R));
  }
  report('detector matrix R = NU.DELTA', worst, 1e-14,
         `${fx.detector_matrices.length} angle sets`);
}

// --- 2. B matrices ---------------------------------------------------------
{
  let worst = 0;
  for (const c of fx.b_matrices) worst = Math.max(worst, matDiff(P.bMatrix(...c.cell), c.B));
  report('B matrix from cell parameters', worst, 1e-13,
         `${fx.b_matrices.length} lattices incl. triclinic`);
}

// --- 3. structure factors --------------------------------------------------
{
  const sf = fx.structure_factors;
  const B = sf.B;
  const hkl = Int32Array.from(sf.hkl.flat());
  const F2 = P.structureFactors(table, cspbbr3.atoms, B, hkl);
  const worst = maxAbs(Array.from(F2).map((v, i) => relDiff(v, sf.F2[i])));
  report('|F(hkl)|^2 vs pytilting', worst, 1e-9,
         `${sf.hkl.length} reflections, ${cspbbr3.atoms.length} atoms`);
  // and the B from the CIF must match the one we derive from the cell
  const Bcell = P.bMatrix(cspbbr3.cell.a, cspbbr3.cell.b, cspbbr3.cell.c,
                          cspbbr3.cell.alpha, cspbbr3.cell.beta, cspbbr3.cell.gamma);
  report('B from exported cell vs CIF B', matDiff(Bcell, B), 1e-12);
}

// --- 4. detector -----------------------------------------------------------
{
  const d = fx.detector;
  const det = new P.Detector(d.spec);
  const f = det.frame();
  let worst = Math.max(
    maxAbs(f.centre.map((v, i) => v - d.centre[i])),
    maxAbs(f.normal.map((v, i) => v - d.normal[i])),
    maxAbs(f.eFast.map((v, i) => v - d.eFast[i])),
    maxAbs(f.eSlow.map((v, i) => v - d.eSlow[i])),
    maxAbs(f.arm.map((v, i) => v - d.arm[i])));
  report('detector frame (centre/axes/arm)', worst, 1e-13);
  report('detector maxQmax', Math.abs(det.maxQmax(0.7293) - d.maxQmax), 1e-12);

  let px = 0, mism = 0;
  for (const r of d.rays) {
    const p = det.projectOne(r.khat);
    px = Math.max(px, Math.abs(p.fast - r.fast), Math.abs(p.slow - r.slow),
                  Math.abs(p.cosInc - r.cosInc));
    if (p.inside !== r.inside) mism++;
  }
  report('detector projection (pixels)', px, 1e-9, `${d.rays.length} rays`);
  report('detector inside/outside flags', mism, 0, `${d.rays.length} rays`);
}

// --- 5. full Ewald pass ----------------------------------------------------
{
  const e = fx.ewald;
  const B = P.bMatrix(...e.cell);
  const det = new P.Detector({ distance: 200, nFast: 1475, nSlow: 1679,
                               pixelSize: 0.172, nu: 0, delta: 0 });
  const hkl = P.hklWithinQmax(B, det.maxQmax(e.wavelength));
  report('reflection list size', Math.abs(hkl.length / 3 - e.n_hkl), 0,
         `${hkl.length / 3} hkl`);

  const F2 = P.latticeStructureFactors(B, hkl);
  const Z = P.sampleMatrix(e.angles.mu, e.angles.eta, e.angles.chi, e.angles.phi);
  const ZU = P.matMul(Z, e.U);
  const r = P.excite({ Qcryst: P.qCryst(B, hkl), F2, hkl, ZU,
                       wavelength: e.wavelength, sigma: e.sigma, nSigma: e.nSigma });

  report('excited reflection count', Math.abs(r.count - e.reflections.length), 0,
         `${r.count} near the sphere`);

  // sort the JS result the same way the fixture was sorted (h, then k, then l)
  const order = [...Array(r.count).keys()].sort((i, j) => {
    for (let c = 0; c < 3; c++) {
      const d2 = r.hkl[3 * i + c] - r.hkl[3 * j + c];
      if (d2) return d2;
    }
    return 0;
  });
  let hklBad = 0, worst = 0;
  order.forEach((idx, n) => {
    const ref = e.reflections[n];
    if (!ref) { hklBad++; return; }
    for (let c = 0; c < 3; c++) if (r.hkl[3 * idx + c] !== ref.hkl[c]) hklBad++;
    worst = Math.max(worst,
      Math.abs(r.khat[3 * idx] - ref.khat[0]),
      Math.abs(r.khat[3 * idx + 1] - ref.khat[1]),
      Math.abs(r.khat[3 * idx + 2] - ref.khat[2]),
      Math.abs(r.eps[idx] - ref.eps),
      Math.abs(r.excitation[idx] - ref.excitation),
      Math.abs(r.twoTheta[idx] - ref.twoTheta));
  });
  report('excited hkl identity', hklBad, 0);
  report('khat / eps / excitation / 2theta', worst, 1e-12,
         `${e.reflections.length} reflections`);
}

// --- 6. orientation --------------------------------------------------------
{
  const o = fx.orientation;
  const B = P.bMatrix(...o.cell);
  const Z = P.sampleMatrix(o.angles.mu, o.angles.eta, o.angles.chi, o.angles.phi);

  let worstR = 0;
  for (const c of o.rotation_between)
    worstR = Math.max(worstR, matDiff(P.rotationBetween(c.from, c.to), c.R));
  report('rotationBetween', worstR, 1e-13, `${o.rotation_between.length} cases`);

  let worstE = 0;
  for (const c of o.euler)
    worstE = Math.max(worstE, matDiff(P.eulerMatrix(...c.rxyz), c.R));
  report('eulerMatrix', worstE, 1e-13, `${o.euler.length} cases`);

  let worstV = 0, worstP = 0, worstS = 0, refuseBad = 0, refusals = 0;
  for (const c of o.cases) {
    worstV = Math.max(worstV, maxAbs(
      P.crystalVector(B, c.indices, c.kind).map((v, i) => v - c.crystal_vector[i])));
    const U1 = P.alignInLab(P.eye3(), B, c.indices, c.target, { kind: c.kind, Z });
    worstP = Math.max(worstP, matDiff(U1, c.U_primary));

    // A secondary parallel to the primary axis carries no information. Python
    // returns False and leaves U alone; JS returns null. Check that the two
    // agree about *when* to refuse, then compare the resulting orientation.
    const U2 = P.alignSecondaryInLab(U1, B, [0, 1, 0], [1, 0, 0], c.target,
                                     { kind: 'hkl', Z });
    const refusedJS = U2 === null;
    const refusedPy = matDiff(c.U_primary, c.U_secondary) < 1e-15;
    if (refusedJS !== refusedPy) refuseBad++;
    if (refusedJS) refusals++;
    worstS = Math.max(worstS, matDiff(refusedJS ? U1 : U2, c.U_secondary));
  }
  report('crystalVector (hkl and uvw)', worstV, 1e-12, `${o.cases.length} cases`);
  report('alignInLab', worstP, 1e-13, `${o.cases.length} cases`);
  report('alignSecondaryInLab', worstS, 1e-13,
         `${o.cases.length} cases, ${refusals} correctly refused`);
  report('secondary refusal agrees with Python', refuseBad, 0);
}

// --- 7. UB conventions -----------------------------------------------------
{
  const u = fx.ub;
  const B = P.bMatrix(...u.cell);
  let worst = 0;
  for (const conv of ['2pi', '1/d', 'lambda'])
    worst = Math.max(worst, matDiff(P.UB(u.U, B, conv, u.wavelength), u[conv]));
  report('UB in all three conventions', worst, 1e-13);
}

// --- 8. solvers ------------------------------------------------------------
{
  const s = fx.solvers;
  const B = P.bMatrix(...s.cell);
  let worstReach = 0, feasBad = 0, solBad = 0, worstSol = 0,
      worstChi = 0, worstAim = 0;

  for (const c of s.cases) {
    const r = P.etaReach(B, s.U, c.hkl, s.angles, s.wavelength);
    if (r.feasible !== c.reach.feasible) feasBad++;
    if (r.inLimitingSphere !== c.reach.in_limiting_sphere) feasBad++;
    worstReach = Math.max(worstReach,
      Math.abs(r.Q - c.reach.Q), Math.abs(r.required - c.reach.required),
      Math.abs(r.lo - c.reach.lo), Math.abs(r.hi - c.reach.hi));

    const sols = r.inLimitingSphere
      ? P.solveEta(B, s.U, c.hkl, s.angles, s.wavelength) : [];
    if (sols.length !== c.solutions.length) solBad++;
    else sols.forEach((e, i) => { worstSol = Math.max(worstSol, Math.abs(e - c.solutions[i])); });

    if (c.suggest_chi !== undefined && c.suggest_chi !== null) {
      const chi = P.suggestChi(B, s.U, c.hkl, s.angles, s.wavelength);
      worstChi = Math.max(worstChi, Math.abs(chi - c.suggest_chi));
    }
    if (c.aim_at_first) {
      const a = P.aimDetectorAt(B, s.U, c.hkl,
                                { ...s.angles, eta: c.aim_at_first.eta }, s.wavelength);
      worstAim = Math.max(worstAim, Math.abs(a.delta - c.aim_at_first.delta),
                          Math.abs(a.gamma - c.aim_at_first.gamma));
    }
  }
  report('etaReach interval and |Q|', worstReach, 1e-12, `${s.cases.length} hkl`);
  report('etaReach feasible / in-sphere flags', feasBad, 0);
  report('solveEta solution count', solBad, 0);
  report('solveEta root positions (deg)', worstSol, 1e-7);
  report('suggestChi', worstChi, 1e-9);
  report('aimDetectorAt (delta, gamma)', worstAim, 1e-11);

  let worstInv = 0;
  for (const c of s.detector_angles_for) {
    const a = P.detectorAnglesFor(c.khat);
    worstInv = Math.max(worstInv, Math.abs(a.delta - c.delta), Math.abs(a.gamma - c.gamma));
  }
  report('detectorAnglesFor round trip', worstInv, 1e-11,
         `${s.detector_angles_for.length} arm directions`);
}

console.log();
const worstOverall = results.filter((r) => r.ok && r.tol > 0)
  .reduce((m, r) => Math.max(m, r.worst), 0);
if (failures === 0) {
  console.log(`ALL PASS  (${results.length} groups, largest deviation anywhere ` +
              `${worstOverall.toExponential(2)})`);
} else {
  console.log(`${failures} of ${results.length} groups FAILED`);
  process.exitCode = 1;
}
