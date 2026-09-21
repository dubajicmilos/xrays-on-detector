// Parity test: the pitch-phi module against the standalone calculator's
// recorded outputs.
//
// The fixture (fixtures_pp.json) was recorded by running the user's verified
// standalone tool in a real browser over 199 cases -- random cells across
// crystal systems, random orientations and reflections, the specular family,
// near-tangency edges and roll probes -- and capturing what its own functions
// returned. This test re-derives every number with js/pitchphi.js and refuses
// to let the two drift apart.
//
// Run from web/:   node test/parity_pp.mjs
import { readFileSync } from "node:fs";
import * as PP from "../js/pitchphi.js";

const fixturePath = new URL("./fixtures_pp.json", import.meta.url);
const dump = JSON.parse(readFileSync(fixturePath, "utf8"));
const azSign = dump.azSign;

let checks = 0;
const fails = [];
const fail = (id, what, msg) => fails.push(`case ${id} [${what}] ${msg}`);
const chk = (id, what, got, want, tol = 0) => {
  checks++;
  if (typeof got === "boolean" || typeof want === "boolean") {
    if (got !== want) fail(id, what, `got ${got} want ${want}`);
    return;
  }
  if (typeof got === "string" || typeof want === "string") {
    if (String(got) !== String(want)) fail(id, what, `got ${got} want ${want}`);
    return;
  }
  if (!(Math.abs(got - want) <= tol))
    fail(id, what, `got ${got} want ${want} diff ${got - want}`);
};

// local linalg so the test does not reach into module internals
const mv = (M, v) => M.map((r) => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
const mm = (A, B) =>
  A.map((r) => [0, 1, 2].map((j) => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
const norm = (v) => Math.hypot(v[0], v[1], v[2]);

const wrapDelta = (a, b) => {
  const d = ((((a - b) + 180) % 360) + 360) % 360 - 180;
  return d === -180 ? 180 : d;
};

function rootSetsMatch(a, b, tol) {
  const aa = a.map(PP.wrap);
  const bb = b.map(PP.wrap);
  if (aa.length !== bb.length) return false;
  const used = new Array(bb.length).fill(false);
  for (const x of aa) {
    let hit = -1;
    for (let j = 0; j < bb.length; j++) {
      if (!used[j] && Math.abs(wrapDelta(x, bb[j])) <= tol) {
        hit = j;
        break;
      }
    }
    if (hit < 0) return false;
    used[hit] = true;
  }
  return true;
}

// -- module identities (frame contract) --------------------------------------

{
  // G and G^T really are inverse orthonormal matrices
  const I = mm(PP.PP_TO_GAME, PP.GAME_TO_PP);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      chk(-1, "G identity", I[i][j], i === j ? 1 : 0, 1e-15);
  // Z_game = G^T . Z_pp . G for a spread of angles
  let worst = 0;
  for (const a of [-80, -12.5, 0, 33.3, 71, 90]) {
    for (const p of [-170, -40, 0, 61, 179]) {
      for (const r of [-120, 0, 95]) {
        const Zg = PP.sampleMatrixGame(a, p, r);
        const Zc = mm(PP.PP_TO_GAME, mm(PP.sampleMatrix(a, p, r), PP.GAME_TO_PP));
        for (let i = 0; i < 3; i++)
          for (let j = 0; j < 3; j++)
            worst = Math.max(worst, Math.abs(Zg[i][j] - Zc[i][j]));
      }
    }
  }
  chk(-1, "Z_game construction", worst, 0, 1e-14);
  // detector direction <-> (2theta, az) roundtrip, both sign conventions
  let rt = 0;
  for (const tt of [5, 40, 120, 179]) {
    for (const az of [-170, -90, 0, 55, 170]) {
      for (const s of [1, -1]) {
        const { tth, az: az2 } = PP.kfToAngles(PP.detectorDir(tt, az, s), s);
        rt = Math.max(rt, Math.abs(tth - tt), Math.abs(PP.wrap(az2 - az)));
      }
    }
  }
  chk(-1, "detectorDir roundtrip", rt, 0, 1e-9);
  // mount: normal to +y, projected reference to +x, and it is a rotation
  const nC = [0.1, 0.2, 0.97];
  const ref = [0.9, 0.1, 0.2];
  const U = PP.mountOrientation(nC, ref, "+x");
  const nu = mv(U, nC);
  chk(-1, "mount n->+y (x)", nu[0], 0, 1e-12);
  chk(-1, "mount n->+y (z)", nu[2], 0, 1e-12);
  chk(-1, "mount n->+y (sign)", nu[1] > 0 ? 1 : -1, 1);
  const detU = U[0][0] * (U[1][1] * U[2][2] - U[1][2] * U[2][1]) -
    U[0][1] * (U[1][0] * U[2][2] - U[1][2] * U[2][0]) +
    U[0][2] * (U[1][0] * U[2][1] - U[1][1] * U[2][0]);
  chk(-1, "mount det", detU, 1, 1e-12);
}

// -- the 199 recorded cases ---------------------------------------------------

for (const r of dump.cases) {
  const id = r.id;
  const UB = [r.UB.slice(0, 3), r.UB.slice(3, 6), r.UB.slice(6, 9)];
  const k = (2 * Math.PI) / r.lam;
  const QU = mv(UB, r.hkl);

  for (let i = 0; i < 3; i++) chk(id, `QU[${i}]`, QU[i], r.QU[i], 1e-9);

  // evaluate vs the page's evaluate
  const e = PP.evaluate(QU, r.alpha, r.phi, r.roll, k, azSign);
  chk(id, "eps", e.eps, r.eval0.eps, 1e-8);
  chk(id, "tth", e.tth, r.eval0.tth, 1e-8);
  chk(id, "az", e.az, r.eval0.az, 1e-8);
  chk(id, "beta", e.beta, r.eval0.beta, 1e-8);
  chk(id, "ok", e.ok, r.eval0.ok);
  chk(id, "flipped", e.flipped, r.eval0.flipped);
  for (let i = 0; i < 3; i++) {
    chk(id, `kf[${i}]`, e.kf[i], r.eval0.kf[i], 1e-8);
    chk(id, `Q[${i}]`, e.Q[i], r.eval0.Ql[i], 1e-8);
    chk(id, `n[${i}]`, e.n[i], r.eval0.n[i], 1e-8);
  }

  // phi solver vs the page's solver and its brute-force scan
  const sp = PP.solvePhi(QU, r.alpha, k);
  const spAny = sp.length === 1 && sp[0] === "any";
  chk(id, "anyPhi flag", spAny, !!r.anyPhi);
  if (!spAny) {
    if (!rootSetsMatch(sp, r.solPhiRaw.map(Number), 1e-6))
      fail(id, "solPhi", `module ${sp} vs page ${r.solPhiRaw}`);
    if (!rootSetsMatch(sp, r.scanPhi, 1e-5))
      fail(id, "scanPhi", `module ${sp} vs page scan ${r.scanPhi}`);
  }
  for (const s of r.solPhi) chk(id, "solPhi eps", s.eps, 0, 1e-8);

  // alpha solver vs page solver and scan
  const sa = PP.solveAlpha(QU, r.phi, k);
  if (!rootSetsMatch(sa, r.solAlpha.map((x) => x.a), 1e-6))
    fail(id, "solAlpha", `module ${sa} vs page ${r.solAlpha.map((x) => x.a)}`);
  if (!rootSetsMatch(sa, r.scanAlpha, 1e-5))
    fail(id, "scanAlpha", `module ${sa} vs page scan ${r.scanAlpha}`);
  for (const s of r.solAlpha) chk(id, "solAlpha eps", s.eps, 0, 1e-8);

  // reachability band
  const myBand = PP.alphaBand(QU, k, r.roll);
  const pb = r.band;
  if (
    myBand.length !== pb.length ||
    myBand.some(
      (run, i) =>
        Math.abs(run[0] - pb[i][0]) > 1e-9 || Math.abs(run[1] - pb[i][1]) > 1e-9,
    )
  )
    fail(id, "band", `module ${JSON.stringify(myBand)} vs page ${JSON.stringify(pb)}`);

  // solutions() in 'alpha' mode: same phi set as the raw branches, all on
  // Bragg, ok-first ordering
  const sols = PP.solutions(QU, "alpha", r.alpha, r.phi, r.roll, k, azSign);
  for (const s of sols) chk(id, "solutions eps", s.eps, 0, 1e-8);
  if (spAny) {
    chk(id, "anyPhi sols count", sols.length, 1);
    chk(id, "anyPhi flag set", !!sols[0] && !!sols[0].anyPhi, true);
  } else {
    const wantCount = new Set(
      r.solPhiRaw.map((x) => Math.round(PP.wrap(Number(x)) * 1e6)),
    ).size;
    chk(id, "solutions count (dedup)", sols.length, wantCount);
    if (!rootSetsMatch(sols.map((s) => s.phi), r.solPhiRaw.map(Number), 1e-6))
      fail(id, "solutions phi set", `${sols.map((s) => s.phi)} vs ${r.solPhiRaw}`);
    for (let i = 1; i < sols.length; i++)
      if (sols[i - 1].ok === false && sols[i].ok === true)
        fail(id, "solutions order", "unblocked solution after blocked one");
  }

  // solutions() in 'phi' mode: same alpha set as the solveAlpha branches
  const sols2 = PP.solutions(QU, "phi", r.alpha, r.phi, r.roll, k, azSign);
  for (const s of sols2) chk(id, "phi-mode solutions eps", s.eps, 0, 1e-8);
  if (!rootSetsMatch(sols2.map((s) => s.alpha), r.solAlpha.map((x) => x.a), 1e-6))
    fail(id, "phi-mode alpha set", `${sols2.map((s) => s.alpha)} vs ${r.solAlpha.map((x) => x.a)}`);

  // Bragg identity on near-zero-eps solutions
  const q = norm(QU);
  if (q <= 2 * k + 1e-12) {
    const tthB = (2 * Math.asin(Math.min(q / (2 * k), 1))) / (Math.PI / 180);
    for (const s of r.solPhi)
      if (Math.abs(s.eps) < 1e-9) chk(id, "bragg tth", s.tth, tthB, 1e-6);
  }
}

// -- roll invariants ----------------------------------------------------------

const rollCases = dump.cases.filter((c) => c.kind === "roll");
if (rollCases.length) {
  const base = rollCases[0];
  for (const r of rollCases.slice(1)) {
    const k = (2 * Math.PI) / r.lam;
    const UB = [r.UB.slice(0, 3), r.UB.slice(3, 6), r.UB.slice(6, 9)];
    const QU = mv(UB, r.hkl);
    const e = PP.evaluate(QU, r.alpha, r.phi, r.roll, k, azSign);
    const b = PP.evaluate(QU, base.alpha, base.phi, base.roll, k, azSign);
    chk(r.id, "roll eps", e.eps, b.eps, 1e-9);
    chk(r.id, "roll tth", e.tth, b.tth, 1e-9);
    chk(r.id, "roll beta", e.beta, b.beta, 1e-9);
    chk(r.id, "roll az", PP.wrap(e.az), PP.wrap(b.az + r.roll), 1e-8);
  }
}

console.log(`cases: ${dump.cases.length}   checks: ${checks}   fails: ${fails.length}`);
for (const f of fails.slice(0, 30)) console.log("  FAIL:", f);
if (fails.length) process.exit(1);
console.log("parity_pp OK");
