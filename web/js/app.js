/*! The Game of Diffraction · © 2026 Miloš Dubajić · MIT · https://github.com/dubajicmilos/xrays-on-detector */

/**
 * The Game of Diffraction: wiring between the physics, the 3D scene and the
 * controls.
 *
 * State lives in one object. Anything that changes it calls requestSim(), which
 * coalesces work into the next animation frame, so dragging a slider never
 * queues more simulations than the display can show.
 */
import * as P from "./physics.js";
import { blockedGeometry, paint, rayGeometry, renderFrame } from "./render.js";
import { CifError, parseCif, setElements } from "./cif.js";
import { InstrumentScene } from "./scene.js";
import { CREDIT, logCredit, mountCredit } from "./credit.js";
import * as PP from "./pitchphi.js";
import { PitchPhiDeck } from "./pitchphi-deck.js";
import { PitchPhiRig } from "./pitchphi-scene.js";

const HC = 12.398419843320026; // keV.Angstrom

const DETECTORS = [
  ["PILATUS3 100K", 487, 195, 0.172],
  ["PILATUS3 300K", 487, 619, 0.172],
  ["PILATUS3 1M", 981, 1043, 0.172],
  ["PILATUS3 2M", 1475, 1679, 0.172],
  ["PILATUS3 6M", 2463, 2527, 0.172],
  ["EIGER2 X 1M", 1028, 1062, 0.075],
  ["EIGER2 X 4M", 2068, 2162, 0.075],
  ["EIGER2 X 9M", 3108, 3262, 0.075],
  ["EIGER2 X 16M", 4148, 4362, 0.075],
  ["LAMBDA 750K", 1554, 516, 0.055],
  ["JUNGFRAU 1M", 1024, 1024, 0.075],
];

const TARGETS = [
  ["beam  +y", [0, 1, 0]],
  ["vertical  +x", [1, 0, 0]],
  ["horizontal  +z", [0, 0, 1]],
  ["upstream  −y", [0, -1, 0]],
];

const MOTORS = [
  ["mu", "mu", -180, 180, "#e85d75"],
  ["eta", "omega", -180, 180, "#f0be46"],
  ["chi", "chi", -180, 180, "#5fc88c"],
  ["phi", "phi", -180, 180, "#5aa0ff"],
  ["delta", "delta", -100, 160, "#c9d3ea"],
  ["gamma", "gamma", -100, 160, "#c9d3ea"],
];

const $ = (id) => document.getElementById(id);

const st = {
  angles: { mu: 0, eta: 0, chi: 0, phi: 0, delta: 0, gamma: 0 },
  // Set at boot to a/9 for the first bundled structure (CsPbBr3, a = 5.87 A,
  // so 19.010 keV). At that wavelength the axis-aligned start is a zone axis
  // with reflections exactly on the Ewald sphere, so the detector has a
  // pattern on it the moment the page opens. Detune the energy and they go
  // out, which is the Bragg condition made visible.
  wavelength: 0.65,
  distance: 200,
  nFast: 1475,
  nSlow: 1679,
  pixelSize: 0.172,
  bin: 4,
  sigma: 0.01,
  mode: "transmission",
  surfaceHkl: [0, 0, 1],
  U: P.eye3(),
  Ubase: P.eye3(),
  rot: { rx: 0, ry: 0, rz: 0 },
  cell: null, // set from the first bundled structure at boot
  atoms: null, // likewise
  B: null,
  hkl: null,
  Qcryst: null, // B . hkl, kept with the list so a frame does not recompute it
  F2: null,
  builtQmax: 0, // the bound hkl was enumerated to; see rebuildReflections
  listCapped: false, // builtQmax stopped at the size limit, not at the panel
  cmap: "inferno",
  log: true,
  gain: 1, // display contrast; see the note on the slider below
  show: {
    rings: true,
    rays: true,
    floor: true,
    axes: true,
    missed: false,
    labels: true,
  },
  polarization: "horizontal",
  nSigma: 4,
  // Which machine is on the floor. The six-circle state above (angles, U,
  // Ubase, rot) stays exactly what it always was; the pitch-phi machine
  // keeps its own angles and its own zero-angle mount, because the two are
  // different physical datums and silently reinterpreting one as the other
  // would be a lie about the instrument.
  instrument: "sixc",
  pp: {
    angles: { pitch: 5, phi: 0, roll: 0, tt: 20, az: 0 },
    U: null, // set from st.U on first entry; then owned by the deck's mount
    init: false,
    cal: { pSign: 1, pOff: 0, fSign: 1, fOff: 0, azSign: 1 },
    mode: "alpha",
    target: [1, 0, 0],
    sols: [],
  },
};

let scene, tables, luts, structures;
let detCanvas,
  pending = false,
  needRebuild = true;
const motorRows = {},
  rotRows = {};
const spinning = new Set();
let chiTarget = null;
let ppRig = null,
  ppDeck = null;

// ---------------------------------------------------------------- helpers

function detector(bin = st.bin, angles = st.angles) {
  return new P.Detector({
    distance: st.distance,
    nFast: st.nFast,
    nSlow: st.nSlow,
    pixelSize: st.pixelSize,
    nu: angles.gamma,
    delta: angles.delta,
  }).binned(bin);
}

/**
 * The pitch-phi panel, expressed as the shared Detector.
 *
 * The instrument places its panel by (2theta, azimuth); the direction is all
 * the simulation needs from that, and the panel's own basis then keeps the
 * site's convention (slow axis as vertical as the arm allows, fast axis to
 * the viewer's right looking downstream) -- the same convention the
 * standalone tool draws, stated here as the virtual-panel convention rather
 * than a claim about the real machine's panel spin, which the tool does not
 * specify.
 */
function ppAnglesToDetector(angles = st.pp.angles) {
  const kh = PP.ppToGame(PP.detectorDir(angles.tt, angles.az, st.pp.cal.azSign));
  const { delta, gamma } = P.detectorAnglesFor(kh);
  return new P.Detector({
    distance: st.distance,
    nFast: st.nFast,
    nSlow: st.nSlow,
    pixelSize: st.pixelSize,
    nu: gamma,
    delta,
  });
}

/** The panel as the active machine stands, binned for display. */
const activeDetector = (bin = st.bin) =>
  st.instrument === "pp" ? ppAnglesToDetector().binned(bin) : detector(bin);

/** The largest |Q| the active panel reaches with the machine where it is. */
const qmaxAt = (angles = st.angles) =>
  (st.instrument === "pp" ? ppAnglesToDetector() : detector(1, angles)).maxQmax(
    st.wavelength,
  );

/**
 * The most the list may hold. The structure-factor sum costs one term per
 * reflection and atom, and every frame sweeps the whole list, so both the
 * count and the product are bounded: a 188-atom cell at 60 keV would
 * otherwise ask for millions of reflections and hold the page for a minute.
 * Past the bound the list stops at a smaller |Q| and the readout says so;
 * what is cut is the weakest, highest-angle tail.
 */
const MAX_HKL = 1e6;
const MAX_TERMS = 4e7;

function qmaxCap() {
  const nAtoms = Math.max(1, st.atoms.length);
  return P.qmaxForCount(st.B, Math.min(MAX_HKL, MAX_TERMS / nAtoms));
}

function rebuildReflections(qmax = qmaxAt()) {
  st.B = P.bMatrix(...st.cell);
  const cap = qmaxCap();
  st.listCapped = qmax > cap;
  st.builtQmax = st.listCapped ? cap : qmax;
  st.hkl = P.hklWithinQmax(st.B, st.builtQmax);
  st.Qcryst = P.qCryst(st.B, st.hkl);
  st.F2 = P.structureFactors(tables, st.atoms, st.B, st.hkl);
  needRebuild = false;
  return st.builtQmax;
}

/**
 * Extend the list to `qmax` if the arm has walked past what it holds.
 *
 * Enumerating hkl costs the cube of the bound, so a list rebuilt to exactly
 * what the arm needs is re-enumerated on nearly every frame of a delta drag.
 * Reaching a little past the requirement makes that occasional instead. The
 * margin is only spent where the arm has actually gone: a rebuild from a
 * change of wavelength, cell or detector geometry goes back to the panel's own
 * reach.
 */
const QMAX_MARGIN = 1.15;

function growReflections(qmax) {
  if (qmax > st.builtQmax && !st.listCapped)
    rebuildReflections(qmax * QMAX_MARGIN);
}

// A CIF that says P 1 asserts nothing, so those structures show no symbol at
// all, and a name that already carries the symbol does not repeat it.
function structureLabel(s) {
  const flat = (t) => t.toLowerCase().replace(/[\s_]/g, "");
  const show = s.spaceGroup && !flat(s.name).includes(flat(s.spaceGroup));
  return (
    `${s.name.replace(/_/g, " ")}` +
    (show ? `  ${s.spaceGroup}` : "") +
    `  (${s.atoms.length} atoms)`
  );
}

function applyStructure(s) {
  const c = s.cell;
  st.cell = [c.a, c.b, c.c, c.alpha, c.beta, c.gamma];
  st.atoms = s.atoms;
  needRebuild = true;
}

function surfaceNormalLab() {
  const n = P.crystalVector(st.B, st.surfaceHkl, "hkl");
  if (P.norm(n) < 1e-12) return [0, 0, 1];
  const { mu, eta, chi, phi } = st.angles;
  return P.unit(P.matVec(P.matMul(P.sampleMatrix(mu, eta, chi, phi), st.U), n));
}

const alphaDeg = () =>
  P.toDegrees(
    Math.asin(Math.max(-1, Math.min(1, -P.dot(P.BEAM, surfaceNormalLab())))),
  );

function requestSim() {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => {
    pending = false;
    simulate();
  });
}

// ---------------------------------------------------------------- simulate

function simulate() {
  // The list reaches out to the Q the panel corners reach, and that depends on
  // where the arm is standing: swinging delta and gamma out moves the window
  // to higher Q. Rebuilding on a geometry change alone left a far detector
  // driven to a peak reading a list that stops short of it, so the frame came
  // up blank until an unrelated edit happened to rebuild it. The list follows
  // the arm out and is only cut back by a real geometry change.
  if (needRebuild || !st.hkl) rebuildReflections();
  else growReflections(qmaxAt());

  // the second machine runs the same list through its own pose and its own
  // visibility rule; everything downstream of the pose is shared
  if (st.instrument === "pp") {
    simulatePp();
    return;
  }

  const det = detector();
  const { mu, eta, chi, phi } = st.angles;
  const ZU = P.matMul(P.sampleMatrix(mu, eta, chi, phi), st.U);
  let refl = P.excite({
    Qcryst: st.Qcryst,
    F2: st.F2,
    hkl: st.hkl,
    ZU,
    wavelength: st.wavelength,
    sigma: st.sigma,
    nSigma: st.nSigma,
  });

  const nNear = refl.count;
  let blockedFlat = null,
    nBlocked = 0;
  const alpha = alphaDeg();

  if (st.mode === "reflection" && refl.count) {
    const n = surfaceNormalLab();
    const keep = [],
      drop = [];
    for (let i = 0; i < refl.count; i++) {
      const beta =
        refl.khat[3 * i] * n[0] +
        refl.khat[3 * i + 1] * n[1] +
        refl.khat[3 * i + 2] * n[2];
      (alpha > 0 && beta > 0 ? keep : drop).push(i);
    }
    blockedFlat = new Float64Array(drop.length * 3);
    drop.forEach((i, j) => {
      blockedFlat[3 * j] = refl.khat[3 * i];
      blockedFlat[3 * j + 1] = refl.khat[3 * i + 1];
      blockedFlat[3 * j + 2] = refl.khat[3 * i + 2];
    });
    nBlocked = drop.length;
    refl = subset(refl, keep);
  }

  const { image, table } = renderFrame(det, refl, {
    wavelength: st.wavelength,
    sigma: st.sigma,
    polarizationMode: st.polarization,
    // A spot narrower than about a pixel disappears when the panel texture is
    // minified in the 3D view, so give the preview a floor of ~1 px.
    minSigmaPx: 1.0,
  });

  paint(detCanvas, image, det.nFast, det.nSlow, luts[st.cmap], {
    log: st.log,
    gain: st.gain,
  });
  fitDetectorCanvas(det);
  scene.touchDetectorImage();
  st.lastDet = det;
  st.lastTable = table;
  drawDetectorOverlay(det, table);

  const missLen = 1.55 * Math.max(st.distance, 60);
  const rays = rayGeometry(det, refl, table, missLen, {
    log: st.log,
    gain: st.gain,
  });
  rays.block = blockedFlat
    ? blockedGeometry(blockedFlat, nBlocked, missLen * 0.5)
    : new Float32Array(0);

  scene.setSceneScale(st.distance);
  const A = P.aMatrix(st.B);
  const Zm = P.matMul(P.sampleMatrix(mu, eta, chi, phi), st.U);
  const axes = [0, 1, 2].map((j) =>
    P.unit(P.matVec(Zm, [A[0][j], A[1][j], A[2][j]])),
  );

  scene.update({
    angles: st.angles,
    U: st.U,
    detector: det,
    frame: det.frame(),
    rays,
    crystalAxes: axes,
    surface: st.mode === "reflection" ? { normal: surfaceNormalLab() } : null,
    show: st.show,
  });

  updateReadouts(det, table, nNear, refl.count, nBlocked, alpha);
}

function subset(r, idx) {
  const n = idx.length;
  const out = {
    count: n,
    idx: new Int32Array(n),
    khat: new Float64Array(3 * n),
    eps: new Float64Array(n),
    excitation: new Float64Array(n),
    twoTheta: new Float64Array(n),
    F2: new Float64Array(n),
    hkl: new Int32Array(3 * n),
  };
  idx.forEach((i, j) => {
    out.eps[j] = r.eps[i];
    out.excitation[j] = r.excitation[i];
    out.twoTheta[j] = r.twoTheta[i];
    out.F2[j] = r.F2[i];
    for (let c = 0; c < 3; c++) {
      out.khat[3 * j + c] = r.khat[3 * i + c];
      out.hkl[3 * j + c] = r.hkl[3 * i + c];
    }
  });
  return out;
}

// ------------------------------------------------------------- pitch–phi

/** Z·U of the pitch-phi machine, in the shared lab frame. */
function ppZUE() {
  const a = st.pp.angles;
  return P.matMul(PP.sampleMatrixGame(a.pitch, a.phi, a.roll), st.pp.U);
}

/**
 * Unit surface normal of the pitch-phi sample, in the shared lab frame.
 *
 * The machine's surface definition: the sample surface is the plane whose
 * normal is +y_pp at zero angles, lifted by pitch and roll (φ is about that
 * normal and cannot move it). The solver in pitchphi.js works in the same
 * datum, which is why solutions and this visibility filter always agree;
 * the mounting block is what ties a chosen crystal plane to this datum, and
 * the crystal-axes gizmo shows how the two relate.
 */
function surfaceNormalPpLab() {
  const a = st.pp.angles;
  return PP.ppToGame(PP.surfaceNormal(a.pitch, a.roll));
}

/**
 * One frame of the pitch-phi machine through the shared pipeline.
 *
 * Same pipeline as the six-circle by design -- excite() over the shared
 * reflection list, the shared panel renderer, ray bundles, detector pane,
 * hover readouts -- but with the machine's own pose and its own visibility
 * rule: the standalone tool's ok flag, verbatim (incidence between 0 and
 * 90 degrees and the exit angle above the surface, otherwise the reflection
 * is drawn as blocked, not on the panel).
 */
function simulatePp() {
  const det = ppAnglesToDetector().binned(st.bin);
  const ZU = ppZUE();
  let refl = P.excite({
    Qcryst: st.Qcryst,
    F2: st.F2,
    hkl: st.hkl,
    ZU,
    wavelength: st.wavelength,
    sigma: st.sigma,
    nSigma: st.nSigma,
  });
  const nNear = refl.count;

  const n = surfaceNormalPpLab();
  const ainc = PP.wrap(st.pp.angles.pitch);
  const alphaOk = ainc > 0 && ainc <= 90;
  let blockedFlat = null,
    nBlocked = 0;
  if (refl.count) {
    const keep = [],
      drop = [];
    for (let i = 0; i < refl.count; i++) {
      const beta =
        refl.khat[3 * i] * n[0] +
        refl.khat[3 * i + 1] * n[1] +
        refl.khat[3 * i + 2] * n[2];
      (alphaOk && beta > 0 ? keep : drop).push(i);
    }
    blockedFlat = new Float64Array(drop.length * 3);
    drop.forEach((i, j) => {
      blockedFlat[3 * j] = refl.khat[3 * i];
      blockedFlat[3 * j + 1] = refl.khat[3 * i + 1];
      blockedFlat[3 * j + 2] = refl.khat[3 * i + 2];
    });
    nBlocked = drop.length;
    refl = subset(refl, keep);
  }

  const { image, table } = renderFrame(det, refl, {
    wavelength: st.wavelength,
    sigma: st.sigma,
    polarizationMode: st.polarization,
    minSigmaPx: 1.0,
  });

  paint(detCanvas, image, det.nFast, det.nSlow, luts[st.cmap], {
    log: st.log,
    gain: st.gain,
  });
  fitDetectorCanvas(det);
  scene.touchDetectorImage();
  st.lastDet = det;
  st.lastTable = table;
  drawDetectorOverlay(det, table);

  const missLen = 1.55 * Math.max(st.distance, 60);
  const rays = rayGeometry(det, refl, table, missLen, {
    log: st.log,
    gain: st.gain,
  });
  rays.block = blockedFlat
    ? blockedGeometry(blockedFlat, nBlocked, missLen * 0.5)
    : new Float32Array(0);

  scene.setSceneScale(st.distance);
  const A = P.aMatrix(st.B);
  const axes = [0, 1, 2].map((j) =>
    P.unit(P.matVec(ZU, [A[0][j], A[1][j], A[2][j]])),
  );

  const frame = det.frame();
  scene.update({
    angles: { mu: 0, eta: 0, chi: 0, phi: 0 }, // the six-circle rig parks
    U: st.U,
    rig: "pp",
    detector: det,
    frame,
    rays,
    crystalAxes: axes,
    surface: { normal: n },
    show: st.show,
  });
  if (ppRig)
    ppRig.update({
      angles: st.pp.angles,
      azSign: st.pp.cal.azSign,
      centre: frame.centre,
      rings: st.show.rings,
    });
  if (ppDeck) ppDeck.sync();
  updateReadouts(det, table, nNear, refl.count, nBlocked, ainc);
}

// ---------------------------------------------------------------- readouts

function updateReadouts(det, table, nNear, nOn, nBlocked, alpha) {
  // d_min goes with the count beside it, so both describe the list rather than
  // one describing the list and the other the panel.
  const qmax = st.builtQmax;
  st.summary =
    `${det.nFast}×${det.nSlow} px (${(det.pixelSize * 1000).toFixed(0)} µm bins)   ` +
    `${st.hkl.length / 3} hkl in range   d_min ${((2 * Math.PI) / qmax).toFixed(3)} Å` +
    (st.listCapped ? " (list capped at its size limit)" : "") +
    `   ${nNear} near the sphere   ${table.length} on the detector` +
    (nBlocked ? `   ${nBlocked} into the sample` : "");
  $("detInfo").textContent = st.summary;

  const legend = [[`#78e6ff`, `on the detector (${table.length})`]];
  if (st.show.missed) {
    legend.push([
      `#8296b9`,
      `misses the panel (${Math.max(nOn - table.length, 0)})`,
    ]);
    if (st.mode === "reflection")
      legend.push([`#eb6e5f`, `into the sample (${nBlocked})`]);
  }
  $("legend").innerHTML = legend
    .map(([c, t]) => `<div><i style="background:${c}"></i>${t}</div>`)
    .join("");

  if (st.mode === "reflection") {
    const el = $("alpha");
    el.textContent =
      `${alpha >= 0 ? "+" : ""}${alpha.toFixed(3)}°` +
      (alpha > 0 ? "" : "   BEAM BELOW SURFACE");
    el.style.color = alpha > 0 ? "var(--good)" : "var(--bad)";
  }

  const [a, b, c, al, be, ga] = st.cell;
  $("cellInfo").textContent =
    `a=${a.toFixed(4)} b=${b.toFixed(4)} c=${c.toFixed(4)} Å\n` +
    `α=${al.toFixed(2)} β=${be.toFixed(2)} γ=${ga.toFixed(2)}°` +
    `\n${st.atoms.length} atoms, |F(hkl)|² from atomic form factors`;

  // the UB matrix in this panel belongs to the six-circle mount, which is
  // what the readout means; the pitch-phi deck displays its own zero-angle
  // UB in its own (native) frame, so this one must not impersonate it
  if (st.instrument === "sixc") refreshUB();
}

function refreshUB() {
  const ub = P.UB(st.U, st.B, $("ubConv").value, st.wavelength);
  // an entry that rounds to zero is printed as +0.000000, not as -0.000000
  const cell = (v) => {
    const r = Math.abs(v) < 5e-7 ? 0 : v;
    return (r >= 0 ? "+" : "") + r.toFixed(6);
  };
  $("ub").textContent = ub.map((r) => r.map(cell).join("  ")).join("\n");
}

// Pixel c of the image covers CSS [c, c+1) of the canvas, so a spot centred on
// coordinate c sits at c + 0.5 on screen. Row 0 is the top (see render.js).
const cssX = (det, fast, sx) => (fast + 0.5) * sx;
const cssY = (det, slow, sy) => (det.nSlow - 1 - slow + 0.5) * sy;

function drawDetectorOverlay(det, table) {
  const box = $("detOverlay");
  // The canvas carries no border (its frame is an outline), so this rect is
  // the image itself and the scale below maps pixels to it exactly.
  const r = detCanvas.getBoundingClientRect();
  const p = box.getBoundingClientRect();
  const sx = r.width / det.nFast,
    sy = r.height / det.nSlow;
  const ox = r.left - p.left,
    oy = r.top - p.top;

  const bcx = ox + cssX(det, det.beamCenterFast, sx);
  const bcy = oy + cssY(det, det.beamCenterSlow, sy);
  let svg =
    `<svg width="100%" height="100%" style="position:absolute;inset:0">` +
    `<line x1="${bcx}" y1="${oy}" x2="${bcx}" y2="${oy + r.height}" stroke="#78c8ff66" stroke-dasharray="4 4"/>` +
    `<line x1="${ox}" y1="${bcy}" x2="${ox + r.width}" y2="${bcy}" stroke="#78c8ff66" stroke-dasharray="4 4"/>`;
  if (st.show.labels) {
    for (const t of [...table]
      .sort((x, y) => y.intensity - x.intensity)
      .slice(0, 22)) {
      const x = ox + cssX(det, t.fast, sx),
        y = oy + cssY(det, t.slow, sy);
      svg +=
        `<circle cx="${x}" cy="${y}" r="5.5" fill="none" stroke="#8cf0d2bb"/>` +
        `<text x="${x + 7}" y="${y - 5}" fill="#8cf0d2dd" font-size="10"` +
        ` font-family="Consolas,monospace">${t.h} ${t.k} ${t.l}</text>`;
    }
  }
  box.innerHTML = svg + "</svg>";
}

// ---------------------------------------------------------------- controls

/**
 * Let a number box be typed into without writing over what is in it.
 *
 * `apply` takes each finished value and `settled` gives the value the box is
 * squared up with once the edit is committed. A number input reports an empty
 * value while it holds "-" or "0.", which is what a minus sign and a decimal
 * point look like on the way in, so a handler that echoed a rounded number
 * back into the box rubbed the character out as it was typed: no negative or
 * fractional angle could be entered at all. Half-typed and out-of-range values
 * are ignored while typing, and nothing is written to the box until it loses
 * focus. On commit a finished value outside the range is clamped to it rather
 * than dropped: "251" in a box that stops at 250 used to leave the 25 that
 * went in on the way, and now gives 250.
 */
function bindTypedNumber(el, lo, hi, apply, settled) {
  el.addEventListener("input", () => {
    const v = parseFloat(el.value);
    if (!Number.isFinite(v) || v < lo || v > hi) return;
    apply(v);
  });
  el.addEventListener("change", () => {
    const v = parseFloat(el.value);
    if (Number.isFinite(v) && (v < lo || v > hi))
      apply(Math.max(lo, Math.min(hi, v)));
    el.value = settled();
  });
}

function buildMotorRows() {
  const host = $("motorRows");
  for (const [name, label, lo, hi, colour] of MOTORS) {
    const row = document.createElement("div");
    row.className = "motor";
    row.innerHTML =
      `<span class="name" style="color:${colour}">${label}</span>` +
      `<input type="range" min="${lo}" max="${hi}" step="0.01" value="0">` +
      `<input type="number" min="${lo}" max="${hi}" step="0.1" value="0">` +
      `<button class="run" title="rotate ${label} continuously">▶</button>`;
    const [range, num, run] = [
      row.children[1],
      row.children[2],
      row.children[3],
    ];
    const set = (v, silent, keepBox) => {
      v = Math.max(lo, Math.min(hi, v));
      range.value = v;
      if (!keepBox) num.value = Number(v.toFixed(2));
      st.angles[name] = v;
      if (!silent) requestSim();
    };
    // Taking hold of a motor, by its slider or its box, takes it off a running
    // move and stops it spinning: a box rewritten every frame by the spin
    // cannot be typed into.
    const takeHold = () => {
      stopAnim();
      if (spinning.has(name)) run.click();
    };
    range.addEventListener("pointerdown", takeHold);
    range.addEventListener("input", () => {
      takeHold();
      set(parseFloat(range.value));
    });
    num.addEventListener("focus", takeHold);
    num.addEventListener("input", takeHold);
    bindTypedNumber(
      num,
      lo,
      hi,
      (v) => set(v, false, true),
      () => Number(st.angles[name].toFixed(2)),
    );
    run.addEventListener("click", () => {
      if (spinning.has(name)) {
        spinning.delete(name);
        run.classList.remove("on");
        run.textContent = "▶";
      } else {
        stopAnim();
        spinning.add(name);
        run.classList.add("on");
        run.textContent = "■";
      }
    });
    motorRows[name] = { set, range, num, run, lo, hi };
    host.appendChild(row);
  }
}

function buildRotRows() {
  const host = $("rotRows");
  for (const [name, colour] of [
    ["rx", "#ff6e6e"],
    ["ry", "#8ceb8c"],
    ["rz", "#82afff"],
  ]) {
    const [lo, hi] = [-180, 180];
    const row = document.createElement("div");
    row.className = "motor";
    row.innerHTML =
      `<span class="name" style="color:${colour}">${name}</span>` +
      `<input type="range" min="${lo}" max="${hi}" step="0.1" value="0">` +
      `<input type="number" min="${lo}" max="${hi}" step="1" value="0">`;
    const [range, num] = [row.children[1], row.children[2]];
    const set = (v, silent, keepBox) => {
      range.value = v;
      if (!keepBox) num.value = Number(v.toFixed(1));
      st.rot[name] = v;
      st.U = P.matMul(P.eulerMatrix(st.rot.rx, st.rot.ry, st.rot.rz), st.Ubase);
      if (!silent) requestSim();
    };
    range.addEventListener("input", () => set(parseFloat(range.value)));
    bindTypedNumber(
      num,
      lo,
      hi,
      (v) => set(v, false, true),
      () => Number(st.rot[name].toFixed(1)),
    );
    rotRows[name] = { set };
    host.appendChild(row);
  }
}

function rebaseOrientation() {
  st.Ubase = st.U.map((r) => r.slice());
  for (const k of ["rx", "ry", "rz"]) rotRows[k].set(0, true);
  st.rot = { rx: 0, ry: 0, rz: 0 };
}

const SIXC_HINT =
  "Six circles: sample on a (mu, eta, chi, phi) cradle, detector on a " +
  "(delta, gamma) arm.";
const PP_HINT =
  "Surface machine: sample on a pitch cradle and a φ spindle, detector on " +
  "(2θ, azimuth). α is the incidence angle from the surface.";

/**
 * Move the second machine onto the floor, or the first one back.
 *
 * Each machine's settings are kept (angles, mount, detector drive), so
 * switching back and forth is free; nothing is reinterpreted between them.
 * Anything moving is stopped first: a spin or an eased move belongs to the
 * machine it was started on, and letting it run against the other one's
 * state would fight over whose angles mean what.
 */
function setInstrument(name) {
  if (name === st.instrument) return;
  for (const n of [...spinning]) motorRows[n].run.click();
  stopAnim();
  st.instrument = name;
  if (name === "pp" && !st.pp.init) {
    // First entry: the crystal starts on the same nominal mount it had on
    // the six-circle; the deck's mounting block re-mounts it properly, and
    // the two mounts are then saved independently.
    st.pp.U = st.U.map((r) => r.slice());
    st.pp.init = true;
  }
  $("instrument").value = name;
  $("instrumentHint").textContent = name === "pp" ? PP_HINT : SIXC_HINT;
  document
    .querySelectorAll(".six-only")
    .forEach((el) => el.classList.toggle("hidden", name !== "sixc"));
  document
    .querySelectorAll(".pp-only")
    .forEach((el) => el.classList.toggle("hidden", name !== "pp"));
  if (ppRig) ppRig.setVisible(name === "pp");
  if (ppDeck) ppDeck.activate(name === "pp");
  // the reachable |Q| is the panel's, and the panel changed
  needRebuild = true;
  requestSim();
}

// -- continuous rotation and animated moves

let animation = null;
function stopAnim() {
  animation = null;
}

/**
 * Drive motors to `targets` over `duration` ms with an ease-in-out, then call
 * `onDone`. Time-based rather than frame-based, so the move takes as long on
 * a 144 Hz display as on a 30 Hz one, and whatever waits on it (Move chi
 * re-running Find omega) runs when the motors have actually arrived.
 */
function animateTo(targets, onDone = null, duration = 450) {
  // Any motor being driven stops spinning: the two would fight over it.
  for (const k of Object.keys(targets))
    if (spinning.has(k)) motorRows[k].run.click();
  const start = {};
  for (const k of Object.keys(targets)) start[k] = st.angles[k];
  // Build for where the arm is going before it sets off, so the reflection it
  // is driving to is on the frame for the whole move rather than appearing at
  // the end of it. simulate() still grows the list on the way, which covers a
  // path that swings further out than either end of it.
  growReflections(Math.max(qmaxAt(), qmaxAt({ ...st.angles, ...targets })));
  animation = { start, targets, t0: null, duration, onDone };
}

let lastTick = null;

function tick(now) {
  requestAnimationFrame(tick);
  // Elapsed time, capped so a tab coming back from the background does not
  // leap: the spin is in degrees per second, not per frame.
  if (!Number.isFinite(now)) now = performance.now();
  const dt = lastTick === null ? 0 : Math.min((now - lastTick) / 1000, 0.1);
  lastTick = now;
  let changed = false;

  if (spinning.size) {
    const step = (parseFloat($("speed").value) || 0) * dt;
    for (const name of spinning) {
      const m = motorRows[name];
      let v = st.angles[name] + step;
      if (v > m.hi) v = m.lo + (v - m.hi);
      else if (v < m.lo) v = m.hi - (m.lo - v);
      m.set(v, true);
      changed = true;
    }
  }
  if (animation) {
    if (animation.t0 === null) animation.t0 = now;
    let t = Math.min(1, (now - animation.t0) / animation.duration);
    const done = t >= 1;
    t = t * t * (3 - 2 * t);
    for (const [k, target] of Object.entries(animation.targets)) {
      motorRows[k].set(
        animation.start[k] + (target - animation.start[k]) * t,
        true,
      );
    }
    changed = true;
    if (done) {
      const { onDone } = animation;
      animation = null;
      if (onDone) onDone();
    }
  }
  if (changed) simulate();
  if (scene.dirty) scene.render();
}

// ---------------------------------------------------------------- wiring

function bindInputs() {
  /**
   * The min and max a number input already carries, as numbers.
   *
   * The browser enforces them for form validation and not for typing, so a
   * field can hold a value its own attributes forbid. Reading them here makes
   * one guard serve every box rather than repeating a range per handler.
   */
  const limitsOf = (el) => [
    el.min === "" ? -Infinity : parseFloat(el.min),
    el.max === "" ? Infinity : parseFloat(el.max),
  ];

  const shownWavelength = () => st.wavelength.toFixed(4);
  const shownEnergy = () => (HC / st.wavelength).toFixed(3);

  // The range matters as much as the finiteness, which is why bindTypedNumber
  // is given both: clearing the wavelength box and typing gives a moment at
  // zero, where k = 2*pi/lambda is infinite, the limiting sphere has no bound,
  // and hklWithinQmax tries to enumerate every integer triple. That crashes
  // the tab rather than drawing nothing.
  const num = (id, key, after, show = () => st[key]) => {
    const el = $(id);
    el.value = st[key];
    const [lo, hi] = limitsOf(el);
    bindTypedNumber(
      el,
      lo,
      hi,
      (v) => {
        st[key] = v;
        if (after) after();
        requestSim();
      },
      show,
    );
  };

  num(
    "wl",
    "wavelength",
    () => {
      $("energy").value = shownEnergy();
      needRebuild = true;
    },
    shownWavelength,
  );
  // both boxes carry the same rounding they get from each other's handler, so
  // the exact default (a/9) shows as a wavelength and not as a raw float
  $("wl").value = shownWavelength();
  $("energy").value = shownEnergy();
  bindTypedNumber(
    $("energy"),
    ...limitsOf($("energy")),
    (e) => {
      st.wavelength = HC / e;
      $("wl").value = shownWavelength();
      needRebuild = true;
      requestSim();
    },
    shownEnergy,
  );
  num("pixel", "pixelSize", () => {
    needRebuild = true;
  });
  num("nFast", "nFast", () => {
    needRebuild = true;
  });
  num("nSlow", "nSlow", () => {
    needRebuild = true;
  });
  num("distance", "distance", () => {
    needRebuild = true;
  });
  num("bin", "bin");
  num("sigma", "sigma");

  const dp = $("detPreset");
  DETECTORS.forEach(([n, f, s, p], i) => {
    const o = document.createElement("option");
    o.value = i;
    o.textContent = `${n}  (${f}×${s}, ${(p * 1000).toFixed(0)} µm)`;
    dp.appendChild(o);
  });
  const custom = document.createElement("option");
  custom.value = "custom";
  custom.textContent = "custom";
  dp.appendChild(custom);
  dp.value = 3;
  dp.addEventListener("change", () => {
    if (dp.value === "custom") return;
    const [, f, s, p] = DETECTORS[dp.value];
    st.nFast = f;
    st.nSlow = s;
    st.pixelSize = p;
    $("nFast").value = f;
    $("nSlow").value = s;
    $("pixel").value = p;
    needRebuild = true;
    requestSim();
  });
  // Editing the boxes by hand leaves the preset name, so the select says
  // "custom" unless the numbers still match one of them.
  const syncPreset = () => {
    const i = DETECTORS.findIndex(
      ([, f, s, p]) => f === st.nFast && s === st.nSlow && p === st.pixelSize,
    );
    dp.value = i < 0 ? "custom" : i;
  };
  for (const id of ["pixel", "nFast", "nSlow"])
    $(id).addEventListener("input", syncPreset);

  const sel = $("structure");
  for (const s of structures) {
    const o = document.createElement("option");
    o.value = s.name;
    o.textContent = structureLabel(s);
    sel.appendChild(o);
  }
  sel.addEventListener("change", () => {
    applyStructure(structures.find((x) => x.name === sel.value));
    requestSim();
  });

  // Reading a CIF here rather than shipping it: the file never leaves the
  // machine, and the reader expands the symmetry, because the structure factor
  // sum applies none of its own.
  const status = $("cifStatus");
  const say = (text, bad) => {
    status.textContent = text;
    status.classList.remove("hidden");
    status.style.color = bad ? "var(--bad)" : "var(--good)";
  };
  $("cifPick").addEventListener("click", () => $("cifFile").click());
  $("cifFile").addEventListener("change", async (ev) => {
    const file = ev.target.files && ev.target.files[0];
    if (!file) return;
    ev.target.value = ""; // so the same file can be loaded again
    try {
      const doc = parseCif(await file.text(), file.name.replace(/\.cif$/i, ""));
      // The reader checks the cell too; this keeps a structure from any
      // source out of the list unless a lattice can be built from it, because
      // the first attempt would otherwise be inside an animation frame.
      const c = doc.cell;
      P.bMatrix(c.a, c.b, c.c, c.alpha, c.beta, c.gamma);
      const clash = structures.findIndex((x) => x.name === doc.name);
      if (clash >= 0) {
        structures[clash] = doc;
        sel.options[clash].textContent = structureLabel(doc);
      } else {
        structures.push(doc);
        const o = document.createElement("option");
        o.value = doc.name;
        o.textContent = structureLabel(doc);
        sel.appendChild(o);
      }
      sel.value = doc.name;
      applyStructure(doc);
      const notes = [`${doc.atoms.length} atoms`];
      if (doc.spaceGroup) notes.push(doc.spaceGroup);
      if (doc.skippedSites)
        notes.push(
          `${doc.skippedSites} atom ${doc.skippedSites === 1 ? "site" : "sites"} ` +
            "without a readable position skipped",
        );
      if (doc.blocksInFile > 1)
        notes.push(
          `${doc.blocksInFile} structures in the file, read ${doc.block}`,
        );
      say(`Loaded ${file.name}: ${notes.join(", ")}`, false);
      requestSim();
    } catch (err) {
      say(
        err instanceof CifError
          ? `Cannot use ${file.name}: ${err.message}`
          : `Cannot read ${file.name}: ${err.message}`,
        true,
      );
    }
  });

  for (const r of document.querySelectorAll("input[name=mode]")) {
    r.addEventListener("change", () => {
      st.mode = r.value;
      $("surfaceBox").classList.toggle("hidden", st.mode !== "reflection");
      requestSim();
    });
  }
  for (const id of ["sh", "sk", "sl"]) {
    $(id).addEventListener("input", () => {
      st.surfaceHkl = [
        +$("sh").value || 0,
        +$("sk").value || 0,
        +$("sl").value || 0,
      ];
      requestSim();
    });
  }
  $("mountFlat").addEventListener("click", () => {
    if (!st.surfaceHkl.some(Boolean)) {
      $("alignMsg").textContent = "Give a non-zero surface (hkl) first.";
      return;
    }
    const U = P.alignInLab(st.U, st.B, st.surfaceHkl, [1, 0, 0], {
      frame: "phi",
    });
    if (!U) {
      $("alignMsg").textContent = "Alignment failed.";
      return;
    }
    st.U = U;
    rebaseOrientation();
    $("alignMsg").textContent =
      `Surface normal (${st.surfaceHkl.join(" ")}) mounted vertical.`;
    requestSim();
  });

  for (const [id, sel2] of [
    ["a1t", TARGETS],
    ["a2t", TARGETS],
  ]) {
    sel2.forEach(([label], i) => {
      const o = document.createElement("option");
      o.value = i;
      o.textContent = label;
      $(id).appendChild(o);
    });
  }
  $("a1t").value = 0;
  $("a2t").value = 1;

  const idxOf = (a, b, c) => [
    +$(a).value || 0,
    +$(b).value || 0,
    +$(c).value || 0,
  ];

  $("a1go").addEventListener("click", () => {
    const idx = idxOf("a1h", "a1k", "a1l");
    if (!idx.some(Boolean)) {
      $("alignMsg").textContent = "Give a non-zero direction.";
      return;
    }
    const Z = P.sampleMatrix(
      st.angles.mu,
      st.angles.eta,
      st.angles.chi,
      st.angles.phi,
    );
    const U = P.alignInLab(st.U, st.B, idx, TARGETS[$("a1t").value][1], {
      kind: $("a1kind").value,
      Z,
    });
    if (!U) {
      $("alignMsg").textContent = "Alignment failed.";
      return;
    }
    st.U = U;
    rebaseOrientation();
    $("alignMsg").textContent =
      `${$("a1kind").value === "hkl" ? "(hkl)" : "[uvw]"} ${idx.join(" ")} now along ` +
      `${TARGETS[$("a1t").value][0].trim()} at these motor positions.`;
    requestSim();
  });

  $("a2go").addEventListener("click", () => {
    const idx = idxOf("a2h", "a2k", "a2l");
    if (!idx.some(Boolean)) {
      $("alignMsg").textContent = "Give a non-zero direction.";
      return;
    }
    const Z = P.sampleMatrix(
      st.angles.mu,
      st.angles.eta,
      st.angles.chi,
      st.angles.phi,
    );
    const axis = TARGETS[$("a1t").value][1];
    const U = P.alignSecondaryInLab(
      st.U,
      st.B,
      idx,
      TARGETS[$("a2t").value][1],
      axis,
      { kind: $("a2kind").value, Z },
    );
    if (!U) {
      $("alignMsg").textContent =
        "That direction is parallel to the primary axis, so spinning about it " +
        "changes nothing. Pick a different one.";
      return;
    }
    st.U = U;
    rebaseOrientation();
    $("alignMsg").textContent =
      `Spun about ${TARGETS[$("a1t").value][0].trim()} to bring ${idx.join(" ")} ` +
      `as close as possible to ${TARGETS[$("a2t").value][0].trim()}.`;
    requestSim();
  });

  $("resetU").addEventListener("click", () => {
    st.U = P.eye3();
    rebaseOrientation();
    $("alignMsg").textContent =
      "U reset to the identity: crystal axes on the phi frame.";
    requestSim();
  });

  $("ubConv").addEventListener("change", refreshUB);
  const copyBtn = $("ubCopy");
  copyBtn.addEventListener("click", async () => {
    // The button reports what happened; a refused clipboard leaves the
    // matrix selected so it can still be copied by hand.
    let label = "Copied";
    try {
      await navigator.clipboard.writeText($("ub").textContent);
    } catch {
      label = "Copy refused: select the text";
      const sel = window.getSelection();
      if (sel) sel.selectAllChildren($("ub"));
    }
    copyBtn.textContent = label;
    setTimeout(() => {
      copyBtn.textContent = "Copy";
    }, 1800);
  });

  $("stopAll").addEventListener("click", () => {
    for (const n of [...spinning]) motorRows[n].run.click();
  });
  $("zeroAll").addEventListener("click", () => {
    for (const n of [...spinning]) motorRows[n].run.click();
    stopAnim();
    for (const [name] of MOTORS) motorRows[name].set(0, true);
    requestSim();
  });

  $("findOmega").addEventListener("click", findOmega);
  $("instrument").addEventListener("change", () => setInstrument($("instrument").value));
  $("driveThere").addEventListener("click", () => {
    const opt = $("solutions").selectedOptions[0];
    if (!opt || !opt.dataset.eta) return;
    animateTo({
      eta: +opt.dataset.eta,
      delta: +opt.dataset.delta,
      gamma: +opt.dataset.gamma,
    });
  });
  $("aimDet").addEventListener("click", () => {
    const hkl = driveHkl();
    const msg = $("reachMsg");
    if (!hkl.some(Boolean)) {
      msg.textContent = "0 0 0 is the direct beam; give a reflection.";
      return;
    }
    const r = P.etaReach(st.B, st.U, hkl, st.angles, st.wavelength);
    if (!r.inLimitingSphere) {
      msg.textContent = outsideSphereText(r);
      return;
    }
    const arm = armWithinLimits(
      P.aimDetectorAt(st.B, st.U, hkl, st.angles, st.wavelength),
    );
    if (!arm) {
      msg.textContent =
        "The arm cannot reach where this reflection scatters at these " +
        "motor positions: it is outside the delta / gamma limits.";
      return;
    }
    // The arm can point at where the reflection would scatter whether or not
    // it is on the sphere now; say which, so an empty panel is not a puzzle.
    const onSphere = Math.abs(qEps(hkl)) <= st.nSigma * st.sigma;
    msg.textContent = onSphere
      ? `Arm to delta ${arm.delta.toFixed(2)}, gamma ${arm.gamma.toFixed(2)}.`
      : `Arm to delta ${arm.delta.toFixed(2)}, gamma ${arm.gamma.toFixed(2)}, ` +
        "but the reflection is not on the Ewald sphere at these motor " +
        "positions, so nothing will be there: use Find omega first.";
    animateTo(arm);
  });
  $("moveChi").addEventListener("click", () => {
    if (chiTarget === null) return;
    animateTo({ chi: chiTarget }, findOmega);
  });

  for (const [id, key] of [
    ["showRings", "rings"],
    ["showRays", "rays"],
    ["showFloor", "floor"],
    ["showAxes", "axes"],
    ["showMissed", "missed"],
    ["showLabels", "labels"],
  ]) {
    $(id).addEventListener("change", () => {
      st.show[key] = $(id).checked;
      requestSim();
    });
  }
  $("logScale").addEventListener("change", () => {
    st.log = $("logScale").checked;
    requestSim();
  });

  // Contrast is a display gain, not a change to the physics: the frame is
  // normalised to its brightest spot, so a reflection 1e-5 of the peak sits
  // below one colour step until you stretch the map. The slider is decades,
  // because that is the range the weak reflections actually span.
  const gainEl = $("gain");
  const showGain = () => {
    const g = st.gain;
    const round = (v) => (v < 10 ? +v.toFixed(1) : Math.round(v));
    $("gainInfo").textContent =
      g >= 1e6
        ? `x${round(g / 1e6)}M`
        : g >= 1e3
          ? `x${round(g / 1e3)}k`
          : `x${round(g)}`;
  };
  gainEl.addEventListener("input", () => {
    st.gain = Math.pow(10, parseFloat(gainEl.value) || 0);
    showGain();
    requestSim();
  });
  showGain();

  const cm = $("cmap");
  for (const name of Object.keys(luts)) {
    const o = document.createElement("option");
    o.value = name;
    o.textContent = name;
    cm.appendChild(o);
  }
  cm.value = st.cmap;
  cm.addEventListener("change", () => {
    st.cmap = cm.value;
    requestSim();
  });

  $("panelToggle").addEventListener("click", () =>
    $("panel").classList.toggle("open"),
  );
}

const driveHkl = () => [
  +$("dh").value || 0,
  +$("dk").value || 0,
  +$("dl").value || 0,
];

/** Excitation error of one reflection at the current motor positions. */
function qEps(hkl) {
  const k = (2 * Math.PI) / st.wavelength;
  const q = P.qLab(st.B, st.U, hkl, st.angles);
  return Math.hypot(q[0], k + q[1], q[2]) - k;
}

const outsideSphereText = (r) =>
  `|Q| = ${r.Q.toFixed(3)} > 2k = ${((4 * Math.PI) / st.wavelength).toFixed(3)} Å⁻¹. ` +
  "No geometry reaches this reflection at this wavelength; you need a shorter one.";

/**
 * The (delta, gamma) pair that points the arm at the same direction and lies
 * within the motor limits, or null if neither does. The arm direction
 * (sin d, cos g cos d, sin g cos d) is unchanged by d -> 180 - d together
 * with g -> g + 180, so a solution past a limit usually has a twin inside it.
 */
function armWithinLimits(a) {
  const lim = Object.fromEntries(
    MOTORS.filter(([n]) => n === "delta" || n === "gamma").map(
      ([n, , lo, hi]) => [n, [lo, hi]],
    ),
  );
  const ok = (x) =>
    x.delta >= lim.delta[0] &&
    x.delta <= lim.delta[1] &&
    x.gamma >= lim.gamma[0] &&
    x.gamma <= lim.gamma[1];
  const wrap = (g) => ((((g + 180) % 360) + 360) % 360) - 180;
  const twin = { delta: 180 - a.delta, gamma: wrap(a.gamma + 180) };
  if (ok(a)) return { delta: a.delta, gamma: a.gamma };
  if (ok(twin)) return twin;
  return null;
}

function findOmega() {
  const hkl = driveHkl();
  const sel = $("solutions");
  sel.innerHTML = "";
  $("moveChi").disabled = true;
  chiTarget = null;

  if (!hkl.some(Boolean)) {
    sel.innerHTML = "<option>give a reflection</option>";
    $("reachMsg").textContent = "0 0 0 is the direct beam; give a reflection.";
    return;
  }
  const r = P.etaReach(st.B, st.U, hkl, st.angles, st.wavelength);
  if (!r.inLimitingSphere) {
    sel.innerHTML = "<option>outside the limiting sphere</option>";
    $("reachMsg").textContent = outsideSphereText(r);
    return;
  }
  if (!r.feasible) {
    sel.innerHTML = "<option>no omega solution at this chi / phi / mu</option>";
    const chi = P.suggestChi(st.B, st.U, hkl, st.angles, st.wavelength);
    $("reachMsg").textContent =
      `Blind cone. Rocking omega holds the part of Q along the omega axis fixed, ` +
      `so k_i·Q can only sweep [${r.lo.toFixed(4)}, ${r.hi.toFixed(4)}] while Bragg ` +
      `needs ${r.required.toFixed(4)} (short by ${r.shortfall.toFixed(4)}). ` +
      (chi === null
        ? "No chi alone fixes it; move phi or mu too."
        : `chi = ${chi.toFixed(2)}° brings it into reach.`);
    if (chi !== null) {
      chiTarget = chi;
      $("moveChi").disabled = false;
      $("moveChi").textContent = `Move chi to ${chi.toFixed(2)}°`;
    }
    return;
  }

  const sols = P.solveEta(st.B, st.U, hkl, st.angles, st.wavelength);
  let unreachable = 0;
  for (const e of sols) {
    const raw = P.aimDetectorAt(
      st.B,
      st.U,
      hkl,
      { ...st.angles, eta: e },
      st.wavelength,
    );
    const a = armWithinLimits(raw);
    const o = document.createElement("option");
    if (a) {
      o.textContent = `omega = ${e.toFixed(3)}  →  delta ${a.delta.toFixed(2)}  gamma ${a.gamma.toFixed(2)}`;
      o.dataset.eta = e;
      o.dataset.delta = a.delta;
      o.dataset.gamma = a.gamma;
    } else {
      // Listed so the omega is known, but not driveable: the arm cannot
      // follow, and a drive that stopped at a limit used to look like a
      // solution that had failed.
      o.textContent = `omega = ${e.toFixed(3)}  →  arm out of reach (delta ${raw.delta.toFixed(1)}, gamma ${raw.gamma.toFixed(1)})`;
      o.disabled = true;
      unreachable++;
    }
    sel.appendChild(o);
  }
  const first = [...sel.options].find((o) => !o.disabled);
  if (first) first.selected = true;
  $("reachMsg").textContent =
    `|Q| = ${r.Q.toFixed(4)} Å⁻¹,  d = ${((2 * Math.PI) / r.Q).toFixed(4)} Å,  ` +
    `${sols.length} omega solution(s)` +
    (unreachable
      ? `, ${unreachable} with the arm outside its delta / gamma limits.`
      : ".");
}

// ------------------------------------------------------- detector zoom / pan

const view = { zoom: 1, x: 0, y: 0 };

/**
 * Size the canvas element so the frame fits the pane at 100%, preserving the
 * detector's aspect ratio. The canvas keeps its intrinsic pixel size (one
 * texel per detector pixel); only its CSS box is fitted, and the zoom
 * transform scales from there.
 */
function fitDetectorCanvas(det) {
  const vp = $("detViewport");
  const w = vp.clientWidth - 20;
  const h = vp.clientHeight - 20;
  if (w <= 0 || h <= 0) return;
  const s = Math.min(w / det.nFast, h / det.nSlow);
  detCanvas.style.width = `${Math.max(1, Math.round(det.nFast * s))}px`;
  detCanvas.style.height = `${Math.max(1, Math.round(det.nSlow * s))}px`;
}

function applyDetTransform() {
  detCanvas.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.zoom})`;
  $("detZoomLabel").textContent = `${Math.round(view.zoom * 100)}%`;
  if (st.lastDet) drawDetectorOverlay(st.lastDet, st.lastTable || []);
}

function setZoom(factor, cx = null, cy = null) {
  const prev = view.zoom;
  const next = Math.max(0.25, Math.min(40, prev * factor));
  if (next === prev) return;
  // keep the point under the cursor fixed
  if (cx !== null) {
    const r = $("detViewport").getBoundingClientRect();
    const mx = cx - (r.left + r.width / 2) - view.x;
    const my = cy - (r.top + r.height / 2) - view.y;
    view.x -= mx * (next / prev - 1);
    view.y -= my * (next / prev - 1);
  }
  view.zoom = next;
  applyDetTransform();
}

function bindDetectorZoom() {
  const vp = $("detViewport");
  vp.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      setZoom(e.deltaY > 0 ? 1 / 1.15 : 1.15, e.clientX, e.clientY);
    },
    { passive: false },
  );

  let drag = null;
  vp.addEventListener("pointerdown", (e) => {
    drag = { x: e.clientX, y: e.clientY };
    vp.setPointerCapture(e.pointerId);
    vp.classList.add("dragging");
  });
  vp.addEventListener("pointermove", (e) => {
    if (!drag) return;
    view.x += e.clientX - drag.x;
    view.y += e.clientY - drag.y;
    drag = { x: e.clientX, y: e.clientY };
    applyDetTransform();
  });
  const end = (e) => {
    drag = null;
    vp.classList.remove("dragging");
    try {
      vp.releasePointerCapture(e.pointerId);
    } catch {}
  };
  vp.addEventListener("pointerup", end);
  vp.addEventListener("pointercancel", end);
  vp.addEventListener("dblclick", () => setZoom(1.8, null));

  $("detZoomIn").addEventListener("click", () => setZoom(1.4));
  $("detZoomOut").addEventListener("click", () => setZoom(1 / 1.4));
  $("detZoomReset").addEventListener("click", () => {
    view.zoom = 1;
    view.x = 0;
    view.y = 0;
    applyDetTransform();
  });
}

// ---------------------------------------------------------------- detector hover

function bindDetectorHover() {
  const pane = $("detPane");
  pane.addEventListener("pointermove", (ev) => {
    const det = activeDetector();
    const r = detCanvas.getBoundingClientRect();
    // CSS position to pixel coordinate: pixel c spans [c, c+1) on screen, so
    // its centre, where the physics puts the pixel, is at c + 0.5.
    const fx = ((ev.clientX - r.left) / r.width) * det.nFast - 0.5;
    const fy = ((ev.clientY - r.top) / r.height) * det.nSlow - 0.5;
    if (
      fx < -0.5 ||
      fy < -0.5 ||
      fx >= det.nFast - 0.5 ||
      fy >= det.nSlow - 0.5
    ) {
      $("detInfo").textContent = st.summary || "";
      return;
    }
    const slow = det.nSlow - 1 - fy;
    const { centre, eFast, eSlow } = det.frame();
    const u = (fx - det.beamCenterFast) * det.pixelSize;
    const v = (slow - det.beamCenterSlow) * det.pixelSize;
    const khat = P.unit([
      centre[0] + u * eFast[0] + v * eSlow[0],
      centre[1] + u * eFast[1] + v * eSlow[1],
      centre[2] + u * eFast[2] + v * eSlow[2],
    ]);
    const tt = P.toDegrees(Math.acos(Math.max(-1, Math.min(1, khat[1]))));
    const q = ((4 * Math.PI) / st.wavelength) * Math.sin(P.toRadians(tt) / 2);
    const d =
      q > 1e-6 ? `${((2 * Math.PI) / q).toFixed(3)} Å` : "∞ (direct beam)";
    $("detInfo").textContent =
      `px (${fx.toFixed(1)}, ${fy.toFixed(1)})   2θ ${tt.toFixed(2)}°   ` +
      `|Q| ${q.toFixed(3)} Å⁻¹   d ${d}`;
  });
  // the readout goes back to the frame summary once the pointer leaves
  pane.addEventListener("pointerleave", () => {
    $("detInfo").textContent = st.summary || "";
  });
}

// ---------------------------------------------------------------- boot

async function boot() {
  const grab = async (p) => {
    const r = await fetch(p);
    if (!r.ok) throw new Error(`${p}: ${r.status} ${r.statusText}`);
    return r.json();
  };
  [tables, luts] = await Promise.all([
    grab("data/scattering_factors.json"),
    grab("data/colormaps.json"),
  ]);
  // data/structures.json is written by tools/export_web_data.py, so bundling
  // another CIF is a re-export rather than a code change here.
  const names = await grab("data/structures.json");
  structures = await Promise.all(names.map((n) => grab(`data/${n}.json`)));

  setElements(Object.keys(tables));
  detCanvas = $("detCanvas");
  scene = new InstrumentScene($("view3d"));
  scene.setDetectorImage(detCanvas);

  // Open at a/9 for the first bundled structure: see the note on st.wavelength.
  st.wavelength = structures[0].cell.a / 9;
  applyStructure(structures[0]);
  buildMotorRows();
  buildRotRows();
  ppRig = new PitchPhiRig(scene.scene);
  scene.attachPpRig(ppRig);
  ppDeck = new PitchPhiDeck(st, { requestSim });
  ppDeck.build();
  bindInputs();
  bindDetectorZoom();
  bindDetectorHover();
  mountCredit($("panel"));
  logCredit();

  // deep link into a machine: ?instrument=pitchphi (allowlisted values only)
  if (new URLSearchParams(location.search).get("instrument") === "pitchphi")
    setInstrument("pp");

  new ResizeObserver(() => {
    scene.resize();
    requestSim();
  }).observe($("stage"));
  window.addEventListener("resize", () => requestSim());

  // The loading screen comes down only once the first frame has drawn, so a
  // failure inside it still lands on the message below.
  simulate();
  $("boot").remove();
  requestAnimationFrame(tick);

  // Scripting handle. Deliberate public API, not a debug leftover: it lets the
  // test harness drive the page without waiting on animation frames, and lets
  // anyone script the instrument from the console.
  window.diffractionGame = {
    credit: CREDIT,
    state: st,
    simulate,
    scene,
    physics: P,
    setAngles(a) {
      Object.assign(st.angles, a);
      for (const k of Object.keys(a)) motorRows[k]?.set(a[k], true);
      simulate();
    },
    setInstrument(name) {
      setInstrument(name === "pitchphi" ? "pp" : name);
    },
    pitchPhi: st.pp,
    render() {
      scene.render();
    },
  };
}

boot().catch((err) => {
  const b =
    $("boot") || document.body.appendChild(document.createElement("div"));
  b.id = "boot";
  b.className = "err";
  // Name the likely cause rather than always blaming file://: a browser
  // without WebGL fails at the 3D scene, and that needs different advice.
  const hint = /webgl/i.test(err.message)
    ? "The 3D view needs WebGL, which this browser could not provide. " +
      "Turn on hardware acceleration or try a current desktop browser."
    : location.protocol === "file:"
      ? "You opened this file directly; serve it over http instead " +
        "(ES modules and fetch do not work from file://)."
      : "Reload the page; if it keeps failing, the message above says what broke.";
  b.textContent = `Failed to start.\n\n${err.message}\n\n${hint}`;
  console.error(err);
});
