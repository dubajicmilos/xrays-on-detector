/**
 * Diffraction Game: wiring between the physics, the 3D scene and the controls.
 *
 * State lives in one object. Anything that changes it calls requestSim(), which
 * coalesces work into the next animation frame, so dragging a slider never
 * queues more simulations than the display can show.
 */
import * as P from "./physics.js";
import { blockedGeometry, paint, rayGeometry, renderFrame } from "./render.js";
import { InstrumentScene } from "./scene.js";

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

const CELLS = [
  ["Pseudocubic perovskite", 5.917, 5.917, 5.917, 90, 90, 90],
  ["MAPbI3 cubic (340 K)", 6.29, 6.29, 6.29, 90, 90, 90],
  ["Silicon", 5.431, 5.431, 5.431, 90, 90, 90],
  ["LaB6", 4.1569, 4.1569, 4.1569, 90, 90, 90],
  ["Tetragonal I4/mcm", 8.37, 8.37, 11.83, 90, 90, 90],
  ["Monoclinic small molecule", 10, 12, 15, 90, 103, 90],
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
  // 0.657444 A = 5.917/9 = a/9 for the default cell, i.e. 18.859 keV. At that
  // wavelength the axis-aligned start is a zone axis with twelve reflections
  // exactly on the Ewald sphere, so the detector has a pattern on it the
  // moment the page opens. Detune the energy and they go out, which is the
  // Bragg condition made visible.
  wavelength: 5.917 / 9,
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
  cell: CELLS[0].slice(1),
  atoms: null, // null => bare lattice
  B: null,
  hkl: null,
  F2: null,
  cmap: "inferno",
  log: true,
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
};

let scene, tables, luts, structures;
let detCanvas,
  detCtx,
  pending = false,
  needRebuild = true;
const motorRows = {},
  rotRows = {};
const spinning = new Set();
let chiTarget = null;

// ---------------------------------------------------------------- helpers

function detector(bin = st.bin) {
  return new P.Detector({
    distance: st.distance,
    nFast: st.nFast,
    nSlow: st.nSlow,
    pixelSize: st.pixelSize,
    nu: st.angles.gamma,
    delta: st.angles.delta,
  }).binned(bin);
}

function rebuildReflections() {
  st.B = P.bMatrix(...st.cell);
  const qmax = detector(1).maxQmax(st.wavelength);
  st.hkl = P.hklWithinQmax(st.B, qmax);
  st.F2 = st.atoms
    ? P.structureFactors(tables, st.atoms, st.B, st.hkl)
    : P.latticeStructureFactors(st.B, st.hkl);
  needRebuild = false;
  return qmax;
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
  if (needRebuild || !st.hkl) rebuildReflections();

  const det = detector();
  const { mu, eta, chi, phi } = st.angles;
  const ZU = P.matMul(P.sampleMatrix(mu, eta, chi, phi), st.U);
  let refl = P.excite({
    Qcryst: P.qCryst(st.B, st.hkl),
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

  paint(detCanvas, image, det.nFast, det.nSlow, luts[st.cmap], { log: st.log });
  fitDetectorCanvas(det);
  scene.touchDetectorImage();
  st.lastDet = det;
  st.lastTable = table;
  drawDetectorOverlay(det, table);

  const missLen = 1.55 * Math.max(st.distance, 60);
  const rays = rayGeometry(det, refl, table, missLen);
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

// ---------------------------------------------------------------- readouts

function updateReadouts(det, table, nNear, nOn, nBlocked, alpha) {
  const qmax = detector(1).maxQmax(st.wavelength);
  $("detInfo").textContent =
    `${det.nFast}×${det.nSlow} px (${(det.pixelSize * 1000).toFixed(0)} µm bins)   ` +
    `${st.hkl.length / 3} hkl in range   d_min ${((2 * Math.PI) / qmax).toFixed(3)} Å   ` +
    `${nNear} near the sphere   ${table.length} on the detector` +
    (nBlocked ? `   ${nBlocked} into the sample` : "");

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
    (st.atoms
      ? `\n${st.atoms.length} atoms, real |F|²`
      : "\nlattice only, no structure");

  refreshUB();
}

function refreshUB() {
  const ub = P.UB(st.U, st.B, $("ubConv").value, st.wavelength);
  $("ub").textContent = ub
    .map((r) => r.map((v) => (v >= 0 ? "+" : "") + v.toFixed(6)).join("  "))
    .join("\n");
}

function drawDetectorOverlay(det, table) {
  const box = $("detOverlay");
  const r = detCanvas.getBoundingClientRect();
  const p = box.getBoundingClientRect();
  const sx = r.width / det.nFast,
    sy = r.height / det.nSlow;
  const ox = r.left - p.left,
    oy = r.top - p.top;

  const bcx = ox + det.beamCenterFast * sx;
  const bcy = oy + (det.nSlow - 1 - det.beamCenterSlow) * sy;
  let svg =
    `<svg width="100%" height="100%" style="position:absolute;inset:0">` +
    `<line x1="${bcx}" y1="${oy}" x2="${bcx}" y2="${oy + r.height}" stroke="#78c8ff66" stroke-dasharray="4 4"/>` +
    `<line x1="${ox}" y1="${bcy}" x2="${ox + r.width}" y2="${bcy}" stroke="#78c8ff66" stroke-dasharray="4 4"/>`;
  if (st.show.labels) {
    for (const t of [...table]
      .sort((x, y) => y.intensity - x.intensity)
      .slice(0, 22)) {
      const x = ox + t.fast * sx,
        y = oy + (det.nSlow - 1 - t.slow) * sy;
      svg +=
        `<circle cx="${x}" cy="${y}" r="5.5" fill="none" stroke="#8cf0d2bb"/>` +
        `<text x="${x + 7}" y="${y - 5}" fill="#8cf0d2dd" font-size="10"` +
        ` font-family="Consolas,monospace">${t.h} ${t.k} ${t.l}</text>`;
    }
  }
  box.innerHTML = svg + "</svg>";
}

// ---------------------------------------------------------------- controls

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
    const set = (v, silent) => {
      v = Math.max(lo, Math.min(hi, v));
      range.value = v;
      num.value = Number(v.toFixed(2));
      st.angles[name] = v;
      if (!silent) requestSim();
    };
    range.addEventListener("input", () => {
      stopAnim();
      set(parseFloat(range.value));
    });
    num.addEventListener("input", () => {
      stopAnim();
      set(parseFloat(num.value) || 0);
    });
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
    const row = document.createElement("div");
    row.className = "motor";
    row.innerHTML =
      `<span class="name" style="color:${colour}">${name}</span>` +
      `<input type="range" min="-180" max="180" step="0.1" value="0">` +
      `<input type="number" min="-180" max="180" step="1" value="0">`;
    const [range, num] = [row.children[1], row.children[2]];
    const set = (v, silent) => {
      range.value = v;
      num.value = Number(v.toFixed(1));
      st.rot[name] = v;
      st.U = P.matMul(P.eulerMatrix(st.rot.rx, st.rot.ry, st.rot.rz), st.Ubase);
      if (!silent) requestSim();
    };
    range.addEventListener("input", () => set(parseFloat(range.value)));
    num.addEventListener("input", () => set(parseFloat(num.value) || 0));
    rotRows[name] = { set };
    host.appendChild(row);
  }
}

function rebaseOrientation() {
  st.Ubase = st.U.map((r) => r.slice());
  for (const k of ["rx", "ry", "rz"]) rotRows[k].set(0, true);
  st.rot = { rx: 0, ry: 0, rz: 0 };
}

// -- continuous rotation and animated moves

let animation = null;
function stopAnim() {
  animation = null;
}

function animateTo(targets, steps = 26) {
  const start = {};
  for (const k of Object.keys(targets)) start[k] = st.angles[k];
  animation = { start, targets, i: 0, steps };
}

function tick() {
  requestAnimationFrame(tick);
  let changed = false;

  if (spinning.size) {
    const step = parseFloat($("speed").value) || 0;
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
    animation.i++;
    let t = animation.i / animation.steps;
    t = t * t * (3 - 2 * t);
    for (const [k, target] of Object.entries(animation.targets)) {
      motorRows[k].set(
        animation.start[k] + (target - animation.start[k]) * t,
        true,
      );
    }
    changed = true;
    if (animation.i >= animation.steps) animation = null;
  }
  if (changed) simulate();
  if (scene.dirty) scene.render();
}

// ---------------------------------------------------------------- wiring

function bindInputs() {
  const num = (id, key, after) => {
    const el = $(id);
    el.value = st[key];
    el.addEventListener("input", () => {
      const v = parseFloat(el.value);
      if (!Number.isFinite(v)) return;
      st[key] = v;
      if (after) after();
      requestSim();
    });
  };

  num("wl", "wavelength", () => {
    $("energy").value = (HC / st.wavelength).toFixed(3);
    needRebuild = true;
  });
  // both boxes carry the same rounding they get from each other's handler, so
  // the exact default (a/9) shows as a wavelength and not as a raw float
  $("wl").value = st.wavelength.toFixed(4);
  $("energy").value = (HC / st.wavelength).toFixed(3);
  $("energy").addEventListener("input", () => {
    const e = parseFloat($("energy").value);
    if (!Number.isFinite(e) || e <= 0) return;
    st.wavelength = HC / e;
    $("wl").value = st.wavelength.toFixed(4);
    needRebuild = true;
    requestSim();
  });
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
  dp.value = 3;
  dp.addEventListener("change", () => {
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

  const sel = $("structure");
  CELLS.forEach(([n, ...cell], i) => {
    const o = document.createElement("option");
    o.value = `cell:${i}`;
    o.textContent = `${n}  (a=${cell[0]} Å)`;
    sel.appendChild(o);
  });
  for (const s of structures) {
    const o = document.createElement("option");
    o.value = `struct:${s.name}`;
    o.textContent = `${s.name}  (${s.atoms.length} atoms, real |F|²)`;
    sel.appendChild(o);
  }
  sel.addEventListener("change", () => {
    const [kind, key] = sel.value.split(":");
    if (kind === "cell") {
      st.cell = CELLS[+key].slice(1);
      st.atoms = null;
    } else {
      const s = structures.find((x) => x.name === key);
      st.cell = [
        s.cell.a,
        s.cell.b,
        s.cell.c,
        s.cell.alpha,
        s.cell.beta,
        s.cell.gamma,
      ];
      st.atoms = s.atoms;
    }
    needRebuild = true;
    requestSim();
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
    const U = P.alignInLab(st.U, st.B, st.surfaceHkl, [1, 0, 0], {
      frame: "phi",
    });
    if (U) {
      st.U = U;
      rebaseOrientation();
      requestSim();
    }
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
  $("ubCopy").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText($("ub").textContent);
    } catch {
      /* denied */
    }
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
    const a = P.aimDetectorAt(st.B, st.U, driveHkl(), st.angles, st.wavelength);
    animateTo({ delta: a.delta, gamma: a.gamma });
  });
  $("moveChi").addEventListener("click", () => {
    if (chiTarget === null) return;
    animateTo({ chi: chiTarget });
    setTimeout(findOmega, 700);
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

function findOmega() {
  const hkl = driveHkl();
  const sel = $("solutions");
  sel.innerHTML = "";
  $("moveChi").disabled = true;
  chiTarget = null;

  const r = P.etaReach(st.B, st.U, hkl, st.angles, st.wavelength);
  if (!r.inLimitingSphere) {
    sel.innerHTML = "<option>outside the limiting sphere</option>";
    $("reachMsg").textContent =
      `|Q| = ${r.Q.toFixed(3)} > 2k = ${((4 * Math.PI) / st.wavelength).toFixed(3)} Å⁻¹. ` +
      "No geometry reaches this reflection at this wavelength; you need a shorter one.";
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
  $("reachMsg").textContent =
    `|Q| = ${r.Q.toFixed(4)} Å⁻¹,  d = ${((2 * Math.PI) / r.Q).toFixed(4)} Å,  ` +
    `${sols.length} omega solution(s).`;
  for (const e of sols) {
    const a = P.aimDetectorAt(
      st.B,
      st.U,
      hkl,
      { ...st.angles, eta: e },
      st.wavelength,
    );
    const o = document.createElement("option");
    o.textContent = `omega = ${e.toFixed(3)}  →  delta ${a.delta.toFixed(2)}  gamma ${a.gamma.toFixed(2)}`;
    o.dataset.eta = e;
    o.dataset.delta = a.delta;
    o.dataset.gamma = a.gamma;
    sel.appendChild(o);
  }
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
  $("detPane").addEventListener("pointermove", (ev) => {
    const det = detector();
    const r = detCanvas.getBoundingClientRect();
    const fx = ((ev.clientX - r.left) / r.width) * det.nFast;
    const fy = ((ev.clientY - r.top) / r.height) * det.nSlow;
    if (fx < 0 || fy < 0 || fx >= det.nFast || fy >= det.nSlow) return;
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
    $("detInfo").textContent =
      `px (${fx.toFixed(1)}, ${fy.toFixed(1)})   2θ ${tt.toFixed(2)}°   ` +
      `|Q| ${q.toFixed(3)} Å⁻¹   d ${((2 * Math.PI) / q).toFixed(3)} Å`;
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
  structures = await Promise.all(
    ["cspbbr3"].map((n) => grab(`data/${n}.json`)),
  );

  detCanvas = $("detCanvas");
  detCtx = detCanvas.getContext("2d");
  scene = new InstrumentScene($("view3d"));
  scene.setDetectorImage(detCanvas);

  buildMotorRows();
  buildRotRows();
  bindInputs();
  bindDetectorZoom();
  bindDetectorHover();

  new ResizeObserver(() => {
    scene.resize();
    requestSim();
  }).observe($("stage"));
  window.addEventListener("resize", () => requestSim());

  $("boot").remove();
  simulate();
  tick();

  // Scripting handle. Deliberate public API, not a debug leftover: it lets the
  // test harness drive the page without waiting on animation frames, and lets
  // anyone script the instrument from the console.
  window.diffractionGame = {
    state: st,
    simulate,
    scene,
    physics: P,
    setAngles(a) {
      Object.assign(st.angles, a);
      for (const k of Object.keys(a)) motorRows[k]?.set(a[k], true);
      simulate();
    },
    render() {
      scene.render();
    },
  };
}

boot().catch((err) => {
  const b = $("boot");
  b.className = "err";
  b.textContent =
    `Failed to start.\n\n${err.message}\n\n` +
    "If you opened this file directly, serve it over http instead " +
    "(ES modules and fetch do not work from file://).";
  console.error(err);
});
