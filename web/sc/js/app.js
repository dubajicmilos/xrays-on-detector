/*! Single-Crystal Diffraction · © 2026 Miloš Dubajić · MIT · https://github.com/dubajicmilos/xrays-on-detector */

/**
 * Single-Crystal Diffraction: the browser build.
 *
 * The model is single_crystal/ ported to JavaScript; tests/parity_sc.mjs
 * checks the numbers against fixtures exported from the validated Python.
 * The CIF reader and the reciprocal lattice are shared outright with the
 * Game of Diffraction next door (../js/cif.js, ../js/physics.js).
 */

import { CifError, parseCif, setElements } from "../../js/cif.js";
import { mountCredit, logCredit } from "../../js/credit.js";
import { collapsibleSections } from "../../js/panel.js";
import {
  electronWavelength,
  RADIATION_LABEL,
  setTables,
  UNITS,
} from "../../js/scatter.js";
import { formatHkl } from "./display.js";
import { describeComparison, pairUp, rotateInPlane, bestTwist } from "./overlay.js";
import { computePowder, peakFamilies } from "./powder.js";
import { PatternView } from "./render.js";
import { computeSection, zoneLaw } from "./section.js";
import { Structure } from "./structure.js";
import { computeTem, dMinForZone } from "./tem.js";

const WORK = Object.freeze({
  title: "Single-Crystal Diffraction",
  home: "https://dubajicmilos.github.io/diffraction/single-crystal/",
});

const $ = (id) => document.getElementById(id);

/** A sensible 2-theta limit per radiation. Electrons need a much smaller one:
 *  at 0.025 Å even a few degrees reaches further into reciprocal space than
 *  120° of Cu K-alpha does. */
const TT_MAX = { xray: 120, neutron: 120, electron: 4 };

const st = {
  mode: "section",
  radiation: "xray",
  uvw: [0, 0, 1],
  layer: 0,
  dMin: 0.8,
  wavelength: 1.5406,
  kv: 200,
  thickness: 50,
  zones: 0,
  ttMax: 120,
  gain: 1,
  // Linear by default. A log stretch spreads six decades and lights up every
  // extinct-but-not-quite reflection, which reads as noise until you know to
  // expect it; linear shows the pattern a diffractometer would.
  log: false,
  labels: true,
  rings: false,
  labelThr: 0.35,
  spotSize: 9,
  colormap: "inferno",
  // The pattern's rotation about the beam, degrees anticlockwise, and the step
  // the rotate buttons turn it by. A section or a SAED pattern only: a powder
  // trace has no orientation to turn.
  rotation: 0,
  rotStep: 15,
  // Two-crystal comparison: a second structure, the angle its pattern is
  // rotated by about the zone axis, and whether coincidences are ringed.
  twist: 0,
  showMatch: true,
  secondIndex: -1,
  matchTol: 0.03,
  compareMode: "overlay",
  // Powder: 0 keeps the relative d-spacing merge, a positive value merges on a
  // 2theta window (degrees) instead.
  mergeTol: 0,
  // Powder peak width, FWHM in degrees 2theta; computePowder's own default.
  fwhm: 0.15,
};

let luts = null;
let structures = [];
let structure = null;
let result = null;
let view = null;
let second = null;
let result2 = null;
/** The powder peak under the cursor, {set, i}, so the plot can pick it out. */
let hoverPeak = null;
/** The overlay boxes the last full draw fitted around, for hover redraws. */
let lastAvoid = [];

// --------------------------------------------------------------- compute

/**
 * The wavelength in force, in Angstrom: the beam's for SAED, the voltage's
 * for electrons, the wavelength box's otherwise. A section is drawn without
 * one, so for a section this is the wavelength its 2θ readout is quoted at.
 */
function wavelengthNow() {
  return st.mode === "saed" || st.radiation === "electron"
    ? electronWavelength(st.kv)
    : st.wavelength;
}

/**
 * The current pattern of one structure. Both crystals go through here, so
 * the second is always computed with exactly the settings of the first.
 */
function computeFor(s) {
  if (st.mode === "section")
    return computeSection(s, {
      uvw: st.uvw,
      layer: st.layer,
      dMin: st.dMin,
      radiation: st.radiation,
    });
  if (st.mode === "saed")
    return computeTem(s, {
      uvw: st.uvw,
      kv: st.kv,
      thickness: st.thickness,
      dMin: st.dMin,
      maxZone: st.zones,
    });
  return computePowder(s, {
    wavelength: wavelengthNow(),
    radiation: st.radiation,
    twoThetaMax: st.ttMax,
    twoThetaTol: st.mergeTol,
    fwhm: st.fwhm,
  });
}

function recompute() {
  if (!structure) return;
  const status = $("status");
  status.className = "";
  try {
    result = computeFor(structure);
    if (st.mode === "section") {
      $("zoneLaw").textContent = zoneLaw(result.uvw, result.layer);
    } else if (st.mode === "saed") {
      $("zoneLaw").textContent = "";
      $("zoneRings").textContent = result.zoneRadii
        .slice(1)
        .map((r) => (r ? r.toFixed(1) : "—"))
        .join(" ");
    }
  } catch (err) {
    result = null;
    lastError = err.message;
    view.message(err.message, "#f0725a");
    status.className = "bad";
    status.textContent = err.message;
    // nothing on the canvas belongs to the previous result any more
    $("title").textContent = "";
    $("hover").textContent = "";
    return;
  }
  lastError = null;
  recomputeSecond();
  draw();
}

/**
 * The second crystal's pattern, when one is loaded.
 *
 * It is computed with every setting of the first (zone axis, layer, d min,
 * radiation, wavelength, voltage, thickness, 2θ range, peak width), so the
 * two patterns are directly comparable: any difference on screen is the
 * structure, not the settings.
 */
function recomputeSecond() {
  hoverPeak = null;
  secondError = null;
  if (!second) {
    result2 = null;
    return;
  }
  try {
    result2 = computeFor(second);
  } catch (err) {
    result2 = null;
    // shown by draw(), after the status line it would otherwise be lost under
    secondError = err.message;
  }
}

/** Why the second crystal has no pattern, when it has none. */
let secondError = null;

/** The message on the canvas when there is no result, redrawn on resize. */
let lastError = null;

/**
 * The page's overlays over the pattern, as [x, y, w, h] in CSS pixels from
 * the canvas corner, for the view to fit the pattern around: the title card
 * and the zoom control where they are, and the status line's whole strip,
 * since its text changes after the pattern is drawn.
 */
function overlayBoxes() {
  const c = $("pattern").getBoundingClientRect();
  const boxes = [[0, c.height - 30, c.width, 30]];
  for (const id of ["title", "patZoom"]) {
    const r = $(id).getBoundingClientRect();
    if (r.width && r.height)
      boxes.push([r.left - c.left, r.top - c.top, r.width, r.height]);
  }
  return boxes;
}

function draw() {
  if (!result) return;
  const lut = luts[st.colormap];
  // The title first: the pattern is fitted around the card it fills.
  writeOverlay();
  if (secondError) {
    $("status").className = "bad";
    $("status").textContent = `second crystal: ${secondError}`;
  }
  const avoid = overlayBoxes();
  lastAvoid = avoid;
  if (st.mode === "powder") {
    drawPowder();
  } else if (second && result2) {
    // Two crystals in one frame. The second is drawn rotated by st.twist about
    // the zone axis, which for a section or a zone-axis pattern is exactly a
    // rotation of the picture; st.rotation turns the two together.
    view.drawOverlay(result, result2, {
      twist: st.twist,
      rotation: st.rotation,
      gain: st.gain,
      log: st.log,
      lut,
      spotScale: st.spotSize,
      labels: st.labels,
      labelThreshold: st.labelThr,
      showMatch: st.showMatch,
      showRings: st.rings && st.mode === "saed",
      tol: st.matchTol,
      mode: st.compareMode,
      structure,
      nameA: structure.name,
      nameB: second.name,
      avoid,
    });
  } else {
    view.drawSpots(result, structure, {
      gain: st.gain,
      log: st.log,
      lut,
      spotScale: st.spotSize,
      labels: st.labels,
      labelThreshold: st.labelThr,
      showRings: st.rings && st.mode === "saed",
      rotation: st.rotation,
      avoid,
    });
  }
  writeMatchInfo();
}

/**
 * The powder plot alone, with the second crystal when there is one and the
 * hovered peak picked out. A hover redraws only this: the title and the
 * fit around it have not changed.
 */
function drawPowder() {
  const two = !!(second && result2);
  view.drawPowder(result, {
    labels: st.labels,
    labelThreshold: Math.max(st.labelThr * 100, 1),
    avoid: lastAvoid,
    second: two ? result2 : null,
    nameA: structure.name,
    nameB: two ? second.name : "",
    hover: hoverPeak,
  });
}

/**
 * Fill the title card: the structure's name, a tag for the radiation, and
 * the lines under them.
 *
 * A bundled or uploaded name is usually a formula with a variant after an
 * underscore ("MAPbI3_Pnma_pseudocubic", "BA2PbI4_100K"), so a leading
 * formula gets its digits as subscripts and the variant follows in a lighter
 * weight; a name that does not start like a formula is shown as it is.
 * Built node by node, never as HTML: an uploaded file names itself.
 */
function setTitle(name, radiation, lines) {
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const [head, ...rest] = String(name).split("_");
  const top = el("div", "hud-name");
  const formula = el("span", "formula");
  if (/^[A-Z][A-Za-z0-9]*$/.test(head) && /\d/.test(head)) {
    for (const part of head.split(/(\d+)/))
      if (part) formula.append(/^\d/.test(part) ? el("sub", "", part) : part);
  } else formula.textContent = head;
  top.append(formula);
  if (rest.length) top.append(el("span", "variant", rest.join(" ")));
  top.append(el("span", `chip ${radiation}`, RADIATION_LABEL[radiation]));
  $("title").replaceChildren(top, ...lines.map((l) => el("div", "hud-meta", l)));
}

function writeOverlay() {
  const status = $("status");
  const unit = UNITS[result.radiation || st.radiation];
  const radiation = result.radiation || st.radiation;

  if (st.mode === "powder") {
    setTitle(structure.name, radiation, [
      `λ ${result.wavelength.toFixed(5)} Å  ·  ${result.count} peaks`,
    ]);
    status.textContent =
      `Lorentz ${st.radiation === "xray" ? "and polarisation " : ""}applied` +
      `   ·   intensities relative to the strongest peak` +
      (st.radiation === "xray" ? "" : "   ·   no polarisation term") +
      "   ·   hover a peak for its reflections";
    return;
  }

  const zone = result.uvw.join(" ");
  if (st.mode === "section") {
    setTitle(structure.name, radiation, [
      `zone [${zone}]  ·  layer ${result.layer}  ·  (${zoneLaw(result.uvw, result.layer)})`,
      `${result.count} reflections  ·  d ≥ ${st.dMin.toFixed(2)} Å  ·  |F|² in ${unit}²`,
    ]);
    const bits = [];
    if (result.zoneFactor > 1)
      bits.push(`zone axis reduced by ${result.zoneFactor} to [${zone}]`);
    if (result.layer !== 0)
      bits.push(
        `layer offset ${result.height.toFixed(3)} Å⁻¹ along the zone axis`,
      );
    status.textContent =
      bits.join("   ·   ") ||
      "hover a spot for its indices and 2θ   ·   wheel to zoom, drag to pan, " +
        "double-click to fit";
  } else {
    setTitle(structure.name, "electron", [
      `beam ∥ [${zone}]  ·  ${result.kv} kV  ·  λ ${result.wavelength.toFixed(5)} Å`,
      `${result.count} reflections  ·  thickness ${result.thickness} Å`,
    ]);
    const first = result.zoneRadii[1];
    if (st.zones > 0 && first && first > result.qMax) {
      status.className = "warn";
      status.textContent =
        `the first Laue ring is at ${first.toFixed(1)} Å⁻¹, outside d min: ` +
        `set d min to ${((2 * Math.PI) / first).toFixed(3)} Å to reach it`;
    } else {
      status.textContent =
        `kinematic, sinc² relrod` +
        `   ·   ${result.zonesVisible} of ${result.zoneRadii.length} Laue zones in range` +
        `   ·   dynamical scattering is not modelled`;
      // The outer zones are orders of magnitude weaker than the zero layer,
      // so on a linear stretch they sit below the drawing threshold and the
      // rings asked for never appear. Say so rather than show a plain section.
      if (st.zones > 0 && result.zonesVisible > 1 && !st.log && holzHidden()) {
        status.className = "warn";
        status.textContent +=
          "   ·   the outer zones are too faint for a linear stretch: tick " +
          "log intensity or raise the contrast";
      }
    }
  }
}

/**
 * The comparison readout: how many reflections the two crystals share, how
 * many are unique to each, and how far apart the shared ones sit.
 *
 * The pairing repeats the one the renderer did, which is a linear pass over a
 * few hundred points and far cheaper than plumbing the result back out.
 */
function writeMatchInfo() {
  const info = $("matchInfo");
  if (!info) return;
  // The pairing is by position in the plane, which a powder trace does not have.
  if (!second || !result2 || st.mode === "powder") {
    info.textContent = "";
    return;
  }
  const rot = rotateInPlane(result2, st.twist);
  const ours = pairUp(result, { count: result2.count, x: rot.x, y: rot.y }, { tol: st.matchTol });
  info.textContent = describeComparison(
    result,
    result2,
    ours,
    structure.name,
    second.name,
  );
}

/** True when no reflection outside the zero layer reaches the draw threshold. */
function holzHidden() {
  let top = 0;
  for (let i = 0; i < result.count; i++)
    top = Math.max(top, result.intensity[i]);
  if (!(top > 0)) return true;
  for (let i = 0; i < result.count; i++)
    if (result.laueZone[i] > 0 && (result.intensity[i] * st.gain) / top > 1e-3)
      return false;
  return true;
}

// ------------------------------------------------------------ structures

function applyStructure(doc) {
  // A different structure is a different picture, so the magnification from
  // the last one would leave you looking at empty space. Changing the zone or
  // the layer keeps it, which is what you want while comparing sections.
  if (view) {
    view.resetView();
    showZoom();
  }
  try {
    structure = new Structure(doc);
  } catch (err) {
    $("status").className = "bad";
    $("status").textContent = err.message;
    return;
  }
  const d = structure.describe();
  $("structInfo").textContent =
    `${doc.name}${doc.spaceGroup ? "   " + doc.spaceGroup : ""}\n` +
    `${d.cell}\n${d.angles}\n${d.volume}   ${d.atoms}`;
  const notes = [];
  if (doc.blocksInFile > 1)
    notes.push(`${doc.blocksInFile} structures in the file; using the first`);
  if (doc.skippedSites)
    notes.push(
      `${doc.skippedSites} atom ${doc.skippedSites === 1 ? "site" : "sites"} ` +
        "without a readable position skipped",
    );
  const species = [...new Set(structure.nuclides)];
  if (species.includes("D") || species.includes("T"))
    notes.push("D and T keep their own neutron scattering lengths");
  $("cifNote").textContent = notes.join(" · ");
  recompute();
}

async function onFile(file) {
  try {
    const doc = parseCif(await file.text(), file.name.replace(/\.cif$/i, ""));
    // A structure joins the list only once it is known to build; a dead
    // entry that fails again every time it is picked helps nobody.
    new Structure(doc);
    structures.push(doc);
    const opt = document.createElement("option");
    opt.value = String(structures.length - 1);
    opt.textContent = `${doc.name} (uploaded)`;
    $("structure").appendChild(opt);
    $("structure").value = opt.value;
    applyStructure(doc);
  } catch (err) {
    const msg =
      err instanceof CifError
        ? err.message
        : `could not read this file: ${err.message}`;
    $("status").className = "bad";
    $("status").textContent = msg;
  }
}

// ------------------------------------------------------------------ second crystal

/**
 * Load the comparison crystal.
 *
 * It joins the same `structures` list as the primary, so either select can
 * pick any structure and the two lists cannot drift apart.
 */
async function onSecondFile(file) {
  try {
    const doc = parseCif(await file.text(), file.name.replace(/\.cif$/i, ""));
    new Structure(doc);
    structures.push(doc);
    const idx = String(structures.length - 1);
    for (const id of ["structure", "structure2"]) {
      const opt = document.createElement("option");
      opt.value = idx;
      opt.textContent = `${doc.name} (uploaded)`;
      $(id).appendChild(opt);
    }
    $("structure2").value = idx;
    st.secondIndex = structures.length - 1;
    second = new Structure(doc);
    openCompare();
    syncRows();
    recompute();
  } catch (err) {
    const msg =
      err instanceof CifError
        ? err.message
        : `could not read this file: ${err.message}`;
    $("status").className = "bad";
    $("status").textContent = msg;
  }
}

/** Unfold the comparison block, if it was folded, once a second crystal is in. */
function openCompare() {
  const grp = $("grpCompare");
  if (grp.classList.contains("collapsed"))
    grp.querySelector(".grp-toggle")?.click();
}

// ------------------------------------------------------------------ export

function stem() {
  const base = structure ? structure.name.replace(/[^\w.-]+/g, "_") : "pattern";
  if (st.mode === "powder") return `${base}_powder_${st.radiation}`;
  // the reduced zone, with separators: [1 -1 0] must not read as 1-10
  const z = `[${(result ? result.uvw : st.uvw).join(",")}]`;
  if (st.mode === "saed") return `${base}_saed_${z}`;
  return `${base}_zone${z}_n${st.layer}_${st.radiation}`;
}

function download(name, blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * The coincidence table: which reflections the two crystals share, how far
 * apart they sit, and which belong to one crystal only.
 *
 * Intensities are given relative to the strongest reflection of each crystal,
 * so the two columns are comparable across structures whose absolute scale
 * differs (a neutron and an X-ray pattern, say).
 */
function saveMatches() {
  if (!result || !second || !result2 || st.mode === "powder") return;
  const rot = rotateInPlane(result2, st.twist);
  const { pairs, onlyA, onlyB } = pairUp(
    result,
    { count: result2.count, x: rot.x, y: rot.y },
    { tol: st.matchTol },
  );
  const peak = (arr) => {
    let m = 0;
    for (const v of arr) if (v > m) m = v;
    return m || 1;
  };
  const maxA = peak(result.intensity);
  const maxB = peak(result2.intensity);

  const L = [
    `# ${structure.name}  vs  ${second.name}`,
    st.mode === "saed"
      ? `# SAED, beam || [${result.uvw.join(" ")}]   ${result.kv} kV   ` +
        `thickness ${result.thickness} A   d min ${st.dMin} A`
      : `# zone [${st.uvw.join(" ")}]   layer ${st.layer}   d min ${st.dMin} A`,
    `# twist ${st.twist.toFixed(2)} deg   match tolerance ${st.matchTol} 1/A`,
    `# ${pairs.length} coincident, ${onlyA.length} only in ${structure.name}, ` +
      `${onlyB.length} only in ${second.name}`,
    "#",
    "#    h    k    l     |     h    k    l      dQ(1/A)    I(0-100) A    I(0-100) B",
  ];
  const hklOf = (res, i) => [
    res.hkl[3 * i],
    res.hkl[3 * i + 1],
    res.hkl[3 * i + 2],
  ];
  const fmt = (h) => h.map((v) => String(v).padStart(4)).join(" ");
  for (const p of pairs)
    L.push(
      `${fmt(hklOf(result, p.a))}     |  ${fmt(hklOf(result2, p.b))}` +
        `      ${p.dq.toFixed(5)}` +
        `      ${((100 * result.intensity[p.a]) / maxA).toFixed(2)}` +
        `      ${((100 * result2.intensity[p.b]) / maxB).toFixed(2)}`,
    );
  for (const i of onlyA) L.push(`${fmt(hklOf(result, i))}     |`);
  for (const i of onlyB) L.push(`              |  ${fmt(hklOf(result2, i))}`);
  download(stem() + "_coincidences.txt", new Blob([L.join("\n") + "\n"], { type: "text/plain" }));
}

function saveTable() {
  if (!result) return;
  const unit = UNITS[result.radiation || st.radiation];
  const L = [
    `# ${structure.name}`,
    `# radiation      ${RADIATION_LABEL[result.radiation || st.radiation]}`,
    `# |F|^2 unit     ${unit}^2`,
  ];
  if (st.mode === "powder") {
    L.push(`# wavelength     ${result.wavelength.toFixed(6)} A`);
    L.push("#", "#   2theta        d      I(rel)  mult   h   k   l");
    for (let i = 0; i < result.count; i++) {
      const [h, k, l] = result.hkl[i];
      L.push(
        `${result.twoTheta[i].toFixed(4).padStart(10)} ${result.d[i]
          .toFixed(5)
          .padStart(9)} ${result.intensity[i].toFixed(3).padStart(9)} ` +
          `${String(result.multiplicity[i]).padStart(5)} ` +
          `${String(h).padStart(4)}${String(k).padStart(4)}${String(l).padStart(4)}`,
      );
    }
  } else {
    L.push(`# zone axis      ${result.uvw.join(" ")}`);
    if (st.mode === "section")
      L.push(
        `# layer          ${result.layer}   (${zoneLaw(result.uvw, result.layer)})`,
      );
    else {
      L.push(
        `# voltage        ${result.kv} kV, lambda ${result.wavelength.toFixed(6)} A`,
      );
      L.push(`# thickness      ${result.thickness} A`);
    }
    L.push(
      "#",
      "#   h   k   l        d        |Q|          I" +
        (st.mode === "saed" ? "          s_g  zone" : ""),
    );
    const order = [...Array(result.count).keys()].sort(
      (a, b) => result.intensity[b] - result.intensity[a],
    );
    for (const i of order) {
      let line =
        `${String(result.hkl[3 * i]).padStart(5)}` +
        `${String(result.hkl[3 * i + 1]).padStart(4)}` +
        `${String(result.hkl[3 * i + 2]).padStart(4)} ` +
        `${result.d[i].toFixed(5).padStart(9)} ${result.q[i].toFixed(5).padStart(10)} ` +
        `${result.intensity[i].toExponential(4).padStart(12)}`;
      if (st.mode === "saed")
        line += ` ${result.sg[i].toFixed(6).padStart(12)} ${String(
          result.laueZone[i],
        ).padStart(5)}`;
      L.push(line);
    }
  }
  download(
    stem() + ".txt",
    new Blob([L.join("\n") + "\n"], { type: "text/plain" }),
  );
}

// ---------------------------------------------------------- hover readouts

const isElectron = () => st.mode === "saed" || st.radiation === "electron";

/**
 * A scattering angle, in the units its radiation is read in: degrees for
 * x-rays and neutrons, and for electrons, whose angles are a few degrees at
 * most, milliradians as well.
 */
function angleText(twoThetaDeg) {
  if (!isElectron()) return `2θ ${twoThetaDeg.toFixed(3)}°`;
  const mrad = (twoThetaDeg * Math.PI * 1000) / 180;
  return `2θ ${twoThetaDeg.toFixed(4)}° = ${mrad.toFixed(2)} mrad`;
}

/**
 * Where a reflection of spacing d diffracts, by Bragg's law, at the
 * wavelength in force. A section has no wavelength of its own, so the line
 * says which one the angle is for; SAED's is in the title already.
 */
function twoThetaLine(d) {
  const lam = wavelengthNow();
  const at =
    st.mode === "saed"
      ? ""
      : isElectron()
        ? `   at ${st.kv} kV`
        : `   at λ ${+lam.toFixed(5)} Å`;
  const s = lam / (2 * d);
  if (!(s <= 1)) return `no 2θ: d is under λ/2 = ${(lam / 2).toFixed(4)} Å${at}`;
  return angleText((2 * Math.asin(s) * 180) / Math.PI) + at;
}

/** A modulus |F|, with the decimals its size calls for. */
function fmtF(v) {
  if (v >= 100) return v.toFixed(1);
  if (v >= 1) return v.toFixed(2);
  return v.toPrecision(3);
}

/** A share of the strongest reflection, as a percentage. */
function pct(x) {
  const v = 100 * x;
  return v >= 0.01 ? `${v.toFixed(2)}%` : `${v.toExponential(1)}%`;
}

/** Pad an hkl string to a width, not counting its combining overbars. */
const padHkl = (text, n) =>
  text + " ".repeat(Math.max(0, n - text.replace(/̅/g, "").length));

/**
 * The readout for one spot of a section or SAED pattern: its indices, d, |Q|
 * and 2θ, its intensity against the strongest reflection, and, in the
 * overlay, which crystal it belongs to and the reflection of the other one
 * it coincides with.
 */
function spotText({ set, i }) {
  const res = set ? result2 : result;
  const own = set ? second : structure;
  if (!res || !own || i >= res.count) return "";
  const hkl = [res.hkl[3 * i], res.hkl[3 * i + 1], res.hkl[3 * i + 2]];
  const unit = UNITS[res.radiation || st.radiation];
  let top = 0;
  for (let k = 0; k < res.count; k++)
    if (res.intensity[k] > top) top = res.intensity[k];
  const lines = [];
  if (second && result2) lines.push(own.name);
  lines.push(`${formatHkl(hkl)}   (${hkl.join(" ")})`);
  lines.push(`d ${res.d[i].toFixed(4)} Å   |Q| ${res.q[i].toFixed(4)} Å⁻¹`);
  lines.push(twoThetaLine(res.d[i]));
  lines.push(
    `|F|² ${res.intensity[i].toExponential(3)} ${unit}²` +
      (top > 0 ? `   ${pct(res.intensity[i] / top)} of the strongest` : ""),
  );
  if (st.mode === "saed")
    lines.push(`s_g ${res.sg[i].toFixed(5)} Å⁻¹   Laue zone ${res.laueZone[i]}`);
  const pair = view.pairs && view.pairs[set].get(i);
  if (pair) {
    const other = set ? result : result2;
    const j = set ? pair.a : pair.b;
    const oh = [other.hkl[3 * j], other.hkl[3 * j + 1], other.hkl[3 * j + 2]];
    lines.push(
      `coincides with ${(set ? structure : second).name} ${formatHkl(oh)}` +
        `, Δq ${pair.dq.toFixed(4)} Å⁻¹`,
    );
  }
  return lines.join("\n");
}

/**
 * The readout for one powder peak, with what a reflection list gives for it:
 * indices and multiplicity, 2θ, d and |Q|, the intensity on the plot's 0-100
 * scale and |F|. A peak holding more than one family (two unrelated sets of
 * planes at one d, or several merged by the 2θ window) lists each, with its
 * own multiplicity, |F| and share of the intensity.
 */
function peakText({ set, i }) {
  const p = set ? result2 : result;
  if (!p || i >= p.count) return "";
  const unit = UNITS[p.radiation];
  const all = peakFamilies(p, i);
  // Extinct families scatter nothing; a reflection list built from the space
  // group would not have them, so they are counted on one line at the end
  // rather than listed, and the multiplicity is that of the ones that diffract.
  const fams = all.filter((f) => !f.absent);
  const extinct = all.reduce((s, f) => s + (f.absent ? f.m : 0), 0);
  const lines = [];
  if (second && result2) lines.push(set ? second.name : structure.name);
  if (!fams.length) {
    lines.push(`${all.map((f) => formatHkl(f.hkl)).join(" + ")}   extinct, |F| = 0`);
    return lines.join("\n");
  }
  // Where the peak is: its families' 2θ, weighted by intensity, so an extinct
  // reflection merged into it does not pull the reading off the one that
  // diffracts.
  const sumI = fams.reduce((s, f) => s + f.I, 0);
  const at = (key) =>
    sumI > 0 ? fams.reduce((s, f) => s + f.I * f[key], 0) / sumI : fams[0][key];
  const tt = at("tt");
  const d = at("d");
  const names = fams.map((f) => formatHkl(f.hkl));
  lines.push(
    (names.length > 3
      ? names.slice(0, 3).join(" + ") + " + …"
      : names.join(" + ")) +
      `   m ${fams.reduce((s, f) => s + f.m, 0)}`,
  );
  lines.push(
    `${angleText(tt)}   d ${d.toFixed(4)} Å   |Q| ${((2 * Math.PI) / d).toFixed(4)} Å⁻¹`,
  );
  lines.push(
    `I ${p.intensity[i].toFixed(1)} of 100` +
      (fams.length === 1 ? `   |F| ${fmtF(Math.sqrt(fams[0].F2))} ${unit}` : ""),
  );
  if (fams.length > 1) {
    for (const f of fams.slice(0, 6))
      lines.push(
        `  ${padHkl(formatHkl(f.hkl), 9)} m ${String(f.m).padStart(3)}` +
          `   |F| ${fmtF(Math.sqrt(f.F2))} ${unit}   I ${f.I.toFixed(1)}` +
          (Math.abs(f.tt - tt) > 5e-4 ? `   ${angleText(f.tt)}` : ""),
      );
    if (fams.length > 6) lines.push(`  … and ${fams.length - 6} more`);
  }
  if (extinct)
    lines.push(
      `  + ${extinct} extinct ${extinct === 1 ? "reflection" : "reflections"}` +
        " in this peak, |F| = 0",
    );
  return lines.join("\n");
}

/** Under the cursor on the powder plot, away from any peak: 2θ and its d. */
function cursorText(tt) {
  const s = Math.sin((tt * Math.PI) / 360);
  const d = s > 0 ? `${(result.wavelength / (2 * s)).toFixed(4)} Å` : "∞";
  return `${angleText(tt)}   d ${d}`;
}

const sameHit = (a, b) =>
  a === b || (!!a && !!b && a.set === b.set && a.i === b.i);

/** The last peak readout, so moving within one peak does not rebuild it. */
let tipCache = null;

/**
 * Hover over the powder plot: pick out the peak under the cursor and float
 * its readout beside the cursor. The plot is redrawn, and the readout built,
 * only when the peak changes, not on every pixel the pointer moves: a peak
 * merged by a wide 2θ window can hold tens of thousands of reflections.
 */
function powderHover(e) {
  const hit = view.pickPowder(e.clientX, e.clientY);
  const peak = hit && hit.i !== undefined ? hit : null;
  if (!sameHit(peak, hoverPeak)) {
    hoverPeak = peak;
    drawPowder();
  }
  let text = "";
  if (peak) {
    const res = peak.set ? result2 : result;
    if (!tipCache || tipCache.res !== res || tipCache.i !== peak.i)
      tipCache = { res, i: peak.i, text: peakText(peak) };
    text = tipCache.text;
  } else if (hit) text = cursorText(hit.tt);
  const box = $("hover");
  box.textContent = text;
  if (text) placeTip(box, e);
}

/**
 * Float the readout beside the pointer, inside the plot pane and clear of
 * the title card, which says what the plot is.
 */
function placeTip(box, e) {
  box.classList.add("tip");
  const ov = $("overlay").getBoundingClientRect();
  const pane = $("stage").getBoundingClientRect();
  const card = $("title").getBoundingClientRect();
  const w = box.offsetWidth;
  const h = box.offsetHeight;
  let x = e.clientX + 16;
  let y = e.clientY - h - 12;
  if (x + w > pane.right - 6) x = e.clientX - 16 - w;
  if (x < pane.left + 6) x = pane.left + 6;
  const overCard = () =>
    x < card.right && x + w > card.left && y < card.bottom && y + h > card.top;
  if (y < pane.top + 6 || overCard()) y = e.clientY + 18;
  if (y + h > pane.bottom - 6) y = pane.bottom - 6 - h;
  box.style.left = `${x - ov.left}px`;
  box.style.top = `${y - ov.top}px`;
}

/** Back to the readout's own place, under the title card. */
function dockHover() {
  const box = $("hover");
  if (!box.classList.contains("tip")) return;
  box.classList.remove("tip");
  box.style.left = "";
  box.style.top = "";
}

// ---------------------------------------------------------------- binding

/** Set while a pan is in progress, so hover does not fight the drag. */
let drag = null;

/** The rotation readout on the pattern and the angle box in the panel. */
function showRotation() {
  const a = st.rotation;
  $("rotReset").textContent = `${Number.isInteger(a) ? a : a.toFixed(1)}°`;
  if (document.activeElement !== $("rotation")) $("rotation").value = String(a);
}

/** An angle in degrees, brought into (-180, 180]. */
function wrapAngle(deg) {
  let a = ((deg % 360) + 360) % 360;
  if (a > 180) a -= 360;
  // so a run of 0.1° steps does not read 29.999999999999996
  return Math.round(a * 1e6) / 1e6;
}

/**
 * The rotate buttons on the pattern, and the angle and step under Display.
 * A rotation turns the picture, not the model, so it only redraws.
 */
function bindRotation() {
  const turn = (deg) => {
    st.rotation = wrapAngle(deg);
    showRotation();
    draw();
  };
  $("rotLeft").addEventListener("click", () => turn(st.rotation + st.rotStep));
  $("rotRight").addEventListener("click", () => turn(st.rotation - st.rotStep));
  $("rotReset").addEventListener("click", () => turn(0));
  // Applied as it is typed; the box itself is tidied only when it is left,
  // so typing 200 is not rewritten to -160 under the cursor.
  $("rotation").addEventListener("input", () => {
    const v = parseFloat($("rotation").value);
    if (!Number.isFinite(v)) return;
    st.rotation = wrapAngle(v);
    $("rotReset").textContent = `${
      Number.isInteger(st.rotation) ? st.rotation : st.rotation.toFixed(1)
    }°`;
    draw();
  });
  $("rotation").addEventListener("change", () => {
    $("rotation").value = String(st.rotation);
  });
  $("rotStep").addEventListener("change", () => {
    st.rotStep = +$("rotStep").value;
  });
  showRotation();
}

function showZoom() {
  $("zoomLabel").textContent = `${Math.round(view.zoom * 100)}%`;
  // A powder trace has its own axes; magnifying it would only stretch a plot.
  $("patZoom").classList.toggle("hidden", st.mode === "powder");
}

/**
 * Wheel to magnify about the pointer, drag to pan, and the usual three
 * buttons. The transform lives on the view, so a redraw after a zone change
 * keeps whatever magnification you were looking at.
 */
function bindZoom() {
  const canvas = $("pattern");

  canvas.addEventListener(
    "wheel",
    (e) => {
      if (!result || st.mode === "powder") return;
      e.preventDefault();
      // Normalise the notch: trackpads report pixels, wheels report lines.
      const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1;
      if (
        view.zoomAt(e.clientX, e.clientY, Math.exp((-e.deltaY * unit) / 420))
      ) {
        draw();
        showZoom();
      }
    },
    { passive: false },
  );

  canvas.addEventListener("pointerdown", (e) => {
    if (!result || st.mode === "powder" || e.button !== 0) return;
    drag = { x: e.clientX, y: e.clientY };
    canvas.setPointerCapture(e.pointerId);
    canvas.classList.add("dragging");
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    view.panBy(e.clientX - drag.x, e.clientY - drag.y);
    drag = { x: e.clientX, y: e.clientY };
    draw();
  });
  const endDrag = (e) => {
    if (!drag) return;
    drag = null;
    canvas.classList.remove("dragging");
    if (e.pointerId !== undefined && canvas.hasPointerCapture(e.pointerId))
      canvas.releasePointerCapture(e.pointerId);
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);

  const step = (f) => () => {
    if (view.zoomCentre(f)) {
      draw();
      showZoom();
    }
  };
  $("zoomIn").addEventListener("click", step(1.35));
  $("zoomOut").addEventListener("click", step(1 / 1.35));
  $("zoomFit").addEventListener("click", () => {
    view.resetView();
    draw();
    showZoom();
  });
  // Double-click is the quick way back, as it is in most viewers.
  canvas.addEventListener("dblclick", () => {
    view.resetView();
    draw();
    showZoom();
  });
  showZoom();
}

function syncRows() {
  const show = (id, on) => $(id).classList.toggle("hidden", !on);
  const powder = st.mode === "powder";
  // a readout left over from the last view describes nothing on this one
  $("hover").textContent = "";
  dockHover();
  show("grpZone", !powder);
  show("rowLayer", st.mode === "section");
  // A powder trace sets its own d min from the 2θ limit and draws lines, not
  // spots, so the controls that would do nothing are not offered.
  show("rowDmin", !powder);
  show("rowGain", !powder);
  show("rowSpot", !powder);
  show("chkLog", !powder);
  show("chkRings", st.mode === "saed");
  show("rowColours", !powder);
  // A section is drawn without a wavelength, but its hover quotes 2θ, so it
  // offers the wavelength (or, for electrons, the voltage) that 2θ is at.
  show("rowWavelength", st.mode !== "saed" && st.radiation !== "electron");
  show("rowKv", st.mode === "saed" || st.radiation === "electron");
  show("rowRotate", !powder);
  show("rowThickness", st.mode === "saed");
  show("rowZones", st.mode === "saed");
  show("rowTtMax", st.mode === "powder");
  show("rowFwhm", st.mode === "powder");
  show("rowMerge", st.mode === "powder");
  // Two crystals compare in every view. The twist and the coincidence pairing
  // need positions in a plane, so a powder trace, which is the same at every
  // orientation, gets the two traces and their peak readouts only.
  show("grpCompare", true);
  // The twist, tolerance and readout mean nothing until there is a second
  // crystal, so they wait for one; so does its export.
  show("compareBody", !!second);
  show("compareNone", !second);
  show("compareSpots", !powder);
  show("saveMatches", !powder && !!second);
  $("compareHint").textContent = COMPARE_HINT[st.mode];
  $("radiation").disabled = st.mode === "saed";
  if (view) showZoom();
  $("modeNote").textContent =
    st.mode === "saed"
      ? "SAED is electron diffraction by definition, so the radiation is fixed."
      : st.mode === "section"
        ? "An undistorted plane of the weighted reciprocal lattice, as a " +
          "precession camera records it. The wavelength (for electrons, the " +
          "voltage) only sets the 2θ a hover reports."
        : "Reflections sharing a d-spacing are summed, so multiplicity is counted rather than looked up.";
}

/** What the comparison holds fixed between the two crystals, per view. */
const COMPARE_HINT = {
  section:
    "Both crystals are cut on the same zone axis, layer and d min. The twist " +
    "rotates the second pattern about the zone axis, which is that crystal's " +
    "orientation relative to the first, so it is the one rotation that " +
    "compares two patterns without inventing a picture no crystal can give.",
  saed:
    "Both crystals are seen down the same zone axis, at the same voltage, " +
    "thickness and d min. The twist rotates the second pattern about the " +
    "beam, which is that crystal's orientation relative to the first.",
  powder:
    "Both traces are computed at the same wavelength, 2θ range, peak width " +
    "and merge rule, each scaled to its own strongest peak. The two rows of " +
    "ticks under the axis mark each crystal's reflections; hover a peak or a " +
    "tick to read them.",
};

function bind() {
  // A box whose min is above zero holds a quantity that must stay positive
  // (a wavelength, a voltage, a thickness); a non-positive entry is left in
  // the box while it is typed and not applied.
  const num = (id, key, after) =>
    $(id).addEventListener("input", () => {
      const v = parseFloat($(id).value);
      if (!Number.isFinite(v)) return;
      if (parseFloat($(id).min) > 0 && !(v > 0)) return;
      st[key] = v;
      after?.();
      recompute();
    });

  $("mode").addEventListener("change", () => {
    st.mode = $("mode").value;
    // SAED is electron diffraction whatever the menu says, so switch it
    // rather than report a radiation that is not being used.
    if (st.mode === "saed") {
      st.radiation = "electron";
      $("radiation").value = "electron";
    }
    if (st.mode === "powder") {
      st.ttMax = TT_MAX[st.radiation];
      $("ttmax").value = String(st.ttMax);
    }
    syncRows();
    recompute();
  });

  $("radiation").addEventListener("change", () => {
    st.radiation = $("radiation").value;
    if (st.mode === "powder") {
      st.ttMax = TT_MAX[st.radiation];
      $("ttmax").value = String(st.ttMax);
    }
    syncRows();
    recompute();
  });

  for (const [i, id] of ["zu", "zv", "zw"].entries())
    $(id).addEventListener("input", () => {
      const v = Number($(id).value);
      if (!Number.isInteger(v)) return;
      st.uvw[i] = v;
      recompute();
    });
  // Only the zone buttons: the twist presets are .quickzone buttons too, and
  // binding them here threw on every click (they carry no data-zone).
  for (const b of document.querySelectorAll(".quickzone button[data-zone]"))
    b.addEventListener("click", () => {
      st.uvw = b.dataset.zone.split(",").map(Number);
      for (const [i, id] of ["zu", "zv", "zw"].entries())
        $(id).value = String(st.uvw[i]);
      recompute();
    });

  num("layer", "layer");
  num("dmin", "dMin");
  num("wl", "wavelength");
  num("kv", "kv");
  num("thickness", "thickness");
  num("ttmax", "ttMax");
  num("fwhm", "fwhm");
  num("mergeTol", "mergeTol");

  // The match tolerance only re-pairs the two patterns, so it redraws without
  // a recompute: the sections themselves have not changed.
  $("matchTol").addEventListener("input", () => {
    const v = parseFloat($("matchTol").value);
    if (!Number.isFinite(v) || !(v > 0)) return;
    st.matchTol = v;
    draw();
  });

  // Asking for a Laue ring is asking for the resolution that shows it. The
  // rings sit far outside any comfortable d min, so leaving the limit alone
  // would answer the request with an unchanged picture. It is only ever
  // widened, so a d min chosen by hand survives.
  $("zones").addEventListener("input", () => {
    const v = parseInt($("zones").value, 10);
    if (!Number.isFinite(v)) return;
    st.zones = v;
    if (structure && v > 0) {
      let need = Infinity;
      try {
        need = dMinForZone(structure, st.uvw, v, st.kv);
      } catch {
        // an unusable zone axis: recompute() reports it
      }
      if (Number.isFinite(need) && need < st.dMin) {
        st.dMin = need;
        $("dmin").value = need.toFixed(3);
      }
    }
    recompute();
  });

  $("gain").addEventListener("input", () => {
    st.gain = Math.pow(10, $("gain").value / 10);
    $("gainLabel").textContent =
      st.gain < 1000
        ? `×${st.gain.toFixed(0)}`
        : `×1e${Math.log10(st.gain).toFixed(0)}`;
    draw();
  });
  $("spotSize").addEventListener("input", () => {
    st.spotSize = +$("spotSize").value;
    draw();
  });
  $("labelThr").addEventListener("input", () => {
    st.labelThr = $("labelThr").value / 100;
    $("labelThrLabel").textContent = `${$("labelThr").value}%`;
    draw();
  });
  for (const [id, key] of [
    ["logScale", "log"],
    ["showLabels", "labels"],
    ["showRings", "rings"],
  ])
    $(id).addEventListener("change", () => {
      st[key] = $(id).checked;
      draw();
    });
  $("colormap").addEventListener("change", () => {
    st.colormap = $("colormap").value;
    draw();
  });

  $("structure").addEventListener("change", () =>
    applyStructure(structures[+$("structure").value]),
  );
  $("uploadBtn").addEventListener("click", () => $("cifFile").click());
  $("cifFile").addEventListener("change", (e) => {
    if (e.target.files[0]) onFile(e.target.files[0]);
    e.target.value = "";
  });
  // Dropping a file anywhere on the plot loads it too.
  const stage = $("stage");
  stage.addEventListener("dragover", (e) => e.preventDefault());
  stage.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]);
  });

  // ---------------------------------------------------- two-crystal compare
  $("structure2").addEventListener("change", () => {
    const v = +$("structure2").value;
    st.secondIndex = v;
    if (v >= 0) {
      try {
        second = new Structure(structures[v]);
      } catch (err) {
        second = null;
        $("status").className = "bad";
        $("status").textContent = err.message;
      }
    } else {
      second = null;
    }
    syncRows();
    recompute();
  });
  $("upload2Btn").addEventListener("click", () => $("cifFile2").click());
  $("cifFile2").addEventListener("change", (e) => {
    if (e.target.files[0]) onSecondFile(e.target.files[0]);
    e.target.value = "";
  });
  // The twist only redraws: it rotates the picture, not the model, so there is
  // nothing to recompute and the slider can stay live while it is dragged.
  $("twist").addEventListener("input", () => {
    st.twist = +$("twist").value;
    $("twistLabel").textContent = `${st.twist.toFixed(1)}°`;
    draw();
  });
  for (const b of document.querySelectorAll("#grpCompare .quickzone button"))
    b.addEventListener("click", () => {
      st.twist = +b.dataset.twist;
      $("twist").value = String(st.twist);
      $("twistLabel").textContent = `${st.twist.toFixed(1)}°`;
      draw();
    });
  $("chkMatch").addEventListener("change", () => {
    st.showMatch = $("chkMatch").checked;
    draw();
  });
  $("compareMode").addEventListener("change", () => {
    st.compareMode = $("compareMode").value;
    draw();
  });
  // Scan the twist and go to the angle that lines the two patterns up best.
  // This is the question a twinned crystal asks, so the answer is reported with
  // the count rather than just moving the slider silently.
  $("findTwist").addEventListener("click", () => {
    if (!result || !result2) return;
    const best = bestTwist(result, result2, { tol: st.matchTol });
    st.twist = best.twist;
    $("twist").value = String(best.twist);
    $("twistLabel").textContent = `${best.twist.toFixed(1)}°`;
    draw();
    $("status").className = "";
    $("status").textContent =
      `best twist ${best.twist.toFixed(1)}° with ${best.count} of ` +
      `${result.count} reflections coincident (tolerance ${st.matchTol} Å⁻¹)`;
  });

  $("savePng").addEventListener("click", () => {
    if (!result) return;
    view.canvas.toBlob((b) => b && download(stem() + ".png", b));
  });
  $("saveTable").addEventListener("click", saveTable);
  $("saveMatches").addEventListener("click", saveMatches);
  $("panelToggle").addEventListener("click", () =>
    $("panel").classList.toggle("open"),
  );

  bindZoom();
  bindRotation();

  $("pattern").addEventListener("mousemove", (e) => {
    if (!result || drag) return;
    if (st.mode === "powder") {
      powderHover(e);
      return;
    }
    dockHover();
    const hit = view.pickSpot(e.clientX, e.clientY);
    $("hover").textContent = hit ? spotText(hit) : "";
  });
  $("pattern").addEventListener("mouseleave", () => {
    $("hover").textContent = "";
    if (hoverPeak) {
      hoverPeak = null;
      if (result && st.mode === "powder") drawPowder();
    }
  });

  new ResizeObserver(() => {
    if (result) draw();
    else if (lastError) view.message(lastError, "#f0725a");
  }).observe($("stage"));
}

// ------------------------------------------------------------------- boot

async function boot() {
  const grab = async (p) => {
    const r = await fetch(p);
    if (!r.ok) throw new Error(`${p}: ${r.status} ${r.statusText}`);
    return r.json();
  };
  const [xray, neutron, electron, maps] = await Promise.all([
    grab("../data/scattering_factors.json"),
    grab("../data/neutron_lengths.json"),
    grab("../data/electron_factors.json"),
    grab("../data/colormaps.json"),
  ]);
  setTables({ xray, neutron, electron });
  setElements(Object.keys(xray));
  luts = maps;

  const names = await grab("../data/structures.json");
  structures = await Promise.all(names.map((n) => grab(`../data/${n}.json`)));

  const sel = $("structure");
  structures.forEach((s, i) => {
    const o = document.createElement("option");
    o.value = String(i);
    o.textContent = s.name;
    sel.appendChild(o);
  });
  const sel2 = $("structure2");
  const none = document.createElement("option");
  none.value = "-1";
  none.textContent = "none";
  sel2.appendChild(none);
  structures.forEach((s, i) => {
    const o = document.createElement("option");
    o.value = String(i);
    o.textContent = s.name;
    sel2.appendChild(o);
  });
  sel2.value = "-1";
  const cmap = $("colormap");
  for (const k of Object.keys(luts)) {
    const o = document.createElement("option");
    o.value = k;
    o.textContent = k;
    cmap.appendChild(o);
  }
  cmap.value = st.colormap;

  view = new PatternView($("pattern"));
  bind();
  syncRows();
  collapsibleSections($("panel"));
  mountCredit($("panel"), WORK);
  logCredit(WORK);
  applyStructure(structures[0]);
  $("boot").remove();

  // Deliberate public API, not a debug leftover: it lets the test harness
  // drive the page without waiting on animation frames, and lets anyone
  // script the app from the console.
  window.singleCrystal = {
    state: st,
    get view() {
      return view;
    },
    get structure() {
      return structure;
    },
    get result() {
      return result;
    },
    get second() {
      return second;
    },
    get result2() {
      return result2;
    },
    /** Load a comparison crystal from a parsed CIF document, or null to clear. */
    setSecond(doc) {
      second = doc ? new Structure(doc) : null;
      st.secondIndex = -1;
      syncRows();
      recompute();
    },
    recompute,
    computeSection,
    computeTem,
    computePowder,
    Structure,
  };
}

boot().catch((err) => {
  const b =
    $("boot") || document.body.appendChild(document.createElement("div"));
  b.id = "boot";
  b.className = "err";
  const hint =
    location.protocol === "file:"
      ? "You opened this file directly; serve it over http instead " +
        "(ES modules and fetch do not work from file://)."
      : "Reload the page; if it keeps failing, the message above says what broke.";
  b.textContent = `Failed to start.\n\n${err.message}\n\n${hint}`;
  console.error(err);
});
