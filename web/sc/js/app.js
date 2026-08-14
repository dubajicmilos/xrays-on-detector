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
import {
  electronWavelength,
  RADIATION_LABEL,
  setTables,
  UNITS,
} from "../../js/scatter.js";
import { formatHkl } from "./display.js";
import { computePowder } from "./powder.js";
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
};

let luts = null;
let structures = [];
let structure = null;
let result = null;
let view = null;

// --------------------------------------------------------------- compute

function recompute() {
  if (!structure) return;
  const status = $("status");
  status.className = "";
  try {
    if (st.mode === "section") {
      result = computeSection(structure, {
        uvw: st.uvw,
        layer: st.layer,
        dMin: st.dMin,
        radiation: st.radiation,
      });
      $("zoneLaw").textContent = zoneLaw(result.uvw, result.layer);
    } else if (st.mode === "saed") {
      result = computeTem(structure, {
        uvw: st.uvw,
        kv: st.kv,
        thickness: st.thickness,
        dMin: st.dMin,
        maxZone: st.zones,
      });
      $("zoneLaw").textContent = "";
      $("zoneRings").textContent = result.zoneRadii
        .slice(1)
        .map((r) => (r ? r.toFixed(1) : "—"))
        .join(" ");
    } else {
      const lam =
        st.radiation === "electron" ? electronWavelength(st.kv) : st.wavelength;
      result = computePowder(structure, {
        wavelength: lam,
        radiation: st.radiation,
        twoThetaMax: st.ttMax,
      });
    }
  } catch (err) {
    result = null;
    view.message(err.message, "#f0725a");
    status.className = "bad";
    status.textContent = err.message;
    return;
  }
  draw();
}

function draw() {
  if (!result) return;
  const lut = luts[st.colormap];
  if (st.mode === "powder") {
    view.drawPowder(result, {
      labels: st.labels,
      labelThreshold: Math.max(st.labelThr * 100, 1),
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
    });
  }
  writeOverlay();
}

function writeOverlay() {
  const title = $("title");
  const status = $("status");
  const unit = UNITS[result.radiation || st.radiation];
  const rad = RADIATION_LABEL[result.radiation || st.radiation];

  if (st.mode === "powder") {
    title.textContent =
      `${structure.name}   ${rad}\n` +
      `λ ${result.wavelength.toFixed(5)} Å   ${result.count} peaks`;
    status.textContent =
      `Lorentz ${st.radiation === "xray" ? "and polarisation " : ""}applied` +
      `   ·   intensities relative to the strongest peak` +
      (st.radiation === "xray" ? "" : "   ·   no polarisation term");
    return;
  }

  const zone = result.uvw.join(" ");
  if (st.mode === "section") {
    title.textContent =
      `${structure.name}   ${rad}\n` +
      `zone [${zone}]   layer ${result.layer}   (${zoneLaw(result.uvw, result.layer)})\n` +
      `${result.count} reflections   d ≥ ${st.dMin.toFixed(2)} Å   |F|² in ${unit}²`;
    const bits = [];
    if (result.zoneFactor > 1)
      bits.push(`zone axis reduced by ${result.zoneFactor} to [${zone}]`);
    if (result.layer !== 0)
      bits.push(
        `layer offset ${result.height.toFixed(3)} Å⁻¹ along the zone axis`,
      );
    status.textContent =
      bits.join("   ·   ") ||
      "hover a spot for its indices   ·   wheel to zoom, drag to pan, " +
        "double-click to fit";
  } else {
    title.textContent =
      `${structure.name}   Electrons\n` +
      `beam ∥ [${zone}]   ${result.kv} kV   λ ${result.wavelength.toFixed(5)} Å\n` +
      `${result.count} reflections   thickness ${result.thickness} Å`;
    const first = result.zoneRadii[1];
    if (st.zones > 0 && first && first > result.qMax) {
      status.className = "warn";
      status.textContent =
        `the first Laue ring is at ${first.toFixed(1)} Å⁻¹, outside d min ` +
        `— set d min to ${((2 * Math.PI) / first).toFixed(3)} Å to reach it`;
    } else {
      status.textContent =
        `kinematic, sinc² relrod` +
        `   ·   ${result.zonesVisible} of ${result.zoneRadii.length} Laue zones in range` +
        `   ·   dynamical scattering is not modelled`;
    }
  }
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
  const species = [...new Set(structure.nuclides)];
  if (species.includes("D") || species.includes("T"))
    notes.push("D and T keep their own neutron scattering lengths");
  $("cifNote").textContent = notes.join(" · ");
  recompute();
}

async function onFile(file) {
  try {
    const doc = parseCif(await file.text(), file.name.replace(/\.cif$/i, ""));
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

// ------------------------------------------------------------------ export

function stem() {
  const base = structure ? structure.name.replace(/[^\w.-]+/g, "_") : "pattern";
  if (st.mode === "powder") return `${base}_powder_${st.radiation}`;
  const z = st.uvw.join("");
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

// ---------------------------------------------------------------- binding

/** Set while a pan is in progress, so hover does not fight the drag. */
let drag = null;

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
  show("grpZone", st.mode !== "powder");
  show("rowLayer", st.mode === "section");
  show("rowWavelength", st.mode === "powder" && st.radiation !== "electron");
  show(
    "rowKv",
    st.mode === "saed" || (st.mode === "powder" && st.radiation === "electron"),
  );
  show("rowThickness", st.mode === "saed");
  show("rowZones", st.mode === "saed");
  show("rowTtMax", st.mode === "powder");
  $("radiation").disabled = st.mode === "saed";
  if (view) showZoom();
  $("modeNote").textContent =
    st.mode === "saed"
      ? "SAED is electron diffraction by definition, so the radiation is fixed."
      : st.mode === "section"
        ? "An undistorted plane of the weighted reciprocal lattice, as a precession camera records it."
        : "Reflections sharing a d-spacing are summed, so multiplicity is counted rather than looked up.";
}

function bind() {
  const num = (id, key, after) =>
    $(id).addEventListener("input", () => {
      const v = parseFloat($(id).value);
      if (!Number.isFinite(v)) return;
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
      const v = parseInt($(id).value, 10);
      if (!Number.isFinite(v)) return;
      st.uvw[i] = v;
      recompute();
    });
  for (const b of document.querySelectorAll(".quickzone button"))
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

  // Asking for a Laue ring is asking for the resolution that shows it. The
  // rings sit far outside any comfortable d min, so leaving the limit alone
  // would answer the request with an unchanged picture. It is only ever
  // widened, so a d min chosen by hand survives.
  $("zones").addEventListener("input", () => {
    const v = parseInt($("zones").value, 10);
    if (!Number.isFinite(v)) return;
    st.zones = v;
    if (structure && v > 0) {
      const need = dMinForZone(structure, st.uvw, v, st.kv);
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

  $("savePng").addEventListener("click", () => {
    view.canvas.toBlob((b) => b && download(stem() + ".png", b));
  });
  $("saveTable").addEventListener("click", saveTable);
  $("panelToggle").addEventListener("click", () =>
    $("panel").classList.toggle("open"),
  );

  bindZoom();

  $("pattern").addEventListener("mousemove", (e) => {
    if (!result || st.mode === "powder" || drag) return;
    const i = view.pick(e.clientX, e.clientY);
    if (i === null) {
      $("hover").textContent = "";
      return;
    }
    const hkl = [
      result.hkl[3 * i],
      result.hkl[3 * i + 1],
      result.hkl[3 * i + 2],
    ];
    const unit = UNITS[result.radiation || st.radiation];
    let text =
      `${formatHkl(hkl)}   (${hkl.join(" ")})\n` +
      `d ${result.d[i].toFixed(4)} Å   |Q| ${result.q[i].toFixed(4)} Å⁻¹\n` +
      `|F|² ${result.intensity[i].toExponential(3)} ${unit}²`;
    if (st.mode === "saed")
      text += `\ns_g ${result.sg[i].toFixed(5)} Å⁻¹   Laue zone ${result.laueZone[i]}`;
    $("hover").textContent = text;
  });
  $("pattern").addEventListener(
    "mouseleave",
    () => ($("hover").textContent = ""),
  );

  new ResizeObserver(() => draw()).observe($("stage"));
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
    recompute,
    computeSection,
    computeTem,
    computePowder,
    Structure,
  };
}

boot().catch((err) => {
  const b = $("boot");
  b.className = "err";
  b.textContent = `${err.message}\n\nServe this folder over http, not file://`;
  console.error(err);
});
