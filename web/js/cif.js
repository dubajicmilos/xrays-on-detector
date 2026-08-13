/*! The Game of Diffraction · © 2026 Miloš Dubajić · MIT · https://github.com/dubajicmilos/xrays-on-detector */

/**
 * CIF reader for the browser: cell, symmetry, atoms, expanded to P1.
 *
 * This is the one piece of the pipeline that used to live only on the desktop
 * side, where ASE did the work. The structure factor sum in physics.js runs
 * over the atoms it is given and applies no symmetry of its own, so a file that
 * lists an asymmetric unit has to be expanded before it can be used. Doing that
 * wrong is silent and severe: skip the expansion and half the reflections come
 * out with the wrong intensity while a spot check still passes.
 *
 * Operators come from the file itself rather than a space-group table, which
 * covers essentially every CIF in the wild and keeps this honest: a listed
 * operator set is complete by definition, centring included, so it is used
 * exactly as given. Only a file with no operators at all falls back to the
 * Hermann-Mauguin symbol, and one naming a symmetry it does not spell out is
 * refused rather than quietly expanded into a structure that looks plausible.
 */

export class CifError extends Error {}

const CENTRING = {
  P: [[0, 0, 0]],
  I: [[0, 0, 0], [0.5, 0.5, 0.5]],
  F: [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
  A: [[0, 0, 0], [0, 0.5, 0.5]],
  B: [[0, 0, 0], [0.5, 0, 0.5]],
  C: [[0, 0, 0], [0.5, 0.5, 0]],
  R: [[0, 0, 0], [2 / 3, 1 / 3, 1 / 3], [1 / 3, 2 / 3, 2 / 3]],
};

/** A CIF number: strip the estimated standard deviation, reject placeholders. */
function num(token) {
  if (token === undefined || token === null) return null;
  const t = String(token).trim().replace(/\((\d+)\)$/, "");
  if (t === "" || t === "?" || t === ".") return null;
  const v = Number(t);
  return Number.isFinite(v) ? v : null;
}

/**
 * Split a CIF into tokens, honouring quotes and semicolon text fields.
 * Returns tokens as {value, quoted} so a quoted "loop_" cannot start a loop.
 */
function tokenize(text) {
  const out = [];
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];
    if (line.startsWith(";")) {
      // multi-line text field: swallow to the closing semicolon
      let body = line.slice(1);
      while (++i < lines.length && !lines[i].startsWith(";")) body += lines[i];
      out.push({ value: body.trim(), quoted: true });
      continue;
    }
    let j = 0;
    while (j < line.length) {
      const c = line[j];
      if (c === " " || c === "\t") {
        j++;
      } else if (c === "#") {
        break; // comment to end of line
      } else if (c === "'" || c === '"') {
        const end = line.indexOf(c, j + 1);
        if (end < 0) {
          out.push({ value: line.slice(j + 1), quoted: true });
          break;
        }
        out.push({ value: line.slice(j + 1, end), quoted: true });
        j = end + 1;
      } else {
        let end = j;
        while (end < line.length && !" \t".includes(line[end])) end++;
        out.push({ value: line.slice(j, end), quoted: false });
        j = end;
      }
    }
  }
  return out;
}

/**
 * Split into data blocks, each {items: Map, loops: [{tags, rows}]}.
 *
 * Published CIFs routinely open with a data_publication_text block carrying
 * only bibliography, so the structure is not always in the first one.
 */
function parseBlocks(tokens) {
  const blocks = [];
  let items = new Map();
  let loops = [];
  let name = "";
  let i = 0;
  const isTag = (t) => !t.quoted && t.value.startsWith("_");
  const isWord = (t, w) => !t.quoted && t.value.toLowerCase() === w;

  while (i < tokens.length) {
    const t = tokens[i];
    if (!t.quoted && t.value.toLowerCase().startsWith("data_")) {
      if (items.size || loops.length) blocks.push({ items, loops, name });
      name = t.value.slice(5);
      items = new Map();
      loops = [];
      i++;
    } else if (isWord(t, "loop_")) {
      i++;
      const tags = [];
      while (i < tokens.length && isTag(tokens[i]))
        tags.push(tokens[i++].value.toLowerCase());
      const rows = [];
      let row = [];
      while (
        i < tokens.length &&
        !isTag(tokens[i]) &&
        !isWord(tokens[i], "loop_") &&
        !(!tokens[i].quoted && tokens[i].value.toLowerCase().startsWith("data_"))
      ) {
        row.push(tokens[i++].value);
        if (row.length === tags.length) {
          rows.push(row);
          row = [];
        }
      }
      loops.push({ tags, rows });
    } else if (isTag(t)) {
      const tag = t.value.toLowerCase();
      i++;
      if (i < tokens.length && !isTag(tokens[i]) && !isWord(tokens[i], "loop_"))
        items.set(tag, tokens[i++].value);
      else items.set(tag, "");
    } else {
      i++;
    }
  }
  if (items.size || loops.length) blocks.push({ items, loops, name });
  return blocks;
}

/** "-x+1/2, y, -z" -> [[a,b,c,t], ...] acting on fractional coordinates. */
function parseOperator(spec) {
  const parts = spec.split(",");
  if (parts.length !== 3)
    throw new CifError(`symmetry operator "${spec}" does not have 3 parts`);
  return parts.map((part) => {
    const row = [0, 0, 0, 0];
    const cleaned = part.replace(/\s+/g, "").toLowerCase();
    const terms = cleaned.match(/[+-]?[^+-]+/g) || [];
    for (const term of terms) {
      const m = term.match(/^([+-]?)([\d./]*)\*?([xyz]?)$/);
      if (!m) throw new CifError(`cannot read symmetry operator "${spec}"`);
      const sign = m[1] === "-" ? -1 : 1;
      let mag = 1;
      if (m[2]) {
        const frac = m[2].split("/");
        mag = frac.length === 2 ? Number(frac[0]) / Number(frac[1]) : Number(m[2]);
        if (!Number.isFinite(mag))
          throw new CifError(`cannot read symmetry operator "${spec}"`);
      }
      const axis = m[3];
      if (axis) row["xyz".indexOf(axis)] += sign * mag;
      else row[3] += sign * mag;
    }
    return row;
  });
}

const ELEMENTS = new Set();
/** Give the reader the element list so an unknown symbol is caught here. */
export function setElements(symbols) {
  ELEMENTS.clear();
  for (const s of symbols) ELEMENTS.add(s);
}

/**
 * "Pb2+", "Cs1", "D" -> the symbols our tables know, as {element, nuclide}.
 *
 * They differ only for deuterium and tritium, and only because the radiation
 * cares: those isotopes scatter x-rays and electrons exactly as hydrogen does,
 * so `element` folds them in, while b_c(H) is -3.739 fm against b_c(D) of
 * +6.671 fm, opposite in sign, so `nuclide` keeps them apart for neutrons.
 */
function element(typeSymbol, label) {
  for (const raw of [typeSymbol, label]) {
    if (!raw) continue;
    const letters = String(raw).replace(/[^A-Za-z]/g, "");
    if (!letters) continue;
    const two = letters.slice(0, 2);
    const cap2 = two.charAt(0).toUpperCase() + two.slice(1).toLowerCase();
    const cap1 = letters.charAt(0).toUpperCase();
    if (ELEMENTS.has(cap2)) return { element: cap2, nuclide: cap2 };
    if (ELEMENTS.has(cap1)) return { element: cap1, nuclide: cap1 };
    if (cap1 === "D" || cap1 === "T") return { element: "H", nuclide: cap1 };
  }
  return null;
}

const wrap = (v) => {
  let w = v - Math.floor(v);
  if (w > 1 - 1e-6 || w < 1e-6) w = 0;
  return w;
};

/**
 * Parse CIF text into the same shape as the bundled structures.
 * Throws CifError with a message meant for the user.
 */
export function parseCif(text, name = "uploaded") {
  if (!text || !/data_/i.test(text))
    throw new CifError("this file has no data_ block, so it is not a CIF");

  const blocks = parseBlocks(tokenize(text));
  const block =
    blocks.find(
      (b) =>
        b.items.has("_cell_length_a") &&
        b.loops.some((L) => L.tags.some((t) => /_atom_site_fract_x$/.test(t))),
    ) || blocks.find((b) => b.items.has("_cell_length_a"));
  if (!block)
    throw new CifError("no data block in this file has a unit cell in it");
  const { items, loops } = block;
  const withStructure = blocks.filter(
    (b) =>
      b.items.has("_cell_length_a") &&
      b.loops.some((L) => L.tags.some((t) => /_atom_site_fract_x$/.test(t))),
  );

  const cellOf = (tag) => num(items.get(tag));
  const a = cellOf("_cell_length_a");
  const b = cellOf("_cell_length_b");
  const c = cellOf("_cell_length_c");
  const alpha = cellOf("_cell_angle_alpha");
  const beta = cellOf("_cell_angle_beta");
  const gamma = cellOf("_cell_angle_gamma");
  if ([a, b, c, alpha, beta, gamma].some((v) => v === null || v <= 0))
    throw new CifError("the unit cell is missing or unreadable");

  // -- symmetry operators, from the file
  let ops = [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]];
  const symLoop = loops.find((L) =>
    L.tags.some((t) =>
      /_(space_group_symop_operation_xyz|symmetry_equiv_pos_as_xyz)$/.test(t),
    ),
  );
  if (symLoop) {
    const col = symLoop.tags.findIndex((t) =>
      /_(space_group_symop_operation_xyz|symmetry_equiv_pos_as_xyz)$/.test(t),
    );
    const specs = symLoop.rows.map((r) => r[col]).filter(Boolean);
    if (specs.length) ops = specs.map(parseOperator);
  }

  // Whether a symmetry is claimed and whether it is written as a readable
  // Hermann-Mauguin symbol are different questions: files name the space group
  // by its International Tables number too, which says nothing about centring
  // and is no use as a label, but does still assert a symmetry we must honour.
  const raw = (items.get("_space_group_name_h-m_alt") ||
    items.get("_symmetry_space_group_name_h-m") ||
    "").replace(/\s+/g, "");
  const itNumber = num(
    items.get("_space_group_it_number") || items.get("_symmetry_int_tables_number"),
  );
  const symbol = /[A-Za-z]/.test(raw) ? raw : null;
  const claimsSymmetry =
    (symbol && !/^P1$/i.test(symbol)) || (itNumber !== null && itNumber !== 1);

  if (!symLoop && claimsSymmetry)
    throw new CifError(
      `this CIF names the space group ${symbol || itNumber} but lists no ` +
        "symmetry operators, so the full cell cannot be built from it. Export " +
        "it as P1 (VESTA, or ASE read/write) and load that.",
    );

  // A file that lists operators lists all of them, centring included, so the
  // symbol is only consulted when there are none to go on.
  const letter = symbol ? symbol.replace(/^[-+]/, "").charAt(0).toUpperCase() : "P";
  const centring = symLoop ? CENTRING.P : CENTRING[letter] || CENTRING.P;

  // -- atom sites
  const atomLoop = loops.find((L) => L.tags.some((t) => /_atom_site_fract_x$/.test(t)));
  if (!atomLoop)
    throw new CifError("no atom sites with fractional coordinates were found");
  const col = (re) => atomLoop.tags.findIndex((t) => re.test(t));
  const cx = col(/_atom_site_fract_x$/);
  const cy = col(/_atom_site_fract_y$/);
  const cz = col(/_atom_site_fract_z$/);
  const cType = col(/_atom_site_type_symbol$/);
  const cLabel = col(/_atom_site_label$/);
  const cOcc = col(/_atom_site_occupancy$/);
  const cB = col(/_atom_site_b_iso_or_equiv$/);
  const cU = col(/_atom_site_u_iso_or_equiv$/);
  if (cy < 0 || cz < 0)
    throw new CifError("the atom sites are missing y or z coordinates");

  const sites = [];
  const unknown = new Set();
  for (const r of atomLoop.rows) {
    const x = num(r[cx]), y = num(r[cy]), z = num(r[cz]);
    if (x === null || y === null || z === null) continue;
    const sym = element(cType >= 0 ? r[cType] : null, cLabel >= 0 ? r[cLabel] : null);
    if (!sym) {
      unknown.add((cType >= 0 ? r[cType] : r[cLabel]) || "?");
      continue;
    }
    const el = sym.element;
    const occ = cOcc >= 0 ? num(r[cOcc]) : null;
    let B = cB >= 0 ? num(r[cB]) : null;
    if (B === null && cU >= 0) {
      const U = num(r[cU]);
      if (U !== null) B = 8 * Math.PI * Math.PI * U;
    }
    sites.push({
      el,
      nuc: sym.nuclide,
      x,
      y,
      z,
      occ: occ === null ? 1 : occ,
      B: B === null ? 0 : B,
    });
  }
  if (unknown.size)
    throw new CifError(
      `these atom types are not in the scattering factor table: ` +
        `${[...unknown].slice(0, 6).join(", ")}`,
    );
  if (!sites.length) throw new CifError("no usable atom sites were found");

  // -- expand to P1, merging positions that coincide
  //
  // An atom on a special position is mapped onto itself by several operators,
  // so the copies have to be merged or it scatters several times over. They
  // are compared with a tolerance and across the cell boundary, not by a
  // rounded key: two copies either side of a rounding step are the same atom.
  // Different elements never merge, which is what keeps a site shared by a
  // disordered C and N as the two contributions it physically is.
  const TOL = 1e-3;
  const near = (p, q) => {
    let s2 = 0;
    for (let i = 0; i < 3; i++) {
      let d = Math.abs(p[i] - q[i]);
      if (d > 0.5) d = 1 - d;
      s2 += d * d;
    }
    return s2 < TOL * TOL;
  };

  const atoms = [];
  const byElement = new Map();
  for (const s of sites) {
    // Keyed by nuclide, not element, so an H site and a D site at the same
    // position stay the two distinct scatterers they are for neutrons.
    const kept = byElement.get(s.nuc) || [];
    byElement.set(s.nuc, kept);
    for (const op of ops) {
      for (const t of centring) {
        const p = [0, 1, 2].map((i) => {
          const row = op[i];
          return wrap(
            row[0] * s.x + row[1] * s.y + row[2] * s.z + row[3] + t[i],
          );
        });
        if (kept.some((q) => near(p, q))) continue;
        kept.push(p);
        // Full precision. Rounding to six decimals costs nothing on a
        // coordinate like 0.25 and 3e-7 on a hexagonal 1/3, which is enough to
        // move |F|^2 by 1e-5 relative -- invisible on screen, but a needless
        // disagreement with any other code.
        atoms.push({
          element: s.el,
          nuclide: s.nuc,
          x: p[0],
          y: p[1],
          z: p[2],
          occ: s.occ,
          B: s.B,
        });
      }
    }
  }

  return {
    name,
    source: name,
    block: block.name || null,
    blocksInFile: withStructure.length,
    spaceGroup: symbol && !/^P1$/i.test(symbol) ? symbol : null,
    cell: { a, b, c, alpha, beta, gamma },
    atoms,
  };
}
