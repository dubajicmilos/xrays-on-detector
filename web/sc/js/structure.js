/*! Single-Crystal Diffraction · © 2026 Miloš Dubajić · MIT · https://github.com/dubajicmilos/xrays-on-detector */

/**
 * Reciprocal lattice and structure factors for a P1 cell.
 *
 * A port of single_crystal/structure.py. The reciprocal convention is 2*pi
 * throughout, matching the rest of the project: Q = B . (h,k,l) and
 * |Q| = 2*pi/d. bMatrix comes from ../../js/physics.js, so this app and the
 * Game of Diffraction build the same lattice from the same cell.
 *
 * Symmetry is applied when the CIF is read (../../js/cif.js expands to P1) and
 * never here. That split is the whole point: the sum below is only correct
 * because every atom of the cell is already in the list.
 */

import { bMatrix, inv3, TWO_PI } from "../../js/physics.js";
import { factorsAt, missingFor, RADIATION_LABEL } from "../../js/scatter.js";

export class Structure {
  /**
   * @param doc a parsed CIF, or a bundled structure from data/*.json:
   *   {name, cell: {a,b,c,alpha,beta,gamma}, atoms: [{element, nuclide?, x, y, z, occ, B}]}
   */
  constructor(doc) {
    if (!doc || !doc.atoms || !doc.atoms.length)
      throw new Error("this structure has no atoms");
    this.name = doc.name || "structure";
    this.spaceGroup = doc.spaceGroup || null;
    this.cell = doc.cell;
    const { a, b, c, alpha, beta, gamma } = doc.cell;
    this.B = bMatrix(a, b, c, alpha, beta, gamma);
    // Direct lattice with a, b, c as columns: A = 2*pi inv(B)^T.
    const bi = inv3(this.B);
    this.A = [0, 1, 2].map((i) => [0, 1, 2].map((j) => TWO_PI * bi[j][i]));

    const n = doc.atoms.length;
    this.n = n;
    this.frac = new Float64Array(3 * n);
    this.occ = new Float64Array(n);
    this.Biso = new Float64Array(n);
    this.elements = [];
    this.nuclides = [];
    for (let i = 0; i < n; i++) {
      const at = doc.atoms[i];
      this.frac[3 * i] = at.x;
      this.frac[3 * i + 1] = at.y;
      this.frac[3 * i + 2] = at.z;
      this.occ[i] = at.occ === undefined ? 1 : at.occ;
      this.Biso[i] = at.B || 0;
      this.elements.push(at.element);
      // Structures bundled before the neutron work have no nuclide field, and
      // for them the element is the right answer: none contains deuterium.
      this.nuclides.push(at.nuclide || at.element);
    }

    // Distinct species, so a table lookup happens once per species per
    // reflection instead of once per atom.
    this._byRadiation = new Map();
  }

  symbolsFor(radiation) {
    return radiation === "neutron" ? this.nuclides : this.elements;
  }

  /** Distinct symbols and, for each atom, which one it is. */
  _species(radiation) {
    let got = this._byRadiation.get(radiation);
    if (!got) {
      const symbols = this.symbolsFor(radiation);
      const uniq = [...new Set(symbols)];
      const index = new Int32Array(symbols.length);
      for (let i = 0; i < symbols.length; i++)
        index[i] = uniq.indexOf(symbols[i]);
      got = { uniq, index };
      this._byRadiation.set(radiation, got);
    }
    return got;
  }

  /** Throws with a readable list if this radiation cannot see some species. */
  check(radiation) {
    const missing = missingFor(radiation, this._species(radiation).uniq);
    if (missing.length)
      throw new Error(
        `no ${RADIATION_LABEL[radiation].toLowerCase()} scattering data for ` +
          missing.slice(0, 6).join(", "),
      );
  }

  /** Q in the crystal Cartesian frame for a flat hkl array (1/Angstrom). */
  q(hkl) {
    const B = this.B;
    const n = hkl.length / 3;
    const out = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) {
      const h = hkl[3 * i],
        k = hkl[3 * i + 1],
        l = hkl[3 * i + 2];
      out[3 * i] = B[0][0] * h + B[0][1] * k + B[0][2] * l;
      out[3 * i + 1] = B[1][0] * h + B[1][1] * k + B[1][2] * l;
      out[3 * i + 2] = B[2][0] * h + B[2][1] * k + B[2][2] * l;
    }
    return out;
  }

  /**
   * |F(hkl)|^2 for a flat hkl array, in the squared unit of the radiation.
   *
   * F = sum_j occ_j f_j(s) exp(-B_j s^2) exp(2 pi i (h x_j + k y_j + l z_j))
   *
   * `qOut`, if given, is filled with |Q| so the caller does not repeat the
   * matrix product it already needs for the plot coordinates.
   */
  intensity(hkl, radiation = "xray", qOut = null) {
    this.check(radiation);
    const { uniq, index } = this._species(radiation);
    const n = hkl.length / 3;
    const m = this.n;
    const Q = this.q(hkl);
    const out = new Float64Array(n);
    const frac = this.frac,
      occ = this.occ,
      Biso = this.Biso;

    for (let i = 0; i < n; i++) {
      const qx = Q[3 * i],
        qy = Q[3 * i + 1],
        qz = Q[3 * i + 2];
      const qn = Math.hypot(qx, qy, qz);
      if (qOut) qOut[i] = qn;
      const s = qn / (4 * Math.PI);
      const s2 = s * s;
      const f = factorsAt(radiation, uniq, s);
      const h = hkl[3 * i],
        k = hkl[3 * i + 1],
        l = hkl[3 * i + 2];

      let re = 0,
        im = 0;
      for (let j = 0; j < m; j++) {
        const w =
          occ[j] * f[index[j]] * (Biso[j] ? Math.exp(-Biso[j] * s2) : 1);
        const ph =
          TWO_PI *
          (h * frac[3 * j] + k * frac[3 * j + 1] + l * frac[3 * j + 2]);
        re += w * Math.cos(ph);
        im += w * Math.sin(ph);
      }
      out[i] = re * re + im * im;
    }
    return out;
  }

  /** Cell volume in Angstrom^3. */
  get volume() {
    const A = this.A;
    return Math.abs(
      A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
        A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
        A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]),
    );
  }

  /**
   * Every integer hkl except 000 with d >= dMin, as a flat Int32Array.
   *
   * Refuses rather than allocating something that will not fit. The limit is
   * easy to cross without meaning to: an electron wavelength of 0.025 Angstrom
   * puts 2*theta = 120 degrees at d = 0.014 Angstrom, which is billions of
   * reflections.
   */
  hklWithin(dMin, maxCandidates = 4e6) {
    if (!(dMin > 0)) throw new Error("d min must be positive");
    const qMax = TWO_PI / dMin;
    const bi = inv3(this.B);
    const bound = bi.map(
      (row) => Math.ceil(qMax * Math.hypot(row[0], row[1], row[2])) + 1,
    );
    const count = bound.reduce((acc, m) => acc * (2 * m + 1), 1);
    if (count > maxCandidates) {
      const shrink = Math.cbrt(maxCandidates / count);
      throw new Error(
        `a d min of ${dMin.toFixed(4)} Å means ${count.toExponential(2)} ` +
          `candidate reflections for this cell, which is more than the browser ` +
          `can hold. Raise d min above about ${(dMin / shrink).toFixed(3)} Å.`,
      );
    }
    const out = [];
    for (let h = -bound[0]; h <= bound[0]; h++)
      for (let k = -bound[1]; k <= bound[1]; k++)
        for (let l = -bound[2]; l <= bound[2]; l++) {
          if (h === 0 && k === 0 && l === 0) continue;
          const qx = this.B[0][0] * h + this.B[0][1] * k + this.B[0][2] * l;
          const qy = this.B[1][0] * h + this.B[1][1] * k + this.B[1][2] * l;
          const qz = this.B[2][0] * h + this.B[2][1] * k + this.B[2][2] * l;
          if (Math.hypot(qx, qy, qz) <= qMax) out.push(h, k, l);
        }
    return Int32Array.from(out);
  }

  describe() {
    const { a, b, c, alpha, beta, gamma } = this.cell;
    return {
      cell: `a ${a.toFixed(4)}  b ${b.toFixed(4)}  c ${c.toFixed(4)} Å`,
      angles: `α ${alpha.toFixed(3)}  β ${beta.toFixed(3)}  γ ${gamma.toFixed(3)}°`,
      volume: `V ${this.volume.toFixed(2)} Å³`,
      atoms: `${this.n} atoms in P1`,
    };
  }
}
