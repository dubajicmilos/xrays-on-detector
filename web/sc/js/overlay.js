/*! Single-Crystal Diffraction · © 2026 Miloš Dubajić · MIT · https://github.com/dubajicmilos/xrays-on-detector */

/**
 * Overlaying two crystals in the same view.
 *
 * Two structures are loaded at once and their sections drawn in the same frame.
 * The second pattern can be rotated about the zone axis, which is the one
 * rotation that keeps the comparison honest: a section is cut perpendicular to
 * the direct-lattice vector u a + v b + w c, so rotating the crystal about that
 * vector rotates the section in its own plane and changes nothing else. The
 * in-plane angle of the pattern therefore *is* the twist of crystal 2 about the
 * zone axis, and no re-indexing is needed to draw it.
 *
 * The alternative reading, rotating the pattern without rotating the crystal,
 * is not offered: it would produce a picture no crystal can give.
 *
 * Everything here is pure, so it can be exercised without a canvas.
 */

/** Rotate a section's in-plane coordinates by phi degrees about the origin. */
export function rotateInPlane(result, phiDeg) {
  const n = result.count;
  const c = Math.cos((phiDeg * Math.PI) / 180);
  const s = Math.sin((phiDeg * Math.PI) / 180);
  const x = new Float64Array(n);
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const xi = result.x[i];
    const yi = result.y[i];
    x[i] = c * xi - s * yi;
    y[i] = s * xi + c * yi;
  }
  return { x, y };
}

/**
 * Pair up reflections of two sections by position in the plane.
 *
 * Both sections are drawn in the same reciprocal-plane coordinates, so two
 * reflections coincide when their (x, y) agree to within `tol` Angstrom^-1.
 * A hash on a tol-sized grid keeps this linear; the neighbour cells are checked
 * too, or a pair straddling a cell boundary would be missed.
 *
 * Returns { pairs, onlyA, onlyB, tolerance }. `pairs` entries are
 * { a, b, dq } with `a`/`b` indices into the respective results.
 */
export function pairUp(resultA, resultB, { tol = 0.03 } = {}) {
  const cells = new Map();
  const key = (i, j) => `${i},${j}`;
  for (let i = 0; i < resultB.count; i++) {
    const gx = Math.floor(resultB.x[i] / tol);
    const gy = Math.floor(resultB.y[i] / tol);
    const k = key(gx, gy);
    let bin = cells.get(k);
    if (!bin) cells.set(k, (bin = []));
    bin.push(i);
  }
  const taken = new Uint8Array(resultB.count);
  const pairs = [];
  const onlyA = [];
  for (let i = 0; i < resultA.count; i++) {
    const ax = resultA.x[i];
    const ay = resultA.y[i];
    const gx = Math.floor(ax / tol);
    const gy = Math.floor(ay / tol);
    let best = -1;
    let bestD = Infinity;
    for (let dx = -1; dx <= 1; dx++)
      for (let dy = -1; dy <= 1; dy++) {
        const bin = cells.get(key(gx + dx, gy + dy));
        if (!bin) continue;
        for (const j of bin) {
          if (taken[j]) continue;
          const d = Math.hypot(resultB.x[j] - ax, resultB.y[j] - ay);
          if (d <= tol && d < bestD) {
            bestD = d;
            best = j;
          }
        }
      }
    if (best >= 0) {
      taken[best] = 1;
      pairs.push({ a: i, b: best, dq: bestD });
    } else {
      onlyA.push(i);
    }
  }
  const onlyB = [];
  for (let j = 0; j < resultB.count; j++) if (!taken[j]) onlyB.push(j);
  return { pairs, onlyA, onlyB, tolerance: tol };
}

/** Mean and worst positional mismatch of the paired reflections. */
export function coincidenceStats(pairs) {
  if (!pairs.length) return { count: 0, mean: 0, worst: 0 };
  let sum = 0;
  let worst = 0;
  for (const p of pairs) {
    sum += p.dq;
    if (p.dq > worst) worst = p.dq;
  }
  return { count: pairs.length, mean: sum / pairs.length, worst };
}

/**
 * The twist that brings the most reflections of B into coincidence with A.
 *
 * Scans [lo, hi) in `step` degree steps and returns { twist, count }. The
 * pairing is linear in the number of reflections, so a whole-degree scan of a
 * few hundred spots costs nothing and answers the question a twinned crystal
 * actually poses: at what angle do the two patterns share spots, and how many.
 */
export function bestTwist(
  resultA,
  resultB,
  { tol = 0.03, step = 0.5, lo = 0, hi = 180 } = {},
) {
  let best = { twist: 0, count: -1 };
  const n = Math.max(1, Math.round((hi - lo) / step));
  for (let i = 0; i <= n; i++) {
    const twist = lo + i * step;
    const rot = rotateInPlane(resultB, twist);
    const { pairs } = pairUp(
      resultA,
      { count: resultB.count, x: rot.x, y: rot.y },
      { tol },
    );
    if (pairs.length > best.count) best = { twist, count: pairs.length };
  }
  return best;
}

/**
 * A few lines describing the comparison, for the panel and the status bar.
 * `A` and `B` are the display names of the two structures.
 */
export function describeComparison(resultA, resultB, ours, A, B) {
  const { pairs, onlyA, onlyB } = ours;
  const stats = coincidenceStats(pairs);
  const pct = (k, n) => (n ? ((100 * k) / n).toFixed(0) + "%" : "—");
  const lines = [
    `${A}  ${resultA.count} reflections`,
    `${B}  ${resultB.count} reflections`,
    "",
    `coincident      ${pairs.length}   (${pct(pairs.length, resultA.count)} of ${A}, ` +
      `${pct(pairs.length, resultB.count)} of ${B})`,
    `only ${A}          ${onlyA.length}`,
    `only ${B}          ${onlyB.length}`,
  ];
  if (stats.count)
    lines.push(
      "",
      `mean offset     ${stats.mean.toFixed(4)} Å⁻¹`,
      `worst offset    ${stats.worst.toFixed(4)} Å⁻¹`,
    );
  return lines.join("\n");
}
