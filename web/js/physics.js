/**
 * Six-circle diffraction physics, ported from the xrays_on_detector Python
 * package (You 1999 convention, via diffcalc-core).
 *
 * This is a deliberate line-for-line port, not a reimplementation: every
 * function here has a counterpart in the package, and test/parity.mjs checks
 * the numbers against fixtures exported from the validated Python.
 *
 * Lab frame: +x vertical (up), +y along the incident beam, +z horizontal.
 * Reciprocal convention: 2*pi throughout, |Q| = 2*pi/d.
 */

export const BEAM = [0, 1, 0];
export const TWO_PI = 2 * Math.PI;

// ---------------------------------------------------------------------------
// 3x3 linear algebra. Matrices are row-major arrays of three rows.
// ---------------------------------------------------------------------------

export const eye3 = () => [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

export function matMul(A, B) {
  const C = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += A[i][k] * B[k][j];
      C[i][j] = s;
    }
  return C;
}

export const matMulN = (...ms) => ms.reduce(matMul);

export function matVec(A, v) {
  return [
    A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2],
    A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2],
    A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2],
  ];
}

export const transpose = (A) => [
  [A[0][0], A[1][0], A[2][0]],
  [A[0][1], A[1][1], A[2][1]],
  [A[0][2], A[1][2], A[2][2]],
];

export function det3(A) {
  return (
    A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
    A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
    A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
  );
}

export function inv3(A) {
  const d = det3(A);
  if (Math.abs(d) < 1e-300) throw new Error("singular matrix");
  const c = [
    [
      A[1][1] * A[2][2] - A[1][2] * A[2][1],
      A[0][2] * A[2][1] - A[0][1] * A[2][2],
      A[0][1] * A[1][2] - A[0][2] * A[1][1],
    ],
    [
      A[1][2] * A[2][0] - A[1][0] * A[2][2],
      A[0][0] * A[2][2] - A[0][2] * A[2][0],
      A[0][2] * A[1][0] - A[0][0] * A[1][2],
    ],
    [
      A[1][0] * A[2][1] - A[1][1] * A[2][0],
      A[0][1] * A[2][0] - A[0][0] * A[2][1],
      A[0][0] * A[1][1] - A[0][1] * A[1][0],
    ],
  ];
  return c.map((row) => row.map((v) => v / d));
}

export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const norm = (a) => Math.hypot(a[0], a[1], a[2]);
export function unit(a) {
  const n = norm(a);
  return n < 1e-300 ? [0, 0, 0] : [a[0] / n, a[1] / n, a[2] / n];
}
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

const rad = (deg) => (deg * Math.PI) / 180;
const deg = (r) => (r * 180) / Math.PI;
export { rad as toRadians, deg as toDegrees };

// ---------------------------------------------------------------------------
// Circle matrices (diffcalc.util + diffcalc.hkl.geometry)
// ---------------------------------------------------------------------------

export const rotX = (t) => [
  [1, 0, 0],
  [0, Math.cos(t), -Math.sin(t)],
  [0, Math.sin(t), Math.cos(t)],
];
export const rotY = (t) => [
  [Math.cos(t), 0, Math.sin(t)],
  [0, 1, 0],
  [-Math.sin(t), 0, Math.cos(t)],
];
export const rotZ = (t) => [
  [Math.cos(t), -Math.sin(t), 0],
  [Math.sin(t), Math.cos(t), 0],
  [0, 0, 1],
];

// diffcalc: MU = x_rotation(mu), ETA = z_rotation(-eta), CHI = y_rotation(chi),
// PHI = z_rotation(-phi), DELTA = z_rotation(-delta), NU = x_rotation(nu).
export const rotMU = (d) => rotX(rad(d));
export const rotETA = (d) => rotZ(-rad(d));
export const rotCHI = (d) => rotY(rad(d));
export const rotPHI = (d) => rotZ(-rad(d));
export const rotDELTA = (d) => rotZ(-rad(d));
export const rotNU = (d) => rotX(rad(d));

/** Sample orientation matrix Z = MU . ETA . CHI . PHI (degrees). */
export const sampleMatrix = (mu, eta, chi, phi) =>
  matMulN(rotMU(mu), rotETA(eta), rotCHI(chi), rotPHI(phi));

/** Detector arm matrix R = NU . DELTA (degrees). */
export const detectorMatrix = (nu, delta) => matMul(rotNU(nu), rotDELTA(delta));

// ---------------------------------------------------------------------------
// Lattice
// ---------------------------------------------------------------------------

/**
 * Reciprocal matrix B in the crystal Cartesian frame, 2*pi convention, so that
 * Q_cart = B . (h,k,l) and |Q| = 2*pi/d. Columns are a*, b*, c*.
 */
export function bMatrix(a, b, c, alpha = 90, beta = 90, gamma = 90) {
  const al = rad(alpha),
    be = rad(beta),
    ga = rad(gamma);
  const cx = c * Math.cos(be);
  const cy = (c * (Math.cos(al) - Math.cos(be) * Math.cos(ga))) / Math.sin(ga);
  const cz2 = c * c - cx * cx - cy * cy;
  if (cz2 <= 0) throw new Error("cell angles do not describe a real lattice");
  // Direct lattice A with a, b, c as columns.
  const A = [
    [a, b * Math.cos(ga), cx],
    [0, b * Math.sin(ga), cy],
    [0, 0, Math.sqrt(cz2)],
  ];
  const invA = inv3(A);
  return transpose(invA).map((row) => row.map((v) => v * TWO_PI));
}

/** Direct lattice with a, b, c as columns: A = 2*pi * inv(B)^T. */
export const aMatrix = (B) =>
  transpose(inv3(B)).map((r) => r.map((v) => v * TWO_PI));

/** A crystal direction as a Cartesian vector in the crystal frame. */
export function crystalVector(B, indices, kind = "hkl") {
  if (kind === "hkl") return matVec(B, indices);
  if (kind === "uvw") return matVec(aMatrix(B), indices);
  throw new Error(`kind must be 'hkl' or 'uvw', got ${kind}`);
}

/** All integer hkl (excluding 000) with |B . hkl| <= Qmax, as a flat Int32Array. */
export function hklWithinQmax(B, Qmax) {
  const Binv = inv3(B);
  const bound = Binv.map(
    (row) => Math.ceil(Qmax * Math.hypot(row[0], row[1], row[2])) + 1,
  );
  const out = [];
  for (let h = -bound[0]; h <= bound[0]; h++)
    for (let k = -bound[1]; k <= bound[1]; k++)
      for (let l = -bound[2]; l <= bound[2]; l++) {
        if (h === 0 && k === 0 && l === 0) continue;
        const qx = B[0][0] * h + B[0][1] * k + B[0][2] * l;
        const qy = B[1][0] * h + B[1][1] * k + B[1][2] * l;
        const qz = B[2][0] * h + B[2][1] * k + B[2][2] * l;
        if (Math.hypot(qx, qy, qz) <= Qmax) out.push(h, k, l);
      }
  return Int32Array.from(out);
}

/** Q in the crystal Cartesian frame for a flat hkl array, as a flat Float64Array. */
export function qCryst(B, hkl) {
  const n = hkl.length / 3;
  const Q = new Float64Array(hkl.length);
  for (let i = 0; i < n; i++) {
    const h = hkl[3 * i],
      k = hkl[3 * i + 1],
      l = hkl[3 * i + 2];
    Q[3 * i] = B[0][0] * h + B[0][1] * k + B[0][2] * l;
    Q[3 * i + 1] = B[1][0] * h + B[1][1] * k + B[1][2] * l;
    Q[3 * i + 2] = B[2][0] * h + B[2][1] * k + B[2][2] * l;
  }
  return Q;
}

// ---------------------------------------------------------------------------
// Structure factors (Cromer-Mann, matching pytilting)
// ---------------------------------------------------------------------------

/**
 * Atomic scattering factor f(s), s = sin(theta)/lambda = |Q| / (4 pi).
 * `table` maps element symbol -> [[a1..a4], [b1..b4], c].
 */
export function scatteringFactor(table, element, s) {
  const base = element.replace(/[0-9+-]/g, "");
  const e = table[base] || table.C;
  const [a, b, c] = e;
  const s2 = s * s;
  let f = c;
  for (let i = 0; i < 4; i++) f += a[i] * Math.exp(-b[i] * s2);
  return f;
}

/**
 * |F(hkl)|^2 for a flat hkl array.
 *
 * F = sum_atoms occ * f(s) * exp(-B_iso s^2) * exp(2 pi i (hx + ky + lz)),
 * which is pytilting's expression exactly. `atoms` is a list of
 * {element, x, y, z, occ, B} in fractional coordinates.
 */
export function structureFactors(table, atoms, B, hkl) {
  const n = hkl.length / 3;
  const out = new Float64Array(n);
  const Q = qCryst(B, hkl);
  for (let i = 0; i < n; i++) {
    const s = Math.hypot(Q[3 * i], Q[3 * i + 1], Q[3 * i + 2]) / (4 * Math.PI);
    const h = hkl[3 * i],
      k = hkl[3 * i + 1],
      l = hkl[3 * i + 2];
    let re = 0,
      im = 0;
    for (const at of atoms) {
      const f = scatteringFactor(table, at.element, s);
      const T = Math.exp(-(at.B || 0) * s * s);
      const w = (at.occ === undefined ? 1 : at.occ) * f * T;
      const ph = TWO_PI * (h * at.x + k * at.y + l * at.z);
      re += w * Math.cos(ph);
      im += w * Math.sin(ph);
    }
    out[i] = re * re + im * im;
  }
  return out;
}

/**
 * |F|^2 for a bare lattice with no structure: an isotropic Debye-Waller
 * falloff only, exp(-2 B_iso (|Q|/4pi)^2). Matches LatticeCrystal, and exists
 * so the app runs on a cell alone. This is not a structure factor and must not
 * be presented as one.
 */
export function latticeStructureFactors(B, hkl, Biso = 1.5) {
  const n = hkl.length / 3;
  const Q = qCryst(B, hkl);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const s = Math.hypot(Q[3 * i], Q[3 * i + 1], Q[3 * i + 2]) / (4 * Math.PI);
    out[i] = Math.exp(-2 * Biso * s * s);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Detector
// ---------------------------------------------------------------------------

export class Detector {
  constructor({
    distance,
    nFast,
    nSlow,
    pixelSize,
    nu = 0,
    delta = 0,
    beamCenterFast = null,
    beamCenterSlow = null,
  }) {
    this.distance = distance;
    this.nFast = nFast;
    this.nSlow = nSlow;
    this.pixelSize = pixelSize;
    this.nu = nu;
    this.delta = delta;
    this.beamCenterFast =
      beamCenterFast === null ? (nFast - 1) / 2 : beamCenterFast;
    this.beamCenterSlow =
      beamCenterSlow === null ? (nSlow - 1) / 2 : beamCenterSlow;
  }

  /**
   * {centre, normal, eFast, eSlow, arm} in the lab frame. Fast is the
   * horizontal lab axis and slow the vertical, matching LabDetector rather
   * than the core Detector (see the note in instrument.py). Fast is -z, so
   * eFast x eSlow = -arm: the image is laid out as seen from the sample
   * looking downstream, which is the view the 3D scene shows.
   */
  frame() {
    const R = detectorMatrix(this.nu, this.delta);
    const arm = matVec(R, BEAM);
    return {
      centre: scale(arm, this.distance),
      normal: scale(arm, -1),
      eFast: matVec(R, [0, 0, -1]),
      eSlow: matVec(R, [1, 0, 0]),
      arm,
    };
  }

  /** Geometrically identical detector with factor x factor pixels merged. */
  binned(factor) {
    if (factor <= 1) return this;
    return new Detector({
      distance: this.distance,
      nFast: Math.max(1, Math.floor(this.nFast / factor)),
      nSlow: Math.max(1, Math.floor(this.nSlow / factor)),
      pixelSize: this.pixelSize * factor,
      nu: this.nu,
      delta: this.delta,
      beamCenterFast: this.beamCenterFast / factor,
      beamCenterSlow: this.beamCenterSlow / factor,
    });
  }

  /** Project one unit diffracted direction. Returns {fast, slow, inside, cosInc}. */
  projectOne(khat) {
    const { centre, normal, eFast, eSlow, arm } = this.frame();
    const denom = dot(khat, normal);
    const t = dot(centre, normal) / denom;
    const hit = scale(khat, t);
    const rel = sub(hit, centre);
    const fast = this.beamCenterFast + dot(rel, eFast) / this.pixelSize;
    const slow = this.beamCenterSlow + dot(rel, eSlow) / this.pixelSize;
    const cosInc = dot(khat, arm);
    const inside =
      denom < 0 &&
      t > 0 &&
      fast >= 0 &&
      fast <= this.nFast - 1 &&
      slow >= 0 &&
      slow <= this.nSlow - 1;
    return { fast, slow, inside, cosInc };
  }

  /** Largest |Q| reachable at any detector corner, 2*pi convention. */
  maxQmax(wavelength) {
    const k = TWO_PI / wavelength;
    const { centre, eFast, eSlow } = this.frame();
    const hf = 0.5 * this.nFast * this.pixelSize;
    const hs = 0.5 * this.nSlow * this.pixelSize;
    let worst = 0;
    for (const sf of [-1, 1])
      for (const ss of [-1, 1]) {
        const c = add(
          centre,
          add(scale(eFast, sf * hf), scale(eSlow, ss * hs)),
        );
        const tt = Math.acos(Math.min(1, Math.max(-1, dot(unit(c), BEAM))));
        worst = Math.max(worst, tt);
      }
    return 2 * k * Math.sin(worst / 2);
  }
}

// ---------------------------------------------------------------------------
// Ewald construction
// ---------------------------------------------------------------------------

/**
 * Reflections within nSigma * sigma of the Ewald sphere.
 *
 * Returns flat typed arrays plus a count; the caller reads the first `count`
 * entries. Buffers are allocated per call, which is cheap next to the sweep.
 */
export function excite({ Qcryst, F2, hkl, ZU, wavelength, sigma, nSigma = 4 }) {
  const n = Qcryst.length / 3;
  const k = TWO_PI / wavelength;
  const kiy = k; // k_i = k * (0, 1, 0)
  const cut = nSigma * sigma;

  const idx = new Int32Array(n);
  const khat = new Float64Array(3 * n);
  const eps = new Float64Array(n);
  const excitation = new Float64Array(n);
  const twoTheta = new Float64Array(n);
  const f2 = new Float64Array(n);
  const outHkl = new Int32Array(3 * n);

  let m = 0;
  for (let i = 0; i < n; i++) {
    const qx = Qcryst[3 * i],
      qy = Qcryst[3 * i + 1],
      qz = Qcryst[3 * i + 2];
    // Q_lab = ZU . Q_cryst
    const lx = ZU[0][0] * qx + ZU[0][1] * qy + ZU[0][2] * qz;
    const ly = ZU[1][0] * qx + ZU[1][1] * qy + ZU[1][2] * qz;
    const lz = ZU[2][0] * qx + ZU[2][1] * qy + ZU[2][2] * qz;

    const fx = lx,
      fy = kiy + ly,
      fz = lz;
    const mag = Math.hypot(fx, fy, fz);
    const e = mag - k;
    if (!(Math.abs(e) <= cut) || !(mag > 0)) continue;

    const ux = fx / mag,
      uy = fy / mag,
      uz = fz / mag;
    idx[m] = i;
    khat[3 * m] = ux;
    khat[3 * m + 1] = uy;
    khat[3 * m + 2] = uz;
    eps[m] = e;
    excitation[m] = Math.exp(-(e * e) / (2 * sigma * sigma));
    twoTheta[m] = Math.acos(Math.min(1, Math.max(-1, uy)));
    f2[m] = F2[i];
    if (hkl) {
      outHkl[3 * m] = hkl[3 * i];
      outHkl[3 * m + 1] = hkl[3 * i + 1];
      outHkl[3 * m + 2] = hkl[3 * i + 2];
    }
    m++;
  }
  return {
    count: m,
    idx,
    khat,
    eps,
    excitation,
    twoTheta,
    F2: f2,
    hkl: outHkl,
  };
}

// ---------------------------------------------------------------------------
// Orientation
// ---------------------------------------------------------------------------

/** Smallest rotation taking unit vector a onto unit vector b. */
export function rotationBetween(vFrom, vTo) {
  const a = unit(vFrom),
    b = unit(vTo);
  const v = cross(a, b);
  const c = dot(a, b);
  const s = norm(v);
  if (s < 1e-12) {
    if (c > 0) return eye3();
    let perp = Math.abs(a[0]) > 0.9 ? [0, 1, 0] : [1, 0, 0];
    const ax = unit(cross(a, perp));
    const K = skew(ax);
    const KK = matMul(K, K);
    return eye3().map((row, i) => row.map((val, j) => val + 2 * KK[i][j]));
  }
  const K = skew(v);
  const KK = matMul(K, K);
  const f = (1 - c) / (s * s);
  return eye3().map((row, i) =>
    row.map((val, j) => val + K[i][j] + KK[i][j] * f),
  );
}

const skew = (v) => [
  [0, -v[2], v[1]],
  [v[2], 0, -v[0]],
  [-v[1], v[0], 0],
];

/** Extrinsic rotation about the lab x, then y, then z axes (degrees). */
export const eulerMatrix = (rx, ry, rz) =>
  matMulN(rotZ(rad(rz)), rotY(rad(ry)), rotX(rad(rx)));

/** Rodrigues rotation about a unit axis by an angle in radians. */
export function axisAngle(axis, angle) {
  const K = skew(unit(axis));
  const KK = matMul(K, K);
  const s = Math.sin(angle),
    c = 1 - Math.cos(angle);
  return eye3().map((row, i) =>
    row.map((v, j) => v + s * K[i][j] + c * KK[i][j]),
  );
}

/**
 * Rotate the crystal so `indices` points along `targetLab`.
 * frame='lab' aligns at the current motor positions (U_new = Z^T R Z U);
 * frame='phi' aligns at the goniometer datum. Returns the new U.
 */
export function alignInLab(
  U,
  B,
  indices,
  targetLab,
  { kind = "hkl", frame = "lab", Z = null } = {},
) {
  const v = crystalVector(B, indices, kind);
  if (norm(v) < 1e-12) return null;
  const Zm = frame === "phi" ? eye3() : Z || eye3();
  const w = matVec(matMul(Zm, U), v);
  const R = rotationBetween(w, targetLab);
  return matMulN(transpose(Zm), R, Zm, U);
}

/**
 * Spin about `axisLab` to bring `indices` as close as it can to `targetLab`,
 * leaving the primary alignment untouched. Returns the new U, or null if the
 * direction is parallel to the axis and so carries no information.
 */
export function alignSecondaryInLab(
  U,
  B,
  indices,
  targetLab,
  axisLab,
  { kind = "hkl", frame = "lab", Z = null } = {},
) {
  const axis = unit(axisLab);
  if (norm(axis) < 1e-12) return null;
  const v = crystalVector(B, indices, kind);
  if (norm(v) < 1e-12) return null;
  const Zm = frame === "phi" ? eye3() : Z || eye3();
  const w = matVec(matMul(Zm, U), v);

  let wp = sub(w, scale(axis, dot(w, axis)));
  let tp = sub(targetLab, scale(axis, dot(targetLab, axis)));
  if (norm(wp) < 1e-9 || norm(tp) < 1e-9) return null;
  wp = unit(wp);
  tp = unit(tp);

  const angle = Math.atan2(dot(cross(wp, tp), axis), dot(wp, tp));
  const R = axisAngle(axis, angle);
  return matMulN(transpose(Zm), R, Zm, U);
}

/** UB with columns a*, b*, c* in the phi frame. */
export function UB(U, B, convention = "2pi", wavelength = 1) {
  const ub = matMul(U, B);
  if (convention === "2pi") return ub;
  if (convention === "1/d") return ub.map((r) => r.map((v) => v / TWO_PI));
  if (convention === "lambda")
    return ub.map((r) => r.map((v) => (v * wavelength) / TWO_PI));
  throw new Error(`unknown convention ${convention}`);
}

// ---------------------------------------------------------------------------
// Inverse problems
// ---------------------------------------------------------------------------

/** Q of one hkl in the lab frame at the given angles. */
export function qLab(B, U, indices, { mu, eta, chi, phi }) {
  const Z = sampleMatrix(mu, eta, chi, phi);
  return matVec(matMul(Z, U), matVec(B, indices));
}

/** |k_i + Q| - k at a trial eta, other circles held. */
export function excitationError(B, U, indices, angles, eta, wavelength) {
  const k = TWO_PI / wavelength;
  const q = qLab(B, U, indices, { ...angles, eta });
  return Math.hypot(q[0], k + q[1], q[2]) - k;
}

/**
 * Every eta in [-180, 180) that puts `indices` on the Ewald sphere, found by
 * scanning for sign changes and bisecting. Bisection rather than Brent keeps
 * this dependency-free; the tolerance is far below anything that matters.
 */
export function solveEta(B, U, indices, angles, wavelength, nGrid = 1441) {
  const f = (e) => excitationError(B, U, indices, angles, e, wavelength);
  const roots = [];
  let prevX = -180,
    prevY = f(-180);
  for (let i = 1; i < nGrid; i++) {
    const x = -180 + (360 * i) / (nGrid - 1);
    const y = f(x);
    if (prevY === 0) roots.push(prevX);
    else if (prevY * y < 0) roots.push(bisect(f, prevX, x));
    prevX = x;
    prevY = y;
  }
  const uniq = [];
  for (const e of roots.sort((a, b) => a - b))
    if (!uniq.length || Math.abs(e - uniq[uniq.length - 1]) > 1e-4)
      uniq.push(e);
  return uniq;
}

function bisect(f, lo, hi, tol = 1e-10, maxIter = 200) {
  let flo = f(lo);
  for (let i = 0; i < maxIter; i++) {
    const mid = 0.5 * (lo + hi);
    const fm = f(mid);
    if (fm === 0 || (hi - lo) / 2 < tol) return mid;
    if (flo * fm < 0) hi = mid;
    else {
      lo = mid;
      flo = fm;
    }
  }
  return 0.5 * (lo + hi);
}

/**
 * Whether `indices` can be rocked onto the Ewald sphere with eta, and why not.
 *
 * Rotating one circle cannot change the component of Q along its axis, so only
 * the perpendicular part can swing into place. Splitting the Bragg condition
 * k_i . Q = -|Q|^2 / 2 about the eta axis gives an achievable interval; the
 * blind cone is everything outside it.
 */
export function etaReach(B, U, indices, { mu, chi, phi }, wavelength) {
  const k = TWO_PI / wavelength;
  const Zin = sampleMatrix(0, 0, chi, phi);
  const v = matVec(matMul(Zin, U), matVec(B, indices));
  const Q = norm(v);
  if (Q < 1e-12)
    return {
      feasible: false,
      Q: 0,
      required: 0,
      lo: 0,
      hi: 0,
      shortfall: 0,
      inLimitingSphere: false,
    };

  const e = [0, 0, -1]; // the eta axis
  const u = matVec(transpose(sampleMatrix(mu, 0, 0, 0)), BEAM);
  const vPar = dot(v, e),
    uPar = dot(u, e);
  const vPerp = Math.sqrt(Math.max(Q * Q - vPar * vPar, 0));
  const uPerp = Math.sqrt(Math.max(1 - uPar * uPar, 0));

  const centre = uPar * vPar;
  const swing = uPerp * vPerp;
  const required = -(Q * Q) / (2 * k);
  const lo = centre - swing,
    hi = centre + swing;
  return {
    feasible: lo <= required && required <= hi,
    Q,
    required,
    lo,
    hi,
    shortfall: Math.max(lo - required, required - hi, 0),
    inLimitingSphere: Q <= 2 * k,
  };
}

/** A chi that makes `indices` reachable by rocking eta, nearest to the current. */
export function suggestChi(B, U, indices, angles, wavelength, nGrid = 721) {
  let best = null,
    bestCost = Infinity;
  for (let i = 0; i < nGrid; i++) {
    const c = -180 + (360 * i) / (nGrid - 1);
    if (!etaReach(B, U, indices, { ...angles, chi: c }, wavelength).feasible)
      continue;
    const d = Math.abs(((((c - angles.chi + 180) % 360) + 360) % 360) - 180);
    if (d < bestCost) {
      best = c;
      bestCost = d;
    }
  }
  return best;
}

/**
 * (delta, gamma) in degrees that centre the arm on a direction.
 * The arm points along (sin d, cos g cos d, sin g cos d), so this inverts it.
 */
export function detectorAnglesFor(khat) {
  const d = Math.asin(Math.min(1, Math.max(-1, khat[0])));
  return { delta: deg(d), gamma: deg(Math.atan2(khat[2], khat[1])) };
}

/** (delta, gamma) that put `indices` on the beam centre at the current angles. */
export function aimDetectorAt(B, U, indices, angles, wavelength) {
  const k = TWO_PI / wavelength;
  const q = qLab(B, U, indices, angles);
  const kf = [q[0], k + q[1], q[2]];
  if (norm(kf) < 1e-9) return { delta: angles.delta, gamma: angles.gamma };
  return detectorAnglesFor(unit(kf));
}

/** True if |Q| <= 2k, i.e. inside the limiting sphere. */
export const reachable = (B, indices, wavelength) =>
  norm(matVec(B, indices)) <= 2 * (TWO_PI / wavelength);
