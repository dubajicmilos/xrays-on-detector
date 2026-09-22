/*! The Game of Diffraction · © 2026 Miloš Dubajić · MIT · https://github.com/dubajicmilos/xrays-on-detector */

/**
 * Pitch–phi surface diffractometer: geometry, in its own frame.
 *
 * The second machine of the site. The sample rides a pitch cradle (rotation
 * about the horizontal axis normal to the beam) that carries a phi spindle
 * about the sample normal; a roll ring about the beam is the outermost sample
 * axis. The detector is placed by a scattering angle (2theta) and an azimuth
 * about the beam, rather than by a (delta, gamma) arm.
 *
 * The instrument's frame, as its operators read it:
 *   +z along the beam, +y up, +x to the left looking downstream; right-handed.
 * The site's lab frame is +x up, +y along the beam, +z horizontal. The two
 * are related by the orthonormal GAME_TO_PP matrix below (x up -> +y_pp,
 * y beam -> +z_pp, z -> +x_pp); it is applied only where this instrument
 * meets the shared simulation and scene, never inside the physics here.
 *
 * Sample rotation (degrees):
 *   Z = Rz(roll) . Rx(-alpha) . Ry(phi)
 *   alpha  incidence angle between beam and surface: 0 = grazing, 90 = normal
 *   phi    right-handed about the sample normal (+y at pitch 0)
 *   roll   about the beam, outermost: shifts azimuth only, never the Bragg
 *          condition (Rz does not change z-components, and rotating sample
 *          and normal together leaves the exit angle invariant)
 * Detector: k_f at 2theta from +z; azimuth from +y, positive toward -x
 * (right-handed about +z) when azSign is +1, mirrored when -1.
 * Convention: 2*pi throughout, |Q| = 2*pi/d. k_i = k z, k = 2*pi/lambda.
 *
 * Framing note: B is the site's reciprocal matrix (x || a, z || c*  -- the
 * same construction as physics.js bMatrix), so a mount built here maps the
 * same crystal vectors the six-circle machine reads.
 *
 * Every function here is a faithful port of the standalone calculator the
 * user wrote and verified (199 cases against an independent oracle, 0
 * failures); test/parity_pp.mjs re-checks this file against that battery's
 * recorded outputs, so the two cannot drift apart silently.
 */

const DEG = Math.PI / 180;

// ---------------------------------------------------------------------------
// 3x3 helpers (row-major, same shape as the site's physics.js)
// ---------------------------------------------------------------------------

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const mv = (M, v) => [dot(M[0], v), dot(M[1], v), dot(M[2], v)];
const mm = (A, B) =>
  A.map((r) => [0, 1, 2].map((j) => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const norm = (a) => Math.hypot(a[0], a[1], a[2]);
const unit = (a) => {
  const n = norm(a);
  return n < 1e-300 ? [0, 0, 0] : [a[0] / n, a[1] / n, a[2] / n];
};

const rotX = (t) => {
  const c = Math.cos(t), s = Math.sin(t);
  return [[1, 0, 0], [0, c, -s], [0, s, c]];
};
const rotY = (t) => {
  const c = Math.cos(t), s = Math.sin(t);
  return [[c, 0, s], [0, 1, 0], [-s, 0, c]];
};
const rotZ = (t) => {
  const c = Math.cos(t), s = Math.sin(t);
  return [[c, -s, 0], [s, c, 0], [0, 0, 1]];
};

/** Wrap degrees into (-180, 180]; the +180 end is chosen so +/-180 are one value. */
export const wrap = (d) => {
  d = (((d + 180) % 360) + 360) % 360 - 180;
  return d === -180 ? 180 : d;
};

// ---------------------------------------------------------------------------
// Frame change between this instrument and the site's lab
// ---------------------------------------------------------------------------

/** Game lab -> pitch-phi frame: forward vectors x, y, z go to y_pp, z_pp, x_pp. */
export const GAME_TO_PP = [
  [0, 0, 1],
  [1, 0, 0],
  [0, 1, 0],
];

/** Pitch-phi frame -> game lab. Orthonormal, so it is the transpose. */
export const PP_TO_GAME = [
  [0, 1, 0],
  [0, 0, 1],
  [1, 0, 0],
];

export const gameToPp = (v) => mv(GAME_TO_PP, v);
export const ppToGame = (v) => mv(PP_TO_GAME, v);

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/** Sample orientation in the instrument's frame: Z = Rz(roll) Rx(-alpha) Ry(phi), degrees. */
export const sampleMatrix = (alpha, phi, roll) =>
  mm(rotZ(roll * DEG), mm(rotX(-alpha * DEG), rotY(phi * DEG)));

/**
 * The same orientation expressed in the shared lab frame:
 *   Z_game = G^T . Rz(roll) . Rx(-alpha) . Ry(phi) . G
 *          = R_y(roll) . R_z(-alpha) . R_x(phi)   (game axes, degrees).
 * test/parity_pp.mjs checks the two constructions agree.
 */
export const sampleMatrixGame = (alpha, phi, roll) =>
  mm(rotY(roll * DEG), mm(rotZ(-alpha * DEG), rotX(phi * DEG)));

/** The surface normal in the instrument's frame: +y rolled and pitched. */
export const surfaceNormal = (alpha, roll) =>
  mv(mm(rotZ(roll * DEG), rotX(-alpha * DEG)), [0, 1, 0]);

/**
 * Everything the cards show for one angle setting.
 *
 * QU: Q at zero angles, 2*pi/Angstrom, in the instrument's frame. alpha/phi/
 * roll in degrees. Returns excitation error (Angstrom^-1), 2theta, azimuth,
 * exit angle beta (degrees; both beam angles measured from the surface), the
 * unit diffracted direction, the lab Q and the normal, and the flags:
 * `flipped` when |alpha| > 90 (beam would enter through the back face) and
 * `ok` when the incident and diffracted beams both stay above the surface.
 */
export function evaluate(QU, alpha, phi, roll, k, azSign = 1) {
  const Z = sampleMatrix(alpha, phi, roll);
  const Q = mv(Z, QU);
  const n = surfaceNormal(alpha, roll);
  const kf = [Q[0], Q[1], k + Q[2]];
  const eps = norm(kf) - k;
  const kh = unit(kf);
  const tth = Math.acos(Math.max(-1, Math.min(1, kh[2]))) / DEG;
  const az = Math.atan2(-azSign * kh[0], kh[1]) / DEG;
  const beta = Math.asin(Math.max(-1, Math.min(1, dot(kh, n)))) / DEG;
  const ainc = wrap(alpha);
  const flipped = !(ainc >= -90 && ainc <= 90);
  return {
    Q,
    n,
    kf: kh,
    eps,
    tth,
    az,
    beta,
    alpha: ainc,
    phi,
    flipped,
    ok: ainc > 0 && !flipped && beta > 0,
  };
}

/** True when Q (at zero angles) is parallel to the surface normal. */
export const isSpecular = (QU) => Math.hypot(QU[0], QU[2]) < 1e-9;

/**
 * Fixed alpha -> the phi values (deg) that put QU on the Ewald sphere.
 *
 * The condition Q_z = -|Q|^2 / (2k) expands to
 *   cos(phi) q_z - sin(phi) q_x = c,  c = (qz0 + q_y sin a) / cos a,
 * which is rho cos(phi + delta) = c with rho, delta the polar coordinates of
 * (q_x, q_z). So phi = +-acos(c/rho) - delta: two branches, or none when
 * |c| > rho. Returns the string 'any' for a specular reflection at
 * alpha = theta_B (every phi works), or [] when nothing does.
 */
export function solvePhi(QU, alpha, k) {
  const qz0 = -dot(QU, QU) / (2 * k);
  const a = alpha * DEG;
  const c = (qz0 + QU[1] * Math.sin(a)) / Math.cos(a);
  const rho = Math.hypot(QU[0], QU[2]);
  const delta = Math.atan2(QU[0], QU[2]);
  if (rho < 1e-9) return Math.abs(c) < 1e-6 ? ['any'] : [];
  const x = c / rho;
  if (Math.abs(x) > 1) return [];
  const t = Math.acos(x);
  return [wrap((t - delta) / DEG), wrap((-t - delta) / DEG)];
}

/**
 * Fixed phi -> the alpha values (deg) that put QU on the Ewald sphere.
 *
 * With v = Ry(phi) QU the condition is -v_y sin a + v_z cos a = qz0, i.e.
 * R cos(a + eps) = qz0 for R = hypot(v_y, v_z), eps = atan2(v_y, v_z), so
 * a = +-acos(qz0/R) - eps. Both branches are returned; the caller decides
 * which are physical (sample not flipped, beams above the surface).
 */
export function solveAlpha(QU, phi, k) {
  const qz0 = -dot(QU, QU) / (2 * k);
  const v = mv(rotY(phi * DEG), QU);
  const R = Math.hypot(v[1], v[2]);
  if (R < 1e-12) return [];
  const eps = Math.atan2(v[1], v[2]);
  const x = qz0 / R;
  if (Math.abs(x) > 1) return [];
  const t = Math.acos(x);
  return [wrap((t - eps) / DEG), wrap((-t - eps) / DEG)];
}

/**
 * The angle settings that make QU diffract, in the calculator's three modes:
 * 'alpha' fixes the incidence and solves phi, 'phi' fixes phi and solves
 * alpha, 'explore' just evaluates what the user set. Solutions carry the
 * full readout plus `anyPhi` for the specular case. Duplicates (both
 * branches landing on one angle) are removed, and solvable settings sort
 * before blocked ones, nearest phi first.
 */
export function solutions(QU, mode, alpha, phi, roll, k, azSign = 1) {
  let out = [];
  if (mode === 'explore') return [evaluate(QU, alpha, phi, roll, k, azSign)];
  if (mode === 'alpha') {
    for (const p of solvePhi(QU, alpha, k)) {
      const ph = p === 'any' ? phi : p;
      const e = evaluate(QU, alpha, ph, roll, k, azSign);
      e.anyPhi = p === 'any';
      out.push(e);
    }
  } else {
    for (const a of solveAlpha(QU, phi, k)) out.push(evaluate(QU, a, phi, roll, k, azSign));
  }
  out = out.filter(
    (e, i) =>
      !out.slice(0, i).some(
        (o) =>
          Math.abs(wrap(o.alpha - e.alpha)) < 1e-6 &&
          Math.abs(wrap(o.phi - e.phi)) < 1e-6,
      ),
  );
  out.sort((a, b) => b.ok - a.ok || Math.abs(a.phi) - Math.abs(b.phi));
  return out;
}

/**
 * The ranges of incidence alpha in [0, 90] where some phi both satisfies
 * Bragg and keeps the two beams above the surface: the reachable band shown
 * on the card. A specular reflection reports only theta_B. Sampled every
 * 0.1 degrees, so the band is an indicator, not an exact gate.
 */
export function alphaBand(QU, k, roll) {
  const runs = [];
  let cur = null;
  for (let a = 0; a <= 90.0001; a += 0.1) {
    const ps = solvePhi(QU, a, k);
    const ok = ps.some((p) => evaluate(QU, a, p === 'any' ? 0 : p, roll, k).ok);
    if (ok) {
      if (!cur) cur = [a, a];
      else cur[1] = a;
    } else if (cur) {
      runs.push(cur);
      cur = null;
    }
  }
  if (cur) runs.push(cur);
  return runs;
}

// ---------------------------------------------------------------------------
// Detector mapping
// ---------------------------------------------------------------------------

/**
 * Unit direction of the diffracted beam for a detector set at (2theta, az).
 *
 * Inverts the azimuth definition: az = atan2(-azSign k_x, k_y), so
 * k = (-s sin(az) sin(2theta), cos(az) sin(2theta), cos(2theta)). The result
 * is in the instrument's frame; kfToAngles recovers (2theta, az) exactly.
 */
export function detectorDir(tth, az, azSign = 1) {
  const t = tth * DEG;
  const a = az * DEG;
  return [
    -azSign * Math.sin(a) * Math.sin(t),
    Math.cos(a) * Math.sin(t),
    Math.cos(t),
  ];
}

/** (2theta, azimuth) of a unit direction, matching evaluate()'s conventions. */
export function kfToAngles(kh, azSign = 1) {
  const tth = Math.acos(Math.max(-1, Math.min(1, kh[2]))) / DEG;
  const az = Math.atan2(-azSign * kh[0], kh[1]) / DEG;
  return { tth, az };
}

// ---------------------------------------------------------------------------
// Mount
// ---------------------------------------------------------------------------

/**
 * Zero-angle mount from a surface normal and an in-plane reference.
 *
 * nC is the surface normal as a crystal-frame vector (B . (hkl), unit taken
 * here); refVec is the reference as a crystal-frame vector (reciprocal or
 * direct, the caller chooses -- physics.js crystalVector does both). The
 * reference is projected into the surface plane first. Returns U, the
 * rotation taking the crystal into the instrument frame at zero angles:
 * nC -> +y (up), the reference -> the chosen in-plane direction.
 * Throws when the reference is parallel to the normal, which carries no
 * in-plane information.
 */
export function mountOrientation(nC, refVec, rdir = '+x') {
  const n = unit(nC);
  let r = sub(refVec, scale(n, dot(refVec, n)));
  if (norm(r) < 1e-9)
    throw new Error('reference direction is parallel to the surface normal');
  r = unit(r);
  const e3 = cross(n, r);
  const dir =
    { '+x': [1, 0, 0], '-x': [-1, 0, 0], '+z': [0, 0, 1], '-z': [0, 0, -1] }[rdir] ||
    [1, 0, 0];
  const f1 = [0, 1, 0];
  const f3 = cross(f1, dir);
  // rows n, r, e3 -> basis coordinates; those go to columns f1, dir, f3
  const M1 = [n.slice(), r.slice(), e3.slice()];
  const P = [
    [f1[0], dir[0], f3[0]],
    [f1[1], dir[1], f3[1]],
    [f1[2], dir[2], f3[2]],
  ];
  return mm(P, M1);
}

/**
 * UB in the instrument's native frame at zero angles, from a mount U given
 * in the game frame: G . U . B. Columns are a*, b*, c* (2*pi/Angstrom when B
 * is the site's bMatrix), so |column| = 2*pi/d.
 */
export const zeroAngleUB = (U, B) => mm(GAME_TO_PP, mm(U, B));

// ---------------------------------------------------------------------------
// Scene support
// ---------------------------------------------------------------------------

const transpose3 = (M) => [
  [M[0][0], M[1][0], M[2][0]],
  [M[0][1], M[1][1], M[2][1]],
  [M[0][2], M[1][2], M[2][2]],
];

/**
 * Orientation of the sample plate for the scene, in the shared lab frame.
 *
 * The plate is the crystal's surface: its thickness must lie along the datum
 * normal for ANY mount (the surface is a property of the machine; the mount
 * only says which crystal plane it is), while its in-plane azimuth must be
 * the crystal's, so phi visibly turns it. Built as: the datum frame (Z, the
 * mount removed from ZU), the shared slab geometry laid flat (rotZ(-90)
 * takes its thickness axis onto the normal), spun by the mount's twist about
 * the normal, which is exactly the crystal's azimuth at zero angles.
 *
 * With this, changing phi acts as an exact rotation of the plate about the
 * datum normal (the twist does not depend on the angles), which is what the
 * scene must show. test/parity_pp.mjs checks both properties.
 */
export function plateOrientation(ZU, U) {
  const Z = mm(ZU, transpose3(U));
  const psi = -Math.atan2(U[1][2] - U[2][1], U[1][1] + U[2][2]);
  // laid flat first (rotZ(-90) takes the slab's thickness axis onto the
  // normal), then spun about the normal by the mount's twist
  return mm(mm(Z, rotX(psi)), rotZ(-Math.PI / 2));
}
