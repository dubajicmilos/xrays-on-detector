"""Simulate and index *real* area-detector frames from a rotation experiment.

This complements the synthetic six-circle model (simulate.py) with the geometry
of a standard flat area detector sitting on the beam axis (2theta = 0), driven
by an external UB (e.g. from CrysAlisPro / an rspace3d reconstruction) and a
single oscillation axis. It was validated against Diamond I19-2 MAPbBr3 Eiger
CBF frames (see examples/validate_I19-2_realframe.py).

Lab frame: beam +z (source -> sample -> detector), detector fast axis +x,
slow axis +y, detector plane at z = distance. Units are the crystallographic
1/d convention to match a CrysAlisPro UB (UB/lambda gives a*,b*,c* with
|a*| = 1/a); the physics is convention-independent as long as k = 1/lambda.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FlatDetector:
    """Flat area detector on the beam axis (2theta = 0)."""
    distance: float                 # sample -> detector (same unit as pixel_size)
    pixel_size: float
    beam_center: tuple              # (fast_x, slow_y) in pixels
    shape: tuple                    # (n_slow, n_fast)
    wavelength: float               # Angstrom
    beam: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    fast: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    slow: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))

    @classmethod
    def from_eiger_cbf(cls, path: str):
        """Build geometry + read the goniometer angles from a Dectris miniCBF.

        Returns (detector, angles, image) where angles is a dict of the header
        goniometer values (deg) and image is the int32 pixel array.
        """
        import fabio

        img = fabio.open(path)
        hdr = img.header.get("_array_data.header_contents", "")
        if not hdr:                                   # fall back to raw text
            with open(path, "rb") as f:
                hdr = f.read(4096).decode("latin-1", "replace")

        def num(key):
            import re
            m = re.search(rf"#\s*{re.escape(key)}\s+\(?\s*([-\d.eE]+)", hdr)
            return float(m.group(1)) if m else None

        def pair(key):
            import re
            m = re.search(rf"#\s*{re.escape(key)}\s*\(\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)",
                          hdr)
            return (float(m.group(1)), float(m.group(2))) if m else None

        dist_m = num("Detector_distance")
        pix_m = num("Pixel_size")
        bc = pair("Beam_xy")
        lam = num("Wavelength")
        ny, nx = img.data.shape
        det = cls(distance=dist_m * 1e3, pixel_size=pix_m * 1e3,
                  beam_center=bc, shape=(ny, nx), wavelength=lam)
        angles = {a: num(a) for a in ("Phi", "Kappa", "Omega", "Start_angle",
                                      "Angle_increment", "Detector_2theta")}
        return det, angles, img.data.astype(np.int32)

    def scattering_vectors(self, px: np.ndarray) -> np.ndarray:
        """Pixel (fast, slow) -> scattering vector r = s1 - s0 (1/Angstrom)."""
        px = np.atleast_2d(np.asarray(px, float))
        u = (px[:, 0] - self.beam_center[0]) * self.pixel_size
        v = (px[:, 1] - self.beam_center[1]) * self.pixel_size
        P = (u[:, None] * self.fast + v[:, None] * self.slow
             + self.distance * self.beam)
        s1 = P / np.linalg.norm(P, axis=1)[:, None] / self.wavelength
        return s1 - self.beam / self.wavelength

    def project(self, kf: np.ndarray):
        """Diffracted wavevector(s) kf (lab) -> pixel (fast_x, slow_y)."""
        kf = np.atleast_2d(np.asarray(kf, float))
        d = kf / np.linalg.norm(kf, axis=1)[:, None]
        denom = d @ self.beam
        t = self.distance / denom
        P = d * t[:, None]
        fast = self.beam_center[0] + (P @ self.fast) / self.pixel_size
        slow = self.beam_center[1] + (P @ self.slow) / self.pixel_size
        return fast, slow, denom > 0


def detect_peaks(image, beam_center, thr=200, rmin=170, smin=3, smax=200):
    """Label pixels above `thr`, return spot centroids (fast_x, slow_y)."""
    from scipy import ndimage

    lbl, n = ndimage.label(image > thr)
    sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    coms = ndimage.center_of_mass(image, lbl, range(1, n + 1))
    out = []
    for c, s in zip(coms, sizes):
        x, y = c[1], c[0]
        if smin <= s <= smax and np.hypot(x - beam_center[0], y - beam_center[1]) > rmin:
            out.append((x, y))
    return np.array(out)


def detector_display(image, bin_factor=2, smooth=0.7):
    """Return (binned_image, vmin, vmax) ready for a clean log-scaled plot.

    Photon-counting frames are ~99.9% single-photon noise with Bragg peaks in
    the rare top ~0.01%, so a naive vmin=1 buries the signal in speckle. This
    bins + lightly smooths, floors the display just above the noise, and caps it
    in the spot range. Plot with LogNorm(vmin, vmax) and origin='upper'.
    """
    from scipy import ndimage

    d = np.asarray(image, float).copy()
    d[d < 0] = np.nan                       # detector gaps / bad pixels
    ny, nx = d.shape
    b = bin_factor
    with np.errstate(invalid="ignore"):
        db = np.nanmean(d[:ny // b * b, :nx // b * b]
                        .reshape(ny // b, b, nx // b, b), axis=(1, 3))
    db = ndimage.gaussian_filter(np.nan_to_num(db, nan=0.0), smooth)
    finite = db[np.isfinite(db) & (db > 0)]
    vmin = max(3.0, np.percentile(finite, 99.5))
    vmax = np.percentile(finite, 99.995)
    return db, float(vmin), float(vmax)


def _kabsch(A, B):
    U, _, Vt = np.linalg.svd(A @ B.T)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


@dataclass
class IndexResult:
    R: np.ndarray            # crystal-to-lab rotation for the frame
    hkl: np.ndarray          # (N,3) int, per input peak
    inliers: np.ndarray      # (N,) bool
    rms: float               # 1/Angstrom


def index_frame(peaks, UB, detector: FlatDetector,
                tol=0.012, dtol=0.02, atol_deg=1.5, min_inliers=5,
                hkl_max=9) -> IndexResult | None:
    """Known-cell pair indexing: find R with r_obs = R @ (UB @ hkl).

    UB columns are a*,b*,c* in 1/d units (i.e. CrysAlis UB / lambda)."""
    r_obs = detector.scattering_vectors(peaks)
    UBinv = np.linalg.inv(UB)
    hs = np.arange(-hkl_max, hkl_max + 1)
    HKL = np.stack(np.meshgrid(hs, hs, hs, indexing="ij"), -1).reshape(-1, 3)
    HKL = HKL[np.any(HKL != 0, 1)]
    G = (UB @ HKL.T).T
    GN = np.linalg.norm(G, axis=1)

    vn = np.linalg.norm(r_obs, axis=1)
    order = np.argsort(vn)
    i0 = order[0]
    cand0 = np.nonzero(np.abs(GN - vn[i0]) < dtol)[0]
    best = None
    for i1 in order[1:]:
        cand1 = np.nonzero(np.abs(GN - vn[i1]) < dtol)[0]
        ang_obs = np.degrees(np.arccos(np.clip(
            r_obs[i0] @ r_obs[i1] / (vn[i0] * vn[i1]), -1, 1)))
        for a in cand0:
            for b in cand1:
                ang = np.degrees(np.arccos(np.clip(
                    G[a] @ G[b] / (GN[a] * GN[b]), -1, 1)))
                if abs(ang - ang_obs) > atol_deg:
                    continue
                R = _kabsch(np.stack([G[a], G[b]], 1),
                            np.stack([r_obs[i0], r_obs[i1]], 1))
                hkl = np.round((UBinv @ R.T @ r_obs.T).T)
                inl = np.linalg.norm((R @ (UB @ hkl.T)).T - r_obs, axis=1) < tol
                if inl.sum() < min_inliers:
                    continue
                R = _kabsch(UB @ hkl[inl].T, r_obs[inl].T)
                resid = np.linalg.norm((R @ (UB @ hkl.T)).T - r_obs, axis=1)
                inl = resid < tol
                rms = float(np.sqrt(np.mean(resid[inl] ** 2)))
                score = (int(inl.sum()), -rms)
                if best is None or score > best[0]:
                    best = (score, IndexResult(R, hkl.astype(int), inl, rms))
    return best[1] if best else None


def _axis_rot(axis, deg):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) * c + s * K + (1 - c) * np.outer(axis, axis)


def find_crysalis_par(folder):
    """Locate the CrysAlisPro .par carrying the refined UB.

    Prefers `*_cracker.par` (the refined orientation matrix) over a bare `*.par`:
    the latter can hold an automatic *reduced* cell (e.g. a monoclinic sub-cell)
    that will NOT index the pseudocubic perovskite frames, whereas the cracker par
    holds the pseudocubic UB actually used for the reconstruction.
    """
    import glob
    import os

    cracker = sorted(glob.glob(os.path.join(folder, "*_cracker.par")))
    if cracker:
        return cracker[0]
    plain = sorted(p for p in glob.glob(os.path.join(folder, "*.par"))
                   if not p.endswith("_cracker.par"))
    return plain[0] if plain else None


def read_crysalis_par(par_path):
    """Parse (UB, wavelength) from a CrysAlisPro .par file.

    UB is returned as columns a*,b*,c* in the 1/d convention (|column| = 1/d), i.e.
    the `CRYSTALLOGRAPHY UB` matrix (row-major 3x3, lambda-scaled) divided by the
    `CRYSTALLOGRAPHY WAVELENGTH`. Use find_crysalis_par to pick the right file; a
    reduced-cell par parses fine here but fails to index (wrong |a*|,|b*|,|c*|).
    """
    import re

    with open(par_path, "r", encoding="latin-1") as f:
        txt = f.read()
    m = re.search(r"CRYSTALLOGRAPHY UB[ \t]+([-\d][^\r\n]*)", txt)
    if not m:
        raise ValueError(f"no 'CRYSTALLOGRAPHY UB' line in {par_path}")
    ub = np.array([float(x) for x in m.group(1).split()[:9]]).reshape(3, 3)
    wm = re.search(r"CRYSTALLOGRAPHY WAVELENGTH[ \t]+([-\d.eE+]+)", txt)
    if wm is None:
        raise ValueError(f"no 'CRYSTALLOGRAPHY WAVELENGTH' line in {par_path}")
    wl = float(wm.group(1))
    return ub / wl, wl


def orient_from_frame(par_path, frame1_path, **index_kw):
    """Full geometry + orientation setup from a CrysAlisPro par + the first frame.

    Returns (detector, UB, R0, phi0):
      * detector, phi0 and the goniometer datum come from the CBF header;
      * UB (1/d) and the wavelength come from the par (read_crysalis_par);
      * R0 (the rotation linking the UB frame to the detector lab frame - the
        "missing rotation") is found by indexing frame 1.

    R0 is derived per dataset on purpose: across mounts it differs by a lattice-
    symmetry branch (e.g. a ~180 deg 4/mmm operation), so it is NOT safely reused
    from another crystal even on the same instrument, whereas the pixel->hkl map
    and the cell (UB) are shared. Indexing is fast (<1 s) and robust (rms ~0.003).
    """
    det, ang, _img = FlatDetector.from_eiger_cbf(frame1_path)
    UB, _wl = read_crysalis_par(par_path)
    res = index_frame(detect_peaks(_img, det.beam_center), UB, det, **index_kw)
    if res is None:
        raise RuntimeError(
            f"index_frame failed for {frame1_path}; check the UB cell "
            f"(use the *_cracker.par, not a reduced-cell *.par)")
    return det, UB, res.R, ang["Phi"]


def predict_recorded(R, UB, detector: FlatDetector, hkl, *, oscillation=0.2,
                     osc_axis=(0, 1, 0), sigma=0.004, n_sigma=3.0, substeps=9):
    """Which reflections cross the Ewald sphere during an oscillation, and where.

    Returns (hkl_rec, fast_px, slow_px, eps) for on-panel recorded reflections.
    hkl are the reciprocal-lattice indices for the supplied UB (use fractional
    indices, e.g. super/2, to place superlattice points on a parent UB)."""
    hkl = np.asarray(hkl, float)
    k = 1.0 / detector.wavelength
    ki = k * detector.beam
    ny, nx = detector.shape
    best = np.full(len(hkl), 1e9); bsub = np.zeros(len(hkl))
    for s_ in np.linspace(-oscillation / 2, oscillation / 2, substeps):
        r = (_axis_rot(osc_axis, s_) @ R @ (UB @ hkl.T)).T
        eps = np.linalg.norm(ki + r, axis=1) - k
        take = np.abs(eps) < np.abs(best); best[take] = eps[take]; bsub[take] = s_
    rec = np.abs(best) < n_sigma * sigma
    hkl_r, sub_r, eps_r = hkl[rec], bsub[rec], best[rec]
    fx = np.zeros(len(hkl_r)); fy = np.zeros(len(hkl_r))
    for i in range(len(hkl_r)):
        kf = ki + (_axis_rot(osc_axis, sub_r[i]) @ R @ (UB @ hkl_r[i]))
        a, b, _ = detector.project(kf)
        fx[i], fy[i] = a[0], b[0]
    on = (fx >= 0) & (fx < nx) & (fy >= 0) & (fy < ny)
    return hkl_r[on], fx[on], fy[on], eps_r[on]
