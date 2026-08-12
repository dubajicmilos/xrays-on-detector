"""Virtual six-circle instrument: state, physics and the motor solvers.

Built on the package core: :mod:`xrays_on_detector.geometry` supplies the You
(1999) circle matrices, :mod:`xrays_on_detector.ewald` the Ewald construction and
:mod:`xrays_on_detector.render` the detector image. This module adds only what a
live instrument needs on top of that:

* a lattice-only crystal, so the app runs with no CIF loaded;
* the transmission / reflection distinction (a sample surface that blocks rays);
* detector pixel binning for interactive frame rates;
* the inverse problems a beamline user actually solves: which eta brings a given
  hkl onto the Ewald sphere, and where the detector arm must go to catch it.

Lab frame throughout is the You / diffcalc one used by the core package:
``+x`` vertical (up), ``+y`` along the incident beam, ``+z`` horizontal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import asin, atan2, cos, degrees, radians, sin, sqrt

import numpy as np

from ..detector import Detector
from ..ewald import Reflections, excite
from ..geometry import BEAM, sample_matrix
from ..render import render

# --------------------------------------------------------------------------
# Detector with lab-natural pixel axes
# --------------------------------------------------------------------------


class LabDetector(Detector):
    """A :class:`~xrays_on_detector.detector.Detector` mounted the way a real
    one is, and read out in the beam's-eye view.

    The core class puts fast along lab ``+x``, which in the You frame is the
    *vertical*, so an image displayed straight from it appears rotated by 90
    degrees relative to how the detector is physically mounted. Overriding
    :meth:`frame` is enough to fix that, because ``project`` and ``max_Qmax``
    both go through it. The core class is left untouched so nothing already
    validated against it changes.

    The fast axis is ``-z``, not ``+z``. Displayed with column 0 on the left
    and row 0 (the largest slow) at the top, that is what an observer standing
    at the sample and looking downstream sees: ``e_fast x e_slow = -arm``, the
    same handedness as :mod:`xrays_on_detector.realframe`. With ``+z`` the
    frame would be laid out as seen from behind the panel, so the pattern in
    the 2D view came out mirrored against the 3D scene it was taken from.
    """

    def frame(self):
        from ..geometry import detector_matrix

        R = detector_matrix(self.nu, self.delta)
        arm = R @ BEAM
        centre = self.distance * arm
        normal = -arm
        e_fast = R @ np.array([0.0, 0.0, -1.0])  # horizontal, right looking downstream
        e_slow = R @ np.array([1.0, 0.0, 0.0])   # vertical (up)
        return centre, normal, e_fast, e_slow, arm

    def binned(self, factor: int) -> "LabDetector":
        """A geometrically identical detector with `factor` x `factor` pixels
        merged, for fast preview rendering."""
        if factor <= 1:
            return self
        return LabDetector(
            distance=self.distance,
            n_fast=max(1, self.n_fast // factor),
            n_slow=max(1, self.n_slow // factor),
            pixel_size=self.pixel_size * factor,
            nu=self.nu,
            delta=self.delta,
            beam_center_fast=self.beam_center_fast / factor,
            beam_center_slow=self.beam_center_slow / factor,
        )


# --------------------------------------------------------------------------
# Lattice-only crystal (no CIF needed)
# --------------------------------------------------------------------------


def b_matrix(a, b, c, alpha, beta, gamma) -> np.ndarray:
    """Reciprocal matrix in the crystal Cartesian frame, 2*pi convention.

    Returns B with ``Q_cart = B @ (h, k, l)`` and ``|Q| = 2*pi/d``, matching
    :attr:`xrays_on_detector.crystal.Crystal.B`.
    """
    al, be, ga = radians(alpha), radians(beta), radians(gamma)
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * cos(ga), b * sin(ga), 0.0])
    cx = c * cos(be)
    cy = c * (cos(al) - cos(be) * cos(ga)) / sin(ga)
    cz2 = c * c - cx * cx - cy * cy
    if cz2 <= 0:
        raise ValueError("cell angles do not describe a real lattice")
    vc = np.array([cx, cy, sqrt(cz2)])
    A = np.column_stack([va, vb, vc])          # direct lattice, columns a b c
    return 2.0 * np.pi * np.linalg.inv(A).T    # columns a* b* c*, 2*pi scaled


@dataclass
class LatticeCrystal:
    """Duck-types :class:`~xrays_on_detector.crystal.Crystal` with no structure.

    Every reflection is allowed with ``|F|^2`` set by an isotropic Debye-Waller
    falloff only, ``exp(-2 B_iso (|Q|/4pi)^2)``. This is not a structure factor
    and is never presented as one: it exists so the instrument is usable before
    a CIF is loaded, and so high-angle spots fade the way real ones do.
    """

    cell: dict
    B: np.ndarray
    B_iso: float = 1.5
    n_atoms: int = 0
    name: str = "lattice only"

    @classmethod
    def from_cell(cls, a, b, c, alpha=90.0, beta=90.0, gamma=90.0, name="lattice only"):
        cell = dict(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
        return cls(cell=cell, B=b_matrix(a, b, c, alpha, beta, gamma), name=name)

    def q_cryst(self, hkl: np.ndarray) -> np.ndarray:
        return np.asarray(hkl, dtype=float) @ self.B.T

    def structure_factor_mag2(self, hkl: np.ndarray) -> np.ndarray:
        Q = np.linalg.norm(self.q_cryst(hkl), axis=1)
        return np.exp(-2.0 * self.B_iso * (Q / (4.0 * np.pi)) ** 2)

    def hkl_within_Qmax(self, Q_max: float) -> np.ndarray:
        Binv = np.linalg.inv(self.B)
        hmax = np.ceil(Q_max * np.linalg.norm(Binv, axis=1)).astype(int) + 1
        ranges = [np.arange(-m, m + 1) for m in hmax]
        H, K, L = np.meshgrid(*ranges, indexing="ij")
        hkl = np.stack([H.ravel(), K.ravel(), L.ravel()], axis=1)
        hkl = hkl[np.any(hkl != 0, axis=1)]
        keep = np.linalg.norm(self.q_cryst(hkl), axis=1) <= Q_max
        return hkl[keep]


# --------------------------------------------------------------------------
# Small rotation helpers
# --------------------------------------------------------------------------


def rotation_between(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Smallest rotation matrix taking unit vector `v_from` onto `v_to`."""
    a = np.asarray(v_from, float)
    b = np.asarray(v_to, float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # antiparallel: rotate by pi about any axis perpendicular to a
        perp = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * K @ K
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def euler_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Extrinsic rotation about the lab x, then y, then z axes (degrees)."""
    from diffcalc.util import x_rotation, y_rotation, z_rotation

    def R(m):
        return np.asarray(m, dtype=float)

    return (R(z_rotation(radians(rz)))
            @ R(y_rotation(radians(ry)))
            @ R(x_rotation(radians(rx))))


# --------------------------------------------------------------------------
# Simulation result
# --------------------------------------------------------------------------


@dataclass
class Shot:
    image: np.ndarray                # (n_slow, n_fast) preview image
    table: list                      # reflections that landed on the panel
    refl: Reflections                # everything near the Ewald sphere (unblocked)
    blocked: Reflections | None      # reflections killed by the sample surface
    detector: LabDetector            # the (possibly binned) detector used
    alpha_deg: float                 # incidence angle onto the sample surface
    n_near: int                      # reflections near the sphere before blocking


# --------------------------------------------------------------------------
# The instrument
# --------------------------------------------------------------------------


@dataclass
class Instrument:
    """Complete state of the virtual diffractometer."""

    # circles (degrees)
    mu: float = 0.0
    eta: float = 0.0
    chi: float = 0.0
    phi: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0

    # beam and detector
    wavelength: float = 0.7293       # Angstrom
    distance: float = 200.0          # mm
    n_fast: int = 1475
    n_slow: int = 1679
    pixel_size: float = 0.172        # mm
    preview_bin: int = 4

    # sample
    crystal: object = None
    U: np.ndarray = field(default_factory=lambda: np.eye(3))
    sigma: float = 0.010             # reciprocal-space peak width, 1/A (2*pi)
    mode: str = "transmission"       # or "reflection"
    surface_hkl: tuple = (0, 0, 1)

    # reflection list cache
    hkl: np.ndarray | None = None
    Fmag2: np.ndarray | None = None

    polarization_mode: str = "horizontal"
    n_sigma: float = 4.0

    def __post_init__(self):
        if self.crystal is None:
            self.crystal = LatticeCrystal.from_cell(5.917, 5.917, 5.917,
                                                    name="pseudocubic 5.917 A")

    # -- geometry ---------------------------------------------------------

    def detector_obj(self, bin_factor: int | None = None) -> LabDetector:
        det = LabDetector(
            distance=self.distance,
            n_fast=self.n_fast,
            n_slow=self.n_slow,
            pixel_size=self.pixel_size,
            nu=self.gamma,
            delta=self.delta,
        )
        b = self.preview_bin if bin_factor is None else bin_factor
        return det.binned(b)

    def sample_M(self) -> np.ndarray:
        return sample_matrix(self.mu, self.eta, self.chi, self.phi)

    def surface_normal_cryst(self) -> np.ndarray:
        n = self.crystal.B @ np.asarray(self.surface_hkl, float)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            return np.array([0.0, 0.0, 1.0])
        return n / norm

    def surface_normal_lab(self) -> np.ndarray:
        return self.sample_M() @ self.U @ self.surface_normal_cryst()

    def alpha_deg(self) -> float:
        """Incidence angle of the beam onto the sample surface (degrees).

        Positive means the beam strikes the front face, so a reflection
        experiment is possible.
        """
        n = self.surface_normal_lab()
        return degrees(asin(float(np.clip(-np.dot(BEAM, n), -1.0, 1.0))))

    # -- reflection list --------------------------------------------------

    def q_max(self) -> float:
        return self.detector_obj(1).max_Qmax(self.wavelength)

    def build_reflection_list(self, progress=None) -> int:
        """Generate hkl within the detector's Q range and their |F|^2.

        `progress` is an optional callable taking (done, total); the CIF path
        is a per-reflection Python loop into pytilting and can take seconds.
        """
        hkl = self.crystal.hkl_within_Qmax(self.q_max())
        if progress is None or isinstance(self.crystal, LatticeCrystal):
            Fmag2 = self.crystal.structure_factor_mag2(hkl)
        else:
            Fmag2 = np.empty(len(hkl))
            chunk = max(1, len(hkl) // 100)
            for i in range(0, len(hkl), chunk):
                Fmag2[i:i + chunk] = self.crystal.structure_factor_mag2(
                    hkl[i:i + chunk])
                progress(min(i + chunk, len(hkl)), len(hkl))
        self.hkl, self.Fmag2 = hkl, Fmag2
        return len(hkl)

    # -- forward simulation ----------------------------------------------

    def _select(self, refl: Reflections, mask: np.ndarray) -> Reflections:
        return Reflections(
            hkl=refl.hkl[mask], khat=refl.khat[mask], eps=refl.eps[mask],
            excitation=refl.excitation[mask], Fmag2=refl.Fmag2[mask],
            two_theta=refl.two_theta[mask],
        )

    def shoot(self, bin_factor: int | None = None) -> Shot:
        """Simulate one frame at the current motor positions."""
        if self.hkl is None:
            self.build_reflection_list()

        det = self.detector_obj(bin_factor)
        Z = self.sample_M()
        refl = excite(self.crystal, self.U, Z, self.wavelength, self.sigma,
                      self.hkl, self.Fmag2, n_sigma=self.n_sigma)
        n_near = len(refl.hkl)

        blocked = None
        alpha = self.alpha_deg()
        if self.mode == "reflection" and n_near:
            n_lab = self.surface_normal_lab()
            beta = refl.khat @ n_lab            # sin(exit angle above surface)
            escapes = beta > 0.0
            if alpha <= 0.0:
                escapes[:] = False              # beam is under the surface
            blocked = self._select(refl, ~escapes)
            refl = self._select(refl, escapes)

        image, table = render(det, refl, self.wavelength, self.sigma,
                              polarization_mode=self.polarization_mode)
        return Shot(image=image, table=table, refl=refl, blocked=blocked,
                    detector=det, alpha_deg=alpha, n_near=n_near)

    # -- inverse problems -------------------------------------------------

    def q_lab(self, hkl, mu=None, eta=None, chi=None, phi=None) -> np.ndarray:
        """Scattering vector of one hkl in the lab frame, at given angles."""
        Z = sample_matrix(
            self.mu if mu is None else mu,
            self.eta if eta is None else eta,
            self.chi if chi is None else chi,
            self.phi if phi is None else phi,
        )
        return Z @ self.U @ (self.crystal.B @ np.asarray(hkl, float))

    def excitation_error(self, hkl, eta: float) -> float:
        """|k_i + Q| - k at eta, with the other sample circles held."""
        k = 2.0 * np.pi / self.wavelength
        kf = k * BEAM + self.q_lab(hkl, eta=eta)
        return float(np.linalg.norm(kf) - k)

    def eta_reach(self, hkl, mu=None, chi=None, phi=None) -> dict:
        """Why an hkl can or cannot be rocked onto the Ewald sphere with eta.

        Rotating one circle moves Q on a cone about that axis, so the component
        of Q along the axis is fixed and only the perpendicular part can swing
        into position. Writing the Bragg condition as ``k_i . Q = -|Q|^2 / 2``
        and splitting Q about the eta axis gives an achievable interval for
        ``k_i . Q``; a solution exists exactly when the required value lies in
        it. Everything outside is the blind cone of a single-axis rotation.

        Returns the required value, the achievable interval and the shortfall.
        """
        k = 2.0 * np.pi / self.wavelength
        mu = self.mu if mu is None else mu
        chi = self.chi if chi is None else chi
        phi = self.phi if phi is None else phi

        # v is what ETA actually rotates: everything inside it is fixed.
        v = (sample_matrix(0.0, 0.0, chi, phi) @ self.U
             @ (self.crystal.B @ np.asarray(hkl, float)))
        Q = float(np.linalg.norm(v))
        if Q < 1e-12:
            return dict(feasible=False, Q=0.0, required=0.0, lo=0.0, hi=0.0,
                        shortfall=0.0, in_limiting_sphere=False)

        # ETA turns about e; MU is applied after it, so fold MU into the beam.
        e = np.array([0.0, 0.0, -1.0])
        u = sample_matrix(mu, 0.0, 0.0, 0.0).T @ BEAM

        v_par, u_par = float(v @ e), float(u @ e)
        v_perp = np.sqrt(max(Q ** 2 - v_par ** 2, 0.0))
        u_perp = np.sqrt(max(1.0 - u_par ** 2, 0.0))

        centre = u_par * v_par
        swing = u_perp * v_perp
        required = -(Q ** 2) / (2.0 * k)
        lo, hi = centre - swing, centre + swing
        return dict(
            feasible=lo <= required <= hi,
            Q=Q, required=required, lo=lo, hi=hi,
            shortfall=max(lo - required, required - hi, 0.0),
            in_limiting_sphere=Q <= 2 * k,
        )

    def suggest_chi(self, hkl, n_grid: int = 721) -> float | None:
        """A chi that makes `hkl` reachable by rocking eta, nearest to current.

        This is what you do on the floor when a reflection sits in the blind
        cone: move an outer circle until it comes into reach.
        """
        best, best_cost = None, np.inf
        for c in np.linspace(-180.0, 180.0, n_grid):
            if not self.eta_reach(hkl, chi=float(c))["feasible"]:
                continue
            d = abs((c - self.chi + 180.0) % 360.0 - 180.0)
            if d < best_cost:
                best, best_cost = float(c), d
        return best

    def direct_axes_lab(self) -> tuple:
        """Unit vectors of the direct cell axes a, b, c in the lab frame.

        Returns (unit vectors as columns, lengths in Angstrom). The direct
        lattice is ``A = 2*pi * inv(B).T``; at U = I and all circles zero the
        crystal Cartesian frame is the phi frame, so a points along lab +x.
        """
        A = 2.0 * np.pi * np.linalg.inv(self.crystal.B).T
        lab = self.sample_M() @ self.U @ A
        lengths = np.linalg.norm(lab, axis=0)
        units = lab / np.where(lengths > 1e-12, lengths, 1.0)
        return units, lengths

    def solve_eta(self, hkl, n_grid: int = 1441) -> list:
        """All eta in [-180, 180) that put `hkl` on the Ewald sphere.

        The other three sample circles stay where they are, which is what you
        do at a beamline: rock one motor and watch the peak come up.
        """
        from scipy.optimize import brentq

        grid = np.linspace(-180.0, 180.0, n_grid)
        vals = np.array([self.excitation_error(hkl, e) for e in grid])
        out = []
        for i in range(len(grid) - 1):
            if vals[i] == 0.0:
                out.append(float(grid[i]))
            elif vals[i] * vals[i + 1] < 0:
                out.append(float(brentq(lambda e: self.excitation_error(hkl, e),
                                        grid[i], grid[i + 1], xtol=1e-9)))
        # de-duplicate solutions that fall within a hair of each other
        uniq = []
        for e in sorted(out):
            if not uniq or abs(e - uniq[-1]) > 1e-4:
                uniq.append(e)
        return uniq

    @staticmethod
    def detector_angles_for(khat: np.ndarray) -> tuple:
        """(delta, gamma) in degrees that centre the arm on direction `khat`.

        The arm points along ``(sin d, cos g cos d, sin g cos d)``, so this
        inverts directly.
        """
        kx, ky, kz = (float(v) for v in khat)
        d = asin(float(np.clip(kx, -1.0, 1.0)))
        return degrees(d), degrees(atan2(kz, ky))

    def aim_detector_at(self, hkl) -> tuple:
        """(delta, gamma) that put `hkl` on the beam centre, at current angles."""
        k = 2.0 * np.pi / self.wavelength
        kf = k * BEAM + self.q_lab(hkl)
        n = np.linalg.norm(kf)
        if n < 1e-9:
            return self.delta, self.gamma
        return self.detector_angles_for(kf / n)

    # -- orientation ------------------------------------------------------

    def set_mount(self, rx: float, ry: float, rz: float) -> None:
        """Set U from three mount misorientation angles about the lab axes."""
        self.U = euler_matrix(rx, ry, rz)

    def A_matrix(self) -> np.ndarray:
        """Direct lattice with a, b, c as columns, in the crystal frame."""
        return 2.0 * np.pi * np.linalg.inv(self.crystal.B).T

    def crystal_vector(self, indices, kind: str = "hkl") -> np.ndarray:
        """A crystal direction as a Cartesian vector in the crystal frame.

        kind='hkl' gives the reciprocal vector, i.e. the normal to the (hkl)
        planes. kind='uvw' gives the real-space direction [uvw]. They coincide
        for cubic lattices and generally do not otherwise, so the caller has to
        say which one is meant.
        """
        idx = np.asarray(indices, float)
        if kind == "hkl":
            return self.crystal.B @ idx
        if kind == "uvw":
            return self.A_matrix() @ idx
        raise ValueError(f"kind must be 'hkl' or 'uvw', got {kind!r}")

    def _frame_matrix(self, frame: str) -> np.ndarray:
        """Z for 'lab' (align at the current motor positions) or I for 'phi'
        (align at the goniometer datum, i.e. how the crystal is mounted)."""
        if frame == "lab":
            return self.sample_M()
        if frame == "phi":
            return np.eye(3)
        raise ValueError(f"frame must be 'lab' or 'phi', got {frame!r}")

    def align_in_lab(self, indices, target_lab, kind: str = "hkl",
                     frame: str = "lab") -> bool:
        """Rotate the crystal so `indices` points along `target_lab`.

        With frame='lab' the alignment holds at the current motor positions: if
        ``w = Z U v`` is where the direction currently points in the lab and R
        takes w onto the target, then ``U_new = Z^T R Z U``. Moving the circles
        afterwards carries the crystal with them, exactly as it would in the
        hutch. With frame='phi' it describes how the crystal sits on the mount,
        so the alignment is true when every circle reads zero.
        """
        v = self.crystal_vector(indices, kind)
        if np.linalg.norm(v) < 1e-12:
            return False
        Z = self._frame_matrix(frame)
        w = Z @ self.U @ v
        R = rotation_between(w, np.asarray(target_lab, float))
        self.U = Z.T @ R @ Z @ self.U
        return True

    def align_secondary_in_lab(self, indices, target_lab, axis_lab,
                               kind: str = "hkl", frame: str = "lab") -> bool:
        """Spin the crystal about `axis_lab` to bring `indices` toward a target.

        Aligning one direction leaves a free rotation about it. This removes
        that, by turning about the primary axis until the secondary direction's
        component perpendicular to the axis lines up as closely as it can. The
        primary alignment is preserved exactly because the rotation is about it.
        """
        axis = np.asarray(axis_lab, float)
        n = np.linalg.norm(axis)
        if n < 1e-12:
            return False
        axis = axis / n

        v = self.crystal_vector(indices, kind)
        if np.linalg.norm(v) < 1e-12:
            return False
        Z = self._frame_matrix(frame)
        w = Z @ self.U @ v
        t = np.asarray(target_lab, float)

        # components perpendicular to the axis
        wp = w - axis * float(w @ axis)
        tp = t - axis * float(t @ axis)
        if np.linalg.norm(wp) < 1e-9 or np.linalg.norm(tp) < 1e-9:
            return False           # secondary is parallel to the axis: no info
        wp /= np.linalg.norm(wp)
        tp /= np.linalg.norm(tp)

        angle = np.arctan2(float(np.cross(wp, tp) @ axis), float(wp @ tp))
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = (np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K))
        self.U = Z.T @ R @ Z @ self.U
        return True

    def UB(self, convention: str = "2pi") -> np.ndarray:
        """The orientation matrix UB, columns a*, b*, c* in the phi frame.

        '2pi'    package convention, |a*| = 2*pi/a
        '1/d'    crystallographic, |a*| = 1/a  (CrysAlisPro, rspace3d)
        'lambda' 1/d scaled by the wavelength, |a*| = lambda/a (CrysAlis UB)
        """
        ub = self.U @ self.crystal.B
        if convention == "2pi":
            return ub
        if convention == "1/d":
            return ub / (2.0 * np.pi)
        if convention == "lambda":
            return ub * self.wavelength / (2.0 * np.pi)
        raise ValueError(f"unknown convention {convention!r}")

    def reachable(self, hkl) -> bool:
        """True if |Q| <= 2k, i.e. the reflection is inside the limiting sphere."""
        k = 2.0 * np.pi / self.wavelength
        return float(np.linalg.norm(self.crystal.B @ np.asarray(hkl, float))) <= 2 * k
