"""Flat area detector on the (nu, delta) arm, and ray projection onto it.

The detector face is perpendicular to the arm direction and centred on it at
``distance`` from the sample. Pixel axes are the arm-rotated lab axes:
fast = +x, slow = +z. In the You frame +x is the *vertical*, so this panel is
mounted with its fast axis running up the wall; the virtual diffractometer
uses :class:`~xrays_on_detector.vdiff.instrument.LabDetector` instead, which
mounts it the way real ones are and reads it out in the beam's-eye view.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import BEAM, detector_matrix


@dataclass
class Detector:
    distance: float          # sample -> detector centre (same length unit as pixel_size)
    n_fast: int              # pixels along the fast axis
    n_slow: int              # pixels along the slow axis
    pixel_size: float        # square pixel edge length
    nu: float = 0.0          # detector circle about the vertical axis (deg): swings the arm horizontally
    delta: float = 0.0       # detector circle about the horizontal axis (deg): swings the arm vertically
    beam_center_fast: float | None = None   # pixel hit by the arm axis (default: centre)
    beam_center_slow: float | None = None

    def __post_init__(self):
        if self.beam_center_fast is None:
            self.beam_center_fast = (self.n_fast - 1) / 2.0
        if self.beam_center_slow is None:
            self.beam_center_slow = (self.n_slow - 1) / 2.0

    def frame(self):
        """Return (centre, normal, e_fast, e_slow, arm_dir) in the lab frame."""
        R = detector_matrix(self.nu, self.delta)
        arm = R @ BEAM                      # detector-centre direction from sample
        centre = self.distance * arm
        normal = -arm                       # faces the sample
        e_fast = R @ np.array([1.0, 0.0, 0.0])
        e_slow = R @ np.array([0.0, 0.0, 1.0])
        return centre, normal, e_fast, e_slow, arm

    def project(self, khat: np.ndarray):
        """Project unit diffracted directions onto the detector.

        Parameters
        ----------
        khat : (N, 3) ndarray
            Unit vectors along the diffracted beams.

        Returns
        -------
        fast_px, slow_px : (N,) ndarrays
            Sub-pixel coordinates (may fall outside the panel).
        inside : (N,) bool ndarray
            True where the ray hits the active area travelling forwards.
        cos_inc : (N,) ndarray
            Cosine of the incidence angle on the detector face.
        """
        khat = np.asarray(khat, dtype=float)
        centre, normal, e_fast, e_slow, arm = self.frame()

        denom = khat @ normal                 # < 0 when travelling toward the face
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (centre @ normal) / denom     # ray parameter, P = t * khat
        hit = t[:, None] * khat
        rel = hit - centre
        u = rel @ e_fast
        v = rel @ e_slow
        fast_px = self.beam_center_fast + u / self.pixel_size
        slow_px = self.beam_center_slow + v / self.pixel_size

        cos_inc = khat @ arm                  # = -(khat . normal)
        inside = (
            (denom < 0) & (t > 0)
            & (fast_px >= 0) & (fast_px <= self.n_fast - 1)
            & (slow_px >= 0) & (slow_px <= self.n_slow - 1)
        )
        return fast_px, slow_px, inside, cos_inc

    def max_Qmax(self, wavelength: float) -> float:
        """Largest |Q| (2*pi convention) reachable anywhere on the panel.

        The largest 2theta on the panel is not always at a corner: once the arm
        swings past 90 degrees it sits on an edge, and if the panel covers the
        back direction it is 180 degrees. Written as a point P = centre +
        u e_fast + v e_slow, cos(2theta) = (a + b u + c v) / sqrt(D^2 + u^2 +
        v^2), which along an edge with one coordinate fixed has a single
        stationary point in closed form, so the minimum over the rectangle is
        found exactly from the corners, those points and the pierce point.
        The panel extends half a pixel beyond the outermost pixel centres.
        """
        k = 2.0 * np.pi / wavelength
        _, _, e_fast, e_slow, arm = self.frame()
        D = self.distance
        a = D * float(arm @ BEAM)
        b = float(e_fast @ BEAM)
        c = float(e_slow @ BEAM)
        u0 = -(self.beam_center_fast + 0.5) * self.pixel_size
        u1 = (self.n_fast - 0.5 - self.beam_center_fast) * self.pixel_size
        v0 = -(self.beam_center_slow + 0.5) * self.pixel_size
        v1 = (self.n_slow - 0.5 - self.beam_center_slow) * self.pixel_size

        if a < 0:
            # The back direction -BEAM meets the panel plane at t = -D/(arm.BEAM).
            t = -D * D / a
            if u0 <= -t * b <= u1 and v0 <= -t * c <= v1:
                return 2.0 * k

        def cos2t(u, v):
            return (a + b * u + c * v) / np.sqrt(D * D + u * u + v * v)

        cands = [cos2t(u, v) for u in (u0, u1) for v in (v0, v1)]
        for v in (v0, v1):                       # edges of constant v
            ap = a + c * v
            if ap != 0:
                us = b * (D * D + v * v) / ap
                if u0 < us < u1:
                    cands.append(cos2t(us, v))
        for u in (u0, u1):                       # edges of constant u
            ap = a + b * u
            if ap != 0:
                vs = c * (D * D + u * u) / ap
                if v0 < vs < v1:
                    cands.append(cos2t(u, vs))
        two_theta = np.arccos(np.clip(min(cands), -1.0, 1.0))
        return 2.0 * k * np.sin(two_theta / 2.0)
