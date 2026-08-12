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
    n_fast: int              # number of horizontal pixels
    n_slow: int              # number of vertical pixels
    pixel_size: float        # square pixel edge length
    nu: float = 0.0          # vertical detector circle (deg)
    delta: float = 0.0       # horizontal detector circle (deg)
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
        """Largest |Q| (2*pi convention) reachable at any detector corner."""
        k = 2.0 * np.pi / wavelength
        centre, _, e_fast, e_slow, _ = self.frame()
        half_f = 0.5 * self.n_fast * self.pixel_size
        half_s = 0.5 * self.n_slow * self.pixel_size
        corners = [
            centre + sf * half_f * e_fast + ss * half_s * e_slow
            for sf in (-1, 1) for ss in (-1, 1)
        ]
        two_theta = [
            np.arccos(np.clip((c / np.linalg.norm(c)) @ BEAM, -1, 1)) for c in corners
        ]
        return 2.0 * k * np.sin(max(two_theta) / 2.0)
