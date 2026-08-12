"""Six-circle diffractometer geometry, You (1999) convention.

The rotation matrices come from diffcalc-core so that the circle conventions,
and any UB / hkl<->angle work the user later does in diffcalc, stay consistent.

Lab frame (diffcalc / You):
    +x  vertical (up)
    +y  along the incident beam (downstream)
    +z  horizontal, perpendicular to the beam

That is the frame diffcalc implements, not the other way round: eta and delta
share an axis (both -z) as omega and 2theta do on a four-circle, chi turns
about the beam, and mu and nu share the vertical, which is why diffcalc calls
nu = 0 with delta free its "vertical 4-circle mode".

Sample circles (phi frame -> lab):   Z = MU . ETA . CHI . PHI
Detector arm (2 circles):            R_det = NU . DELTA
    delta swings the detector in the vertical (x, y) plane (about -z),
    nu    swings it horizontally (about +x); most beamlines call it gamma.
"""
from __future__ import annotations

from math import radians

import numpy as np
from diffcalc.hkl.geometry import (
    rot_CHI,
    rot_DELTA,
    rot_ETA,
    rot_MU,
    rot_NU,
    rot_PHI,
)

# Incident beam direction in the lab frame (unit vector).
BEAM = np.array([0.0, 1.0, 0.0])


def _R(m) -> np.ndarray:
    return np.asarray(m, dtype=float)


def sample_matrix(mu: float, eta: float, chi: float, phi: float,
                  degrees: bool = True) -> np.ndarray:
    """Sample orientation matrix Z mapping the phi (crystal) frame to the lab."""
    if degrees:
        mu, eta, chi, phi = (radians(a) for a in (mu, eta, chi, phi))
    return _R(rot_MU(mu)) @ _R(rot_ETA(eta)) @ _R(rot_CHI(chi)) @ _R(rot_PHI(phi))


def detector_matrix(nu: float, delta: float, degrees: bool = True) -> np.ndarray:
    """Detector arm matrix R_det = NU . DELTA."""
    if degrees:
        nu, delta = radians(nu), radians(delta)
    return _R(rot_NU(nu)) @ _R(rot_DELTA(delta))
