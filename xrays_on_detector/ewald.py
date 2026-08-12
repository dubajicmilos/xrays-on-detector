"""Monochromatic Ewald construction with finite-size (Gaussian) Bragg peaks.

For fixed sample angles, each reciprocal-lattice point (RLP) sits at Q in the
lab frame. Elastic scattering requires k_f = k_i + Q with |k_f| = k = 2*pi/lam.
The signed excitation error

    eps = |k_i + Q| - k

is the distance of the RLP from the Ewald sphere. A Gaussian peak of width
sigma (in reciprocal space) contributes with weight exp(-eps^2 / (2 sigma^2)),
placed along the diffracted direction khat = (k_i + Q) / |k_i + Q|.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import BEAM


@dataclass
class Reflections:
    hkl: np.ndarray          # (N, 3) int
    khat: np.ndarray         # (N, 3) unit diffracted directions
    eps: np.ndarray          # (N,) excitation error (1/Angstrom, 2*pi convention)
    excitation: np.ndarray   # (N,) exp(-eps^2 / 2 sigma^2)
    Fmag2: np.ndarray        # (N,) |F|^2
    two_theta: np.ndarray    # (N,) scattering angle (rad)


def excite(crystal, orient, sample_M, wavelength, sigma, hkl, Fmag2,
           n_sigma: float = 4.0) -> Reflections:
    """Select and characterise reflections near the Ewald sphere.

    Parameters
    ----------
    crystal : Crystal
    orient : (3, 3) ndarray
        Orientation matrix U (crystal Cartesian -> phi frame). Identity aligns
        the crystal frame with the phi frame at zero sample angles.
    sample_M : (3, 3) ndarray
        Sample matrix Z from geometry.sample_matrix.
    wavelength : float
        In Angstrom.
    sigma : float
        Gaussian peak width in reciprocal space (1/Angstrom, 2*pi convention).
    hkl : (M, 3) int ndarray
    Fmag2 : (M,) ndarray
        |F(hkl)|^2 for each row of hkl.
    n_sigma : float
        Keep reflections with |eps| <= n_sigma * sigma.
    """
    k = 2.0 * np.pi / wavelength
    ki = k * BEAM

    Q_cryst = crystal.q_cryst(hkl)                    # (M, 3)
    ZU = sample_M @ np.asarray(orient, dtype=float)
    Q_lab = Q_cryst @ ZU.T                            # (M, 3)

    kf = ki + Q_lab
    kf_mag = np.linalg.norm(kf, axis=1)
    eps = kf_mag - k

    sel = (np.abs(eps) <= n_sigma * sigma) & (kf_mag > 0)
    kf_s = kf[sel]
    kf_mag_s = kf_mag[sel]
    khat = kf_s / kf_mag_s[:, None]
    eps_s = eps[sel]
    excitation = np.exp(-(eps_s ** 2) / (2.0 * sigma ** 2))
    two_theta = np.arccos(np.clip(khat @ BEAM, -1.0, 1.0))

    return Reflections(
        hkl=hkl[sel],
        khat=khat,
        eps=eps_s,
        excitation=excitation,
        Fmag2=Fmag2[sel],
        two_theta=two_theta,
    )
