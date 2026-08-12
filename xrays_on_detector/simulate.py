"""Top-level: simulate one detector frame for a crystal on a six-circle setup."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .crystal import Crystal
from .detector import Detector
from .ewald import excite
from .geometry import sample_matrix
from .render import render


@dataclass
class Frame:
    image: np.ndarray                 # (n_slow, n_fast)
    table: list                       # rendered reflections (on the panel)
    all_reflections: object           # ewald.Reflections near the sphere
    wavelength: float
    sample_angles: dict
    detector: Detector


def simulate_frame(
    crystal: Crystal,
    detector: Detector,
    wavelength: float,
    sigma: float,
    *,
    mu: float = 0.0,
    eta: float = 0.0,
    chi: float = 0.0,
    phi: float = 0.0,
    U: np.ndarray | None = None,
    hkl: np.ndarray | None = None,
    Fmag2: np.ndarray | None = None,
    n_sigma: float = 4.0,
    polarization_mode: str = "unpolarized",
    min_sigma_px: float = 0.6,
) -> Frame:
    """Simulate the diffraction image for one set of angles.

    Parameters
    ----------
    crystal, detector : the sample and the area detector (carries nu, delta).
    wavelength : float
        Incident wavelength (Angstrom); monochromatic.
    sigma : float
        Gaussian reciprocal-space peak width (1/Angstrom, 2*pi convention).
    mu, eta, chi, phi : float
        Sample circle angles (degrees), You convention.
    U : (3, 3) ndarray, optional
        Crystal orientation matrix; identity if omitted.
    hkl, Fmag2 : optional precomputed reflection list and |F|^2 (reused across a
        scan). If omitted they are generated from the detector's Q coverage.
    n_sigma : keep reflections within n_sigma * sigma of the sphere.
    """
    if U is None:
        U = np.eye(3)
    if hkl is None:
        Q_max = detector.max_Qmax(wavelength)
        hkl = crystal.hkl_within_Qmax(Q_max)
    if Fmag2 is None:
        Fmag2 = crystal.structure_factor_mag2(hkl)

    Z = sample_matrix(mu, eta, chi, phi)
    refl = excite(crystal, U, Z, wavelength, sigma, hkl, Fmag2, n_sigma=n_sigma)
    image, table = render(
        detector, refl, wavelength, sigma,
        polarization_mode=polarization_mode, min_sigma_px=min_sigma_px,
    )
    return Frame(
        image=image,
        table=table,
        all_reflections=refl,
        wavelength=wavelength,
        sample_angles=dict(mu=mu, eta=eta, chi=chi, phi=phi),
        detector=detector,
    )
