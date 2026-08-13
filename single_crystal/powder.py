"""Powder pattern for the same structure and the same three radiations.

Every reflection out to d_min is enumerated and those sharing a d-spacing are
summed, so the multiplicity is counted by construction rather than looked up
from the Laue class. Accidental overlaps -- two unrelated families at the same
d -- merge too, which is what a real powder diffractometer does as well.

The Lorentz factor 1 / (sin^2(theta) cos(theta)) applies to all three
radiations. The polarisation factor (1 + cos^2(2 theta)) / 2 is X-rays only:
neutrons scatter off nuclei with no polarisation dependence, and it is not
meaningful for the electron case here either.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TWO_PI = 2 * math.pi


@dataclass
class PowderPattern:
    two_theta: np.ndarray  # (P,) peak positions, degrees
    intensity: np.ndarray  # (P,) peak intensities, scaled to 100 at the max
    d: np.ndarray  # (P,) d-spacing, Angstrom
    multiplicity: np.ndarray  # (P,) reflections summed into each peak
    hkl: list  # (P,) a representative hkl per peak
    x: np.ndarray  # (n,) trace abscissa, degrees
    y: np.ndarray  # (n,) trace ordinate
    wavelength: float
    radiation: str

    def __len__(self) -> int:
        return len(self.two_theta)


def compute_powder(
    structure,
    wavelength: float = 1.5406,
    radiation: str = "xray",
    two_theta_max: float = 120.0,
    fwhm: float = 0.15,
    n_points: int = 2400,
    d_tol: float = 1e-5,
) -> PowderPattern:
    """Peak list and a Gaussian-broadened trace.

    wavelength in Angstrom; for electrons use scatter.electron_wavelength(kV).
    fwhm is a constant Gaussian width in degrees, which is a stand-in for the
    instrument, not a Caglioti fit.
    """
    if wavelength <= 0:
        raise ValueError("wavelength must be positive")
    tt_max = min(float(two_theta_max), 179.9)
    # Bragg: lambda = 2 d sin(theta), so the largest reachable 2 theta fixes
    # the smallest d worth enumerating.
    d_min = wavelength / (2 * math.sin(math.radians(tt_max / 2)))

    hkl = structure.hkl_within(d_min)
    if len(hkl) == 0:
        raise ValueError(
            f"no reflections with d >= {d_min:.3f} Å; try a shorter wavelength "
            "or a larger 2-theta limit"
        )
    d = structure.d_spacing(hkl)
    inten = structure.intensity(hkl, radiation)

    sin_theta = wavelength / (2 * d)
    ok = sin_theta <= 1.0
    hkl, d, inten, sin_theta = hkl[ok], d[ok], inten[ok], sin_theta[ok]
    theta = np.arcsin(sin_theta)
    two_theta = np.degrees(2 * theta)
    ok = two_theta <= tt_max
    hkl, d, inten, theta, two_theta = (
        hkl[ok], d[ok], inten[ok], theta[ok], two_theta[ok]
    )
    if len(hkl) == 0:
        raise ValueError("no reflections fall inside the 2-theta range")

    lorentz = 1.0 / (np.sin(theta) ** 2 * np.cos(theta))
    if radiation == "xray":
        lorentz = lorentz * (1 + np.cos(2 * theta) ** 2) / 2
    inten = inten * lorentz

    # Merge families sharing a d-spacing. Sorting by d and cutting where the
    # relative gap exceeds the tolerance groups them without an O(N^2) compare.
    order = np.argsort(-d, kind="stable")
    hkl, d, inten, two_theta = hkl[order], d[order], inten[order], two_theta[order]
    cut = np.nonzero(np.abs(np.diff(d)) > d_tol * d[:-1])[0] + 1
    groups = np.split(np.arange(len(d)), cut)

    peaks_tt, peaks_I, peaks_d, peaks_m, peaks_hkl = [], [], [], [], []
    for g in groups:
        if len(g) == 0:
            continue
        peaks_tt.append(float(two_theta[g].mean()))
        peaks_I.append(float(inten[g].sum()))
        peaks_d.append(float(d[g].mean()))
        peaks_m.append(int(len(g)))
        # Representative index: the one a crystallographer would write, i.e.
        # the most positive of the family.
        peaks_hkl.append(tuple(int(v) for v in hkl[g][np.lexsort(hkl[g].T[::-1])][-1]))

    peaks_tt = np.array(peaks_tt)
    peaks_I = np.array(peaks_I)
    scale = peaks_I.max() if peaks_I.size and peaks_I.max() > 0 else 1.0
    peaks_I = 100.0 * peaks_I / scale

    x = np.linspace(0.0, tt_max, int(n_points))
    sigma = fwhm / (2 * math.sqrt(2 * math.log(2)))
    y = np.zeros_like(x)
    # Only peaks with something to contribute are painted; a structure can
    # easily have thousands of extinct ones.
    live = peaks_I > 1e-6
    for tt, I in zip(peaks_tt[live], peaks_I[live]):
        y += I * np.exp(-0.5 * ((x - tt) / sigma) ** 2)

    return PowderPattern(
        two_theta=peaks_tt,
        intensity=peaks_I,
        d=np.array(peaks_d),
        multiplicity=np.array(peaks_m),
        hkl=peaks_hkl,
        x=x,
        y=y,
        wavelength=wavelength,
        radiation=radiation,
    )
