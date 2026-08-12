"""Render excited reflections onto the detector as finite Gaussian spots.

Each reflection's total intensity is

    I = |F|^2 * excitation(eps) * polarization(2theta)

and it is spread over the detector as a 2D Gaussian normalised to that total, so
grazing-incidence spots are broader and fainter per pixel but carry the same
integrated counts. The spot width follows from the reciprocal-space peak width
sigma: an angular spread ~ sigma / k maps to a detector size sigma * D /
(k * cos_incidence).
"""
from __future__ import annotations

import numpy as np


def polarization(two_theta: np.ndarray, mode: str = "unpolarized",
                 khat: np.ndarray | None = None) -> np.ndarray:
    """Polarization factor P(2theta).

    'unpolarized' : (1 + cos^2 2theta) / 2
    'none'        : 1
    'horizontal'  : synchrotron beam polarized along lab x; 1 - (khat_x)^2
    """
    if mode == "none":
        return np.ones_like(two_theta)
    if mode == "unpolarized":
        return 0.5 * (1.0 + np.cos(two_theta) ** 2)
    if mode == "horizontal":
        if khat is None:
            raise ValueError("horizontal polarization needs khat")
        return 1.0 - khat[:, 0] ** 2
    raise ValueError(f"unknown polarization mode {mode!r}")


def render(detector, refl, wavelength, sigma, *, polarization_mode="unpolarized",
           min_sigma_px=0.6, dtype=np.float64):
    """Accumulate reflections into a detector image.

    Returns
    -------
    image : (n_slow, n_fast) ndarray
        Row 0 is the top of the detector (+z downwards in array rows).
    table : list of dict
        Per-rendered-reflection record (hkl, pixel, eps, 2theta, intensity).
    """
    k = 2.0 * np.pi / wavelength
    image = np.zeros((detector.n_slow, detector.n_fast), dtype=dtype)

    fast_px, slow_px, inside, cos_inc = detector.project(refl.khat)
    P = polarization(refl.two_theta, polarization_mode, refl.khat)
    intensity = refl.Fmag2 * refl.excitation * P

    table = []
    for i in np.nonzero(inside)[0]:
        cx = float(fast_px[i])
        cy = float(slow_px[i])
        ci = max(float(cos_inc[i]), 1e-3)
        s_px = max(min_sigma_px, (sigma / k) * detector.distance
                   / (detector.pixel_size * ci))
        total = float(intensity[i])
        _add_gaussian(image, cx, cy, s_px, total)

        h, kk, l = (int(x) for x in refl.hkl[i])
        table.append({
            "h": h, "k": kk, "l": l,
            "fast_px": cx, "slow_px": cy,
            "eps": float(refl.eps[i]),
            "two_theta_deg": float(np.degrees(refl.two_theta[i])),
            "intensity": total,
        })
    return image, table


def _add_gaussian(image, cx, cy, s_px, total, n_sigma=4.0):
    """Add a normalised 2D Gaussian (integral = total) centred at (cx, cy).

    cx indexes the fast (column) axis, cy the slow axis. The slow axis is
    written top-down so +z (up) maps to decreasing row index.
    """
    n_slow, n_fast = image.shape
    rad = int(np.ceil(n_sigma * s_px))
    col0, col1 = max(0, int(cx) - rad), min(n_fast, int(cx) + rad + 1)
    row_center = (n_slow - 1) - cy
    row0, row1 = max(0, int(row_center) - rad), min(n_slow, int(row_center) + rad + 1)
    if col0 >= col1 or row0 >= row1:
        return
    cols = np.arange(col0, col1)
    rows = np.arange(row0, row1)
    gx = np.exp(-((cols - cx) ** 2) / (2.0 * s_px ** 2))
    gy = np.exp(-((rows - row_center) ** 2) / (2.0 * s_px ** 2))
    patch = np.outer(gy, gx)
    patch *= total / (2.0 * np.pi * s_px ** 2)
    image[row0:row1, col0:col1] += patch
