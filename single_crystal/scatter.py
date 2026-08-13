"""Atomic scattering factors for X-rays, neutrons and electrons.

Tables live in data/ and are written by tools/export_scattering.py; the browser
app reads byte-identical copies under web/data/, so the two cannot drift.

    X-ray     f(s), Cromer-Mann four-Gaussian plus constant, in electrons
    electron  f(s), four-Gaussian, in Angstrom
    neutron   b_c, one number per nuclide, in fm

s = sin(theta) / lambda = |Q| / 4 pi, in 1/Angstrom, throughout.

The three are in different units, so intensities are comparable within a
radiation and not across one. Nothing here normalises them; the apps display
intensity relative to the strongest reflection of the pattern in view.

Neutron scattering lengths are signed and some are negative (H, Ti, V, Mn).
That is physical -- it is what makes contrast variation work -- so the sign is
carried through the structure factor sum and never taken as a magnitude.
"""
from __future__ import annotations

import json
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

RADIATIONS = ("xray", "neutron", "electron")
RADIATION_LABEL = {
    "xray": "X-rays",
    "neutron": "Neutrons",
    "electron": "Electrons",
}
#: What |F|^2 is measured in, per radiation. Shown in the UI so a number
#: copied out of the app carries its unit.
UNITS = {"xray": "e", "neutron": "fm", "electron": "Å"}

_CACHE: dict[str, dict] = {}


def _table(name: str) -> dict:
    if name not in _CACHE:
        path = os.path.join(DATA, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} is missing. Run: python tools/export_scattering.py"
            )
        with open(path, encoding="utf-8") as fh:
            _CACHE[name] = json.load(fh)
    return _CACHE[name]


def xray_table() -> dict:
    return _table("scattering_factors.json")


def electron_table() -> dict:
    return _table("electron_factors.json")


def neutron_table() -> dict:
    return _table("neutron_lengths.json")


def known_elements() -> list[str]:
    """Every symbol the X-ray table covers, which is what the CIF reader
    validates against. Neutrons and electrons are checked separately, when a
    radiation is actually chosen, so a structure is never refused for lacking
    a table it is not going to use."""
    return sorted(xray_table())


def missing_for(radiation: str, symbols) -> list[str]:
    """Symbols with no entry for this radiation, in the order first seen."""
    if radiation == "neutron":
        table = neutron_table()
    elif radiation == "electron":
        table = electron_table()
    elif radiation == "xray":
        table = xray_table()
    else:
        raise ValueError(f"unknown radiation {radiation!r}")
    seen, out = set(), []
    for s in symbols:
        if s not in table and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def factors(radiation: str, symbols, s) -> np.ndarray:
    """Scattering factor of each symbol at each s.

    symbols : sequence of n symbols (element for X-ray and electron, nuclide
              for neutron -- see cif.Atom).
    s       : scalar or array of m values of sin(theta)/lambda.

    Returns (n, m), or (n,) for scalar s. Missing entries raise rather than
    silently scattering as zero, which would look like a systematic absence.
    """
    s = np.asarray(s, dtype=float)
    scalar = s.ndim == 0
    s = np.atleast_1d(s)
    symbols = list(symbols)

    missing = missing_for(radiation, symbols)
    if missing:
        raise KeyError(
            f"no {RADIATION_LABEL[radiation].lower()} scattering data for "
            + ", ".join(missing[:6])
        )

    if radiation == "neutron":
        table = neutron_table()
        out = np.repeat(
            np.array([table[e] for e in symbols], dtype=float)[:, None], s.size, axis=1
        )
    else:
        table = xray_table() if radiation == "xray" else electron_table()
        s2 = s * s
        out = np.empty((len(symbols), s.size), dtype=float)
        for i, e in enumerate(symbols):
            a, b, c = table[e]
            out[i] = c + (np.asarray(a)[:, None] * np.exp(
                -np.asarray(b)[:, None] * s2[None, :]
            )).sum(axis=0)
    return out[:, 0] if scalar else out


def electron_wavelength(kv: float) -> float:
    """Relativistic electron wavelength in Angstrom for an accelerating
    voltage in kV. 200 kV -> 0.02508 Angstrom."""
    V = kv * 1e3
    h, m, e, c = 6.62607015e-34, 9.1093837015e-31, 1.602176634e-19, 2.99792458e8
    lam = h / np.sqrt(2 * m * e * V * (1 + e * V / (2 * m * c * c)))
    return float(lam * 1e10)
