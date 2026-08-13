"""Reciprocal lattice and structure factors for a P1 cell.

The reciprocal convention is 2*pi throughout, matching the rest of this
project: Q = B @ (h, k, l), |Q| = 2*pi/d. bmatrix() is a port of the same
function in web/js/physics.js and returns the same numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import scatter
from .cif import CifStructure

TWO_PI = 2 * math.pi


def bmatrix(a, b, c, alpha=90.0, beta=90.0, gamma=90.0) -> np.ndarray:
    """Reciprocal matrix in the crystal Cartesian frame, 2*pi convention.

    Columns are a*, b*, c*, so Q_cart = B @ (h, k, l) with |Q| = 2*pi/d.
    """
    al, be, ga = math.radians(alpha), math.radians(beta), math.radians(gamma)
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz2 = c * c - cx * cx - cy * cy
    if cz2 <= 0:
        raise ValueError("cell angles do not describe a real lattice")
    # Direct lattice A with a, b, c as columns.
    A = np.array(
        [
            [a, b * math.cos(ga), cx],
            [0.0, b * math.sin(ga), cy],
            [0.0, 0.0, math.sqrt(cz2)],
        ]
    )
    return TWO_PI * np.linalg.inv(A).T


@dataclass
class Structure:
    """A P1 cell ready for reciprocal-space queries."""

    name: str
    cell: tuple
    B: np.ndarray
    frac: np.ndarray  # (M, 3) fractional coordinates
    elements: list  # (M,) symbols for X-ray and electron
    nuclides: list  # (M,) symbols for neutron
    occ: np.ndarray  # (M,)
    Biso: np.ndarray  # (M,) in Angstrom^2
    space_group: str | None = None

    @classmethod
    def from_cif(cls, doc: CifStructure) -> "Structure":
        if not doc.atoms:
            raise ValueError("this structure has no atoms")
        frac = np.array([[a.x, a.y, a.z] for a in doc.atoms], dtype=float)
        return cls(
            name=doc.name,
            cell=doc.cell.as_tuple(),
            B=bmatrix(*doc.cell.as_tuple()),
            frac=frac,
            elements=[a.element for a in doc.atoms],
            nuclides=[a.nuclide for a in doc.atoms],
            occ=np.array([a.occ for a in doc.atoms], dtype=float),
            Biso=np.array([a.B for a in doc.atoms], dtype=float),
            space_group=doc.space_group,
        )

    # -- geometry ---------------------------------------------------------

    @property
    def A(self) -> np.ndarray:
        """Direct lattice with a, b, c as columns: A = 2*pi inv(B)^T."""
        return TWO_PI * np.linalg.inv(self.B).T

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.A)))

    def q(self, hkl) -> np.ndarray:
        """Q in the crystal Cartesian frame, (N, 3) in 1/Angstrom."""
        return np.asarray(hkl, dtype=float) @ self.B.T

    def d_spacing(self, hkl) -> np.ndarray:
        qn = np.linalg.norm(self.q(hkl), axis=-1)
        with np.errstate(divide="ignore"):
            return TWO_PI / qn

    def s_of(self, hkl) -> np.ndarray:
        """sin(theta)/lambda = |Q| / 4 pi, in 1/Angstrom."""
        return np.linalg.norm(self.q(hkl), axis=-1) / (4 * math.pi)

    def symbols_for(self, radiation: str) -> list:
        return self.nuclides if radiation == "neutron" else self.elements

    # -- structure factors ------------------------------------------------

    def structure_factor(self, hkl, radiation="xray", chunk=4096) -> np.ndarray:
        """Complex F(hkl), one per row of hkl.

        F = sum_j occ_j f_j(s) exp(-B_j s^2) exp(2 pi i (h x_j + k y_j + l z_j))

        The phase runs over every atom in the P1 cell; symmetry is applied when
        the CIF is read, never here.
        """
        hkl = np.atleast_2d(np.asarray(hkl, dtype=float))
        symbols = self.symbols_for(radiation)
        uniq = sorted(set(symbols))
        index = np.array([uniq.index(sym) for sym in symbols])

        out = np.empty(len(hkl), dtype=complex)
        for lo in range(0, len(hkl), chunk):
            block = hkl[lo:lo + chunk]
            s = self.s_of(block)  # (n,)
            f = scatter.factors(radiation, uniq, s)  # (n_uniq, n)
            # (M, n): each atom's weight at each reflection
            w = f[index] * np.exp(-np.outer(self.Biso, s * s)) * self.occ[:, None]
            phase = TWO_PI * (self.frac @ block.T)  # (M, n)
            out[lo:lo + chunk] = (w * np.exp(1j * phase)).sum(axis=0)
        return out

    def intensity(self, hkl, radiation="xray") -> np.ndarray:
        """|F(hkl)|^2, in the squared unit of the radiation (see scatter.UNITS)."""
        F = self.structure_factor(hkl, radiation)
        return (F.real * F.real + F.imag * F.imag)

    # -- reflection lists -------------------------------------------------

    #: Refuse to enumerate more than this many candidate hkl at once. The
    #: search box is cubic while the useful volume is a sphere, so this is
    #: roughly twice the reflection count it admits.
    MAX_CANDIDATES = 8_000_000

    def hkl_within(self, d_min: float) -> np.ndarray:
        """Every integer hkl except 000 with d >= d_min, i.e. |Q| <= 2 pi/d_min.

        Raises rather than allocating an array that will not fit. The limit is
        easy to cross without meaning to: an electron wavelength of 0.025 A
        puts 2 theta = 120 degrees at d = 0.014 A, which is half a billion
        reflections and 12 GB.
        """
        if d_min <= 0:
            raise ValueError("d_min must be positive")
        q_max = TWO_PI / d_min
        binv = np.linalg.inv(self.B)
        bound = np.ceil(q_max * np.linalg.norm(binv, axis=1)).astype(int) + 1

        count = float(np.prod(2 * bound.astype(float) + 1))
        if count > self.MAX_CANDIDATES:
            shrink = (self.MAX_CANDIDATES / count) ** (1 / 3)
            raise ValueError(
                f"d min of {d_min:.4f} Å means {count:.3g} candidate "
                f"reflections for this cell, which will not fit in memory. "
                f"Raise d min above about {d_min / shrink:.3f} Å "
                f"(for a powder pattern, lower the 2θ limit or use a longer "
                f"wavelength)."
            )

        grids = np.meshgrid(*[np.arange(-m, m + 1) for m in bound], indexing="ij")
        hkl = np.stack([g.ravel() for g in grids], axis=1)
        hkl = hkl[np.any(hkl != 0, axis=1)]
        return hkl[np.linalg.norm(self.q(hkl), axis=1) <= q_max]

    def describe(self) -> str:
        a, b, c, al, be, ga = self.cell
        sg = f"  {self.space_group}" if self.space_group else ""
        return (
            f"{self.name}{sg}\n"
            f"a {a:.4f}  b {b:.4f}  c {c:.4f} Å\n"
            f"α {al:.3f}  β {be:.3f}  γ {ga:.3f}°\n"
            f"V {self.volume:.2f} Å³   {len(self.frac)} atoms in P1"
        )
