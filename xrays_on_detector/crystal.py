"""Crystal structure and structure factors from a CIF, via pytilting's engine.

Both the reciprocal-lattice geometry and |F(hkl)| come from pytilting's
``StructureFactorCalculator`` (Cromer-Mann form factors, isotropic B, and the
2*pi reciprocal convention |Q| = 2*pi/d).

That engine sums only over the atoms *listed* in the CIF; it does not apply
symmetry operations itself. So an arbitrary CIF (space group + asymmetric unit)
is first expanded to an explicit all-atom P1 cell with ASE. This is reported,
never silent: the number of atoms actually used is stored on the Crystal.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np


# pytilting is not on PyPI, so it has to be a checkout somewhere on disk.
# Locate it by environment variable and fail loudly if it is not there.
def _import_pytilting():
    path = os.environ.get("PYTILTING_PATH", "")
    tests = os.path.join(path, "tests")
    src = os.path.join(path, "src")
    if not os.path.isfile(os.path.join(tests, "structure_factor_calculator.py")):
        raise FileNotFoundError(
            "pytilting not found"
            + (f" at {path!r}" if path else " (PYTILTING_PATH is not set)")
            + ". Set the PYTILTING_PATH environment variable to the pytilting "
            "directory: the one whose tests/ folder contains "
            "structure_factor_calculator.py."
        )
    for p in (src, tests):
        if p not in sys.path:
            sys.path.insert(0, p)
    import structure_factor_calculator as sfc  # noqa: E402

    return sfc


def _expand_to_p1(cif_path: str) -> tuple[str, int]:
    """Read a CIF with ASE (which applies symmetry) and rewrite it as an
    explicit all-atom P1 CIF. Returns (temp_path, n_atoms)."""
    from ase.io import read, write

    atoms = read(cif_path)
    tmp = tempfile.NamedTemporaryFile(suffix=".cif", delete=False, mode="w")
    tmp.close()
    write(tmp.name, atoms)
    return tmp.name, len(atoms)


@dataclass
class Crystal:
    """A crystal ready for structure-factor and reciprocal-space queries.

    Attributes
    ----------
    B : (3, 3) ndarray
        Reciprocal matrix in the crystal Cartesian frame, 2*pi convention, so
        that ``Q_cart = B @ (h, k, l)`` and |Q| = 2*pi/d (units 1/Angstrom).
    cell : dict
        Cell parameters a, b, c (Angstrom) and alpha, beta, gamma (degrees).
    n_atoms : int
        Number of atoms actually summed over (post symmetry expansion).
    """

    calc: object
    B: np.ndarray
    cell: dict
    n_atoms: int
    _F_cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_cif(cls, cif_path: str, expand_symmetry: bool = True) -> "Crystal":
        sfc = _import_pytilting()
        used_cif = cif_path
        if expand_symmetry:
            used_cif, _ = _expand_to_p1(cif_path)
        with contextlib.redirect_stdout(io.StringIO()):
            calc = sfc.StructureFactorCalculator(used_cif)
        B = np.asarray(calc.G, dtype=float)
        return cls(
            calc=calc,
            B=B,
            cell=dict(calc.cif.cell_params),
            n_atoms=len(calc.cif.atoms),
        )

    def q_cryst(self, hkl: np.ndarray) -> np.ndarray:
        """Reciprocal vectors in the crystal Cartesian frame (2*pi, 1/Angstrom).

        hkl : (N, 3) -> (N, 3)."""
        return np.asarray(hkl, dtype=float) @ self.B.T

    def structure_factor_mag2(self, hkl: np.ndarray) -> np.ndarray:
        """|F(hkl)|^2 for each row of hkl, via pytilting (cached per hkl)."""
        hkl = np.asarray(hkl, dtype=int)
        out = np.empty(len(hkl), dtype=float)
        for i, (h, k, l) in enumerate(hkl):
            key = (int(h), int(k), int(l))
            F = self._F_cache.get(key)
            if F is None:
                F, _ = self.calc.calculate_structure_factor(*key)
                self._F_cache[key] = F
            out[i] = abs(F) ** 2
        return out

    def hkl_within_Qmax(self, Q_max: float) -> np.ndarray:
        """All integer hkl (excluding 000) with |B @ hkl| <= Q_max."""
        Binv = np.linalg.inv(self.B)  # hkl = Binv @ Q
        # |h_i| <= Q_max * ||row_i(Binv)||  (Cauchy-Schwarz bound).
        hmax = np.ceil(Q_max * np.linalg.norm(Binv, axis=1)).astype(int) + 1
        ranges = [np.arange(-m, m + 1) for m in hmax]
        H, K, L = np.meshgrid(*ranges, indexing="ij")
        hkl = np.stack([H.ravel(), K.ravel(), L.ravel()], axis=1)
        hkl = hkl[np.any(hkl != 0, axis=1)]  # drop 000
        Q = self.q_cryst(hkl)
        keep = np.linalg.norm(Q, axis=1) <= Q_max
        return hkl[keep]
