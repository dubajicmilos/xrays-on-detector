"""Crystal structure and structure factors from a CIF.

The reciprocal lattice is in the standard setting, a along x and b in the xy
plane, with the 2*pi convention: Q = B @ (h, k, l) and |Q| = 2*pi/d. The
structure factor is the kinematic sum over the atoms of the unit cell,

    F(hkl) = sum_j occ_j f_j(s) exp(-B_j s^2) exp(2 pi i (h x_j + k y_j + l z_j)),

with s = |Q| / 4 pi and f_j the Cromer-Mann X-ray form factor from
single_crystal's table (single_crystal.scatter); an element the table does not
cover raises a KeyError when |F|^2 is computed. F(000) is returned as 0.

The sum applies no symmetry of its own. By default ASE reads the CIF and
applies its space-group symmetry, and every atom of the full cell takes the
occupancy and the isotropic displacement parameter of the site it came from:
B_iso_or_equiv, or 8 pi^2 U_iso_or_equiv where the CIF gives U instead, or 0
where it gives neither. With expand_symmetry=False the atoms are used exactly
as the CIF lists them, with the same rule for B. Crystal.n_atoms reports how
many atoms the sum runs over. The browser build's bundled structures are
exported from this same atom list (tools/export_web_data.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from single_crystal import scatter
# The tokenizer and number rule (e.s.d.s stripped) of single_crystal's CIF
# reader. Its parse_cif also applies the file's symmetry operators, which the
# listed-atom reading here must not do, so these pieces are used directly.
from single_crystal.cif import _num, _parse_blocks, _tokenize
from single_crystal.structure import bmatrix

_CELL = ("a", "b", "c", "alpha", "beta", "gamma")

# Atoms x reflections evaluated at once. At 0.5 MB per float64 array this
# measured faster than larger blocks, and memory stays small however many
# atoms the cell holds.
_BLOCK = 1 << 16


def _expanded_sites(cif_path: str):
    """The CIF expanded by ASE to its full P1 cell.

    ASE applies the space-group symmetry and records, for every atom of the
    cell, the listed site it came from (``spacegroup_kinds``). Each atom takes
    that site's occupancy from ASE and that site's B from _listed_sites, since
    ASE keeps occupancies but drops displacement parameters.

    Returns (cell, elements, frac, occ, B_iso), as _listed_sites does.
    """
    from ase.io import read

    atoms = read(cif_path)
    with open(cif_path, encoding="utf-8", errors="replace") as fh:
        B_site = _listed_sites(fh.read())[4]
    kinds = atoms.arrays.get("spacegroup_kinds")
    occ_site = atoms.info.get("occupancy", {})
    elements = atoms.get_chemical_symbols()
    occ, B_iso = [], []
    for i, element in enumerate(elements):
        site = int(kinds[i]) if kinds is not None else i
        if site >= len(B_site):
            raise ValueError(
                f"ASE assigns atom {i} to site {site}, but the CIF lists "
                f"{len(B_site)} sites"
            )
        occ.append(float(occ_site.get(str(site), {}).get(element, 1.0)))
        B_iso.append(float(B_site[site]))
    cell = tuple(float(v) for v in atoms.cell.cellpar())
    return cell, elements, atoms.get_scaled_positions(), np.array(occ), np.array(B_iso)


def _listed_sites(text: str):
    """Cell and atom sites exactly as a CIF lists them; no symmetry is applied.

    Returns (cell, elements, frac, occ, B_iso). An element is the site's type
    symbol with any charge stripped, or the leading letters of its label when
    the file gives no type symbol. B_iso is B_iso_or_equiv, or 8 pi^2
    U_iso_or_equiv where B is absent or zero and U is positive, or else 0.
    """
    for items, loops, _ in _parse_blocks(_tokenize(text)):
        sites = next((L for L in loops if "_atom_site_fract_x" in L[0]), None)
        if "_cell_length_a" in items and sites is not None:
            break
    else:
        raise ValueError("no data block has a unit cell and fractional atom sites")
    cell = tuple(
        _num(items.get(f"_cell_{k}"))
        for k in ("length_a", "length_b", "length_c",
                  "angle_alpha", "angle_beta", "angle_gamma")
    )
    if None in cell:
        raise ValueError("the unit cell is missing or unreadable")

    tags, rows = sites

    def column(row, tag):
        i = tags.index(tag) if tag in tags else -1
        return row[i] if 0 <= i < len(row) else None

    elements, frac, occ, B_iso = [], [], [], []
    for row in rows:
        xyz = [_num(column(row, f"_atom_site_fract_{axis}")) for axis in "xyz"]
        if None in xyz:
            raise ValueError(f"atom site {row[0]!r} has no readable position")
        symbol = column(row, "_atom_site_type_symbol")
        if symbol is None:  # label only: 'C1A' -> 'C'
            label = column(row, "_atom_site_label") or ""
            symbol = re.match(r"[A-Za-z]{0,2}", label)[0]
        elements.append(re.sub(r"[0-9+-]", "", symbol))  # 'Pb2+' -> 'Pb'
        frac.append(xyz)
        o = _num(column(row, "_atom_site_occupancy"))
        occ.append(1.0 if o is None else o)
        B = _num(column(row, "_atom_site_b_iso_or_equiv")) or 0.0
        U = _num(column(row, "_atom_site_u_iso_or_equiv")) or 0.0
        B_iso.append(8 * np.pi**2 * U if B == 0 and U > 0 else B)
    if not elements:
        raise ValueError("the CIF lists no atom sites")
    return cell, elements, np.array(frac, dtype=float), np.array(occ), np.array(B_iso)


@dataclass
class Crystal:
    """A crystal ready for structure-factor and reciprocal-space queries.

    Made by :meth:`from_cif`. The form factors are the Cromer-Mann X-ray
    factors of single_crystal's table, and each atom's isotropic displacement
    parameter B_iso enters its scattering as the Debye-Waller factor
    exp(-B s^2). With expand_symmetry=True (the default) ASE expands the CIF to
    its full P1 cell, and each atom keeps its site's occupancy and B.

    Attributes
    ----------
    B : (3, 3) ndarray
        Reciprocal matrix in the crystal Cartesian frame, 2*pi convention, so
        that ``Q_cart = B @ (h, k, l)`` and |Q| = 2*pi/d (units 1/Angstrom).
    cell : dict
        Cell parameters a, b, c (Angstrom) and alpha, beta, gamma (degrees).
    elements : list of str
        Element symbol of each atom, as looked up in the form-factor table.
    frac : (n, 3) ndarray
        Fractional coordinates of each atom.
    occ : (n,) ndarray
        Site occupancy of each atom.
    B_iso : (n,) ndarray
        Isotropic displacement parameter of each atom, in Angstrom^2.
    n_atoms : int
        Number of atoms summed over (after the symmetry expansion, if any).
    """

    B: np.ndarray
    cell: dict
    elements: list
    frac: np.ndarray
    occ: np.ndarray
    B_iso: np.ndarray

    @property
    def n_atoms(self) -> int:
        return len(self.elements)

    @classmethod
    def from_cif(cls, cif_path: str, expand_symmetry: bool = True) -> "Crystal":
        """Read a CIF. With expand_symmetry, ASE first expands it to the full
        P1 cell; without, its atom sites are used exactly as listed."""
        if expand_symmetry:
            cell, elements, frac, occ, B_iso = _expanded_sites(cif_path)
        else:
            with open(cif_path, encoding="utf-8", errors="replace") as fh:
                cell, elements, frac, occ, B_iso = _listed_sites(fh.read())
        return cls(
            B=bmatrix(*cell),
            cell=dict(zip(_CELL, cell)),
            elements=elements,
            frac=frac,
            occ=occ,
            B_iso=B_iso,
        )

    def q_cryst(self, hkl: np.ndarray) -> np.ndarray:
        """Reciprocal vectors in the crystal Cartesian frame (2*pi, 1/Angstrom).

        hkl : (N, 3) -> (N, 3)."""
        return np.asarray(hkl, dtype=float) @ self.B.T

    def structure_factor_mag2(self, hkl: np.ndarray) -> np.ndarray:
        """|F(hkl)|^2 in electrons^2 for each row of hkl, and 0 for (0, 0, 0)."""
        hkl = np.asarray(hkl, dtype=int).reshape(-1, 3)
        out = np.zeros(len(hkl))
        kinds, kind = np.unique(self.elements, return_inverse=True)
        step = max(1, _BLOCK // self.n_atoms)
        for lo in range(0, len(hkl), step):
            h = hkl[lo:lo + step]
            s = np.linalg.norm(self.q_cryst(h), axis=1) / (4 * np.pi)
            w = scatter.factors("xray", kinds, s)[kind]  # (n_atoms, len(h))
            w *= self.occ[:, None]
            if self.B_iso.any():  # skipped when no site gives a B
                w *= np.exp(-np.outer(self.B_iso, s * s))
            phase = 2 * np.pi * (self.frac @ h.T)
            F_re = np.einsum("jm,jm->m", w, np.cos(phase))
            F_im = np.einsum("jm,jm->m", w, np.sin(phase))
            out[lo:lo + step] = F_re * F_re + F_im * F_im
        out[~hkl.any(axis=1)] = 0.0
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
