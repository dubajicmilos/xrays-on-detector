"""CIF reader: cell, symmetry, atoms, expanded to P1.

A deliberate port of web/js/cif.js, kept line-for-line comparable so the
desktop app and the browser app read a file the same way. tests/ checks the
two against each other; if you change a rule here, change it there.

The structure factor sum applies no symmetry of its own, so a file listing an
asymmetric unit has to be expanded before it can be used. Getting that wrong is
silent and severe: skip the expansion and half the reflections come out with
the wrong intensity while a spot check still passes. That is the bug that made
the earlier streamlit viewer untrustworthy for a general CIF.

Operators come from the file itself rather than a space-group table, which
covers essentially every CIF in the wild: a listed operator set is complete by
definition, centring included. Only a file with no operators falls back to the
Hermann-Mauguin symbol, and one naming a symmetry it does not spell out is
refused rather than quietly expanded into a structure that looks plausible.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field


class CifError(Exception):
    """A CIF we will not read, with a message meant for the user."""


CENTRING = {
    "P": [(0, 0, 0)],
    "I": [(0, 0, 0), (0.5, 0.5, 0.5)],
    "F": [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)],
    "A": [(0, 0, 0), (0, 0.5, 0.5)],
    "B": [(0, 0, 0), (0.5, 0, 0.5)],
    "C": [(0, 0, 0), (0.5, 0.5, 0)],
    "R": [(0, 0, 0), (2 / 3, 1 / 3, 1 / 3), (1 / 3, 2 / 3, 2 / 3)],
}

# Every element our scattering tables cover. Set by scatter.py at import so an
# unknown atom type is caught while reading rather than scattering as zero.
_ELEMENTS: set[str] = set()


def set_elements(symbols) -> None:
    _ELEMENTS.clear()
    _ELEMENTS.update(symbols)


@dataclass
class Atom:
    """One atom in the P1 cell.

    element : symbol for the X-ray and electron tables, where an isotope is
        indistinguishable from its element.
    nuclide : symbol for the neutron table, where it is not. b_c(H) is
        -3.739 fm and b_c(D) is +6.671 fm, opposite in sign, so a CIF that
        writes D has to keep saying D.
    """

    element: str
    nuclide: str
    x: float
    y: float
    z: float
    occ: float = 1.0
    B: float = 0.0


@dataclass
class Cell:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float

    def as_tuple(self):
        return (self.a, self.b, self.c, self.alpha, self.beta, self.gamma)


@dataclass
class CifStructure:
    name: str
    cell: Cell
    atoms: list[Atom]
    space_group: str | None = None
    block: str | None = None
    blocks_in_file: int = 1
    source: str = ""
    _extra: dict = field(default_factory=dict, repr=False)


def _num(token):
    """A CIF number: strip the estimated standard deviation, reject placeholders."""
    if token is None:
        return None
    t = re.sub(r"\((\d+)\)$", "", str(token).strip())
    if t in ("", "?", "."):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _tokenize(text: str):
    """Split into (value, quoted) tokens, honouring quotes and ; text fields.

    Tokens carry their quoting so a quoted "loop_" cannot start a loop.
    """
    out = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\r")
        if line.startswith(";"):
            body = line[1:]
            i += 1
            while i < len(lines) and not lines[i].startswith(";"):
                body += lines[i].rstrip("\r")
                i += 1
            out.append((body.strip(), True))
            i += 1
            continue
        j = 0
        while j < len(line):
            c = line[j]
            if c in " \t":
                j += 1
            elif c == "#":
                break
            elif c in "'\"":
                end = line.find(c, j + 1)
                if end < 0:
                    out.append((line[j + 1:], True))
                    break
                out.append((line[j + 1:end], True))
                j = end + 1
            else:
                end = j
                while end < len(line) and line[end] not in " \t":
                    end += 1
                out.append((line[j:end], False))
                j = end
        i += 1
    return out


def _parse_blocks(tokens):
    """Split into data blocks, each (items, loops, name).

    Published CIFs routinely open with a data_publication_text block carrying
    only bibliography, so the structure is not always in the first one.
    """
    blocks = []
    items, loops, name = {}, [], ""

    def is_tag(t):
        return not t[1] and t[0].startswith("_")

    def is_word(t, w):
        return not t[1] and t[0].lower() == w

    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        if not t[1] and t[0].lower().startswith("data_"):
            if items or loops:
                blocks.append((items, loops, name))
            name = t[0][5:]
            items, loops = {}, []
            i += 1
        elif is_word(t, "loop_"):
            i += 1
            tags = []
            while i < n and is_tag(tokens[i]):
                tags.append(tokens[i][0].lower())
                i += 1
            rows, row = [], []
            while (
                i < n
                and not is_tag(tokens[i])
                and not is_word(tokens[i], "loop_")
                and not (not tokens[i][1] and tokens[i][0].lower().startswith("data_"))
            ):
                row.append(tokens[i][0])
                i += 1
                if len(row) == len(tags):
                    rows.append(row)
                    row = []
            loops.append((tags, rows))
        elif is_tag(t):
            tag = t[0].lower()
            i += 1
            if i < n and not is_tag(tokens[i]) and not is_word(tokens[i], "loop_"):
                items[tag] = tokens[i][0]
                i += 1
            else:
                items[tag] = ""
        else:
            i += 1
    if items or loops:
        blocks.append((items, loops, name))
    return blocks


_TERM = re.compile(r"^([+-]?)([\d./]*)\*?([xyz]?)$")


def _parse_operator(spec: str):
    """'-x+1/2, y, -z' -> 3 rows of (a, b, c, t) acting on fractional coords."""
    parts = spec.split(",")
    if len(parts) != 3:
        raise CifError(f'symmetry operator "{spec}" does not have 3 parts')
    rows = []
    for part in parts:
        row = [0.0, 0.0, 0.0, 0.0]
        cleaned = re.sub(r"\s+", "", part).lower()
        for term in re.findall(r"[+-]?[^+-]+", cleaned):
            m = _TERM.match(term)
            if not m:
                raise CifError(f'cannot read symmetry operator "{spec}"')
            sign = -1.0 if m.group(1) == "-" else 1.0
            mag = 1.0
            if m.group(2):
                frac = m.group(2).split("/")
                try:
                    mag = (
                        float(frac[0]) / float(frac[1])
                        if len(frac) == 2
                        else float(m.group(2))
                    )
                except (ValueError, ZeroDivisionError):
                    raise CifError(f'cannot read symmetry operator "{spec}"') from None
            axis = m.group(3)
            if axis:
                row["xyz".index(axis)] += sign * mag
            else:
                row[3] += sign * mag
        rows.append(row)
    return rows


def _element(type_symbol, label):
    """'Pb2+', 'Cs1', 'D' -> (element, nuclide) our tables know.

    Deuterium and tritium are returned as hydrogen for the element, because
    they scatter X-rays and electrons exactly as hydrogen does, and as
    themselves for the nuclide, because they do not scatter neutrons as
    hydrogen does.
    """
    for raw in (type_symbol, label):
        if not raw:
            continue
        letters = re.sub(r"[^A-Za-z]", "", str(raw))
        if not letters:
            continue
        two = letters[:2]
        cap2 = two[0].upper() + two[1:].lower()
        cap1 = letters[0].upper()
        if cap2 in _ELEMENTS:
            return cap2, cap2
        if cap1 in _ELEMENTS:
            return cap1, cap1
        if cap1 in ("D", "T"):
            return "H", cap1
    return None, None


def _wrap(v: float) -> float:
    w = v - math.floor(v)
    if w > 1 - 1e-6 or w < 1e-6:
        w = 0.0
    return w


def parse_cif(text: str, name: str = "uploaded") -> CifStructure:
    """Parse CIF text into a P1 structure. Raises CifError with a user message."""
    if not text or not re.search(r"data_", text, re.I):
        raise CifError("this file has no data_ block, so it is not a CIF")

    blocks = _parse_blocks(_tokenize(text))

    def has_cell(b):
        return "_cell_length_a" in b[0]

    def has_atoms(b):
        return any(
            any(re.search(r"_atom_site_fract_x$", t) for t in tags) for tags, _ in b[1]
        )

    with_structure = [b for b in blocks if has_cell(b) and has_atoms(b)]
    block = next(
        (b for b in blocks if has_cell(b) and has_atoms(b)),
        next((b for b in blocks if has_cell(b)), None),
    )
    if block is None:
        raise CifError("no data block in this file has a unit cell in it")
    items, loops, block_name = block

    vals = [
        _num(items.get(f"_cell_{k}"))
        for k in (
            "length_a",
            "length_b",
            "length_c",
            "angle_alpha",
            "angle_beta",
            "angle_gamma",
        )
    ]
    if any(v is None or v <= 0 for v in vals):
        raise CifError("the unit cell is missing or unreadable")
    cell = Cell(*vals)

    # -- symmetry operators, from the file
    ops = [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]]
    sym_re = re.compile(
        r"_(space_group_symop_operation_xyz|symmetry_equiv_pos_as_xyz)$"
    )
    sym_loop = next((L for L in loops if any(sym_re.search(t) for t in L[0])), None)
    if sym_loop:
        tags, rows = sym_loop
        col = next(i for i, t in enumerate(tags) if sym_re.search(t))
        specs = [r[col] for r in rows if col < len(r) and r[col]]
        if specs:
            ops = [_parse_operator(s) for s in specs]

    # Whether a symmetry is claimed and whether it is written as a readable
    # Hermann-Mauguin symbol are different questions: files name the space
    # group by its International Tables number too, which says nothing about
    # centring and is no use as a label, but does still assert a symmetry.
    raw = re.sub(
        r"\s+",
        "",
        items.get("_space_group_name_h-m_alt")
        or items.get("_symmetry_space_group_name_h-m")
        or "",
    )
    it_number = _num(
        items.get("_space_group_it_number") or items.get("_symmetry_int_tables_number")
    )
    symbol = raw if re.search(r"[A-Za-z]", raw) else None
    claims_symmetry = (symbol and not re.fullmatch(r"P1", symbol, re.I)) or (
        it_number is not None and it_number != 1
    )

    if not sym_loop and claims_symmetry:
        raise CifError(
            f"this CIF names the space group {symbol or it_number} but lists no "
            "symmetry operators, so the full cell cannot be built from it. "
            "Export it as P1 (VESTA, or ASE read/write) and load that."
        )

    # A file that lists operators lists all of them, centring included, so the
    # symbol is only consulted when there are none to go on.
    letter = symbol.lstrip("-+")[0].upper() if symbol else "P"
    centring = CENTRING["P"] if sym_loop else CENTRING.get(letter, CENTRING["P"])

    # -- atom sites
    atom_loop = next(
        (L for L in loops if any(re.search(r"_atom_site_fract_x$", t) for t in L[0])),
        None,
    )
    if atom_loop is None:
        raise CifError("no atom sites with fractional coordinates were found")
    tags, rows = atom_loop

    def col(pattern):
        return next((i for i, t in enumerate(tags) if re.search(pattern, t)), -1)

    cx = col(r"_atom_site_fract_x$")
    cy = col(r"_atom_site_fract_y$")
    cz = col(r"_atom_site_fract_z$")
    c_type = col(r"_atom_site_type_symbol$")
    c_label = col(r"_atom_site_label$")
    c_occ = col(r"_atom_site_occupancy$")
    c_b = col(r"_atom_site_b_iso_or_equiv$")
    c_u = col(r"_atom_site_u_iso_or_equiv$")
    if cy < 0 or cz < 0:
        raise CifError("the atom sites are missing y or z coordinates")

    def cellval(row, i):
        return row[i] if 0 <= i < len(row) else None

    sites = []
    unknown = set()
    for r in rows:
        x, y, z = _num(cellval(r, cx)), _num(cellval(r, cy)), _num(cellval(r, cz))
        if x is None or y is None or z is None:
            continue
        el, nuc = _element(cellval(r, c_type), cellval(r, c_label))
        if not el:
            unknown.add(cellval(r, c_type) or cellval(r, c_label) or "?")
            continue
        occ = _num(cellval(r, c_occ))
        B = _num(cellval(r, c_b))
        if B is None and c_u >= 0:
            U = _num(cellval(r, c_u))
            if U is not None:
                B = 8 * math.pi * math.pi * U
        sites.append(
            dict(
                el=el,
                nuc=nuc,
                x=x,
                y=y,
                z=z,
                occ=1.0 if occ is None else occ,
                B=0.0 if B is None else B,
            )
        )
    if unknown:
        raise CifError(
            "these atom types are not in the scattering factor table: "
            + ", ".join(sorted(unknown)[:6])
        )
    if not sites:
        raise CifError("no usable atom sites were found")

    # -- expand to P1, merging positions that coincide
    #
    # An atom on a special position is mapped onto itself by several operators,
    # so the copies have to be merged or it scatters several times over. They
    # are compared with a tolerance and across the cell boundary, not by a
    # rounded key: two copies either side of a rounding step are the same atom.
    # Different elements never merge, which keeps a site shared by a disordered
    # C and N as the two contributions it physically is.
    TOL = 1e-3

    def near(p, q):
        s2 = 0.0
        for i in range(3):
            d = abs(p[i] - q[i])
            if d > 0.5:
                d = 1 - d
            s2 += d * d
        return s2 < TOL * TOL

    atoms: list[Atom] = []
    by_element: dict[str, list] = {}
    for s in sites:
        kept = by_element.setdefault(s["nuc"], [])
        for op in ops:
            for t in centring:
                p = [
                    _wrap(
                        op[i][0] * s["x"]
                        + op[i][1] * s["y"]
                        + op[i][2] * s["z"]
                        + op[i][3]
                        + t[i]
                    )
                    for i in range(3)
                ]
                if any(near(p, q) for q in kept):
                    continue
                kept.append(p)
                # Full precision. Rounding to six decimals costs nothing on a
                # coordinate like 0.25 and 3e-7 on a hexagonal 1/3, which is
                # enough to move |F|^2 by 1e-5 relative -- invisible on screen,
                # but a needless disagreement with any other code.
                atoms.append(
                    Atom(
                        element=s["el"],
                        nuclide=s["nuc"],
                        x=p[0],
                        y=p[1],
                        z=p[2],
                        occ=s["occ"],
                        B=s["B"],
                    )
                )

    return CifStructure(
        name=name,
        cell=cell,
        atoms=atoms,
        space_group=symbol if symbol and not re.fullmatch(r"P1", symbol, re.I) else None,
        block=block_name or None,
        blocks_in_file=len(with_structure) or 1,
        source=name,
    )


def read_cif(path: str) -> CifStructure:
    """Read a CIF from disk. The name is the file stem."""
    import os

    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    return parse_cif(text, os.path.splitext(os.path.basename(path))[0])
