"""Export everything the browser build needs, from the validated Python.

JSON is written compact. These are generated lookup tables, not source, so the
deploy script adds them to the site's .prettierignore rather than formatting
them.

Writes three kinds of file under web/:

  data/scattering_factors.json  Cromer-Mann coefficients, taken from pytilting
                                so the JavaScript cannot drift from it
  data/<name>.json              a bundled structure: cell plus the P1 atom list
  test/fixture.json             reference values for test/parity.mjs

Run from the project root:  python tools/export_web_data.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xrays_on_detector.geometry import detector_matrix, sample_matrix  # noqa: E402
from xrays_on_detector.vdiff.instrument import (Instrument,  # noqa: E402
                                                LabDetector, LatticeCrystal,
                                                b_matrix, euler_matrix,
                                                rotation_between)

WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")


def _m(a):
    return [[float(v) for v in row] for row in np.asarray(a)]


def _v(a):
    return [float(v) for v in np.asarray(a).ravel()]


# ---------------------------------------------------------------------------
# Scattering factors
# ---------------------------------------------------------------------------


def export_colormaps(out_dir, names=("inferno", "viridis", "magma", "turbo",
                                     "gray")):
    """256-entry RGB lookup tables, so the web build looks like the desktop app."""
    import matplotlib

    lut = {}
    for name in names:
        cmap = matplotlib.colormaps[name]
        lut[name] = [int(round(255 * v))
                     for i in range(256)
                     for v in cmap(i / 255.0)[:3]]
    path = os.path.join(out_dir, "colormaps.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(lut, fh, separators=(",", ":"))
    print(f"  {len(lut)} colour maps -> {os.path.relpath(path, WEB)}")


def export_scattering_factors(out_dir):
    from xrays_on_detector.crystal import _import_pytilting

    sfc = _import_pytilting()
    table = {el: [list(a), list(b), float(c)]
             for el, (a, b, c) in sfc.SCATTERING_FACTORS.items()}
    path = os.path.join(out_dir, "scattering_factors.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(table, fh, separators=(",", ":"), sort_keys=True)
    print(f"  {len(table)} elements -> {os.path.relpath(path, WEB)}")
    return table


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------


def _site_displacements(cif_path):
    """Per-site B from the CIF, in the order the atom_site loop lists them.

    ASE expands the symmetry and keeps occupancies, but drops the displacement
    parameters, and a structure factor without them is wrong at high angle. So
    read them here and map them back through ``spacegroup_kinds``. A site given
    as U gets B = 8 pi^2 U; a site giving neither gets 0.
    """
    text = open(cif_path, encoding="utf-8", errors="replace").read()
    out = []
    for block in re.finditer(r"loop_\s*((?:\s*_atom_site_\S+\s*\n)+)(.*?)(?=\n\s*(?:loop_|_|$))",
                             text, re.S):
        headers = re.findall(r"_atom_site_(\S+)", block.group(1))
        if "fract_x" not in headers:
            continue
        b_col = next((headers.index(h) for h in headers if h == "B_iso_or_equiv"), None)
        u_col = next((headers.index(h) for h in headers if h == "U_iso_or_equiv"), None)
        for line in block.group(2).splitlines():
            line = line.strip()
            if not line or line.startswith(("_", "#", "loop_")):
                continue
            vals = line.split()
            if len(vals) < len(headers):
                continue

            def num(col):
                if col is None:
                    return None
                v = re.sub(r"\(\d+\)$", "", vals[col])   # strip the e.s.d.
                try:
                    return float(v)
                except ValueError:
                    return None

            b, u = num(b_col), num(u_col)
            if b is None and u is not None:
                b = 8.0 * np.pi ** 2 * u
            out.append(b or 0.0)
        break
    return out


def export_structure(cif_path, name, out_dir):
    """Expand a CIF to P1 with ASE and write cell + atoms as compact JSON."""
    from ase.io import read

    atoms = read(cif_path)
    cell = atoms.cell.cellpar()
    frac = atoms.get_scaled_positions()
    syms = atoms.get_chemical_symbols()

    # ASE maps every expanded atom back to the site it came from, which is how
    # occupancies and B values follow the symmetry expansion.
    kinds = atoms.arrays.get("spacegroup_kinds")
    occ_by_site = atoms.info.get("occupancy", {})
    b_by_site = _site_displacements(cif_path)

    def site_of(i):
        return int(kinds[i]) if kinds is not None else i

    def occ_of(i, sym):
        site = occ_by_site.get(str(site_of(i)), {})
        return float(site.get(sym, 1.0)) if site else 1.0

    def b_of(i):
        s = site_of(i)
        return float(b_by_site[s]) if s < len(b_by_site) else 0.0

    doc = {
        "name": name,
        "source": os.path.basename(cif_path),
        "cell": {"a": float(cell[0]), "b": float(cell[1]), "c": float(cell[2]),
                 "alpha": float(cell[3]), "beta": float(cell[4]),
                 "gamma": float(cell[5])},
        "atoms": [
            {"element": s,
             "x": round(float(p[0]), 6),
             "y": round(float(p[1]), 6),
             "z": round(float(p[2]), 6),
             "occ": round(occ_of(i, s), 4),
             "B": round(b_of(i), 4)}
            for i, (s, p) in enumerate(zip(syms, frac))
        ],
    }
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    size = os.path.getsize(path)
    print(f"  {name}: {len(doc['atoms'])} atoms, {size / 1024:.1f} kB "
          f"-> {os.path.relpath(path, WEB)}")
    return doc


# ---------------------------------------------------------------------------
# Parity fixture
# ---------------------------------------------------------------------------


def build_fixture(structure_doc):
    """Reference values covering every ported function."""
    fx = {}

    # -- circle matrices at a set of awkward angles
    fx["rotations"] = []
    for mu, eta, chi, phi in [(0, 0, 0, 0), (10, -20, 30, -40),
                              (123.4, -56.7, 89.1, 12.3), (-90, 90, -90, 90)]:
        fx["rotations"].append({
            "angles": [mu, eta, chi, phi],
            "Z": _m(sample_matrix(mu, eta, chi, phi)),
        })
    fx["detector_matrices"] = []
    for nu, delta in [(0, 0), (12.5, -7.25), (-33, 41), (90, 90)]:
        fx["detector_matrices"].append({
            "nu_delta": [nu, delta],
            "R": _m(detector_matrix(nu, delta)),
        })

    # -- B matrices for several lattices
    fx["b_matrices"] = []
    for cell in [(5.917, 5.917, 5.917, 90, 90, 90),
                 (8.37, 8.37, 11.83, 90, 90, 90),
                 (10.0, 12.0, 15.0, 90, 103.0, 90),
                 (7.1, 8.2, 9.3, 88.0, 95.5, 101.2)]:
        fx["b_matrices"].append({"cell": list(cell), "B": _m(b_matrix(*cell))})

    # -- structure factors on the bundled structure
    from xrays_on_detector.crystal import Crystal
    crystal = Crystal.from_cif(structure_doc["_cif_path"])
    hkl = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0], [2, 1, 1],
                    [2, 2, 0], [3, 1, 0], [3, 2, 1], [4, 0, 0], [0, 0, 2],
                    [-1, 2, -3], [5, 1, 2]], dtype=int)
    fx["structure_factors"] = {
        "structure": structure_doc["name"],
        "B": _m(crystal.B),
        "hkl": [[int(x) for x in r] for r in hkl],
        "F2": [float(v) for v in crystal.structure_factor_mag2(hkl)],
    }

    # -- detector frame and projection
    det = LabDetector(distance=200.0, n_fast=1475, n_slow=1679,
                      pixel_size=0.172, nu=8.5, delta=-12.25)
    centre, normal, e_fast, e_slow, arm = det.frame()
    dirs = []
    rng = np.random.default_rng(7)
    for _ in range(40):
        d = rng.normal(size=3)
        d[1] = abs(d[1]) + 0.6            # forward-ish
        dirs.append(d / np.linalg.norm(d))
    dirs = np.array(dirs)
    f_px, s_px, inside, cos_inc = det.project(dirs)
    fx["detector"] = {
        "spec": {"distance": 200.0, "nFast": 1475, "nSlow": 1679,
                 "pixelSize": 0.172, "nu": 8.5, "delta": -12.25},
        "centre": _v(centre), "normal": _v(normal),
        "eFast": _v(e_fast), "eSlow": _v(e_slow), "arm": _v(arm),
        "maxQmax": float(det.max_Qmax(0.7293)),
        "rays": [{"khat": _v(d), "fast": float(f), "slow": float(s),
                  "inside": bool(i), "cosInc": float(c)}
                 for d, f, s, i, c in zip(dirs, f_px, s_px, inside, cos_inc)],
    }

    # -- a full Ewald pass
    inst = Instrument(wavelength=0.7293, distance=200.0, n_fast=1475,
                      n_slow=1679, pixel_size=0.172)
    inst.crystal = LatticeCrystal.from_cell(5.917, 5.917, 5.917)
    inst.mu, inst.eta, inst.chi, inst.phi = 3.0, 14.0, -22.0, 47.0
    inst.U = euler_matrix(7.0, -13.0, 21.0)
    inst.build_reflection_list()
    shot = inst.shoot(bin_factor=1)
    order = np.lexsort((shot.refl.hkl[:, 2], shot.refl.hkl[:, 1],
                        shot.refl.hkl[:, 0]))
    fx["ewald"] = {
        "cell": [5.917, 5.917, 5.917, 90, 90, 90],
        "wavelength": 0.7293, "sigma": inst.sigma, "nSigma": inst.n_sigma,
        "angles": {"mu": 3.0, "eta": 14.0, "chi": -22.0, "phi": 47.0},
        "U": _m(inst.U),
        "n_hkl": int(len(inst.hkl)),
        "reflections": [
            {"hkl": [int(x) for x in shot.refl.hkl[i]],
             "khat": _v(shot.refl.khat[i]),
             "eps": float(shot.refl.eps[i]),
             "excitation": float(shot.refl.excitation[i]),
             "twoTheta": float(shot.refl.two_theta[i])}
            for i in order
        ],
    }

    # -- orientation tools
    ori = Instrument(wavelength=0.7293, distance=200.0)
    ori.crystal = LatticeCrystal.from_cell(10.0, 12.0, 15.0, 90, 103.0, 90)
    ori.mu, ori.eta, ori.chi, ori.phi = 12.0, 7.0, -20.0, 33.0
    cases = []
    for indices, kind, target in [((1, 1, 0), "hkl", [0, 1, 0]),
                                  ((1, 0, 1), "uvw", [0, 0, 1]),
                                  ((0, 0, 1), "hkl", [1, 0, 0])]:
        ori.U = np.eye(3)
        ori.align_in_lab(indices, np.array(target, float), kind=kind)
        U1 = ori.U.copy()
        ori.align_secondary_in_lab((0, 1, 0), np.array([1, 0, 0], float),
                                   np.array(target, float), kind="hkl")
        cases.append({
            "indices": list(indices), "kind": kind, "target": target,
            "U_primary": _m(U1), "U_secondary": _m(ori.U),
            "crystal_vector": _v(ori.crystal_vector(indices, kind)),
        })
    fx["orientation"] = {
        "cell": [10.0, 12.0, 15.0, 90, 103.0, 90],
        "angles": {"mu": 12.0, "eta": 7.0, "chi": -20.0, "phi": 33.0},
        "cases": cases,
        "rotation_between": [
            {"from": _v(a), "to": _v(b), "R": _m(rotation_between(a, b))}
            for a, b in [(np.array([1.0, 0, 0]), np.array([0, 1.0, 0])),
                         (np.array([1.0, 2, 3]), np.array([-3.0, 1, 2])),
                         (np.array([0, 0, 1.0]), np.array([0, 0, -1.0])),
                         (np.array([1.0, 0, 0]), np.array([1.0, 0, 0]))]],
        "euler": [{"rxyz": [rx, ry, rz], "R": _m(euler_matrix(rx, ry, rz))}
                  for rx, ry, rz in [(0, 0, 0), (10, 20, 30), (-95, 40, 175)]],
    }

    # -- UB conventions
    ub = Instrument(wavelength=0.7293)
    ub.crystal = LatticeCrystal.from_cell(5.917, 5.917, 5.917)
    ub.U = euler_matrix(11.0, -6.0, 23.0)
    fx["ub"] = {
        "cell": [5.917, 5.917, 5.917, 90, 90, 90],
        "wavelength": 0.7293, "U": _m(ub.U),
        "2pi": _m(ub.UB("2pi")), "1/d": _m(ub.UB("1/d")),
        "lambda": _m(ub.UB("lambda")),
    }

    # -- solvers
    sv = Instrument(wavelength=0.7293, distance=200.0)
    sv.crystal = LatticeCrystal.from_cell(5.917, 5.917, 5.917)
    sv.mu, sv.chi, sv.phi = 4.0, 13.0, -27.0
    sv.U = euler_matrix(11.0, -6.0, 23.0)
    solves = []
    for hkl_t in [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 0, 0), (2, 1, 1),
                  (0, 0, 1), (0, 0, 3), (3, 1, 0), (9, 9, 9)]:
        r = sv.eta_reach(hkl_t)
        entry = {"hkl": list(hkl_t),
                 "reach": {k: (bool(v) if isinstance(v, (bool, np.bool_))
                               else float(v)) for k, v in r.items()},
                 "solutions": [float(e) for e in sv.solve_eta(hkl_t)]
                 if r["in_limiting_sphere"] else []}
        if not r["feasible"] and r["in_limiting_sphere"]:
            c = sv.suggest_chi(hkl_t)
            entry["suggest_chi"] = None if c is None else float(c)
        if entry["solutions"]:
            saved = sv.eta
            sv.eta = entry["solutions"][0]
            d, g = sv.aim_detector_at(hkl_t)
            entry["aim_at_first"] = {"eta": entry["solutions"][0],
                                     "delta": float(d), "gamma": float(g)}
            sv.eta = saved
        solves.append(entry)
    fx["solvers"] = {
        "cell": [5.917, 5.917, 5.917, 90, 90, 90], "wavelength": 0.7293,
        "angles": {"mu": 4.0, "eta": 0.0, "chi": 13.0, "phi": -27.0},
        "U": _m(sv.U), "cases": solves,
        "detector_angles_for": [
            {"khat": _v(LabDetector(distance=100, n_fast=10, n_slow=10,
                                    pixel_size=1, delta=d, nu=g).frame()[4]),
             "delta": d, "gamma": g}
            for d, g in [(0, 0), (15, -20), (-42.5, 33.25), (60, 60)]],
    }
    return fx


def main():
    data_dir = os.path.join(WEB, "data")
    test_dir = os.path.join(WEB, "test")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print("scattering factors:")
    export_scattering_factors(data_dir)

    print("colour maps:")
    export_colormaps(data_dir)

    # Any CIF given on the command line is bundled as a selectable structure;
    # with none given, the CsPbBr3 example is the default. The index file is
    # what the app reads, so adding a structure needs no JavaScript edit.
    print("structures:")
    root = os.path.dirname(WEB)
    default = sorted(glob.glob(os.path.join(root, "examples", "structures", "*.cif")))
    cifs = sys.argv[1:] or default
    docs = []
    for cif in cifs:
        if not os.path.isfile(cif):
            sys.exit(f"no such CIF: {cif}")
        name = os.path.splitext(os.path.basename(cif))[0]
        d = export_structure(cif, name, data_dir)
        d["_cif_path"] = cif
        docs.append(d)
    index = os.path.join(data_dir, "structures.json")
    with open(index, "w", encoding="utf-8") as fh:
        json.dump([d["name"] for d in docs], fh, separators=(",", ":"))
    print(f"  index: {len(docs)} structure(s) -> "
          f"{os.path.relpath(index, WEB)}")
    doc = docs[0]

    print("parity fixture:")
    fx = build_fixture(doc)
    path = os.path.join(test_dir, "fixture.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(fx, fh, separators=(",", ":"))
    print(f"  {os.path.getsize(path) / 1024:.1f} kB -> "
          f"{os.path.relpath(path, WEB)}")


if __name__ == "__main__":
    main()
