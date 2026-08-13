"""Export reference values for web/test/parity_sc.mjs from the Python.

    python tools/export_sc_fixture.py

The Python single_crystal package is the reference: it is the one checked
against pymatgen in tests/test_single_crystal.py. This writes what it computes
so the JavaScript port can be held to the same numbers, and also writes the
bundled structures out in the shape the browser reads, so both sides start
from an identical atom list rather than from two readings of the same CIF.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import single_crystal as sc  # noqa: E402
from single_crystal import display, scatter  # noqa: E402

STRUCTURES = os.path.join(ROOT, "examples", "structures")
OUT = os.path.join(ROOT, "web", "test", "fixture_sc.json")

# Small enough to keep the fixture readable, varied enough to catch a real
# error: a cubic cell where a wrong projection still looks fine, and a
# 188-atom low-symmetry one where it does not.
CASES = [
    ("CsPbBr3", [(0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 2, 3)]),
    ("PEA2PbBr4", [(0, 0, 1), (1, 1, 0), (0, 1, 2)]),
]


def flat(a):
    return [float(v) for v in np.asarray(a).ravel()]


def structure_doc(doc):
    """The bundled-structure shape the browser reads."""
    return {
        "name": doc.name,
        "spaceGroup": doc.space_group,
        "cell": {
            "a": doc.cell.a, "b": doc.cell.b, "c": doc.cell.c,
            "alpha": doc.cell.alpha, "beta": doc.cell.beta, "gamma": doc.cell.gamma,
        },
        "atoms": [
            {
                "element": a.element, "nuclide": a.nuclide,
                "x": a.x, "y": a.y, "z": a.z, "occ": a.occ, "B": a.B,
            }
            for a in doc.atoms
        ],
    }


def main() -> int:
    fixture = {"structures": {}, "sections": [], "tem": [], "powder": [],
               "factors": [], "stretch": [], "hkl_labels": []}

    for name, zones in CASES:
        doc = sc.read_cif(os.path.join(STRUCTURES, name + ".cif"))
        xtal = sc.Structure.from_cif(doc)
        fixture["structures"][name] = structure_doc(doc)

        for uvw in zones:
            for layer in (0, 1):
                for radiation in ("xray", "neutron", "electron"):
                    s = sc.compute_section(
                        xtal, uvw=uvw, layer=layer, d_min=1.1, radiation=radiation
                    )
                    fixture["sections"].append({
                        "structure": name, "uvw": list(uvw), "layer": layer,
                        "d_min": 1.1, "radiation": radiation,
                        "count": len(s),
                        "g1": [int(v) for v in s.g1],
                        "g2": [int(v) for v in s.g2],
                        "height": float(s.height),
                        "zone_factor": int(s.zone_factor),
                        "hkl": [int(v) for v in s.hkl.ravel()],
                        "x": flat(s.x), "y": flat(s.y),
                        "q": flat(s.q), "d": flat(s.d),
                        "intensity": flat(s.intensity),
                    })

        t = sc.compute_tem(xtal, uvw=(0, 0, 1), kv=200, thickness=50,
                           d_min=0.8, max_zone=1)
        fixture["tem"].append({
            "structure": name, "uvw": [0, 0, 1], "kv": 200, "thickness": 50,
            "d_min": 0.8, "max_zone": 1, "count": len(t),
            "wavelength": float(t.wavelength),
            "zone_radii": [None if r is None else float(r) for r in t.zone_radii],
            "hkl": [int(v) for v in t.hkl.ravel()],
            "x": flat(t.x), "y": flat(t.y),
            "s_g": flat(t.s_g), "intensity": flat(t.intensity),
            "laue_zone": [int(v) for v in t.laue_zone],
        })

        for radiation, wl, tt in (("xray", 1.5406, 70.0), ("neutron", 1.5406, 70.0)):
            p = sc.compute_powder(xtal, wavelength=wl, radiation=radiation,
                                  two_theta_max=tt)
            fixture["powder"].append({
                "structure": name, "radiation": radiation, "wavelength": wl,
                "two_theta_max": tt, "count": len(p),
                "two_theta": flat(p.two_theta), "intensity": flat(p.intensity),
                "d": flat(p.d),
                "multiplicity": [int(v) for v in p.multiplicity],
                "hkl": [list(map(int, h)) for h in p.hkl],
            })

    # Atomic factors on their own, so a table mismatch is not diagnosed as a
    # structure factor bug.
    for radiation in ("xray", "neutron", "electron"):
        for sym in ("H", "D", "C", "O", "Br", "Cs", "Pb", "Sn", "V"):
            if sym in scatter.missing_for(radiation, [sym]):
                continue
            for s in (0.0, 0.15, 0.4, 0.9):
                fixture["factors"].append({
                    "radiation": radiation, "symbol": sym, "s": s,
                    "f": float(scatter.factors(radiation, [sym], s)[0]),
                })

    # The display stretch, which both apps must agree on or the same pattern
    # looks different in each.
    I = np.array([0.0, 1e-9, 1e-6, 1e-3, 0.1, 0.5, 1.0, 3.0])
    for gain in (1.0, 10.0, 1000.0):
        for log in (True, False):
            fixture["stretch"].append({
                "intensity": flat(I), "gain": gain, "log": log,
                "value": flat(display.stretch(I, gain=gain, log=log)),
            })

    for hkl in ([1, 0, 0], [-1, 0, 0], [-4, 4, 3], [1, -1, 0], [0, 0, -6],
                [12, -1, 0], [-11, 2, -3]):
        fixture["hkl_labels"].append({"hkl": hkl, "text": display.format_hkl(hkl)})

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, separators=(",", ":"))
    size = os.path.getsize(OUT) / 1e6
    print(f"wrote {os.path.relpath(OUT, ROOT)}  ({size:.2f} MB)")
    print(f"  {len(fixture['sections'])} sections, {len(fixture['tem'])} TEM, "
          f"{len(fixture['powder'])} powder, {len(fixture['factors'])} factors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
