"""Crystal.from_cif keeps each site's occupancy and displacement parameter.

The browser build bundles the same P1 atom list (tools/export_web_data.py), so
the Python and the JavaScript compute |F(hkl)|^2 from the same structure,
Debye-Waller factors included.

Run:  python -m pytest tests/test_crystal.py
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from xrays_on_detector.crystal import Crystal  # noqa: E402


def test_expanded_atoms_match_the_bundled_structure():
    pytest.importorskip("ase")
    c = Crystal.from_cif(os.path.join(ROOT, "examples", "structures", "MAPbI3_Pm-3m.cif"))
    with open(os.path.join(ROOT, "web", "data", "MAPbI3_Pm-3m.json"), encoding="utf-8") as fh:
        atoms = json.load(fh)["atoms"]

    assert c.n_atoms == len(atoms)
    assert c.elements == [a["element"] for a in atoms]
    # the bundle rounds positions to six decimals, occupancy and B to four
    assert np.allclose(c.frac, [[a["x"], a["y"], a["z"]] for a in atoms], rtol=0, atol=6e-7)
    assert np.allclose(c.occ, [a["occ"] for a in atoms], rtol=0, atol=6e-5)
    assert np.allclose(c.B_iso, [a["B"] for a in atoms], rtol=0, atol=6e-5)
    # every site of this CIF gives a displacement parameter (U, converted to B)
    assert c.B_iso.min() > 3.0
