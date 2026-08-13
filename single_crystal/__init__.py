"""single_crystal: kinematic single-crystal diffraction from any CIF.

Plane sections of the weighted reciprocal lattice (the precession picture),
selected-area electron diffraction, and powder patterns, for X-rays, neutrons
and electrons.

    from single_crystal import read_cif, Structure, compute_section
    doc = read_cif("mystructure.cif")          # expands the symmetry to P1
    xtal = Structure.from_cif(doc)
    sec = compute_section(xtal, uvw=(0, 0, 1), layer=0, radiation="neutron")

The desktop GUI is `python -m single_crystal`; the browser build of the same
model is web/sc/.

This package deliberately depends on numpy alone, so it neither needs nor
loads the diffcalc goniometer machinery in xrays_on_detector.
"""
from .cif import CifError, CifStructure, parse_cif, read_cif, set_elements
from .powder import PowderPattern, compute_powder
from .scatter import RADIATIONS, known_elements
from .section import Section, compute_section, reduce_zone, zone_basis
from .structure import Structure, bmatrix
from .tem import TEMPattern, compute_tem

# The CIF reader validates atom types against whatever the tables cover, so it
# has to be told before any file is read.
set_elements(known_elements())

__all__ = [
    "CifError",
    "CifStructure",
    "PowderPattern",
    "RADIATIONS",
    "Section",
    "Structure",
    "TEMPattern",
    "bmatrix",
    "compute_powder",
    "compute_section",
    "compute_tem",
    "known_elements",
    "parse_cif",
    "read_cif",
    "reduce_zone",
    "set_elements",
    "zone_basis",
]
