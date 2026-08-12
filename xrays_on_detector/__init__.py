"""xrays_on_detector: simulate the single-crystal diffraction image on an area
detector of a six-circle (You 1999) diffractometer.

Pipeline
--------
CIF --(pytilting)--> |F(hkl)|^2 and reciprocal lattice (2*pi convention)
    --(diffcalc You matrices)--> reciprocal-lattice points in the lab frame
    --(monochromatic Ewald, Gaussian peaks)--> excited reflections
    --(flat area detector on the nu/delta arm)--> rendered pixel image.
"""
from .crystal import Crystal
from .detector import Detector
from .simulate import Frame, simulate_frame

__all__ = ["Crystal", "Detector", "Frame", "simulate_frame"]
