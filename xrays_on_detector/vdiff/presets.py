"""Detector and sample presets for the virtual diffractometer.

Pixel counts are (fast, slow) = (columns, rows) as the modules are usually
mounted, and match the Dectris datasheet module layouts. Sizes in mm.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorPreset:
    name: str
    n_fast: int
    n_slow: int
    pixel_size: float        # mm
    note: str = ""

    @property
    def width_mm(self) -> float:
        return self.n_fast * self.pixel_size

    @property
    def height_mm(self) -> float:
        return self.n_slow * self.pixel_size

    def label(self) -> str:
        return (f"{self.name}  ({self.n_fast}x{self.n_slow}, "
                f"{self.pixel_size * 1000:.0f} um, "
                f"{self.width_mm:.0f}x{self.height_mm:.0f} mm)")


DETECTORS = [
    DetectorPreset("PILATUS3 100K", 487, 195, 0.172),
    DetectorPreset("PILATUS3 300K", 487, 619, 0.172),
    DetectorPreset("PILATUS3 1M", 981, 1043, 0.172),
    DetectorPreset("PILATUS3 2M", 1475, 1679, 0.172, "as used on Diamond I15"),
    DetectorPreset("PILATUS3 6M", 2463, 2527, 0.172),
    DetectorPreset("EIGER2 X 1M", 1028, 1062, 0.075),
    DetectorPreset("EIGER2 X 4M", 2068, 2162, 0.075, "as used on Diamond I19-2"),
    DetectorPreset("EIGER2 X 9M", 3108, 3262, 0.075),
    DetectorPreset("EIGER2 X 16M", 4148, 4362, 0.075),
    DetectorPreset("LAMBDA 750K", 1554, 516, 0.055),
    DetectorPreset("JUNGFRAU 1M", 1024, 1024, 0.075),
]

DETECTOR_BY_NAME = {d.name: d for d in DETECTORS}


@dataclass(frozen=True)
class CellPreset:
    """A bare lattice used when no CIF is loaded, so the app runs immediately."""
    name: str
    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0

    def label(self) -> str:
        return f"{self.name}  (a={self.a:g} b={self.b:g} c={self.c:g} A)"


CELLS = [
    CellPreset("Pseudocubic perovskite", 5.917, 5.917, 5.917),
    CellPreset("MAPbI3 cubic (340 K)", 6.29, 6.29, 6.29),
    CellPreset("Silicon", 5.431, 5.431, 5.431),
    CellPreset("LaB6", 4.1569, 4.1569, 4.1569),
    CellPreset("Tetragonal I4/mcm", 8.37, 8.37, 11.83),
    CellPreset("Small molecule", 10.0, 12.0, 15.0, 90.0, 103.0, 90.0),
]
