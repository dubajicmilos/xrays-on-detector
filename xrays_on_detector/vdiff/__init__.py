"""Virtual six-circle diffractometer: an interactive 3D instrument simulator.

Run it with::

    python -m xrays_on_detector.vdiff

The physics is the package core (You/diffcalc circle matrices, the Ewald
construction, Gaussian reciprocal-space peaks, pytilting structure factors from
a CIF). This subpackage adds the instrument state, the transmission/reflection
distinction, the motor solvers and the Qt front end.
"""
from .instrument import Instrument, LabDetector, LatticeCrystal, b_matrix
from .presets import CELLS, DETECTORS

__all__ = ["Instrument", "LabDetector", "LatticeCrystal", "b_matrix",
           "CELLS", "DETECTORS", "run"]


def disable_wmi_probe():
    """Stop ``platform`` from asking WMI for the OS version and CPU family.

    Workaround for a *broken machine*, not a fix. On Windows ``platform.uname()``
    queries WMI, ``numpy.testing`` calls ``platform.machine()`` at import time,
    and ``scipy`` imports ``numpy.testing``. So if the Winmgmt service is hung,
    ``import scipy`` blocks forever and takes diffcalc, and therefore this whole
    package, with it.

    CPython already treats a failing WMI query as normal and falls back to the
    PROCESSOR_ARCHITECTURE environment variables, so making the query fail fast
    yields the same answer rather than a wrong one. Repair the service instead
    when you can; this only exists so a hung WMI does not stop you working.
    """
    import platform

    def _unavailable(*args, **kwargs):
        raise OSError("WMI probe disabled (see vdiff.disable_wmi_probe)")

    platform._wmi_query = _unavailable


def run(no_wmi: bool = False):
    """Launch the GUI (imports PyQt6 lazily so the physics stays importable)."""
    if no_wmi:
        disable_wmi_probe()
    from .gui import main
    return main()
