"""Virtual six-circle diffractometer: an interactive 3D instrument simulator.

Run it with::

    python -m xrays_on_detector.vdiff [--no-wmi] [--sq VOLUME.h5]

The physics is the package core (You/diffcalc circle matrices, the Ewald
construction, Gaussian reciprocal-space peaks, structure factors from a CIF).
This subpackage adds the instrument state, the transmission/reflection
distinction, the motor solvers and the Qt front end.
"""
from .instrument import Instrument, LabDetector, LatticeCrystal, b_matrix
from .presets import CELLS, DETECTORS

__all__ = ["Instrument", "LabDetector", "LatticeCrystal", "b_matrix",
           "CELLS", "DETECTORS", "run", "run_cli"]


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


def run(no_wmi: bool = False, volume: str | None = None):
    """Launch the GUI (imports PyQt6 lazily so the physics stays importable).

    `volume` is an rspace3d S(q) reconstruction (.h5) to load on start, the same
    as pressing Load S(q) ... once the window is up.
    """
    if no_wmi:
        disable_wmi_probe()
    from .gui import main
    return main(volume=volume)


def run_cli(argv=None):
    """Parse the command line and launch; shared by ``python -m`` and the example."""
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m xrays_on_detector.vdiff",
        description="Interactive virtual six-circle diffractometer.")
    p.add_argument("--no-wmi", action="store_true",
                   help="work around a hung Windows WMI service that makes "
                        "'import scipy' block forever (see disable_wmi_probe)")
    p.add_argument("--sq", metavar="VOLUME.h5",
                   help="load this rspace3d / CrysAlisPro S(q) reconstruction "
                        "on start, so the panel shows measured data")
    a = p.parse_args(argv)
    return run(no_wmi=a.no_wmi, volume=a.sq)
