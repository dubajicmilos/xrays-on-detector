"""Launch the interactive virtual six-circle diffractometer.

    python examples/virtual_diffractometer.py [--no-wmi] [--sq VOLUME.h5]

Equivalent to ``python -m xrays_on_detector.vdiff``; this file exists so the app
can be started from the examples folder like the other demos.

``--sq VOLUME.h5`` opens the app on a measured rspace3d / CrysAlisPro S(q)
reconstruction, the same as pressing Load S(q) ... once the window is up.

``--no-wmi`` is a workaround for a hung Windows WMI service, which otherwise
makes ``import scipy`` (and so the whole package) block forever. See
``xrays_on_detector.vdiff.disable_wmi_probe``.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xrays_on_detector.vdiff import run_cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_cli())
