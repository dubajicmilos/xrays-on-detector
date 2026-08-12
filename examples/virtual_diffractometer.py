"""Launch the interactive virtual six-circle diffractometer.

    python examples/virtual_diffractometer.py [--no-wmi]

Equivalent to ``python -m xrays_on_detector.vdiff``; this file exists so the app
can be started from the examples folder like the other demos.

``--no-wmi`` is a workaround for a hung Windows WMI service, which otherwise
makes ``import scipy`` (and so the whole package) block forever. See
``xrays_on_detector.vdiff.disable_wmi_probe``.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xrays_on_detector.vdiff import run  # noqa: E402

if __name__ == "__main__":
    sys.exit(run(no_wmi="--no-wmi" in sys.argv))
