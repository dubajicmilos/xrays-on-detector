"""python -m xrays_on_detector.vdiff [--no-wmi]"""
import sys

from . import run

sys.exit(run(no_wmi="--no-wmi" in sys.argv))
