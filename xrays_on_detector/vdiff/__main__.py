"""python -m xrays_on_detector.vdiff [--no-wmi] [--sq VOLUME.h5]"""
import sys

from . import run_cli

sys.exit(run_cli())
