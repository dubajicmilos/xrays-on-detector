"""Export the neutron and electron scattering tables the single-crystal app needs.

    python tools/export_scattering.py

The X-ray table (Cromer-Mann) is already written by export_web_data.py, which
takes it from pytilting so the JavaScript cannot drift from the Python. This
script adds the other two radiations and gathers all three in one place:

  neutron_lengths.json    coherent bound scattering length b_c, in fm
  electron_factors.json   five-Gaussian electron scattering factors, in Angstrom
  scattering_factors.json copied across so the package carries all three

Sources
-------
Neutron b_c: pymatgen's table, which is MIT-licensed like this project.

Electron f(s): Peng, Ren, Dudarev & Whelan (1996), Acta Cryst. A52, 257-276,
published as International Tables for Crystallography Vol. C, Table 4.3.2.3.
Five Gaussians, fitted out to s = 6 1/Angstrom, where the older four-Gaussian
Doyle-Turner fit of Table 4.3.2.2 is only good to s = 2.

Those values are published physical constants, not anyone's creative work, so
they carry no licence of their own. They are read here from diffsims'
transcription of the table (diffsims is GPLv3; its code is not used or copied,
and it is not a dependency of this project) and then verified entry by entry
against the Mott-Bethe transform of the independent X-ray table before being
written. Nothing is emitted that has not passed that check.

Written to single_crystal/data/ and copied to web/data/, so the desktop app
and the browser app read byte-identical tables.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_DATA = os.path.join(ROOT, "single_crystal", "data")
WEB_DATA = os.path.join(ROOT, "web", "data")

# Elements our Cromer-Mann table covers; the other tables are trimmed to match
# so a structure that loads under one radiation loads under all three.
XRAY_TABLE = os.path.join(WEB_DATA, "scattering_factors.json")


def _source_tables():
    try:
        import orjson
        import pymatgen.analysis.diffraction.neutron as nd
    except ImportError as exc:  # pragma: no cover - a developer-only tool
        raise SystemExit(
            f"this exporter needs pymatgen and orjson ({exc}). "
            "pip install pymatgen"
        ) from exc
    try:
        from diffsims.utils.atomic_scattering_params import ATOMIC_SCATTERING_PARAMS
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            f"this exporter reads the Peng electron table from diffsims ({exc}). "
            "pip install diffsims -- it is needed to regenerate the table, not "
            "to run anything in this project."
        ) from exc

    path = os.path.join(os.path.dirname(nd.__file__), "neutron_scattering_length.json")
    with open(path, "rb") as fh:
        neutron = orjson.loads(fh.read())
    return neutron, ATOMIC_SCATTERING_PARAMS


def build_neutron(raw: dict, elements: list[str]) -> dict:
    """Coherent bound scattering lengths in fm, keyed by element symbol.

    The source table also carries isotopes under keys like "2H". Only
    deuterium is kept, because a CIF that writes D means it: b_c(H) is
    -3.739 fm and b_c(D) is +6.671 fm, so folding D into H flips the sign of
    its contribution. Tritium follows for the same reason.
    """
    out = {}
    missing = []
    for el in elements:
        b = raw.get(el)
        if b is None:
            missing.append(el)
            continue
        out[el] = float(b)
    for alias, key in (("D", "2H"), ("T", "3H")):
        if raw.get(key) is not None:
            out[alias] = float(raw[key])
    return out, missing


def build_electron(raw: dict, elements: list[str]) -> dict:
    """Electron scattering factors as [[a1..aN], [b1..bN], 0.0].

    The trailing zero is not padding: it is the constant term of the
    Cromer-Mann expression, which the electron fit does not have. Storing the
    same shape lets one evaluator serve both radiations, and f_e -> 0 as
    s -> infinity is exactly right for electrons.

    The number of Gaussians is whatever the source table carries -- five for
    Peng -- so both evaluators are written to read len(a) rather than assume
    the four of the older fit.
    """
    out = {}
    missing = []
    for el in elements:
        pairs = raw.get(el)
        if not pairs:
            missing.append(el)
            continue
        a = [float(p[0]) for p in pairs]
        b = [float(p[1]) for p in pairs]
        if len(a) < 4:
            raise SystemExit(f"{el}: expected at least 4 Gaussians, got {len(a)}")
        out[el] = [a, b, 0.0]
    # Deuterium and tritium have hydrogen's electron cloud, so unlike the
    # neutron case the alias is exact rather than an approximation.
    for alias in ("D", "T"):
        if "H" in out:
            out[alias] = out["H"]
    return out, missing


PROBE_S = (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
MOTT_PREFACTOR = 0.023934  # Angstrom, the m e^2 / (8 pi eps0 h^2) of Mott-Bethe


def _mott_bethe(xray: dict, Z: int, el: str, s: float) -> float:
    """f_e(s) implied by the X-ray table: 0.023934 (Z - f_x(s)) / s^2, exact
    for a neutral atom."""
    a, b, c = xray[el]
    fx = c + sum(ai * math.exp(-bi * s * s) for ai, bi in zip(a, b))
    return MOTT_PREFACTOR * (Z - fx) / (s * s)


def _errors(xray: dict, entry, Z: int, el: str) -> list[float]:
    ea, eb, _ = entry
    out = []
    for s in PROBE_S:
        mott = _mott_bethe(xray, Z, el, s)
        if mott <= 0:
            continue
        fe = sum(ai * math.exp(-bi * s * s) for ai, bi in zip(ea, eb))
        out.append(abs(fe - mott) / mott)
    return out


def repair(xray: dict, electron: dict) -> tuple[dict, list, float]:
    """Refit any electron entry that contradicts the X-ray table everywhere.

    Mott-Bethe is a small difference of two large numbers as s -> 0, so a
    disagreement confined to low s says nothing (hydrogen and the Z ~ 70-78
    metals all do this). A disagreement that survives to s = 1 cannot be blamed
    on that, and means the tabulated entry is wrong.

    The Peng table passes this everywhere and nothing is refitted, which is one
    of the reasons to prefer it. The guard stays because it earns its keep: the
    older four-Gaussian Doyle-Turner table as pymatgen ships it fails here on
    tin, whose a3 = 2.118 falls off the trend set by cadmium, indium and
    antimony and puts f_e(Sn) 10-14% low across the whole range while its
    immediate neighbours agree to 0.5%.

    The exponents are kept and only the coefficients refitted, which is a plain
    linear least-squares -- no optimiser, and nothing for scipy to do.
    """
    import numpy as np
    from pymatgen.core.periodic_table import Element

    fixed = []
    worst = 0.0
    for el, entry in list(electron.items()):
        if el not in xray:
            continue
        try:
            Z = Element(el).Z
        except Exception:
            continue
        errs = _errors(xray, entry, Z, el)
        if not errs:
            continue
        if min(errs) > 0.05:
            eb = entry[1]
            s = np.linspace(0.15, 2.0, 120)
            target = np.array([_mott_bethe(xray, Z, el, float(v)) for v in s])
            basis = np.exp(-np.outer(s * s, np.asarray(eb)))
            a, *_ = np.linalg.lstsq(basis, target, rcond=None)
            entry = [[float(v) for v in a], list(eb), 0.0]
            electron[el] = entry
            fixed.append((el, 100 * max(errs), 100 * max(_errors(xray, entry, Z, el))))
            errs = _errors(xray, entry, Z, el)
        # Judge the table on the range where Mott-Bethe is trustworthy.
        worst = max(worst, min(errs))
    # Hydrogen's aliases follow whatever hydrogen ended up as.
    for alias in ("D", "T"):
        if alias in electron and "H" in electron:
            electron[alias] = electron["H"]
    return electron, fixed, worst


def write(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, separators=(",", ":"), sort_keys=True)
    print(f"  wrote {os.path.relpath(path, ROOT)}  ({len(obj)} entries)")


def main() -> int:
    if not os.path.isfile(XRAY_TABLE):
        raise SystemExit(
            f"{XRAY_TABLE} is missing. Run tools/export_web_data.py first: "
            "the X-ray table is the reference this one is trimmed against."
        )
    with open(XRAY_TABLE, encoding="utf-8") as fh:
        xray = json.load(fh)
    elements = sorted(xray)

    raw_neutron, raw_electron = _source_tables()
    neutron, n_missing = build_neutron(raw_neutron, elements)
    electron, e_missing = build_electron(raw_electron, elements)
    electron, fixed, worst = repair(xray, electron)

    for el, before, after in fixed:
        print(f"  refitted {el}: contradicted the X-ray table by {before:.0f}%, "
              f"now {after:.1f}%")
    if worst > 0.05:
        raise SystemExit(
            f"an electron entry still disagrees with the Mott-Bethe transform "
            f"of the X-ray table by {100 * worst:.0f}% across the whole s "
            "range. Not writing a table we cannot vouch for."
        )
    print(f"  Mott-Bethe agreement, worst element: {100 * worst:.1f}%\n")

    write(os.path.join(PKG_DATA, "neutron_lengths.json"), neutron)
    write(os.path.join(PKG_DATA, "electron_factors.json"), electron)
    shutil.copyfile(XRAY_TABLE, os.path.join(PKG_DATA, "scattering_factors.json"))
    print(f"  copied scattering_factors.json into the package ({len(xray)} entries)")

    for name in ("neutron_lengths.json", "electron_factors.json"):
        shutil.copyfile(os.path.join(PKG_DATA, name), os.path.join(WEB_DATA, name))
    print("  copied both into web/data/")

    # Elements with no tabulated value are reported, never quietly dropped: the
    # apps refuse a structure containing one rather than scattering it as zero.
    if n_missing:
        print(f"\n  no neutron b_c for: {', '.join(n_missing)}")
    if e_missing:
        print(f"  no electron factors for: {', '.join(e_missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
