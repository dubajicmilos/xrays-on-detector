"""Validate the xrays_on_detector pipeline against analytic physics.

Run:  python tests/validate.py
Checks:
  A. detector ray projection vs the exact identity r = D * tan(psi);
  B. Ewald / Bragg self-consistency  sin(theta) = |Q| / (2k);
  C. a NaCl frame renders sensible spots (and NaCl |F| systematics).
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xrays_on_detector import Crystal, Detector, simulate_frame
from xrays_on_detector.geometry import BEAM, detector_matrix


def test_detector_geometry():
    print("== Test A: detector projection identity r = D tan(psi) ==")
    rng = np.random.default_rng(0)
    for nu, delta in [(0.0, 0.0), (10.0, 20.0), (-15.0, 5.0)]:
        det = Detector(distance=100.0, n_fast=4001, n_slow=4001,
                       pixel_size=0.05, nu=nu, delta=delta)
        arm = detector_matrix(nu, delta) @ BEAM
        dirs = arm + 0.3 * rng.standard_normal((2000, 3))
        khat = dirs / np.linalg.norm(dirs, axis=1)[:, None]
        fpx, spx, inside, _ = det.project(khat)
        r_meas = det.pixel_size * np.hypot(fpx[inside] - det.beam_center_fast,
                                           spx[inside] - det.beam_center_slow)
        psi = np.arccos(np.clip(khat[inside] @ arm, -1, 1))
        err = np.max(np.abs(r_meas - det.distance * np.tan(psi)))
        print(f"  nu={nu:6.1f} delta={delta:6.1f}: n={inside.sum():4d} maxerr={err:.2e} mm")
        assert err < 1e-6
    print("  PASS\n")


def test_bragg_consistency(crystal, wavelength):
    print("== Test B: Ewald / Bragg self-consistency ==")
    k = 2 * np.pi / wavelength
    det = Detector(distance=100.0, n_fast=6001, n_slow=6001, pixel_size=0.1)
    hkl = crystal.hkl_within_Qmax(det.max_Qmax(wavelength))
    Fmag2 = crystal.structure_factor_mag2(hkl)
    print(f"  reflections in range: {len(hkl)}")

    best = {}
    for eta in np.arange(0, 180, 0.25):
        r = simulate_frame(crystal, det, wavelength, sigma=0.03, eta=eta,
                           hkl=hkl, Fmag2=Fmag2, n_sigma=6.0).all_reflections
        Qmag = np.linalg.norm(crystal.q_cryst(r.hkl), axis=1)
        for j in range(len(r.hkl)):
            key = tuple(int(x) for x in r.hkl[j])
            ae = abs(r.eps[j])
            if key not in best or ae < best[key][0]:
                best[key] = (ae, r.two_theta[j], Qmag[j])

    on = [(tt, Q) for (ae, tt, Q) in best.values() if ae < 2e-3]
    tt = np.array([t for t, _ in on])
    Q = np.array([q for _, q in on])
    err = np.abs(np.sin(tt / 2.0) - Q / (2.0 * k))
    print(f"  reflections reaching sphere: {len(tt)}, "
          f"2theta {np.degrees(tt).min():.1f}..{np.degrees(tt).max():.1f} deg")
    print(f"  max |sin_theta_geom - sin_theta_bragg| = {err.max():.2e}")
    assert err.max() < 1e-3
    print("  PASS\n")


def test_structure_factors(crystal):
    print("== Test C: NaCl |F| systematics ==")
    checks = {(2, 0, 0): "strong", (2, 2, 0): "strong", (4, 0, 0): "strong",
              (1, 1, 1): "weak", (1, 0, 0): "extinct", (2, 1, 0): "extinct"}
    F2 = crystal.structure_factor_mag2(np.array(list(checks)))
    for (hkl, kind), f2 in zip(checks.items(), F2):
        mag = f2 ** 0.5
        print(f"  {hkl} |F|={mag:7.2f}  ({kind})")
        if kind == "extinct":
            assert mag < 1e-6
        elif kind == "strong":
            assert mag > 40
    print("  PASS\n")


if __name__ == "__main__":
    test_detector_geometry()

    # B and C need |F(hkl)| from a CIF, which goes through pytilting and ASE.
    # Test A is pure geometry, so a clone without those still gets a result:
    # say what was skipped rather than fail as if the physics were wrong.
    try:
        from ase.build import bulk
        cif = os.path.join(tempfile.gettempdir(), "xod_nacl.cif")
        bulk("NaCl", crystalstructure="rocksalt", a=5.64, cubic=True).write(cif)
        crystal = Crystal.from_cif(cif, expand_symmetry=True)
    except (ImportError, FileNotFoundError) as exc:
        print(f"SKIP tests B and C: {exc}\n")
        print("GEOMETRY VALIDATION PASSED (structure-factor tests skipped)")
        sys.exit(0)

    print(f"NaCl loaded: {crystal.n_atoms} atoms\n")
    test_structure_factors(crystal)
    test_bragg_consistency(crystal, wavelength=0.7)
    print("ALL VALIDATION PASSED")
