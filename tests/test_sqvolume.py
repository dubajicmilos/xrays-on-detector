"""Validate loading and projecting a measured S(q) volume.

Run:  python tests/test_sqvolume.py
Checks:
  A. trilinear sampling is exact on a linear field, and `inside` is right at
     the grid boundary;
  B. pixel -> hkl -> pixel round trip through project_volume's map agrees with
     Detector.project to machine precision (so the image lines up with the
     rendered spot overlay, row convention included);
  C. a synthetic peak planted at an integer hkl lands on the pixel the forward
     six-circle model predicts;
  E. oversampling a pixel matches an equally-fine panel exactly;
  D. a real rspace3d file loads and its Bragg peaks project onto the predicted
     pixels. SKIPPED unless XOD_SQ_H5 points at one.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xrays_on_detector.geometry import BEAM, sample_matrix
from xrays_on_detector.sqvolume import SqVolume, pixel_khat, project_volume
from xrays_on_detector.vdiff.instrument import (Instrument, LabDetector,
                                                LatticeCrystal, b_matrix)

SQ_H5 = os.environ.get("XOD_SQ_H5", "")


def make_volume(step=0.05, lo=-4.0, hi=4.0, cell=(5.9, 5.9, 5.9)):
    """An empty cubic-cell volume on a regular grid, ready to plant peaks in."""
    axis = np.arange(lo, hi + 0.5 * step, step)
    n = axis.size
    a, b, c = cell
    recip = np.diag([1.0 / a, 1.0 / b, 1.0 / c])       # columns a*, b*, c* in 1/d
    return SqVolume(
        data=np.zeros((n, n, n), np.float32),
        H=axis, K=axis.copy(), L=axis.copy(),
        recip=recip,
        cell=dict(a=a, b=b, c=c, alpha=90.0, beta=90.0, gamma=90.0),
        wavelength=0.7, name="synthetic",
    )


def test_trilinear():
    print("== Test A: trilinear sampling ==")
    rng = np.random.default_rng(0)
    vol = make_volume(step=0.05, lo=-2.0, hi=2.0)

    # A linear field must be reproduced exactly by trilinear interpolation.
    Hg, Kg, Lg = np.meshgrid(vol.H, vol.K, vol.L, indexing="ij")
    vol.data = (3.0 + 2.0 * Hg - 5.0 * Kg + 0.5 * Lg).astype(np.float32)

    pts = rng.uniform(-1.9, 1.9, size=(4000, 3))
    got, inside = vol.sample(pts)
    want = 3.0 + 2.0 * pts[:, 0] - 5.0 * pts[:, 1] + 0.5 * pts[:, 2]
    assert inside.all()
    err = np.max(np.abs(got - want))
    print(f"  linear field, {len(pts)} points: max error {err:.2e}")
    assert err < 1e-5           # float32 storage sets the floor

    # Exact voxel centres, including the last one on each axis.
    corners = np.array([[h, k, l] for h in (vol.H[0], vol.H[-1])
                        for k in (vol.K[0], vol.K[-1])
                        for l in (vol.L[0], vol.L[-1])])
    got, inside = vol.sample(corners)
    want = 3.0 + 2.0 * corners[:, 0] - 5.0 * corners[:, 1] + 0.5 * corners[:, 2]
    assert inside.all(), "grid corners must count as measured"
    err = np.max(np.abs(got - want))
    print(f"  grid corners: max error {err:.2e}")
    assert err < 1e-5

    # Just outside is not measured, and reads zero rather than an edge value.
    out = corners * 1.0
    out[:, 0] += np.sign(out[:, 0]) * 0.6 * (vol.H[1] - vol.H[0])
    got, inside = vol.sample(out)
    assert not inside.any(), "points beyond the last voxel must be outside"
    assert np.all(got == 0.0)
    print("  outside the grid: inside=False, value=0")
    print("  PASS\n")


def test_pixel_roundtrip():
    print("== Test B: pixel -> hkl -> pixel round trip ==")
    vol = make_volume()
    B = b_matrix(**vol.cell)
    rng = np.random.default_rng(1)
    wl = 0.7

    for nu, delta, (mu, eta, chi, phi) in [
        (0.0, 0.0, (0, 0, 0, 0)),
        (12.0, -7.0, (3.0, 40.0, 15.0, -25.0)),
        (-20.0, 30.0, (0.0, 130.0, -60.0, 88.0)),
    ]:
        det = LabDetector(distance=180.0, n_fast=61, n_slow=47, pixel_size=1.0,
                          nu=nu, delta=delta)
        U = np.linalg.qr(rng.standard_normal((3, 3)))[0]
        if np.linalg.det(U) < 0:
            U[:, 0] *= -1.0
        ZUB = sample_matrix(mu, eta, chi, phi) @ U @ B

        k = 2.0 * np.pi / wl
        khat = pixel_khat(det)                      # (n_slow, n_fast, 3)
        hkl = (k * (khat - BEAM)) @ np.linalg.inv(ZUB).T

        # Forward again: hkl -> Q -> kf -> the pixel the renderer would use.
        Q = hkl.reshape(-1, 3) @ ZUB.T
        kf = k * BEAM + Q
        fast_px, slow_px, inside, _ = det.project(kf / np.linalg.norm(kf, axis=1)[:, None])
        row = (det.n_slow - 1) - slow_px

        rows, cols = np.mgrid[0:det.n_slow, 0:det.n_fast]
        e_row = np.max(np.abs(row - rows.ravel()))
        e_col = np.max(np.abs(fast_px - cols.ravel()))
        # The panel edge is exactly on Detector.project's inclusive bound, so
        # a 1e-14 round-off can tip a border pixel out; interior ones cannot.
        interior = ((rows > 0) & (rows < det.n_slow - 1)
                    & (cols > 0) & (cols < det.n_fast - 1)).ravel()
        print(f"  nu={nu:6.1f} delta={delta:6.1f}: max |drow|={e_row:.2e} "
              f"|dcol|={e_col:.2e} px, interior all on the panel="
              f"{bool(inside[interior].all())}")
        assert e_row < 1e-9 and e_col < 1e-9
        assert inside[interior].all()
    print("  PASS\n")


def test_planted_peak():
    print("== Test C: a planted peak lands where the six-circle model says ==")
    vol = make_volume(step=0.02, lo=-3.0, hi=3.0)
    inst = Instrument(wavelength=0.7, distance=200.0, n_fast=487, n_slow=619,
                      pixel_size=0.172, preview_bin=1)
    inst.crystal = LatticeCrystal.from_cell(**vol.cell, name="synthetic")

    # eta turns about lab -z, so a reflection lying along it (00l at U = I) sits
    # in the blind cone of a single circle and is deliberately not in this list.
    peaks = [(1, 1, 0), (0, 2, 0), (2, -1, 1), (1, 0, -1)]
    Hg, Kg, Lg = np.meshgrid(vol.H, vol.K, vol.L, indexing="ij")
    for h, k, l in peaks:                      # narrow Gaussian at each peak
        r2 = (Hg - h) ** 2 + (Kg - k) ** 2 + (Lg - l) ** 2
        vol.data += np.exp(-r2 / (2.0 * 0.03 ** 2)).astype(np.float32)

    n_checked = 0
    for hkl in peaks:
        etas = inst.solve_eta(hkl)
        assert etas, f"{hkl} cannot be brought onto the sphere"
        inst.eta = etas[0]
        aim_delta, aim_gamma = inst.aim_detector_at(hkl)

        # Once on the beam centre, once well off it: aiming alone would only
        # ever test the middle pixel, which cannot catch a transposed or
        # flipped axis.
        for d_off, g_off in ((0.0, 0.0), (-3.5, 2.5)):
            inst.delta, inst.gamma = aim_delta + d_off, aim_gamma + g_off
            det = inst.detector_obj(1)
            ZUB = inst.sample_M() @ inst.U @ inst.crystal.B
            image, coverage = project_volume(det, vol, inst.wavelength, ZUB)

            r, c = np.unravel_index(int(np.argmax(image)), image.shape)
            # where the forward model puts it
            kvec = 2.0 * np.pi / inst.wavelength * BEAM + inst.q_lab(hkl)
            f_px, s_px, inside, _ = det.project(
                (kvec / np.linalg.norm(kvec))[None, :])
            want_r = (det.n_slow - 1) - s_px[0]
            d = np.hypot(r - want_r, c - f_px[0])
            print(f"  {hkl}: eta={inst.eta:8.3f} delta={inst.delta:7.3f} "
                  f"gamma={inst.gamma:7.3f}  brightest px ({c:4d},{r:4d})  "
                  f"predicted ({f_px[0]:7.2f},{want_r:7.2f})  off {d:.2f} px  "
                  f"coverage {coverage:.2f}")
            assert bool(inside[0])
            assert d <= 1.5, f"{hkl} landed {d:.2f} px from the prediction"
            n_checked += 1
    print(f"  {n_checked} placements, all within 1.5 px")
    print("  PASS\n")


def test_oversampling():
    print("== Test E: oversampling averages the right sub-pixels ==")
    rng = np.random.default_rng(2)
    vol = make_volume(step=0.05, lo=-3.0, hi=3.0)
    vol.data = rng.random(vol.data.shape).astype(np.float32)   # nothing smooth
    B = b_matrix(**vol.cell)
    ZUB = sample_matrix(2.0, 35.0, -10.0, 20.0) @ np.eye(3) @ B

    n = 3
    coarse = LabDetector(distance=150.0, n_fast=40, n_slow=32, pixel_size=0.6,
                         nu=8.0, delta=-5.0)
    # A panel whose pixel centres sit exactly on the coarse panel's sub-samples:
    # pixel i' = j*n + m of pitch p/n lands where sub-sample m of pixel j does.
    fine = LabDetector(
        distance=coarse.distance, n_fast=coarse.n_fast * n,
        n_slow=coarse.n_slow * n, pixel_size=coarse.pixel_size / n,
        nu=coarse.nu, delta=coarse.delta,
        beam_center_fast=coarse.beam_center_fast * n + n / 2 - 0.5,
        beam_center_slow=coarse.beam_center_slow * n + n / 2 - 0.5,
    )
    got, _ = project_volume(coarse, vol, 0.7, ZUB, oversample=n)
    fine_img, _ = project_volume(fine, vol, 0.7, ZUB)
    want = fine_img.reshape(coarse.n_slow, n, coarse.n_fast, n).mean(axis=(1, 3))

    err = float(np.max(np.abs(got - want)))
    print(f"  {n}x{n} sub-samples vs an {n}x finer panel: max error {err:.2e}")
    assert err < 1e-12
    single, _ = project_volume(coarse, vol, 0.7, ZUB)
    print("  and it does change the image: mean |os=3 - os=1| = "
          f"{float(np.abs(got - single).mean()):.4f}")
    assert not np.allclose(got, single)
    print("  PASS\n")


def test_real_file():
    print("== Test D: a real rspace3d volume ==")
    if not SQ_H5 or not os.path.isfile(SQ_H5):
        print("  SKIP (set XOD_SQ_H5 to an rspace3d .h5)\n")
        return
    vol = SqVolume.from_h5(SQ_H5)
    print("  " + vol.describe().replace("\n", "\n  "))
    print(f"  NaN voxels replaced: {vol.nan_frac:.3%}")

    inst = Instrument(wavelength=vol.wavelength, distance=200.0,
                      n_fast=487, n_slow=619, pixel_size=0.172, preview_bin=1)
    inst.crystal = LatticeCrystal.from_cell(**vol.cell, name=vol.name)

    # Take the strongest measured Bragg peaks and check each lands where the
    # forward model predicts once the motors are driven onto it.
    hkl_int = np.array([[h, k, l] for h in range(-3, 4) for k in range(-3, 4)
                        for l in range(-3, 4) if (h, k, l) != (0, 0, 0)])
    vals, inside = vol.sample(hkl_int.astype(float))
    order = np.argsort(-np.where(inside, vals, -np.inf))[:6]

    worst = 0.0
    for i in order:
        hkl = tuple(int(x) for x in hkl_int[i])
        etas = inst.solve_eta(hkl)
        if not etas:
            print(f"  {hkl}: unreachable by eta, skipped")
            continue
        inst.eta = etas[0]
        inst.delta, inst.gamma = inst.aim_detector_at(hkl)
        det = inst.detector_obj(1)
        ZUB = inst.sample_M() @ inst.U @ inst.crystal.B
        image, coverage = project_volume(det, vol, inst.wavelength, ZUB)

        r, c = np.unravel_index(int(np.argmax(image)), image.shape)
        d = np.hypot(r - (det.n_slow - 1) / 2.0, c - (det.n_fast - 1) / 2.0)
        print(f"  {hkl}: I(hkl)={vals[i]:10.1f}  brightest px ({c},{r})  "
              f"beam centre ({(det.n_fast - 1) / 2:.1f},{(det.n_slow - 1) / 2:.1f})"
              f"  off {d:.2f} px  coverage {coverage:.2f}")
        worst = max(worst, d)
    assert worst < 4.0, f"aimed peaks land up to {worst:.1f} px off the centre"
    print(f"  worst offset {worst:.2f} px")
    print("  PASS\n")


if __name__ == "__main__":
    test_trilinear()
    test_pixel_roundtrip()
    test_planted_peak()
    test_oversampling()
    test_real_file()
    print("all S(q) volume tests passed")
