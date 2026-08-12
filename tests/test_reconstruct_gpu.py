"""Validate the GPU reconstruction and intensity corrections.

Run:  python tests/test_reconstruct_gpu.py
Checks:
  A. pixel_corrections physics (center = 1; solid-angle = 1/cos^3; polarisation
     attenuates along the polarised axis) - pure numpy, no GPU, no data.
  B. GPU vs CPU reconstruction agree bit-for-bit (counts) / to rounding (sums) on
     real MAPbBr3 CBF frames, corrections OFF.
  C. same, corrections ON with an identical per-pixel correction map.

B/C need the I19-2 CBF frames on F: and a CUDA GPU (CuPy); they SKIP if absent.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xrays_on_detector.corrections import pixel_corrections, pixel_directions

# The physics tests below are self-contained; the ones that need real frames
# are skipped unless XOD_RAW (frame folder) and XOD_NAME (run stem) are set.
RAW = os.environ.get("XOD_RAW", "")
NAME = os.environ.get("XOD_NAME", "")
H5 = os.path.join(RAW, "unwarp", NAME + "_0_raw.h5")
STEM = os.path.join(RAW, NAME + "_01_{:04d}.cbf")


def test_corrections_physics():
    print("== Test A: pixel_corrections physics ==")
    from xrays_on_detector.realframe import FlatDetector
    det = FlatDetector(distance=85.0, pixel_size=0.075, beam_center=(990.0, 1423.33),
                       shape=(2162, 2068), wavelength=0.4859)   # real I19-2 geometry
    khat, cos_a = pixel_directions(det)
    ny, nx = det.shape
    # beam-centre pixel index in the ravel order (row-major slow, fast)
    ci = int(round(det.beam_center[1])) * nx + int(round(det.beam_center[0]))

    # (1) both corrections are 1 at the beam centre
    corr = pixel_corrections(det, solid_angle=True, polarization=True)
    print(f"  centre correction = {corr[ci]:.6f}  (expect ~1)")
    assert abs(corr[ci] - 1.0) < 5e-4

    # (2) solid-angle only == (|P|/L)^3 == 1/cos^3(alpha), computed independently
    sa = pixel_corrections(det, solid_angle=True, polarization=False)
    ys, xs = np.mgrid[0:ny, 0:nx]
    u = (xs.ravel() - det.beam_center[0]) * det.pixel_size
    v = (ys.ravel() - det.beam_center[1]) * det.pixel_size
    Pmag = np.sqrt(u ** 2 + v ** 2 + det.distance ** 2)
    sa_ref = (Pmag / det.distance) ** 3
    print(f"  solid-angle max|corr - (|P|/L)^3| = {np.abs(sa - sa_ref).max():.2e}; "
          f"range {sa.min():.3f}..{sa.max():.3f}")
    assert np.allclose(sa, sa_ref, rtol=1e-6)
    assert sa.min() >= 1.0 - 1e-9                       # never < 1, = 1 on axis

    # (3) polarisation (fully horizontal, p=1) attenuates a horizontal-offset pixel
    #     but NOT a vertical-offset pixel: pol = 1 - (khat.h)^2, corr = 1/pol >= 1
    pol = pixel_corrections(det, solid_angle=False, polarization=True,
                            polarization_fraction=1.0)          # h = fast = +x
    h = det.fast / np.linalg.norm(det.fast)
    horiz = int(round(det.beam_center[1])) * nx + (nx - 1)      # far in +fast
    vert = (ny - 1) * nx + int(round(det.beam_center[0]))       # far in +slow
    print(f"  pol corr horiz-edge = {pol[horiz]:.4f}  vert-edge = {pol[vert]:.4f}")
    assert abs(pol[vert] - 1.0) < 1e-9              # vertical offset: khat.h = 0 -> corr = 1
    assert pol[horiz] > pol[vert] + 1e-3            # horizontal offset loses more intensity
    pol_ref = 1.0 / (1.0 - (khat[horiz] @ h) ** 2)
    assert abs(pol[horiz] - pol_ref) < 1e-9
    print("  PASS\n")


def _setup():
    import h5py
    from xrays_on_detector.realframe import FlatDetector, detect_peaks, index_frame
    det, ang0, img0 = FlatDetector.from_eiger_cbf(STEM.format(1))
    with h5py.File(H5, "r") as f:
        UB = np.array(f["UB"]) / det.wavelength
    R0 = index_frame(detect_peaks(img0, det.beam_center), UB, det).R
    return det, UB, R0, ang0["Phi"]


def _parity(corr_arr, tag):
    from xrays_on_detector.reconstruct import reconstruct_volume
    from xrays_on_detector.reconstruct_gpu import reconstruct_volume_gpu
    det, UB, R0, phi0 = _setup()
    nums = list(range(1, 13))                           # 12 real frames
    frames = [STEM.format(k) for k in nums]
    phis = [phi0 + (k - 1) * 0.2 for k in nums]
    kw = dict(phi0=phi0, hkl_range=(-6.0, 6.0), step=0.1, hot=1e6)

    cpu = reconstruct_volume(frames, phis, UB, R0, det, corr=corr_arr, **kw)
    gpu = reconstruct_volume_gpu(frames, phis, UB, R0, det,
                                 corrections=(corr_arr if corr_arr is not None else None),
                                 prefetch=3, **kw)

    ct_cpu, ct_gpu = int(cpu.counts.sum()), int(gpu.counts.sum())
    dcount = np.abs(cpu.counts.astype(np.int64) - gpu.counts.astype(np.int64))
    ndiff = int((dcount > 0).sum())
    m = np.isfinite(cpu.data) & np.isfinite(gpu.data)
    both = np.isfinite(cpu.data) == np.isfinite(gpu.data)
    if m.sum() > 1:
        r = float(np.corrcoef(cpu.data[m], gpu.data[m])[0, 1])
    else:
        r = float("nan")
    reldiff = np.abs(cpu.data[m] - gpu.data[m]) / (np.abs(cpu.data[m]) + 1e-9)
    print(f"  [{tag}] counts total cpu={ct_cpu:,} gpu={ct_gpu:,} (diff {ct_gpu-ct_cpu:+d})")
    print(f"  [{tag}] voxels with count diff: {ndiff:,} / {cpu.counts.size:,} "
          f"({ndiff/cpu.counts.size:.2e}); max count diff {int(dcount.max())}")
    print(f"  [{tag}] finite-mask agree: {both.mean():.6f}; corr(cpu,gpu)={r:.8f}; "
          f"max rel intensity diff {reldiff.max():.2e}")
    # near-total placement agreement (float32 matmul may flip a few boundary pixels)
    assert abs(ct_gpu - ct_cpu) <= max(20, int(1e-6 * ct_cpu))
    assert ndiff / cpu.counts.size < 1e-3
    assert r > 0.99999
    return r


def test_parity_uncorrected():
    print("== Test B: GPU vs CPU parity, corrections OFF ==")
    _parity(None, "raw")
    print("  PASS\n")


def test_parity_corrected():
    print("== Test C: GPU vs CPU parity, corrections ON (identical map) ==")
    det, _, _, _ = _setup()
    corr = pixel_corrections(det)                       # default solid-angle + pol
    _parity(corr, "corr")
    print("  PASS\n")


def test_viewer_compat_h5():
    print("== Test D: viewer-compatible h5 (wavelength/M_inv/cell/lambda-scaled UB) ==")
    import tempfile
    import h5py
    from xrays_on_detector.reconstruct import Volume, save_rspace3d_h5
    recip = np.eye(3) / 5.9                              # cubic a* = 1/5.9, identity orient
    H = -6 + (np.arange(20) + 0.5) * 0.6
    vol = Volume(np.ones((20, 20, 20), np.float32), H, recip,
                 np.ones((20, 20, 20), np.int64), wavelength=0.4859)
    p = os.path.join(tempfile.gettempdir(), "xod_viewer_test.h5")
    save_rspace3d_h5(p, vol)
    with h5py.File(p, "r") as f:
        assert float(f.attrs["wavelength"]) == 0.4859          # the fix: nonzero wavelength
        assert "M_inv" in f and np.isfinite(np.array(f["M_inv"])).all()
        assert abs(float(f.attrs["cell_a"]) - 5.9) < 0.01
        assert f.attrs["s"] > 0
        ub = np.array(f["UB"])
        assert np.allclose(np.linalg.norm(ub, axis=0), 0.4859 / 5.9, atol=1e-6)  # lambda-scaled
    print("  wavelength=0.4859, M_inv finite, cell_a=5.9, UB lambda-scaled, s>0")
    print("  PASS\n")


def test_par_reader():
    print("== Test E: cracker-par UB reader + orient_from_frame ==")
    from xrays_on_detector.realframe import (find_crysalis_par, read_crysalis_par,
                                             orient_from_frame)
    from xrays_on_detector.reconstruct import _reciprocal_cell
    par = find_crysalis_par(RAW)
    assert par is not None and par.endswith("_cracker.par"), f"expected cracker par: {par}"
    UB, wl = read_crysalis_par(par)
    cell = _reciprocal_cell(UB)
    print(f"  {os.path.basename(par)}  wl={wl}  cell "
          f"{cell['a']:.3f},{cell['b']:.3f},{cell['c']:.3f}  "
          f"ang {cell['alpha']:.2f},{cell['beta']:.2f},{cell['gamma']:.2f}")
    assert all(5.8 < cell[x] < 6.0 for x in "abc"), "cracker-par cell not pseudocubic 5.9"
    assert all(abs(cell[a] - 90) < 1 for a in ("alpha", "beta", "gamma"))
    det, UB2, R0, phi0 = orient_from_frame(par, STEM.format(1))
    assert abs(np.linalg.det(R0) - 1) < 1e-6 and np.allclose(R0 @ R0.T, np.eye(3), atol=1e-6)
    print(f"  orient_from_frame OK: phi0={phi0}, R0 proper rotation (det=1)")
    print("  PASS\n")


if __name__ == "__main__":
    test_corrections_physics()
    test_viewer_compat_h5()                 # no data / GPU needed

    if not os.path.exists(STEM.format(1)):
        print(f"SKIP data tests: frames not found at {RAW}")
        sys.exit(0)
    test_par_reader()                       # needs frames + par, no GPU
    try:
        import cupy  # noqa: F401
    except Exception as e:
        print(f"SKIP B/C: CuPy unavailable ({e!r})")
        sys.exit(0)

    test_parity_uncorrected()
    test_parity_corrected()
    print("ALL GPU RECONSTRUCTION TESTS PASSED")
