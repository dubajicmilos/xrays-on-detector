"""Validate xrays_on_detector against a real I19-2 MAPbBr3 Eiger CBF frame,
using the package's realframe API.

Steps: read the detector geometry from the CBF header, index one frame's spots
against the CrysAlisPro UB, cross-validate the rotation convention across frames
at different phi, and overlay predicted Bragg positions on the measured frame.
Edit the paths below for another dataset.
"""
import os
import numpy as np
import h5py

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xrays_on_detector.realframe import (
    FlatDetector, detect_peaks, index_frame, predict_recorded, detector_display)

# Point these at your own data; no frames are bundled with the repository.
#   XOD_RAW     folder holding the CBF frames
#   XOD_NAME    run stem, so frame n is <XOD_RAW>/<XOD_NAME>_01_000n.cbf
#   XOD_REF_H5  reference rspace3d/CrysAlisPro volume, for the UB
RAW = os.environ.get("XOD_RAW", "")
NAME = os.environ.get("XOD_NAME", "")
H5 = os.environ.get("XOD_REF_H5", "")
if not (RAW and NAME and H5):
    sys.exit("set XOD_RAW, XOD_NAME and XOD_REF_H5; see the header of this file")
STEM = os.path.join(RAW, NAME + "_01_{:04d}.cbf")
OSC_AXIS = (0, 1, 0)      # oscillation axis in the lab frame (validated for this setup)
OUT = os.path.dirname(os.path.abspath(__file__))


def load(frame_no):
    return FlatDetector.from_eiger_cbf(STEM.format(frame_no))


def main():
    det, ang, img = load(1)
    print(f"geometry: D={det.distance:.1f} mm, pixel={det.pixel_size} mm, "
          f"beam={det.beam_center}, lambda={det.wavelength} A")
    print(f"frame 1 angles: phi={ang['Phi']}, omega={ang['Omega']}, "
          f"kappa={ang['Kappa']}, dphi={ang['Angle_increment']}")
    with h5py.File(H5, "r") as f:
        UB = np.array(f["UB"]) / det.wavelength

    peaks = detect_peaks(img, det.beam_center)
    res = index_frame(peaks, UB, det)
    pxerr = res.rms * det.wavelength * det.distance / det.pixel_size
    print(f"\nindexed frame 1: {res.inliers.sum()}/{len(peaks)} spots, "
          f"rms={res.rms:.4f} 1/A (~{pxerr:.1f} px)")
    for i in np.nonzero(res.inliers)[0]:
        print(f"  ({peaks[i,0]:7.1f},{peaks[i,1]:7.1f}) -> hkl {tuple(res.hkl[i])}")

    # cross-validate: frame1 -> frameN should be a pure phi rotation about OSC_AXIS
    print("\ncross-validation vs the recorded phi increment:")
    R1, phi1 = res.R, ang["Phi"]
    from xrays_on_detector.realframe import _axis_rot
    for N in [26, 51, 151]:
        detN, angN, imgN = load(N)
        rN = index_frame(detect_peaks(imgN, detN.beam_center), UB, detN)
        if rN is None:
            print(f"  frame {N}: too few spots"); continue
        dphi = angN["Phi"] - phi1
        # relative rotation, allowing a lattice symmetry op (near-cubic metric)
        best = _rel_angle(rN.R, R1, UB, dphi)
        print(f"  frame {N}: dphi={dphi:+.1f} deg -> measured {best:.2f} deg")

    _overlay(det, img, UB, res, peaks)


def _rel_angle(R2, R1, UB, dphi):
    """Smallest |angle - dphi| over lattice symmetry ops (m-3m holohedry)."""
    import sys
    # rspace3d is a separate package; set RSPACE3D_PATH if it is not importable
    rs = os.environ.get("RSPACE3D_PATH", "")
    if rs and rs not in sys.path:
        sys.path.insert(0, rs)
    from rspace3d.volume_builder import get_symmetry_operations
    UBi = np.linalg.inv(UB)
    best = 1e9
    for G in get_symmetry_operations("m-3m"):
        Rrel = R2 @ (UB @ np.linalg.inv(G) @ UBi) @ R1.T
        ang = np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1)))
        if abs(ang - abs(dphi)) < abs(best - abs(dphi)):
            best = ang
    return best


def _overlay(det, img, UB, res, peaks):
    hs = np.arange(-10, 11)
    hkl = np.stack(np.meshgrid(hs, hs, hs, indexing="ij"), -1).reshape(-1, 3)
    hkl = hkl[np.any(hkl != 0, 1)]
    hkl = hkl[np.linalg.norm((UB @ hkl.T).T, axis=1) <= 1.6]
    _, fx, fy, _ = predict_recorded(res.R, UB, det, hkl, osc_axis=OSC_AXIS)
    matched = sum(np.min(np.hypot(fx - x, fy - y)) < 15 for x, y in peaks)
    print(f"\noverlay: {matched}/{len(peaks)} observed spots matched by a prediction")

    db, vmin, vmax = detector_display(img)      # noise-floored, binned, smoothed
    b = 2
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(figsize=(9, 9), dpi=120)
    cmap = plt.get_cmap("gray").copy(); cmap.set_bad("#0a0a0a")
    ax.imshow(np.where(db > 0, db, np.nan), origin="upper", cmap=cmap,
              norm=LogNorm(vmin=vmin, vmax=vmax))
    ax.scatter(peaks[:, 0] / b, peaks[:, 1] / b, s=160, facecolors="none",
               edgecolors="#ff3b3b", lw=1.6, label=f"observed ({len(peaks)})")
    ax.scatter(fx / b, fy / b, s=26, c="#00e5ff", marker="+", lw=1.0, label=f"predicted ({len(fx)})")
    ax.set_title(f"MAPbBr3 frame 01_0001  |  {matched}/{len(peaks)} observed matched")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.4)
    ax.set_xticks([]); ax.set_yticks([])
    out = os.path.join(OUT, "I19-2_overlay_frame1.png")
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    main()
