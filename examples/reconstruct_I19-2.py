"""Full 3D reciprocal-space reconstruction of the I19-2 MAPbBr3 phi scan from
raw CBF frames, using the validated orientation (xrays_on_detector).

Output: an rspace3d-format HDF5 volume + a comparison of the L=0 HK plane
against the existing CrysAlisPro/rspace3d reconstruction.
"""
import os, sys, time
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xrays_on_detector.realframe import FlatDetector, detect_peaks, index_frame
from xrays_on_detector.reconstruct import reconstruct_volume, save_rspace3d_h5

# Point these at your own data; no frames are bundled with the repository.
#   XOD_RAW     folder holding the CBF frames (and CrysAlisPro's unwarp/)
#   XOD_NAME    run stem, so frame n is <XOD_RAW>/<XOD_NAME>_01_000n.cbf
#   XOD_REF_H5  reference rspace3d/CrysAlisPro volume to compare against
#   XOD_OUT     output folder (default: ./out beside this script)
RAW = os.environ.get("XOD_RAW", "")
NAME = os.environ.get("XOD_NAME", "")
H5_REF = os.environ.get("XOD_REF_H5", "")
if not (RAW and NAME and H5_REF):
    sys.exit("set XOD_RAW, XOD_NAME and XOD_REF_H5; see the header of this file")
STEM = os.path.join(RAW, NAME + "_01_{:04d}.cbf")
OUTDIR = os.environ.get(
    "XOD_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
os.makedirs(OUTDIR, exist_ok=True)

RANGE, STEP = (-9.0, 9.0), 0.05
NFRAMES = 1750

det, ang0, img0 = FlatDetector.from_eiger_cbf(STEM.format(1))
with h5py.File(H5_REF, "r") as f:
    UB = np.array(f["UB"]) / det.wavelength
R0 = index_frame(detect_peaks(img0, det.beam_center), UB, det).R
phi0 = ang0["Phi"]

nums = list(range(1, NFRAMES + 1))
frames = [STEM.format(n) for n in nums]
phis = [phi0 + (n - 1) * 0.2 for n in nums]
print(f"reconstructing {len(frames)} frames, phi {phis[0]:.0f}..{phis[-1]:.0f}, "
      f"grid {RANGE} step {STEP}", flush=True)

t = time.time()
vol = reconstruct_volume(frames, phis, UB, R0, det, phi0=phi0,
                         hkl_range=RANGE, step=STEP, hot=1e6, progress=100)
dt = time.time() - t
print(f"done in {dt/60:.1f} min; measured voxels {(vol.counts>0).sum():,} / {vol.data.size:,}",
      flush=True)

out_h5 = os.path.join(OUTDIR, "mapbbr3_215K_800V_recon_xrays.h5")
save_rspace3d_h5(out_h5, vol)
print("saved volume:", out_h5, flush=True)

# ---- compare L=0 HK plane: mine vs the existing CrysAlisPro reconstruction ----
H = vol.H
li = np.argmin(np.abs(H))
mine = np.nanmean(vol.data[:, :, li-1:li+2], axis=2).T   # (k,h)

with h5py.File(H5_REF, "r") as f:
    ref = np.array(f["data"]); Href = np.array(f["H"]); Kref = np.array(f["K"]); Lref = np.array(f["L"])
lir = np.argmin(np.abs(Lref))
ref_sl = ref[:, :, lir].T                                # (k,h)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
fig, ax = plt.subplots(1, 2, figsize=(15, 7.5), dpi=115)
for a, sl, HH, KK, title in [
        (ax[0], mine, H, H, f"THIS reconstruction (raw CBF, {NFRAMES} frames)"),
        (ax[1], ref_sl, Href, Kref, "existing CrysAlisPro / rspace3d reconstruction")]:
    fin = sl[np.isfinite(sl) & (sl > 0)]
    vmin, vmax = np.percentile(fin, 60), np.percentile(fin, 99.6)
    cmap = plt.get_cmap("magma").copy(); cmap.set_bad("#000010")
    a.imshow(sl, origin="lower", cmap=cmap, extent=[HH[0], HH[-1], KK[0], KK[-1]],
             norm=LogNorm(vmin=max(vmin, 1), vmax=vmax), interpolation="nearest")
    a.set_xlim(-6, 6); a.set_ylim(-6, 6)
    a.set_xticks(range(-6, 7)); a.set_yticks(range(-6, 7)); a.grid(alpha=0.2, lw=0.3)
    a.set_xlabel("h"); a.set_ylabel("k"); a.set_title(title, fontsize=11)
fig.suptitle("MAPbBr3 215 K / 800 V, HK plane at L=0")
fig.tight_layout()
cmp = os.path.join(OUTDIR, "compare_L0_mine_vs_crysalis.png")
fig.savefig(cmp, bbox_inches="tight"); print("saved comparison:", cmp, flush=True)
