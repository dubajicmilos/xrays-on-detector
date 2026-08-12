"""Quantitative validation of the GPU reconstruction against the reference
`..._0_raw.h5`, using metrics that are not swamped by the large-area diffuse /
background differences (unsubtracted low-|Q| scatter, single-axis blind wedge,
powder-ring arc) that a raw whole-plane Pearson r is dominated by.

Metric 1 (headline): Bragg-peak intensity correlation. Sample the peak intensity
at every integer node (h,k,l) in [-6,6]^3 that is well measured in both volumes
and correlate log10 intensities - does the reconstruction reproduce the reference
reflections? Reconstructs an UNcorrected volume so it is compared like-for-like
against the (uncorrected) reference (no |Q|-dependent correction tilt).

Metric 2: background-masked log-intensity plane correlation at L=0 and L=1.5.
"""
import os
import sys
import warnings

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xrays_on_detector.realframe import FlatDetector, detect_peaks, index_frame
from xrays_on_detector.reconstruct_gpu import reconstruct_volume_gpu

# Point these at your own data; no frames are bundled with the repository.
#   XOD_RAW   folder holding the CBF frames (and CrysAlisPro's unwarp/)
#   XOD_NAME  run stem, so frame n is <XOD_RAW>/<XOD_NAME>_01_000n.cbf
#   XOD_OUT   output folder (default: ./out beside this script)
RAW = os.environ.get("XOD_RAW", "")
NAME = os.environ.get("XOD_NAME", "")
if not (RAW and NAME):
    sys.exit("set XOD_RAW and XOD_NAME; see the header of this file")
H5_RAW = os.path.join(RAW, "unwarp", NAME + "_0_raw.h5")
STEM = os.path.join(RAW, NAME + "_01_{:04d}.cbf")
OUTDIR = os.environ.get(
    "XOD_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
RANGE, STEP, NFRAMES = (-6.0, 6.0), 0.025, 1750
WIN = 0.06            # +-rlu window for a node's peak
MIN_COUNT = 10        # a node must have a voxel with >= this many contributing pixels

# ------------------------------------------------- reconstruct raw (uncorrected)
det, ang0, img0 = FlatDetector.from_eiger_cbf(STEM.format(1))
with h5py.File(H5_RAW, "r") as f:
    UB = np.array(f["UB"]) / det.wavelength
R0 = index_frame(detect_peaks(img0, det.beam_center), UB, det).R
phi0 = ang0["Phi"]
frames = [STEM.format(n) for n in range(1, NFRAMES + 1)]
phis = [phi0 + (n - 1) * 0.2 for n in range(1, NFRAMES + 1)]
print("reconstructing raw volume for like-for-like comparison...", flush=True)
vol = reconstruct_volume_gpu(frames, phis, UB, R0, det, phi0=phi0, hkl_range=RANGE,
                             step=STEP, hot=1e6, corrections=None, prefetch=4, progress=500)
mine, cnt, H = vol.data, vol.counts, vol.H
print(f"raw volume {mine.shape}; measured voxels {(cnt>0).sum():,}", flush=True)


def box(vol3, axH, axK, axL, h, k, l, win=WIN):
    ih = np.abs(axH - h) <= win; ik = np.abs(axK - k) <= win; il = np.abs(axL - l) <= win
    if not (ih.any() and ik.any() and il.any()):
        return None
    return vol3[np.ix_(ih, ik, il)]


# ------------------------------------------ Metric 1: Bragg-node correlation
nodes = np.arange(-6, 7)
with h5py.File(H5_RAW, "r") as f:
    Href, Kref, Lref = np.array(f["H"]), np.array(f["K"]), np.array(f["L"])
    data = f["data"]
    ref_planes = {float(l): np.array(data[:, :, int(np.argmin(np.abs(Lref - l)))])
                  for l in list(nodes) + [1.5]}

mv, rv = [], []
for l in nodes:
    plane = ref_planes[float(l)]
    for h in nodes:
        for k in nodes:
            if h == 0 and k == 0 and l == 0:
                continue
            b = box(mine, H, H, H, h, k, l)
            bc = box(cnt, H, H, H, h, k, l)
            if b is None or not np.isfinite(b).any() or bc.max() < MIN_COUNT:
                continue
            vm = float(np.nanmax(b))
            ih = np.abs(Href - h) <= WIN; ik = np.abs(Kref - k) <= WIN
            if not (ih.any() and ik.any()):
                continue
            bx = plane[np.ix_(ih, ik)]
            if not np.isfinite(bx).any():
                continue
            vr = float(np.nanmax(bx))
            if vm > 0 and vr > 0:
                mv.append(vm); rv.append(vr)

mv, rv = np.array(mv), np.array(rv)
lm, lr = np.log10(mv), np.log10(rv)
r_pear = float(np.corrcoef(lm, lr)[0, 1])
rank = lambda a: np.argsort(np.argsort(a))
r_spear = float(np.corrcoef(rank(mv), rank(rv))[0, 1])
print(f"\nMetric 1  Bragg-node intensity correlation over {len(mv)} well-measured "
      f"integer reflections (count>={MIN_COUNT}):")
print(f"   log10-log10 Pearson r = {r_pear:.3f};  Spearman rank r = {r_spear:.3f}")

# ------------------------------- Metric 2: masked log plane correlation
from scipy.interpolate import RegularGridInterpolator

def plane_logcorr(L0, excl_lowQ):
    itp = RegularGridInterpolator((Href, Kref), ref_planes[float(L0)],
                                  bounds_error=False, fill_value=np.nan)
    hh, kk = np.meshgrid(H, H, indexing="ij")
    ref_on = itp(np.stack([hh, kk], -1))
    li = np.where(np.abs(H - L0) <= STEP / 2 + 1e-9)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mine_pl = np.nanmean(mine[:, :, li], axis=2)
    good = np.isfinite(mine_pl) & np.isfinite(ref_on) & (mine_pl > 0) & (ref_on > 0)
    if excl_lowQ:
        good &= (hh ** 2 + kk ** 2) > 1.5 ** 2
    a, b = np.log10(mine_pl[good]), np.log10(ref_on[good])
    return float(np.corrcoef(a, b)[0, 1]), int(good.sum())

for L0 in (0.0, 1.5):
    r_all, n_all = plane_logcorr(L0, False)
    r_ex, n_ex = plane_logcorr(L0, True)
    print(f"Metric 2  L={L0}: log-plane r = {r_all:.3f} (n={n_all:,}); "
          f"excluding |Q_hk|<1.5 r = {r_ex:.3f} (n={n_ex:,})")

# --------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(13, 6.0), dpi=120)
ax[0].scatter(rv, mv, s=10, alpha=0.5, edgecolor="none")
lim = [min(rv.min(), mv.min()) * 0.7, max(rv.max(), mv.max()) * 1.4]
scale = 10 ** np.median(lm - lr)
ax[0].plot(lim, [scale * x for x in lim], "r-", lw=0.9, alpha=0.7,
           label=f"median scale x{scale:.2f}")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel("reference _raw.h5 peak intensity")
ax[0].set_ylabel("GPU reconstruction peak intensity (raw)")
ax[0].set_title(f"Bragg-node intensities ({len(mv)} reflections)\n"
                f"log Pearson r={r_pear:.3f}, Spearman r={r_spear:.3f}", fontsize=10)
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.2, which="both", lw=0.3)

ratio = lm - lr
ax[1].hist(ratio, bins=40, color="#4060c0", alpha=0.8)
ax[1].axvline(np.median(ratio), color="r", lw=1, label=f"median {np.median(ratio):.2f} dex")
ax[1].set_xlabel("log10(mine / reference)  per reflection")
ax[1].set_ylabel("count"); ax[1].legend(fontsize=8)
ax[1].set_title(f"per-reflection intensity ratio (spread {ratio.std():.2f} dex)", fontsize=10)
fig.suptitle("MAPbBr3 215 K / 800 V - GPU reconstruction vs reference: Bragg-peak agreement")
fig.tight_layout()
png = os.path.join(OUTDIR, "validate_bragg_nodes.png")
fig.savefig(png, bbox_inches="tight")
print("\nsaved:", png)
