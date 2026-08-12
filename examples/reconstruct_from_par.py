"""GPU reconstruction of a CrysAlisPro rotation dataset, driven entirely by the
dataset's own files (no prior reconstruction needed):

  * UB + wavelength      <- the *_cracker.par  (read_crysalis_par)
  * detector + phi datum <- the first CBF header (FlatDetector.from_eiger_cbf)
  * R0 (missing rotation)<- indexing frame 1    (index_frame, inside orient_from_frame)

Then reconstructs the -6..6 / 0.025-rlu volume on the GPU with intensity
corrections, saves it in the rsp_viewer-compatible HDF5 layout, and (if the
dataset has an unwarp/..._raw.h5) validates against it (integer-Bragg
registration + Bragg-node intensity correlation + HK-plane comparison).

Point it at any dataset with DATASET=<folder>; defaults to 148 (221 K).
"""
import os
import sys
import time
import glob
import warnings

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xrays_on_detector.realframe import (FlatDetector, find_crysalis_par,
                                         read_crysalis_par, orient_from_frame)
from xrays_on_detector.reconstruct import save_rspace3d_h5
from xrays_on_detector.reconstruct_gpu import reconstruct_volume_gpu
from xrays_on_detector.corrections import pixel_corrections

# DATASET is a CrysAlisPro run folder: the frames are named after the folder
# itself, and the .par beside them carries the geometry. XOD_OUT sets where the
# reconstruction is written (default: ./out beside this script).
FOLDER = os.environ.get("DATASET", "")
if not FOLDER:
    sys.exit("set DATASET to a CrysAlisPro run folder (frames + .par)")
NFRAMES = int(os.environ.get("NFRAMES", "1750"))
RANGE, STEP = (-6.0, 6.0), 0.025
OUTDIR = os.environ.get(
    "XOD_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
os.makedirs(OUTDIR, exist_ok=True)

name = os.path.basename(os.path.normpath(FOLDER))
stem = os.path.join(FOLDER, name + "_01_{:04d}.cbf")
par = find_crysalis_par(FOLDER)
print(f"dataset {name}\n  par  = {os.path.basename(par)}", flush=True)

# ---- geometry + UB (par) + R0 (index frame 1) ----
_det0, ang0, _ = FlatDetector.from_eiger_cbf(stem.format(1))
incr = ang0["Angle_increment"]
det, UB, R0, phi0 = orient_from_frame(par, stem.format(1))
cell_norm = np.linalg.norm(UB, axis=0)
print(f"  UB from par: |a*|,|b*|,|c*| = {cell_norm.round(4)} (1/d); "
      f"wavelength {det.wavelength}", flush=True)
print(f"  phi0 {phi0}, increment {incr} deg/frame; R0 found by indexing frame 1", flush=True)

nums = list(range(1, NFRAMES + 1))
frames = [stem.format(n) for n in nums]
phis = [phi0 + (n - 1) * incr for n in nums]

# ---- reconstruct corrected + raw ----
corr = pixel_corrections(det, solid_angle=True, polarization=True, polarization_fraction=0.95)
t = time.time()
vol_c = reconstruct_volume_gpu(frames, phis, UB, R0, det, phi0=phi0, hkl_range=RANGE,
                               step=STEP, hot=1e6, corrections=corr, prefetch=4, progress=500)
print(f"corrected volume: {time.time()-t:.1f}s; measured voxels "
      f"{(vol_c.counts>0).sum():,}/{vol_c.data.size:,}", flush=True)
t = time.time()
vol_r = reconstruct_volume_gpu(frames, phis, UB, R0, det, phi0=phi0, hkl_range=RANGE,
                               step=STEP, hot=1e6, corrections=None, prefetch=4)
print(f"raw volume: {time.time()-t:.1f}s", flush=True)

out_h5 = os.path.join(OUTDIR, f"{name}_recon_gpu_corr.h5")
save_rspace3d_h5(out_h5, vol_c, source_folder=FOLDER)
print("saved (viewer-compatible):", out_h5, flush=True)

# ---- integer-Bragg registration (reference-free geometry check) ----
H = vol_c.H
hh, kk, ll = np.meshgrid(H, H, H, indexing="ij")
d = vol_c.data.copy()
d[vol_c.counts < 20] = np.nan
d[(hh**2 + kk**2 + ll**2) < 0.6**2] = np.nan
idx = np.argsort(np.where(np.isfinite(d.ravel()), d.ravel(), -np.inf))[::-1][:300]
coords = np.stack([hh.ravel()[idx], kk.ravel()[idx], ll.ravel()[idx]], 1)
dist = np.linalg.norm(coords - np.round(coords), axis=1)
print(f"integer-Bragg registration (top-300 voxels): RMS {np.sqrt(np.mean(dist**2)):.4f} rlu, "
      f"median {np.median(dist):.4f} rlu (voxel {STEP})", flush=True)

# ---- validate vs the dataset's own unwarp/..._raw.h5 (if present) ----
raw_h5 = glob.glob(os.path.join(FOLDER, "unwarp", "*_raw.h5"))
if not raw_h5:
    print("no unwarp/_raw.h5 for this dataset -> skipping reference comparison")
    print("DONE", flush=True); sys.exit(0)
REF = raw_h5[0]
print("reference:", os.path.basename(REF), flush=True)

nodes = np.arange(-6, 7)
WIN, MIN_COUNT = 0.06, 10
with h5py.File(REF, "r") as f:
    Href, Kref, Lref = np.array(f["H"]), np.array(f["K"]), np.array(f["L"])
    data = f["data"]
    ref_planes = {float(l): np.array(data[:, :, int(np.argmin(np.abs(Lref - l)))])
                  for l in list(nodes) + [1.5]}

def boxvals(vol3, cnt3, h, k, l):
    ih = np.abs(H - h) <= WIN; ik = np.abs(H - k) <= WIN; il = np.abs(H - l) <= WIN
    if not (ih.any() and ik.any() and il.any()):
        return None
    b = vol3[np.ix_(ih, ik, il)]; c = cnt3[np.ix_(ih, ik, il)]
    return (np.nanmax(b), c.max()) if np.isfinite(b).any() else None

mv, rv = [], []
for l in nodes:
    plane = ref_planes[float(l)]
    for h in nodes:
        for k in nodes:
            if h == k == l == 0:
                continue
            got = boxvals(vol_r.data, vol_r.counts, h, k, l)
            if got is None or got[1] < MIN_COUNT or not np.isfinite(got[0]) or got[0] <= 0:
                continue
            ih = np.abs(Href - h) <= WIN; ik = np.abs(Kref - k) <= WIN
            if not (ih.any() and ik.any()):
                continue
            bx = plane[np.ix_(ih, ik)]
            if not np.isfinite(bx).any() or np.nanmax(bx) <= 0:
                continue
            mv.append(got[0]); rv.append(float(np.nanmax(bx)))
mv, rv = np.array(mv), np.array(rv)
r_log = float(np.corrcoef(np.log10(mv), np.log10(rv))[0, 1])
rank = lambda a: np.argsort(np.argsort(a))
r_spear = float(np.corrcoef(rank(mv), rank(rv))[0, 1])
print(f"Bragg-node correlation vs reference over {len(mv)} reflections: "
      f"log Pearson r={r_log:.3f}, Spearman r={r_spear:.3f}", flush=True)

# ---- HK-plane comparison figure (L=0, L=1.5) ----
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def slice_at(vol_data, target):
    sel = np.where(np.abs(H - target) <= STEP / 2 + 1e-9)[0]
    if sel.size == 0:
        sel = [int(np.argmin(np.abs(H - target)))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(vol_data[:, :, sel], axis=2)

def show(ax, plane, extent, title):
    fin = plane[np.isfinite(plane) & (plane > 0)]
    vmin = max(np.percentile(fin, 55), 1.0) if fin.size else 1
    vmax = np.percentile(fin, 99.6) if fin.size else 10
    cmap = plt.get_cmap("magma").copy(); cmap.set_bad("#000010")
    ax.imshow(plane.T, origin="lower", cmap=cmap, extent=extent,
              norm=LogNorm(vmin=vmin, vmax=vmax), interpolation="nearest")
    ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.set_xlabel("h"); ax.set_ylabel("k")
    ax.set_xticks(range(-6, 7, 2)); ax.set_yticks(range(-6, 7, 2))
    ax.grid(alpha=0.15, lw=0.3); ax.set_title(title, fontsize=10)

ext_mine = [H[0], H[-1], H[0], H[-1]]
ext_ref = [Href[0], Href[-1], Kref[0], Kref[-1]]
fig, ax = plt.subplots(2, 3, figsize=(15, 10), dpi=110)
for row, L in enumerate((0.0, 1.5)):
    show(ax[row, 0], slice_at(vol_r.data, L), ext_mine, f"GPU raw, L={L}")
    show(ax[row, 1], slice_at(vol_c.data, L), ext_mine, f"GPU corrected, L={L}")
    show(ax[row, 2], ref_planes[float(L)], ext_ref, f"reference _raw.h5, L={L}")
fig.suptitle(f"{name}: GPU reconstruction (UB from par) vs reference   "
             f"[Bragg-node log-r={r_log:.3f}]", fontsize=12)
fig.tight_layout()
png = os.path.join(OUTDIR, f"{name}_compare_vs_raw.png")
fig.savefig(png, bbox_inches="tight")
print("saved comparison:", png, flush=True)
print("DONE", flush=True)
