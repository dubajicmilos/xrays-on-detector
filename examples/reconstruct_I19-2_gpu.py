"""GPU 3D reciprocal-space reconstruction of the I19-2 MAPbBr3 215 K / 800 V phi
scan (1750 raw CBF frames) on an RTX 3090, with optional intensity corrections,
and validation against the reference `..._0_raw.h5` reconstruction.

Outputs (to OUTDIR):
  * mapbbr3_215K_800V_recon_gpu_corr.h5   - corrected volume (rspace3d layout)
  * compare_gpu_vs_raw.png                - L=0 and L=1.5 planes: mine vs reference
Console: GPU-vs-CPU speedup, integer-Bragg registration RMS, plane correlations.
"""
import os
import sys
import time
import warnings

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xrays_on_detector.realframe import FlatDetector, detect_peaks, index_frame
from xrays_on_detector.reconstruct import reconstruct_volume, save_rspace3d_h5
from xrays_on_detector.reconstruct_gpu import reconstruct_volume_gpu
from xrays_on_detector.corrections import pixel_corrections

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
os.makedirs(OUTDIR, exist_ok=True)

RANGE, STEP = (-6.0, 6.0), 0.025          # user target: -6..6, 0.025 rlu, 480^3
NFRAMES = int(os.environ.get("NFRAMES", "1750"))   # override for a fast smoke test

# ---------------------------------------------------------------- geometry setup
det, ang0, img0 = FlatDetector.from_eiger_cbf(STEM.format(1))
with h5py.File(H5_RAW, "r") as f:
    UB = np.array(f["UB"]) / det.wavelength          # -> 1/d columns a*,b*,c*
R0 = index_frame(detect_peaks(img0, det.beam_center), UB, det).R
phi0 = ang0["Phi"]

nums = list(range(1, NFRAMES + 1))
frames = [STEM.format(n) for n in nums]
phis = [phi0 + (n - 1) * 0.2 for n in nums]
n_axis = int(round((RANGE[1] - RANGE[0]) / STEP))
print(f"grid {RANGE} step {STEP} -> {n_axis}^3 = {n_axis**3:,} voxels; "
      f"{NFRAMES} frames, phi {phis[0]:.0f}..{phis[-1]:.0f}", flush=True)

# -------------------- GPU-vs-CPU histogram-compute benchmark (disk I/O excluded)
# Preload frames and warm up the GPU (kernel JIT + pool alloc) so the timing
# reflects steady-state accumulation, not one-off disk/warmup. GPU is timed over
# more frames than CPU because the CPU path is ~70x slower per frame.
import fabio
NB_G, NB_C = 60, 15
imgs = [fabio.open(f).data for f in frames[:NB_G]]
rd = lambda i: imgs[i]
_ = reconstruct_volume_gpu(list(range(5)), phis[:5], UB, R0, det, phi0=phi0,
                           hkl_range=RANGE, step=STEP, hot=1e6, prefetch=0, read_fn=rd)
t = time.time()
_ = reconstruct_volume_gpu(list(range(NB_G)), phis[:NB_G], UB, R0, det, phi0=phi0,
                           hkl_range=RANGE, step=STEP, hot=1e6, prefetch=0, read_fn=rd)
tg = (time.time() - t) / NB_G
t = time.time()
_ = reconstruct_volume(list(range(NB_C)), phis[:NB_C], UB, R0, det, phi0=phi0,
                       hkl_range=RANGE, step=STEP, hot=1e6, read_fn=rd)
tc = (time.time() - t) / NB_C
print(f"compute benchmark @ {n_axis}^3 (disk excluded, steady-state):  "
      f"GPU {tg*1e3:.1f} ms/frame  CPU {tc*1e3:.0f} ms/frame  histogram speedup "
      f"x{tc/tg:.0f}  (full {NFRAMES}-frame scan: GPU ~{tg*NFRAMES:.0f}s vs "
      f"CPU ~{tc*NFRAMES/60:.0f}min)", flush=True)

# -------------------------------------------------------- full corrected + raw runs
corr = pixel_corrections(det, solid_angle=True, polarization=True,
                         polarization_fraction=0.95)      # synchrotron default
print("corrections: solid-angle + polarisation(p=0.95, h=fast); "
      f"per-pixel range {corr.min():.3f}..{corr.max():.3f}", flush=True)

t = time.time()
vol_c = reconstruct_volume_gpu(frames, phis, UB, R0, det, phi0=phi0, hkl_range=RANGE,
                               step=STEP, hot=1e6, corrections=corr, prefetch=4,
                               progress=250)
print(f"corrected volume: {time.time()-t:.1f}s; measured voxels "
      f"{(vol_c.counts>0).sum():,}/{vol_c.data.size:,}", flush=True)

t = time.time()
vol_r = reconstruct_volume_gpu(frames, phis, UB, R0, det, phi0=phi0, hkl_range=RANGE,
                               step=STEP, hot=1e6, corrections=None, prefetch=4,
                               progress=250)
print(f"raw volume: {time.time()-t:.1f}s", flush=True)

out_h5 = os.path.join(OUTDIR, "mapbbr3_215K_800V_recon_gpu_corr.h5")
t = time.time()
save_rspace3d_h5(out_h5, vol_c)
print(f"saved corrected volume ({time.time()-t:.0f}s):", out_h5, flush=True)

# --------------------------------------------- integer-Bragg registration check
def bragg_rms(vol, min_count=20, K=300, r_excl=0.6):
    H = vol.H
    hh, kk, ll = np.meshgrid(H, H, H, indexing="ij")
    d = vol.data.copy()
    d[vol.counts < min_count] = np.nan
    d[(hh**2 + kk**2 + ll**2) < r_excl**2] = np.nan          # skip origin
    flat = d.ravel()
    idx = np.argsort(np.where(np.isfinite(flat), flat, -np.inf))[::-1][:K]
    coords = np.stack([hh.ravel()[idx], kk.ravel()[idx], ll.ravel()[idx]], 1)
    frac = coords - np.round(coords)
    dist = np.linalg.norm(frac, axis=1)
    return float(np.sqrt(np.mean(dist**2))), float(np.median(dist)), coords

rms_c, med_c, top_c = bragg_rms(vol_c)
print(f"integer-Bragg registration (top-300 corrected voxels): "
      f"RMS {rms_c:.4f} rlu, median {med_c:.4f} rlu (voxel {STEP})", flush=True)

# ----------------------------------------------- compare vs reference _raw.h5
from scipy.interpolate import RegularGridInterpolator

def slice_at(vol_data, axis, target):
    sel = np.where(np.abs(axis - target) <= (axis[1]-axis[0])/2 + 1e-9)[0]
    if sel.size == 0:
        sel = [int(np.argmin(np.abs(axis - target)))]
    with warnings.catch_warnings():                               # all-NaN columns
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(vol_data[:, :, sel], axis=2)            # (h,k)

def ref_plane_on(mine_axis, Href, Kref, plane):
    itp = RegularGridInterpolator((Href, Kref), plane, bounds_error=False,
                                  fill_value=np.nan)
    hh, kk = np.meshgrid(mine_axis, mine_axis, indexing="ij")
    return itp(np.stack([hh, kk], -1))                            # (h,k) on my grid

def corr2d(a, b):
    # log-intensity correlation: a raw-linear r is dominated by the few brightest
    # voxels and the large-area diffuse/background differences; see
    # validate_gpu_vs_raw.py for the Bragg-peak-resolved validation.
    m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if m.sum() < 10:
        return float("nan"), 0
    return float(np.corrcoef(np.log10(a[m]), np.log10(b[m]))[0, 1]), int(m.sum())

H = vol_c.H
with h5py.File(H5_RAW, "r") as f:
    Href, Kref, Lref = np.array(f["H"]), np.array(f["K"]), np.array(f["L"])
    data = f["data"]
    planes = {}
    for L in (0.0, 1.5):
        m = int(np.argmin(np.abs(Lref - L)))
        planes[L] = np.array(data[:, :, m])                       # (H,K) native

results = {}
for L in (0.0, 1.5):
    ref_on = ref_plane_on(H, Href, Kref, planes[L])
    mine_r = slice_at(vol_r.data, H, L)
    mine_c = slice_at(vol_c.data, H, L)
    rc_raw, npx = corr2d(mine_r, ref_on)
    rc_cor, _ = corr2d(mine_c, ref_on)
    results[L] = (rc_raw, rc_cor, npx)
    print(f"L={L}: corr(mine_raw, ref)={rc_raw:.4f}  corr(mine_corr, ref)={rc_cor:.4f}  "
          f"(n={npx:,} common voxels)", flush=True)

# ------------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def show(ax, plane, extent, title):
    fin = plane[np.isfinite(plane) & (plane > 0)]
    if fin.size:
        vmin = max(np.percentile(fin, 55), 1.0)
        vmax = np.percentile(fin, 99.6)
    else:
        vmin, vmax = 1, 10
    cmap = plt.get_cmap("magma").copy(); cmap.set_bad("#000010")
    ax.imshow(plane.T, origin="lower", cmap=cmap, extent=extent,
              norm=LogNorm(vmin=vmin, vmax=vmax), interpolation="nearest")
    ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
    ax.set_xticks(range(-6, 7, 2)); ax.set_yticks(range(-6, 7, 2))
    ax.grid(alpha=0.15, lw=0.3); ax.set_xlabel("h"); ax.set_ylabel("k")
    ax.set_title(title, fontsize=10)

ext_mine = [H[0], H[-1], H[0], H[-1]]
ext_ref = [Href[0], Href[-1], Kref[0], Kref[-1]]
fig, ax = plt.subplots(2, 3, figsize=(15, 10), dpi=110)
for row, L in enumerate((0.0, 1.5)):
    rc_raw, rc_cor, _ = results[L]
    show(ax[row, 0], slice_at(vol_r.data, H, L), ext_mine,
         f"GPU raw, L={L}  (log-r={rc_raw:.3f})")
    show(ax[row, 1], slice_at(vol_c.data, H, L), ext_mine,
         f"GPU corrected, L={L}  (log-r={rc_cor:.3f})")
    show(ax[row, 2], planes[L], ext_ref, f"reference _raw.h5, L={L}")
fig.suptitle("MAPbBr3 215 K / 800 V - GPU reconstruction vs reference (HK planes)",
             fontsize=12)
fig.tight_layout()
png = os.path.join(OUTDIR, "compare_gpu_vs_raw.png")
fig.savefig(png, bbox_inches="tight")
print("saved comparison:", png, flush=True)
print("DONE", flush=True)
