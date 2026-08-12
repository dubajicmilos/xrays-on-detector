"""Predict the I4/mcm octahedral-tilt superlattice (all 3 twin domains) on a
real MAPbBr3 frame, with |F| from the 2x2x2 CIF (pytilting), and test whether
those positions carry real intensity.

Uses the xrays_on_detector.realframe API for geometry/indexing/prediction.
The 3 pseudo-merohedral twins share the parent UB; the tilt lights up different
(odd,odd,odd) super-reflections = cyclic index permutations of the CIF |F|.
"""
import os, sys, io, contextlib
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xrays_on_detector.realframe import (
    FlatDetector, detect_peaks, index_frame, predict_recorded, detector_display)

# Point these at your own data; nothing here is bundled with the repository.
#   XOD_RAW     folder holding the CBF frames
#   XOD_NAME    run stem, so the first frame is <XOD_RAW>/<XOD_NAME>_01_0001.cbf
#   XOD_REF_H5  reference rspace3d/CrysAlisPro volume, for the UB
#   XOD_CIF     CIF of the superstructure (2x2x2 pseudocubic here)
RAW = os.environ.get("XOD_RAW", "")
NAME = os.environ.get("XOD_NAME", "")
H5 = os.environ.get("XOD_REF_H5", "")
CIF = os.environ.get("XOD_CIF", "")
if not (RAW and NAME and H5 and CIF):
    sys.exit("set XOD_RAW, XOD_NAME, XOD_REF_H5 and XOD_CIF; see the header")
CBF = os.path.join(RAW, NAME + "_01_0001.cbf")
OUT = os.path.dirname(os.path.abspath(__file__))

# structure factors from the 2x2x2 pseudocubic CIF (super-index basis)
PT = os.environ.get("PYTILTING_PATH", "")
if not PT:
    sys.exit("set PYTILTING_PATH to a pytilting checkout (for |F(hkl)|)")
sys.path.insert(0, os.path.join(PT, "tests"))
from structure_factor_calculator import StructureFactorCalculator
with contextlib.redirect_stdout(io.StringIO()):
    CALC = StructureFactorCalculator(CIF)
_F = {}
def Fabs(t):
    if t not in _F: _F[t] = abs(CALC.calculate_structure_factor(*t)[0])
    return _F[t]

det, ang, img = FlatDetector.from_eiger_cbf(CBF)
with h5py.File(H5, "r") as f:
    UB = np.array(f["UB"]) / det.wavelength
R1 = index_frame(detect_peaks(img, det.beam_center), UB, det).R

# super reflections within reach; positions use pseudocubic = super/2
S = np.arange(-14, 15)
sup = np.stack(np.meshgrid(S, S, S, indexing="ij"), -1).reshape(-1, 3)
sup = sup[np.any(sup != 0, 1)]
sup = sup[np.linalg.norm((UB @ (sup / 2).T).T, axis=1) <= 1.65]
# predict_recorded returns the (filtered) fractional hkl; recover super indices
hkl_frac, fx, fy, _ = predict_recorded(R1, UB, det, sup / 2.0,
                                       oscillation=ang["Angle_increment"],
                                       osc_axis=(0, 1, 0), sigma=0.006)
supr = np.round(hkl_frac * 2).astype(int)

img = img.astype(np.float32)
ny, nx = det.shape

def measure(x, y, rs=3, rb=18):
    xi, yi = int(round(x)), int(round(y))
    if xi < rb or xi >= nx - rb or yi < rb or yi >= ny - rb: return None
    box = img[yi-rb:yi+rb+1, xi-rb:xi+rb+1]
    if (box < 0).mean() > 0.1: return None
    peak = np.percentile(img[yi-rs:yi+rs+1, xi-rs:xi+rs+1], 98)
    ring = box.copy(); ring[rb-rs:rb+rs+1, rb-rs:rb+rs+1] = -1
    bg = np.median(ring[ring >= 0])
    return (peak - bg) / np.sqrt(bg + 1.0)

main, doms = [], [[], [], []]
snr_main, F2_super, snr_super = [], [], []
for (H, K, L), x, y in zip(supr, fx, fy):
    is_main = H % 2 == 0 and K % 2 == 0 and L % 2 == 0
    if is_main:
        main.append((x, y)); s = measure(x, y)
        if s is not None: snr_main.append(s)
    else:
        Fd = [Fabs((H, K, L)), Fabs((K, L, H)), Fabs((L, H, K))]
        d = int(np.argmax(Fd))
        if max(Fd) > 1:
            doms[d].append((x, y))
            s = measure(x, y)
            if s is not None:
                F2_super.append(max(Fd) ** 2); snr_super.append(s)

# background-control SNR level
rng = np.random.default_rng(1); ctrl = []
while len(ctrl) < 300:
    s = measure(rng.uniform(50, nx-50), rng.uniform(50, ny-50))
    if s is not None: ctrl.append(s)
thr = np.percentile(ctrl, 99)
snr_main = np.array(snr_main); F2_super = np.array(F2_super); snr_super = np.array(snr_super)
print(f"main Bragg: {(snr_main>thr).sum()}/{len(snr_main)} above background-99pct")
print(f"superlattice: {(snr_super>thr).sum()}/{len(snr_super)} above background-99pct")
if len(F2_super) > 3:
    c = np.corrcoef(np.log10(F2_super+1), np.log10(np.clip(snr_super*10, 1, None)))[0, 1]
    print(f"superlattice corr(log|F|^2, log intensity) = {c:+.2f}")

# overlay
b = 2
db, vmin, vmax = detector_display(img)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
fig, ax = plt.subplots(figsize=(9.5, 9.5), dpi=120)
cmap = plt.get_cmap("gray").copy(); cmap.set_bad("#0a0a0a")
ax.imshow(np.where(db > 0, db, np.nan), origin="upper", cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
if main:
    m = np.array(main); ax.scatter(m[:,0]/b, m[:,1]/b, s=40, c="#00e5ff", marker="+", lw=1.1, label=f"main Bragg ({len(main)})")
for d, col in enumerate(["#ff3b3b", "#37ff37", "#5b8bff"]):
    if doms[d]:
        dd = np.array(doms[d]); ax.scatter(dd[:,0]/b, dd[:,1]/b, s=55, marker="x", c=col, lw=1.4, label=f"tilt domain {d+1} ({len(dd)})")
ax.set_title("MAPbBr3 frame 01_0001: main Bragg + I4/mcm tilt superlattice (3 twin domains)")
ax.legend(loc="upper right", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
out = os.path.join(OUT, "I19-2_superlattice_frame1.png")
fig.savefig(out, bbox_inches="tight"); print("saved", out)
