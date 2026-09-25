"""Project a measured S(q) volume onto a detector: the reverse of rspace3d.

rspace3d turns a series of detector frames into a 3D reciprocal-space volume.
This runs that backwards: it reads such a volume and asks what an area detector
would record from it, with the crystal at given angles. Bragg peaks, superlattice
peaks and diffuse scattering all appear wherever the Ewald sphere cuts the
measured data.

Output: a still frame, a small rotation series, and a check that the strongest
measured Bragg peak lands on the pixel the six-circle forward model predicts.

    XOD_SQ_H5   the rspace3d / CrysAlisPro reconstruction (.h5)
    XOD_OUT     output folder (default: ./out beside this script)

Run:  python examples/project_sq_volume.py [volume.h5]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xrays_on_detector.geometry import BEAM
from xrays_on_detector.sqvolume import SqVolume, fill_distance, project_volume
from xrays_on_detector.vdiff.instrument import Instrument, LatticeCrystal

H5 = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("XOD_SQ_H5", "")
if not H5:
    sys.exit("pass a volume path or set XOD_SQ_H5; see the header of this file")
OUTDIR = os.environ.get(
    "XOD_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------- the sample
vol = SqVolume.from_h5(H5)
print(vol.describe(), flush=True)

# The volume is indexed in the hkl of the cell it was measured with, so the
# instrument gets that lattice. Its UB is *not* used as an orientation: a
# CrysAlisPro UB lives in CrysAlisPro's frame, which differs from this lab frame
# by a rotation the file does not record. U starts at the identity instead.
inst = Instrument(wavelength=vol.wavelength, n_fast=1475, n_slow=1679,
                  pixel_size=0.172, preview_bin=1)
inst.crystal = LatticeCrystal.from_cell(**vol.cell, name=vol.name)
inst.volume = vol

# ------------------------------------------------------- put it on the panel
# A short wavelength makes the Ewald sphere nearly flat, so a volume a few
# r.l.u. across subtends a small 2theta and needs a long arm to fill a panel.
inst.distance = float(np.clip(
    fill_distance(vol, inst.detector_obj(1), inst.wavelength), 20.0, 5000.0))
shot = inst.shoot()
print(f"\ndistance {inst.distance:.0f} mm -> {shot.coverage:.0%} of the panel "
      f"is inside the measured volume", flush=True)

# ------------------------------------------- does it land where physics says?
# Drive the strongest measured Bragg peak onto the Ewald sphere, aim the arm at
# it, and compare the brightest pixel with the forward model's prediction.
hkl_int = np.array([[h, k, l] for h in range(-4, 5) for k in range(-4, 5)
                    for l in range(-4, 5) if (h, k, l) != (0, 0, 0)])
vals, inside = vol.sample(hkl_int.astype(float))
best = hkl_int[np.argmax(np.where(inside, vals, -np.inf))]
best = tuple(int(x) for x in best)

etas = inst.solve_eta(best)
if etas:
    inst.eta = etas[0]
    inst.delta, inst.gamma = inst.aim_detector_at(best)
    det = inst.detector_obj(1)
    image, _ = project_volume(det, vol, inst.wavelength,
                              inst.sample_M() @ inst.U @ inst.crystal.B)
    r, c = np.unravel_index(int(np.argmax(image)), image.shape)
    kf = 2.0 * np.pi / inst.wavelength * BEAM + inst.q_lab(best)
    f_px, s_px, _, _ = det.project((kf / np.linalg.norm(kf))[None, :])
    off = np.hypot(r - ((det.n_slow - 1) - s_px[0]), c - f_px[0])

    # Read that against the voxel size, not against zero: these grids put their
    # voxel centres half a step off the integer lattice, so the brightest voxel
    # of a symmetric peak is always about half a voxel to one side.
    step = float(vol.H[1] - vol.H[0])
    kf2 = 2.0 * np.pi / inst.wavelength * BEAM + inst.q_lab(
        (best[0] + step, best[1], best[2]))
    f2, s2, _, _ = det.project((kf2 / np.linalg.norm(kf2))[None, :])
    px_per_voxel = float(np.hypot(f2[0] - f_px[0], s2[0] - s_px[0]))

    print(f"strongest measured peak {best}: brightest pixel ({c}, {r}), "
          f"forward model ({f_px[0]:.1f}, {(det.n_slow - 1) - s_px[0]:.1f}), "
          f"off by {off:.2f} px = {off / px_per_voxel:.2f} voxel "
          f"(1 voxel = {px_per_voxel:.1f} px here)", flush=True)
    inst.eta = inst.delta = inst.gamma = 0.0
else:
    print(f"strongest measured peak {best} is in the blind cone of eta")

# ------------------------------------------------------------------ pictures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def show(ax, img, title):
    finite = img[img > 0]
    vmax = float(np.percentile(finite, 99.99)) if finite.size else 1.0
    vmin = max(vmax * 1e-4, float(finite.min()) if finite.size else 1e-3)
    ax.imshow(img, cmap="inferno", norm=LogNorm(vmin=vmin, vmax=vmax),
              interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]), ax.set_yticks([])


shot = inst.shoot()
fig, ax = plt.subplots(figsize=(6, 6.6), dpi=140)
show(ax, shot.image, f"{vol.name}\nall circles at zero, "
                     f"{inst.distance:.0f} mm, {inst.wavelength:.4f} A")
out = os.path.join(OUTDIR, "sq_still.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print("\nsaved", out)

etas = [0.0, 10.0, 20.0, 30.0, 45.0, 60.0]
fig, axes = plt.subplots(2, 3, figsize=(11, 8), dpi=130)
for ax, e in zip(axes.ravel(), etas):
    inst.eta = e
    show(ax, inst.shoot().image, f"omega = {e:.0f} deg")
fig.suptitle(f"{vol.name}: the measured S(q) swept across the detector", y=0.98)
out = os.path.join(OUTDIR, "sq_rotation.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print("saved", out)
