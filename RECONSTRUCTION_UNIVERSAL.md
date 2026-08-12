# Universal reciprocal-space reconstruction from CrysAlisPro data

How the working raw-CBF → 3D reciprocal-space reconstruction was made possible,
what it would take to make it universal for any CrysAlisPro dataset, and honest
answers to the open questions. Written 2026-07-14 as a knowledge file for future
sessions. **No code changes were made when writing this; it is notes only.**

Code lives in `C:\Claude projects\X-rays-on-detector\xrays_on_detector`
(`realframe.py`, `reconstruct.py`); run script `examples\reconstruct_I19-2.py`;
outputs + figures in
`...\rspace3d\validation\cbf-reconstruct-xrays\`.

---

## 1. What works now

Given raw Eiger CBF frames from a phi scan, the CrysAlisPro UB, and the detector
geometry, we reconstruct a 3D reciprocal-space volume in crystallographic (hkl)
coordinates. Validated against the existing CrysAlisPro/rspace3d reconstruction:
Bragg peaks land on the integer grid (L=0), and the tilt superlattice + diffuse
rods reproduce on the half-integer plane (L=1.5). Dataset: MAPbBr3 215 K / 800 V,
I19-2, single 350° phi scan (omega=-90, kappa=0 fixed).

---

## 2. The core idea: the one missing rotation

The reconstruction is the **inverse of the forward diffraction model**. For every
detector pixel we form the scattering vector in the lab frame

    r_lab = s1 - s0 ,   |s1| = |s0| = 1/lambda        (1/d, "crystallographic" units)

and map it to Miller indices with

    hkl = (R_n . UB)^-1 . r_lab ,   R_n = R_osc(phi_n - phi0) . R0 .          (★)

Everything on the right is known **except R0**, the crystal-to-lab orientation of
one reference frame. That is the whole game.

### Why the previous attempt failed
CrysAlisPro's UB is expressed in **CrysAlisPro's internal frame**, not in the
frame where you naturally build `r_lab` from detector pixel geometry (beam,
distance, beam centre). Using UB directly gives hkl that are rotated/mirrored, so
Bragg peaks do **not** land on integer positions. The missing piece is the fixed
rotation between those two coordinate systems. That is exactly what `R0`
supplies. (The removed `cbf_reconstruct.py` in rspace3d hit this wall.)

### How R0 is obtained here
By **indexing one frame**: find the rotation `R0` that makes `R0 . UB . hkl`
match the observed scattering vectors of that frame's Bragg spots
(`realframe.index_frame`, a known-cell pair-indexing search). It was then
**validated by pure forward prediction**: `R0` + the header phi increment (no
refit) predicts every spot on frames out to 60° of rotation, and the opposite
rotation sense fails — so `R0`, the oscillation axis, and the sense are all
correct.

---

## 3. Exact geometry / conventions used

- **Lab frame:** beam along +z (source→sample→detector), detector fast axis +x,
  slow axis +y, detector plane at z = distance. Flat detector, **2θ = 0**
  (on-axis). If a dataset has the detector at nonzero 2θ, add the detector-arm
  rotation (already available as `realframe.detector_matrix` logic).
- **Units:** crystallographic 1/d (no 2π). `UB_used = UB_crysalis / lambda`, so
  the columns are a*, b*, c* with |a*| = 1/a. `k = 1/lambda`.
- **Oscillation:** rotation about lab **+y**, sense **+1** (determined
  empirically, then validated). For a kappa goniometer at kappa=0 the phi axis is
  collinear with the base spindle, which is why a single lab-axis rotation works.
- **Reference frame:** `phi0` = phi of the frame `R0` was indexed on.

---

## 4. Answers to the questions

### Do we need to figure out R?
**Yes — but it is cheap, and it does NOT need a CIF.** Indexing uses only the UB
(the cell + its orientation) and a handful of Bragg spots on one frame. For a
strong crystal that is a few seconds and fully automatic. For a weak crystal,
accumulate spots over several frames first.

### Is R0 "just a fixed thing in CrysAlisPro" that we can reuse universally?
**Partly — and this is the important insight.** Decompose the forward map:

    r_lab = R_conv . R_gonio(omega,kappa,phi) . UB . hkl

- `R_conv` = the FIXED rotation between CrysAlisPro's frame and our detector-lab
  frame. Fixed for a given **beamline + detector + CrysAlisPro import config**.
- `R_gonio(omega,kappa,phi)` = the goniometer rotation in CrysAlisPro's KM4 kappa
  convention.
- `UB` carries the **crystal** (cell + mounting orientation).

At the reference frame this gives `R0 = R_conv . R_gonio(omega0,kappa0,phi0)`.
**Crucially, R0 does not contain the crystal** — the crystal is entirely inside
UB. So in principle:

> R0 is a property of the instrument and the reference goniometer angles, **not**
> of the crystal. Two different crystals measured on the same instrument at the
> same goniometer datum should give the **same R0** (up to a lattice-symmetry
> branch, see caveat).

If true, you calibrate `R_conv` **once** and never index again. **Caveats (be
honest):**
1. This is a **theoretical** consequence of the standard Busing–Levy convention
   (UB defined at datum, `Q_lab = R_gonio . UB . hkl`). It is **not yet tested
   across crystals** — we only have one dataset. The test: index frame 1 of a
   *different* crystal at the same datum and check R0 matches.
2. Indexing can land on any **symmetry-equivalent** orientation (near-cubic
   metric → up to 48 branches). So a fitted R0 is fixed only *up to a lattice
   symmetry operation*. For a truly reusable `R_conv` you must pin the branch.
3. `R_conv` is fixed only per beamline/detector/CrysAlisPro-config. A different
   instrument (or a detector at a different 2θ) needs its own calibration.

### Do we need a CIF file?
**No.** The reconstruction needs UB + detector geometry + R0. The CIF was only
used for a *separate* analysis (predicting tilt-superlattice **intensities** via
pytilting). Reconstruction is purely geometric.

### What is needed per new dataset vs one-time?
| Per dataset (from files, no fitting) | One-time per instrument (optional) |
|---|---|
| detector distance, pixel, beam centre, 2θ, wavelength (CBF header) | `R_conv` calibration (or the full KM4 convention) |
| per-frame omega/kappa/phi (CBF header) | goniometer axis directions + senses + zero corrections |
| UB (CrysAlisPro `.par` or an rspace3d `.h5`) | detector axis handedness |
| R0 via indexing one frame (unless R_conv is calibrated) | |

---

## 5. Making it universal — concrete plan

**Tier 1 (works today, robust, recommended default).** For each dataset: read
geometry from the CBF header, read UB from the `.par`/`.h5`, auto-index one frame
per *sweep* to get R0, reconstruct via (★). No CIF, no convention derivation.
Handles any single- or multi-sweep CrysAlisPro phi/omega scan. The only
per-dataset unknown (R0) is found automatically.
Gaps to close: parse UB directly from `.par` (currently read from the rspace3d
`.h5`); auto-pick the reference frame with the most spots; support omega scans
and nonzero-2θ detector positions.

**Tier 2 (fully blind, no indexing).** Calibrate `R_conv` once (from this dataset:
`R_conv = R0 . R_gonio(omega0,kappa0,phi0)^-1`, which needs the KM4 model), and
encode `R_gonio(omega,kappa,phi)`. Then R0 for any future dataset is computed
from its header angles — no indexing. More work, and must be validated against
Tier 1 before trust. Worth it only if you want reconstruction with zero Bragg
spots available (e.g. very weak crystals) or full automation.

---

## 6. What engine is used for real↔reciprocal conversion?

**Our own ~150-line numpy code** (`xrays_on_detector/reconstruct.py`). Not a
third-party engine. Per frame it is one 3×3 matrix multiply over all pixels
(pixel→hkl) plus a weighted 3D histogram (`np.bincount`) into the voxel grid.

### We are NOT using Yell — and Yell would not reconstruct anyway
Verified: **Yell is a 3D-ΔPDF refinement program** — it *models* diffuse
scattering by fitting a real-space pair-distribution model to an **already
reconstructed, already corrected** reciprocal-space volume. It does **not** read
detector frames and does **not** do the reconstruction or the intensity
corrections. So "Yell gives the fastest reconstruction" / "Yell has the intensity
corrections" is a mix-up: those belong to the **reconstruction** step, which is a
different program.

The reconstruction tool in that ecosystem is **Meerkat** (same author, Arkadiy
Simonov): "a python program for performing reciprocal space reconstruction from
single-crystal X-ray measurements," in crystallographic coordinates, with
symmetry averaging — i.e. it does essentially what our `reconstruct.py` does, and
it already implements the corrections. The pipeline is:

    raw frames --[Meerkat: reconstruct + correct + symmetrise]--> hdf5 --[Yell: 3D-dPDF model]--> disorder model

So our code is a **Meerkat-equivalent** reconstruction step. For the corrections
and a battle-tested path, Meerkat is the reference to compare against or adopt;
Yell would sit downstream of whatever reconstruction we use.

---

## 7. Efficiency and GPU

Current run: **16 min for 1750 frames** on a 47M-voxel grid (−9..9 r.l.u.,
0.05 step), single-threaded. Cost breakdown, slowest first:
1. **Histogram accumulation** (`np.bincount` with `minlength = 47M`, twice per
   frame). Dominant. Allocating/summing a 47M array every frame is the killer.
2. **Frame I/O + decode** (7.9 GB read from the F: drive + fabio byte-offset
   decode). Second biggest; embarrassingly parallel.
3. Pixel→hkl matmul (small, ~4.5M×3 @ 3×3). Cheap.

**Speedups (not yet done):**
- **GPU (biggest win):** move the matmul + scatter-add to CuPy
  (`cupyx.scatter_add` / a bincount kernel). rspace3d already uses CuPy for
  symmetrisation, so the dependency is present. Expect ~10–50× on the histogram.
- **Smaller grid:** −6..6 at 0.05 is 240³ = 13.8M voxels → ~3× faster than the
  −9..9 grid used, for the same physics if you don't need |hkl|>6.
- **Threaded frame reading** to overlap I/O with compute (I/O is a large fraction).
- **Persistent accumulator** instead of re-allocating a `minlength` array each
  frame (use scatter-add into a fixed buffer).
With GPU + threaded I/O, minutes → tens of seconds is realistic. Meerkat is also
numpy-based, so it is not automatically faster than our code; a GPU version would
likely beat it.

---

## 8. Intensity corrections (where they belong, not done yet)

Corrections are part of **reconstruction**, not ΔPDF. For a photometrically
quantitative diffuse map, apply per pixel before/at accumulation:
- **Solid angle** (pixels off-centre subtend less; ∝ cos³ of the obliquity for a
  flat detector),
- **Polarization** (synchrotron beam is ~horizontally polarized; header gave
  Polarization = 0.99),
- **Lorentz** (rotation method: ∝ 1/|sin(2θ)·(component of the reflection
  velocity through the sphere)|; the standard rotation Lorentz factor),
- **Detector efficiency / flat field, air+sensor absorption** (CdTe sensor,
  0.75 mm),
- **Per-voxel normalisation** by the number of contributing measurements
  (already done: we store counts and output the mean).
Meerkat implements these; our reconstruction currently outputs the plain mean of
raw counts (geometrically correct, not yet corrected).

---

## 9. Recommendation for "universal"

1. Keep **Tier 1** (auto-index one frame per sweep) as the default — it is
   already universal for CrysAlisPro phi/omega scans and needs no CIF.
2. Add the **corrections** (Section 8) and a **GPU** path (Section 7) to
   `reconstruct.py`; or, equivalently, **benchmark/adopt Meerkat**, which already
   has both and outputs Yell-ready hdf5.
3. Optionally pursue **Tier 2** (calibrate `R_conv` once) to drop the indexing
   step — but first *test the crystal-independence of R0* across two crystals, and
   pin the symmetry branch.
4. Output stays in rspace3d hdf5 layout so it feeds rspace3d's symmetriser and
   viewer, and (once corrected) Yell.

---

## 11. Meerkat2 (C++) — assessment (checked the source 2026-07-14)

Repo: https://github.com/aglie/Meerkat2 — a C++17 rewrite (Eigen + HDF5 + CBFlib),
reads CBF/HDF5, outputs Yell-format hdf5.

**Efficiency.** Compiled C++, but **single-threaded — no OpenMP / TBB / GPU**
(verified in `CMakeLists.txt`). Faster than Python Meerkat / our numpy per frame
(no interpreter overhead, Eigen-vectorised), but not parallel and not GPU, so a
**CuPy/GPU version of our reconstruction would likely match or beat it**. It has a
`RECONSTRUCT_EVERY_NTH_FRAME` frame-skip option for speed (same trick as our
prototype). Bottom line: modestly faster than us today; not the fastest possible.

**Corrections (from `Corrections.cpp` — this is the useful part).** A single
per-pixel coefficient `= solid_angle × polarization × air_transmission ×
sensor_absorption`, **multiplied into** each pixel:
- **Solid angle / obliquity:** `cos³(detected-ray angle)` (flat-detector
  foreshortening + projected solid angle).
- **Polarization:** `(1-P)·(1-(n̂·ŝ)²) + P·(1-(p̂·ŝ)²)`, with configurable factor
  `P` and plane normal (synchrotron P≈1).
- **Air absorption:** `exp(-μ·path)`, μ from mass-attenuation at the photon energy.
- **Sensor absorption / quantum efficiency:** `1 - transmission(material, λ, t)`
  (Silicon for Pilatus; for our Eiger use **CdTe, 0.75 mm** from the header).
- **No Lorentz — deliberate.** Diffuse reconstruction normalises each voxel by its
  **measurement count**, which plays the Lorentz role for continuous scattering.
  Our `reconstruct.py` already does this (outputs sum/count). So the ONLY
  corrections we lack vs Meerkat2 are the **three per-pixel multipliers above**
  (solid angle, polarization, absorptions) — each ~10 lines.

**Big caveat for our goal.** Meerkat2 takes orientation from **XDS**
(`XPARM.XDS`/`GXPARM.XDS`), **not CrysAlisPro**. So it does not solve the
CrysAlis-UB ambiguity — it sidesteps it by using XDS's self-consistent geometry.
To use Meerkat2 on our data: either (a) re-index the frames in XDS, or (b) write a
**CrysAlis-UB → `XPARM.XDS` converter** (documented text format: rotation axis,
beam, detector geometry, and the A/UB matrix). Option (b) is the clean bridge:
CrysAlisPro indexing + Meerkat2 reconstruction/corrections.

**Recommendation (two viable paths, both give corrected Yell-ready volumes):**
1. **Port the three corrections** into our `reconstruct.py` (formulas above) and
   optionally GPU-accelerate. Stays fully CrysAlisPro-native, no XDS.
2. **Bridge to Meerkat2** via a UB→`XPARM.XDS` writer, then run the maintained C++
   reconstruction for free.

## 10. Sources
- Yell (3D-ΔPDF refinement): https://github.com/yellprogram/Yell
- Meerkat (Python reconstruction from frames): https://github.com/aglie/meerkat
- Meerkat2 (C++ reconstruction from frames): https://github.com/aglie/Meerkat2
- Our code: `xrays_on_detector/reconstruct.py`, `realframe.py`;
  `examples/reconstruct_I19-2.py`.
