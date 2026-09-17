# Reciprocal-space reconstruction from CrysAlisPro data

How the raw-CBF to 3D reciprocal-space reconstruction works, what it would take
to make it universal for any CrysAlisPro dataset, and honest answers to the
open questions.

Code: `xrays_on_detector/realframe.py` (detector geometry, `.par` reading,
frame indexing), `reconstruct.py` (CPU engine, HDF5 output),
`reconstruct_gpu.py` (CuPy engine), `corrections.py` (per-pixel intensity
corrections). Run scripts: `examples/reconstruct_I19-2.py`,
`examples/reconstruct_I19-2_gpu.py`, `examples/reconstruct_from_par.py`.

---

## 1. What works now

Given raw Eiger CBF frames from a phi scan, the CrysAlisPro UB, and the detector
geometry, we reconstruct a 3D reciprocal-space volume in crystallographic (hkl)
coordinates. Validated against a reference reconstruction of the same dataset:
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
rotation sense fails, so `R0`, the oscillation axis, and the sense are all
correct.

---

## 3. Exact geometry / conventions used

- **Lab frame:** beam along +z (source to sample to detector), detector fast
  axis +x, slow axis +y, detector plane at z = distance. Flat detector, **2θ = 0**
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
**Yes, but it is cheap, and it does NOT need a CIF.** Indexing uses only the UB
(the cell + its orientation) and a handful of Bragg spots on one frame. For a
strong crystal that is a few seconds and fully automatic. For a weak crystal,
accumulate spots over several frames first.

### Is R0 "just a fixed thing in CrysAlisPro" that we can reuse universally?
**Partly, and this is the important insight.** Decompose the forward map:

    r_lab = R_conv . R_gonio(omega,kappa,phi) . UB . hkl

- `R_conv` = the FIXED rotation between CrysAlisPro's frame and our detector-lab
  frame. Fixed for a given **beamline + detector + CrysAlisPro import config**.
- `R_gonio(omega,kappa,phi)` = the goniometer rotation in CrysAlisPro's KM4 kappa
  convention.
- `UB` carries the **crystal** (cell + mounting orientation).

At the reference frame this gives `R0 = R_conv . R_gonio(omega0,kappa0,phi0)`.
**Crucially, R0 does not contain the crystal**: the crystal is entirely inside
UB. So in principle:

> R0 is a property of the instrument and the reference goniometer angles, **not**
> of the crystal. Two different crystals measured on the same instrument at the
> same goniometer datum should give the **same R0** (up to a lattice-symmetry
> branch, see caveat).

If true, you calibrate `R_conv` **once** and never index again. **Caveats (be
honest):**
1. This is a **theoretical** consequence of the standard Busing-Levy convention
   (UB defined at datum, `Q_lab = R_gonio . UB . hkl`). Checked so far on two
   datasets of the same crystal at different temperatures: their fitted R0
   differ by a 4/mmm symmetry operation, which is exactly the branch ambiguity
   of caveat 2. So R0 is not reusable across mounts as fitted; re-indexing per
   dataset (fast) stays the default.
2. Indexing can land on any **symmetry-equivalent** orientation (near-cubic
   metric gives up to 48 branches). So a fitted R0 is fixed only *up to a lattice
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

## 5. Making it universal: concrete plan

**Tier 1 (works today, robust, recommended default).** For each dataset: read
geometry from the CBF header, read UB from the `.par` (`realframe.read_crysalis_par`)
or an `.h5`, auto-index one frame per *sweep* to get R0
(`realframe.orient_from_frame`), reconstruct via (★). No CIF, no convention
derivation. Handles any single- or multi-sweep CrysAlisPro phi/omega scan. The
only per-dataset unknown (R0) is found automatically.
Gaps to close: auto-pick the reference frame with the most spots; support omega
scans and nonzero-2θ detector positions.

**Tier 2 (fully blind, no indexing).** Calibrate `R_conv` once (from this dataset:
`R_conv = R0 . R_gonio(omega0,kappa0,phi0)^-1`, which needs the KM4 model), and
encode `R_gonio(omega,kappa,phi)`. Then R0 for any future dataset is computed
from its header angles, with no indexing. More work, and must be validated against
Tier 1 before trust. Worth it only if you want reconstruction with zero Bragg
spots available (e.g. very weak crystals) or full automation.

---

## 6. The engine

Our own numpy code (`xrays_on_detector/reconstruct.py`), about 150 lines, with a
CuPy twin (`reconstruct_gpu.py`) that shares the pixel to hkl map and the voxel
grid. Per frame it is one 3×3 matrix multiply over all pixels (pixel to hkl)
plus a weighted 3D histogram into the voxel grid. Each voxel stores the sum of
counts and the number of contributing pixels; the output is their ratio, so
voxels visited many times and voxels visited once are on the same footing.

---

## 7. Efficiency and GPU

CPU path, measured: **16 min for 1750 frames** on a 47M-voxel grid (−9..9 r.l.u.,
0.05 step), single-threaded. Cost breakdown, slowest first:
1. **Histogram accumulation** (`np.bincount` with `minlength = 47M`, twice per
   frame). Dominant. Allocating/summing a 47M array every frame is the killer.
2. **Frame I/O + decode** (7.9 GB read from disk + fabio byte-offset decode).
   Second biggest; embarrassingly parallel.
3. Pixel to hkl matmul (small, ~4.5M×3 @ 3×3). Cheap.

The GPU path (`reconstruct_volume_gpu`) removes the first two costs: one
persistent float64 sum + int32 count accumulator lives on the GPU and each frame
is a `scatter_add` into it (no per-frame voxel-sized temporaries), the pixel
scattering vectors and the correction map are uploaded once, and frames are
read ahead on a thread pool so disk I/O overlaps compute. It is bit-identical
to the CPU path (`tests/test_reconstruct_gpu.py`). On the 480³ grid used for the
I19-2 validation (−6..6 at 0.025 r.l.u.) it takes about half a minute for 1750
frames against about half an hour on the CPU.

If the CPU path has to be used, a smaller grid is the easy win: −6..6 at 0.05 is
240³ = 13.8M voxels, about 3× faster than the −9..9 grid for the same physics if
you do not need |hkl| > 6.

---

## 8. Intensity corrections

Corrections are part of **reconstruction**, not of any later modelling. For a
photometrically quantitative diffuse map they are applied per pixel at
accumulation time. `corrections.pixel_corrections` implements the two
geometric ones as a single multiplier folded into the pixel weights:
- **Solid angle** (a flat-detector pixel off-centre subtends less; the factor is
  cos³ of the obliquity),
- **Polarization** (synchrotron beam is mostly horizontally polarized; the I19-2
  header gave Polarization = 0.99), in the Thomson form
  `p (1 − (h·k̂)²) + (1 − p) (1 − (v·k̂)²)`.

**Lorentz is deliberately not applied.** A rotation-scan reconstruction that
normalises each voxel by its number of contributing pixels already accounts for
the varying time a reciprocal-space point spends on the Ewald sphere, so
dividing by the count is the geometric Lorentz correction.

Not implemented (each is a further per-pixel multiplier of ~10 lines):
- **Air absorption**, `exp(−μ · path)`, μ from the mass attenuation at the
  photon energy,
- **Sensor absorption / quantum efficiency**, `1 − transmission(material, λ, t)`
  (CdTe, 0.75 mm for the Eiger, from the header),
- **Detector flat field.**

The polarization axis is taken as the detector `fast` axis (see the module
docstring); confirm it for a new instrument before trusting the correction at
the few-percent level.

---

## 9. Recommendation for "universal"

1. Keep **Tier 1** (auto-index one frame per sweep) as the default: it is
   already universal for CrysAlisPro phi/omega scans and needs no CIF.
2. Use the GPU path (Section 7) with the corrections on (Section 8); add the
   absorption terms if the diffuse intensities are to be used quantitatively.
3. Optionally pursue **Tier 2** (calibrate `R_conv` once) to drop the indexing
   step, but first pin the symmetry branch (Section 4, caveats 1 and 2).
4. Output stays in the rspace3d HDF5 layout (`reconstruct.save_rspace3d_h5`) so it
   feeds rspace3d's symmetriser and viewer unchanged.
