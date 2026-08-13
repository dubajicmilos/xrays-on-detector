# xrays_on_detector

Simulate the single-crystal diffraction **image on an area detector** of a
six-circle diffractometer: given a CIF, a set of diffractometer angles, a
detector (distance, size, pixel, arm angles) and a wavelength, compute which
reflections are excited and where their spots land.

It stitches together three existing pieces:

| Role | Package |
|------|---------|
| Structure factors `|F(hkl)|²` and reciprocal lattice (2π convention) | **pytilting** `StructureFactorCalculator` |
| Six-circle rotation matrices, You (1999) convention | **diffcalc-core** |
| (optional) cross-checking reciprocal-space / area-detector conversions | xrayutilities |

> Note: `escape-fel` was considered but is **not** used. It is a FEL
> data-handling / lazy-array framework and carries no diffractometer or
> detector geometry.

## Physics

Monochromatic beam, `k = 2π/λ`, incident along the lab **+y** axis. The diffcalc
/ You (1999) frame is **+x vertical (up), +y along the beam, +z horizontal**;
`mu` and `nu` rotate about +x, `eta`, `phi` and `delta` about −z, and `chi`
about +y (the beam). So `delta` moves the detector up and down and `nu` (called
`gamma` at most beamlines) moves it left and right.

The virtual diffractometer and the browser build read the panel out in the
**beam's-eye view**: fast is `−z` and slow is `+x`, so `e_fast × e_slow = −arm`
and the frame displayed with column 0 on the left and row 0 at the top is what
you see standing at the sample looking downstream. That is the same picture the
3D scene shows on the detector face, and the same handedness as `realframe`.

1. Each reflection has a reciprocal-lattice point `Q = Z·U·B·(h,k,l)`, where
   `B` is the reciprocal matrix from the CIF, `U` the crystal orientation, and
   `Z = MU·ETA·CHI·PHI` the sample circles.
2. Elastic scattering: `k_f = k_i + Q`, `|k_f| = k`. The signed **excitation
   error** `ε = |k_i + Q| − k` measures the distance from the Ewald sphere.
3. Bragg peaks have finite size, modelled as **isotropic 3D Gaussians** of
   width `σ` in reciprocal space. A reflection contributes with weight
   `exp(−ε²/2σ²)`, placed along `k̂_f = (k_i + Q)/|k_i + Q|`.
4. The diffracted ray is projected onto a flat detector on the `NU·DELTA` arm;
   the finite `σ` gives each spot a finite size.

Per-reflection intensity: `I = |F|² · exp(−ε²/2σ²) · polarization(2θ)`,
rendered as a Gaussian normalised to that integrated value.

## Install

```bash
git clone https://github.com/dubajicmilos/xrays-on-detector
cd xrays-on-detector
pip install -e .
```

That gets the simulation core (`numpy`, `diffcalc-core`). The rest are extras,
so you only install what you actually use:

| Extra | `pip install -e ".[extra]"` | For |
|-------|------------------------------|-----|
| `vdiff` | PyQt6, matplotlib, scipy | the virtual diffractometer app |
| `data`  | fabio, h5py, scipy | reading real frames, writing volumes |
| `cif`   | ase | symmetry expansion of a CIF |
| `gpu`   | cupy | CUDA reconstruction (match your toolkit, e.g. `cupy-cuda12x`) |

**Structure factors need `pytilting`, which is not on PyPI.** Point
`PYTILTING_PATH` at a checkout (the directory whose `tests/` holds
`structure_factor_calculator.py`). Without it the geometry, Ewald construction,
reconstruction and the whole browser build still work; only `|F(hkl)|` from a
CIF is unavailable, and `Crystal.from_cif` says so rather than failing quietly.

The example and validation scripts read their paths from the environment, since
no experimental data ships with the repository:

| Variable | Meaning |
|----------|---------|
| `XOD_RAW` | folder of CBF frames (and CrysAlisPro's `unwarp/`) |
| `XOD_NAME` | run stem, so frame *n* is `<XOD_RAW>/<XOD_NAME>_01_000n.cbf` |
| `XOD_REF_H5` | reference rspace3d/CrysAlisPro volume to compare against |
| `XOD_CIF` | CIF for the superlattice example |
| `XOD_OUT` | output folder (default `./out` beside the script) |

## Usage

```python
from xrays_on_detector import Crystal, Detector, simulate_frame

crystal  = Crystal.from_cif("mystructure.cif")          # arbitrary CIF
detector = Detector(distance=120.0, n_fast=1024, n_slow=1024,
                    pixel_size=0.2, nu=0.0, delta=0.0)   # mm

frame = simulate_frame(crystal, detector, wavelength=0.7, sigma=0.04,
                       mu=0, eta=9.5, chi=0, phi=0)

frame.image      # (n_slow, n_fast) float array, ready to display or save
frame.table      # list of dicts: h,k,l, fast_px, slow_px, eps, two_theta_deg, intensity
```

Move the diffractometer by changing the sample angles (`mu, eta, chi, phi`) or
the detector (`Detector(..., nu=, delta=, distance=)`). See `examples/demo.py`
for a two-panel CsPbBr₃ example including detector motion, and a CSV export.

## Conventions

- **Units**: 2π reciprocal convention throughout, `|Q| = 2π/d`, matching
  pytilting's `B`. diffcalc's own `B`/`UB` use 1/d, so only its (unit-free)
  rotation matrices are used here.
- **Angles** in degrees; **lengths** (`distance`, `pixel_size`) share one unit.
- **`U`** defaults to identity (crystal Cartesian frame == phi frame at zero
  angles). Pass your own orientation matrix for a mounted crystal.
- Arbitrary CIFs are expanded to an explicit all-atom P1 cell with ASE before
  the structure-factor sum; `crystal.n_atoms` reports how many atoms were used.

## Validated

`tests/validate.py` checks, against analytic physics:
- detector ray projection `r = D·tan(ψ)` to ~1e-12 (machine precision);
- Ewald/Bragg self-consistency `sin θ = |Q|/2k` to ~6e-5 over hundreds of
  reflections spanning 7–77° in 2θ (NaCl, structure factors also verified:
  strong 200/220/400, weak all-odd 111, extinct mixed-parity).

## Interactive virtual diffractometer (`vdiff/`)

```bash
python -m xrays_on_detector.vdiff
```

A PyQt6 app that puts the whole forward model behind a set of motors. The left
column is the setup, the centre is a 3D view of the instrument, the right is the
simulated frame. Needs only PyQt6 on top of the package requirements; the 3D is
a small software renderer (QPainter, depth sort, near-plane clip) so there is no
OpenGL dependency, and the live frame is projectively texture-mapped onto the
detector face as the arm swings.

- **Detector presets**: PILATUS3 100K/300K/1M/2M/6M, EIGER2 X 1M/4M/9M/16M,
  LAMBDA 750K, JUNGFRAU 1M, or type in any pixel count and pitch. Distance,
  wavelength/energy and preview binning are live.
- **Motors**: `mu`, `omega`(=eta), `chi`, `phi` for the sample and `delta`,
  `gamma`(=nu) for the detector arm, as sliders and spin boxes. Each row has a
  run button that turns that circle continuously, and any number of them can
  run at once; one shared signed speed sets the rate and the direction, and
  angles wrap so a circle keeps going.
- **Sample**: a lattice preset (runs with no CIF at all) or **Load CIF ...** for
  real `|F(hkl)|²` through pytilting.
- **Orientation without writing a UB by hand**. Three free-rotation sliders turn
  the crystal about the lab axes, or point a direction where you want it: *put
  (110) along the beam*, then optionally *spin about that axis until (001) is
  vertical*, which removes the leftover degree of freedom. `(hkl)` means the
  plane normal and `[uvw]` the real-space direction, which matters as soon as
  the lattice is not cubic. The resulting **UB is displayed live** and can be
  copied, in the 2π, 1/d or λ-scaled (CrysAlisPro) convention. The sliders
  compose on top of an alignment rather than discarding it.
- **Two geometries**. *Transmission* is the ordinary single-crystal rotation
  case. *Reflection* adds a sample surface with a chosen `(hkl)`: the incidence
  angle `alpha` is displayed, and any reflection whose incoming or outgoing beam
  is below the surface horizon is removed rather than drawn, which is the actual
  physical difference between the two cases.
- **Drive to a reflection**: type `h k l`, press *Find omega*, and it solves
  `|k_i + Q(omega)| = k` for every `omega` that puts that reflection on the
  Ewald sphere at the current `chi`/`phi`/`mu`, listing the `delta`/`gamma` the
  arm needs for each. *Drive there* moves the motors onto it.

Verified end to end: a solved `omega` lands the reflection on the Ewald sphere
to `|eps| < 1e-11 1/A`, and *Aim detector* puts it on the beam centre to
0.000 px. `examples/virtual_diffractometer.py` is the same launcher.

## Browser build: the Diffraction Game (`web/`)

A client-side JavaScript port of the same forward model, deployed as a tab on
<https://dubajicmilos.github.io/diffraction/>. No backend: everything runs in the
visitor's browser. The physics is checked against this package by a Node harness
(`node web/test/parity.mjs`, 27 groups, machine precision). See
[web/README.md](web/README.md) for the architecture, and `tools/deploy_to_site.py`
to re-sync it into the Jekyll site after a change.

## Single-crystal patterns from any CIF (`single_crystal/`, `web/sc/`)

A separate, self-contained package for the other kind of question: not "where
does this reflection land on my detector" but "what does this structure's
diffraction pattern look like". It reads any CIF and computes:

- **reciprocal-lattice sections** — the undistorted plane a precession camera
  records, named by a zone axis `[uvw]` and a layer `n`, so `[100]` layer 0 is
  the *0kl* section, layer 3 the *3kl* one, and `[110]` or `[123]` are the
  diagonal cuts;
- **selected-area electron diffraction** — the same zone through a curved
  Ewald sphere, with a `sinc²` relrod, so the higher-order Laue zones appear;
- **powder patterns** — multiplicity summed by construction rather than looked
  up from the Laue class.

Each for **X-rays, neutrons or electrons**. Deuterium keeps its own neutron
scattering length: `b(H)` is −3.739 fm and `b(D)` is +6.671 fm, opposite in
sign, so folding D into H would invert its contribution.

```bash
python -m single_crystal structure.cif      # PyQt6 desktop viewer
```
```python
from single_crystal import read_cif, Structure, compute_section
xtal = Structure.from_cif(read_cif("rutile.cif"))   # symmetry expanded to P1
sec = compute_section(xtal, uvw=(1, 1, 0), layer=0, radiation="neutron")
```

The browser build is `web/sc/`, deployed at `/diffraction/single-crystal/`. It
shares the CIF reader and the reciprocal lattice with the Game of Diffraction
next door, and the CIF you upload never leaves your machine.

**Why this is a rewrite and not a port.** The obvious ancestor,
`pytilt-diffraction`, sums only over the atoms *listed* in the CIF and never
applies the symmetry operators, so any file giving an asymmetric unit comes out
wrong while a spot check still passes. On rutile written as its 2-site
asymmetric unit with 16 operators, ignoring the expansion puts (200) out by
+353%, (101) by +71% and (211) by −26%. Here the expansion happens in the
reader, and the sum runs over the full P1 cell.

Verified against **pymatgen** — an independent CIF reader, symmetry expansion
and form factor table — to better than 0.2 on a 0-100 intensity scale on every
bundled structure and on rutile (`python tests/test_single_crystal.py`). The
JavaScript is held to the Python by `node web/test/parity_sc.mjs`, which agrees
to ~1e-14 over 7008 section reflections, 124 SAED reflections and 3826 powder
peaks.

## Real experimental frames (`realframe.py`)

`xrays_on_detector.realframe` simulates and indexes real rotation-method frames
on a flat on-axis detector, driven by an external UB (e.g. a CrysAlisPro /
rspace3d UB) and a single oscillation axis. Lab frame: beam +z, detector fast
+x / slow +y at `distance`; crystallographic 1/d units to match `UB/lambda`.

```python
from xrays_on_detector.realframe import FlatDetector, detect_peaks, index_frame, predict_recorded

det, angles, img = FlatDetector.from_eiger_cbf("frame_0001.cbf")   # geometry from the header
UB = ...                                    # 3x3, columns a*,b*,c* in 1/d (CrysAlis UB / lambda)
peaks = detect_peaks(img, det.beam_center)
res   = index_frame(peaks, UB, det)         # -> R, hkl, inliers, rms
hkl_rec, fast, slow, eps = predict_recorded(res.R, UB, det, hkl, osc_axis=(0, 1, 0))
```

Validated on Diamond I19-2 MAPbBr3 Eiger frames (`examples/validate_I19-2_realframe.py`):
- **8/8** observed spots on one frame indexed and predicted to **~1.6 px** rms;
- frame-to-frame orientation matches the recorded phi increment to **<0.11°**
  (Δphi 5–30°), validating the rotation convention;
- observed spot |Q| match the reciprocal lattice to a median **0.18%**.

`examples/I19-2_superlattice.py` adds the I4/mcm octahedral-tilt superlattice
for all three twin domains (|F| from a 2×2×2 CIF via pytilting). On a single
0.2° still the main Bragg peaks are ~80% detectable, and predicted-|F|² vs
measured superlattice intensity correlate at ~+0.6 to +0.7 (faint superlattice
shows up far more clearly in the integrated 3D reconstruction than on one still).

## Current scope and limits

Built: **monochromatic, single frame, You six-circle, Gaussian peaks.**
Not yet (natural extensions):
- rotation series / oscillation movies as circles sweep;
- Lorentz factor for integrated (rotation) intensities; only polarization is
  applied to a still;
- structure factors ignore anomalous dispersion (f′, f″) and use isotropic B;
- mosaic / anisotropic peak shapes, and polychromatic (Laue/pink) beam;
- UB refinement from reference reflections (diffcalc can supply this).
```

## Licence

MIT, see [LICENSE](LICENSE). The bundled Cromer-Mann coefficients under
`web/data/` were exported from `pytilting`'s tables; they are the published
International Tables values, but check that provenance before redistributing
them under a different licence.

The neutron scattering lengths and electron scattering factors
(`single_crystal/data/`, mirrored into `web/data/`) come from **pymatgen**,
which is MIT-licensed like this project. `diffsims` ships a better electron
table — the five-Gaussian Peng fit, which holds above s = 2 Å⁻¹ — but it is
GPLv3, so its numbers are deliberately *not* vendored here. One entry, tin, was
refitted: pymatgen's a₃ = 2.118 falls off the trend set by cadmium, indium and
antimony, and made f_e(Sn) 10-14% low across the whole range while its
neighbours agreed to 0.5%. `tools/export_scattering.py` detects that against
the Mott-Bethe transform of the X-ray table, refits the coefficients, and
refuses to write a table it cannot vouch for.
