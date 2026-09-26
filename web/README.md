# The Game of Diffraction (browser build)

A client-side port of the `xrays_on_detector` simulator: a six-circle
diffractometer and a pitch–phi surface diffractometer you drive in a browser.
Everything runs in the visitor's tab, so the site is static and there is no
backend.

Live page: <https://dubajicmilos.github.io/diffraction/>

## Why a port rather than Pyodide

PyQt6 cannot run in a browser at all, so the UI had to be rewritten whichever
route was taken. Pyodide's advantage, reusing the Python code, therefore mostly
disappears: the UI is the bulk of the code, and the physics is only ~700 lines
of linear algebra. Porting the physics as well gives a ~190 kB page instead of
a 15-30 MB one.

## Layout

| Path | Role |
|---|---|
| `js/physics.js` | the port: circle matrices, B matrix, Cromer-Mann, Ewald, orientation, solvers |
| `js/render.js` | frame rendering (Gaussian spots), colour mapping, ray geometry |
| `js/scene.js` | Three.js scene, own orbit/pinch controls |
| `js/app.js` | state, controls, the simulate loop |
| `js/pitchphi.js` | the pitch–phi surface machine: sample (pitch, φ), detector (2θ, azimuth) and solvers, ported from the standalone calculator |
| `js/pitchphi-deck.js` | its control deck: solve modes, solution list, batch table, dial calibration, mounting |
| `js/pitchphi-scene.js` | its Three.js rig: roll/pitch/φ rings and the detector azimuth arc |
| `js/cif.js` | CIF reader with symmetry expansion to P1, shared with `sc/` |
| `js/scatter.js` | scattering tables for X-rays, neutrons and electrons (used by `sc/`) |
| `js/credit.js` | the authorship notice, on screen and in the console |
| `sc/` | the single-crystal viewer (sections, SAED, powder), a sibling page |
| `img/` | poster images for the two pages |
| `lib/three.module.js` | vendored Three.js r169 (see the naming note below) |
| `data/*.json` | generated: form factors, colour maps, bundled structures, neutron and electron tables |
| `test/parity.mjs`, `test/parity_sc.mjs` | Node harnesses comparing the JS against Python fixtures |
| `test/parity_pp.mjs`, `test/fixtures_pp.json` | the pitch–phi module against the standalone tool's recorded browser outputs |

`../tools/export_web_data.py` generates the X-ray form factors, colour maps,
bundled structures and `test/fixture.json`; `../tools/export_scattering.py`
generates the neutron and electron tables, and `../tools/export_sc_fixture.py`
generates `test/fixture_sc.json`. `../tools/deploy_to_site.py` syncs this folder
into the Jekyll site, and `../tools/devserver.py` serves it locally with a
screenshot sink.

### The folder is `lib/`, not `vendor/`

al-folio's `.gitignore` and the exclude list in its `_config.yml` both carry a
bare `vendor` entry, which matches at any depth. A `vendor/` folder here would be
silently dropped from both the commit and the built site, and the deployed page
would get a 404 error for Three.js. Do not rename it back.

## Running it

```bash
python tools/devserver.py          # serves web/ on http://localhost:8777
```

ES modules and `fetch` do not work from `file://`, so the folder must be served
over http.

The page exposes a scripting handle, `window.diffractionGame`, with
`{state, simulate, scene, physics, setAngles, setInstrument, pitchPhi, render}`.
The handle lets a script drive the app without waiting for animation frames
(useful when a headless or hidden browser never fires `requestAnimationFrame`),
and it lets anyone script the instrument from the console.

## Verifying it

```bash
python tools/export_web_data.py    # regenerate fixtures from the Python
node web/test/parity.mjs           # 30 groups, JS vs Python
node web/test/parity_sc.mjs        # the single-crystal viewer
node web/test/parity_pp.mjs        # pitch–phi vs the standalone tool's recorded run
```

`parity.mjs` compares circle matrices, B matrices, `|F(hkl)|²` (against
`xrays_on_detector.crystal`), the detector frame, projection, reach and
binning, a full Ewald pass
(identical hkl, `khat`, `eps`, excitation, 2θ, polarization),
`rotationBetween`/`eulerMatrix`, the align tools, UB in all three conventions,
and every solver. The largest deviation anywhere is 2e-10, which is the bisection
tolerance; all other quantities agree to machine precision.

`parity_pp.mjs` is a different kind of check. It recomputes the pitch–phi
machine's numbers with `js/pitchphi.js` and compares them, case by case, with
what the standalone calculator actually returned: 199 cases across cells,
orientations, the specular family, near-tangency edges and roll probes, recorded
from a real browser before this port existed (`fixtures_pp.json` is that
recording).

Terser-minified output was checked with the same harness and still passes, so
the site's build step does not change the numbers.

## Performance

Measured in Node, full Ewald pass per frame:

| Case | reflections | ms/frame |
|---|---|---|
| Perovskite, PILATUS 2M @ 200 mm | 924 | 0.06 |
| Perovskite, EIGER 4M @ 85 mm | 4 944 | 0.27 |
| 2×2×2 supercell, EIGER 4M @ 85 mm | 40 098 | 1.5 |

This is faster than the numpy original (5.4 ms for the supercell case) because
the typed-array loop does not allocate intermediate arrays. The frame rate is
limited by rendering, not by the physics.

## Two implementation notes

The detector texture needs mipmaps. With `LinearFilter` and no mipmaps, a
368×419 frame minified onto a ~200 px quad is point-sampled, and single-pixel
Bragg spots vanish. This looks exactly like "the pattern is not updating", even
though the texture is uploaded every frame. The fix is
`LinearMipmapLinearFilter` + `generateMipmaps` + anisotropy, together with a
~1 px floor on the preview spot size.

Structures ship as atom lists, not `|F|²` tables. The Cromer-Mann form factors
need only nine coefficients per element and a simple sum, so the browser
computes the structure factors itself. CsPbBr3 takes 0.4 kB as atoms, whereas a
precomputed table would take hundreds of kB and be fixed at one `Q_max`. The
coefficient table is exported from `single_crystal/data/`, so the JS cannot
drift from the Python.

## Still open

- Pane balance: inside a 1345 px iframe the 3D view only gets ~492 px, because
  the control panel takes 396 and the detector pane 34vw.
- Touch: the panes take single-pointer drags and the 3D view pinches, but the
  detector pane has no pinch zoom.
