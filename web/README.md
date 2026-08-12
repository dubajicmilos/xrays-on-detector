# Diffraction Game (browser build)

A client-side port of the `xrays_on_detector` simulator: a six-circle
diffractometer you drive in a browser. Everything runs in the visitor's tab, so
the site is static and there is no backend.

**Live:** <https://dubajicmilos.github.io/diffraction/>

## Why a port rather than Pyodide

PyQt6 cannot run in a browser under any option, so the UI had to be rewritten
whichever route was taken. Once that is true, Pyodide's advantage (reusing the
Python) mostly disappears, because the UI is the bulk of the code and the
physics is only ~700 lines of linear algebra. Porting the physics too buys a
~190 kB page instead of a 15-30 MB one.

## Layout

| Path | Role |
|---|---|
| `js/physics.js` | the port: circle matrices, B matrix, Cromer-Mann, Ewald, orientation, solvers |
| `js/render.js` | frame rendering (Gaussian spots), colour mapping, ray geometry |
| `js/scene.js` | Three.js scene, own orbit/pinch controls |
| `js/app.js` | state, controls, the simulate loop |
| `lib/three.module.js` | vendored Three.js r169 (see the naming note below) |
| `data/*.json` | generated: form factors, colour maps, bundled structures |
| `test/parity.mjs` | Node harness comparing the JS against Python fixtures |

`../tools/export_web_data.py` generates everything under `data/` and
`test/fixture.json`. `../tools/deploy_to_site.py` syncs this folder into the
Jekyll site. `../tools/devserver.py` serves it locally with a screenshot sink.

### The folder is `lib/`, not `vendor/`

al-folio's `.gitignore` **and** its `_config.yml` exclude list both carry a bare
`vendor` entry, which matches at any depth. A `vendor/` folder here would be
silently dropped from both the commit and the built site, and the deployed page
would 404 on Three.js. Do not rename it back.

## Running it

```bash
python tools/devserver.py          # serves web/ on http://localhost:8777
```

ES modules and `fetch` do not work from `file://`, so it must be served over
http.

The page exposes a scripting handle, `window.diffractionGame`, with
`{state, simulate, scene, physics, setAngles, render}`. It exists so the app can
be driven without waiting on animation frames (useful when a headless or hidden
browser never fires `requestAnimationFrame`), and so anyone can script the
instrument from the console.

## Verifying it

```bash
python tools/export_web_data.py    # regenerate fixtures from the Python
node web/test/parity.mjs           # 27 groups, JS vs Python
```

The harness compares circle matrices, B matrices, `|F(hkl)|²` against pytilting,
the detector frame and projection, a full Ewald pass (identical hkl, `khat`,
`eps`, excitation, 2θ), `rotationBetween`/`eulerMatrix`, the align tools, UB in
all three conventions, and every solver. Largest deviation anywhere is 2e-10,
which is the bisection tolerance; the rest sit at machine precision.

Terser-minified output was checked through the same harness and still passes, so
the site's build step does not change the numbers.

## Performance

Measured in Node, full Ewald pass per frame:

| Case | reflections | ms/frame |
|---|---|---|
| Perovskite, PILATUS 2M @ 200 mm | 924 | 0.06 |
| Perovskite, EIGER 4M @ 85 mm | 4 944 | 0.27 |
| 2×2×2 supercell, EIGER 4M @ 85 mm | 40 098 | 1.5 |

Faster than the numpy original (5.4 ms for the supercell case), because the
typed-array loop avoids allocating intermediates. Rendering, not physics, is the
frame-rate limit.

## Two implementation notes worth keeping

**Detector texture needs mipmaps.** With `LinearFilter` and no mipmaps, a
368×419 frame minified onto a ~200 px quad is point-sampled and single-pixel
Bragg spots vanish. It looks exactly like "the pattern is not updating" even
though the texture uploads every frame. Fixed with
`LinearMipmapLinearFilter` + `generateMipmaps` + anisotropy, and a ~1 px floor on
the preview spot size.

**Structures ship as atom lists, not `|F|²` tables.** Cromer-Mann is nine
coefficients per element and a simple sum, so the browser computes structure
factors itself. CsPbBr3 is 0.4 kB as atoms; a precomputed table would be
hundreds of kB and frozen at one `Q_max`. The coefficient table is exported
straight from pytilting so the JS cannot drift from it.

## Still open

- Mobile pass: the responsive CSS is written but untested on a real narrow viewport.
- A short in-page guide for visitors who have never seen a diffractometer.
- Pane balance: inside a 1345 px iframe the 3D view only gets ~492 px, because
  the control panel takes 396 and the detector pane 34vw.
- CIF upload (needs a JS CIF parser plus symmetry expansion).
