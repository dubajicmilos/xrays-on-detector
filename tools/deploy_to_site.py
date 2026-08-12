"""Sync the Diffraction Game into an al-folio Jekyll site.

    python tools/deploy_to_site.py <path-to-site-repo> [--nav] [--page-only]

Copies web/ into <site>/assets/diffraction/ and writes the two Markdown pages:
<site>/_pages/diffraction.md (the game) and diffraction-guide.md (how to use).
Run it again after any tweak; it is idempotent.

  --nav        set nav: true so the tab appears in the site menu
               (leave it off to publish the page without advertising it yet)
  --page-only  rewrite only the Markdown page, leave the app files alone

Nothing is committed or pushed; that stays a manual step.
"""
from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
SUBDIR = os.path.join("assets", "diffraction")
COPY = ["index.html", "css", "js", "lib", "data"]        # note: test/ is not shipped
# NB: the folder is "lib", not "vendor". al-folio's .gitignore and its
# _config.yml exclude list both carry a bare "vendor" entry, which matches at
# any depth, so a vendor/ folder here would be neither committed nor built.

BUTTON_CSS = """  .game-launch {
    display: inline-block;
    margin: 0 0.5rem 0 0;
    padding: 0.55rem 1.15rem;
    border: 1px solid var(--global-theme-color);
    border-radius: 6px;
    color: var(--global-theme-color);
    font-weight: 600;
    text-decoration: none;
    cursor: pointer;
  }
  .game-launch:hover {
    background: var(--global-theme-color);
    color: var(--global-bg-color);
    text-decoration: none;
  }
  .game-buttons { margin: 0.1rem 0 1.1rem; }
"""

PAGE = """---
layout: page
permalink: /diffraction/
title: The Game of Diffraction
nav: {nav}
nav_order: 6
description: Ever wondered how single-crystal diffraction works, and how we can predict where the Bragg peaks will emerge on the detector?
_styles: >
  .game-shell {
    width: 100vw;
    margin-left: calc(50% - 50vw);
    height: min(86vh, 940px);
    min-height: 520px;
    border: 0;
  }
  .game-shell iframe { width: 100%; height: 100%; border: 0; display: block; }
  .game-note { font-size: 0.85rem; opacity: 0.75; margin-top: 0.6rem; }
__BUTTON_CSS__---

<div class="game-buttons">
<a class="game-launch" href="{{ '/assets/diffraction/index.html' | relative_url }}"
   target="_blank" rel="noopener">Open full screen &#8599;</a>
<a class="game-launch" href="{{ '/diffraction/guide/' | relative_url }}">How to use</a>
</div>

Try the game, made using
[xrays_on_detector](https://github.com/dubajicmilos/xrays-on-detector), an
open-source package for six-circle diffraction.

You are free to reorient the sample, drive the goniometer circles, change the
detector model and its distance, and tune the X-ray energy, then watch the
diffraction pattern each configuration produces for a range of crystals. You
can also drive the detector arm onto a particular hkl reflection.

<div class="game-shell">
  <iframe src="{{ '/assets/diffraction/index.html' | relative_url }}"
          title="The Game of Diffraction"
          loading="lazy"
          allow="fullscreen"></iframe>
</div>

<p class="game-note">Works best on a desktop browser.</p>
"""

GUIDE = """---
layout: page
permalink: /diffraction/guide/
title: How to use
nav: false
description: A guide to The Game of Diffraction, its panels, circles and controls.
_styles: >
__BUTTON_CSS__---

<div class="game-buttons">
<a class="game-launch" href="{{ '/diffraction/' | relative_url }}">&#8592; Back to the game</a>
<a class="game-launch" href="{{ '/assets/diffraction/index.html' | relative_url }}"
   target="_blank" rel="noopener">Open full screen &#8599;</a>
</div>

## The workspace

- **Control panel (left).** Beamline parameters, sample structure and
  orientation, and the motor positions.
- **Instrument view (centre).** The diffractometer in three dimensions, seen
  from upstream looking downstream along the beam. The cyan rays are the
  diffracted beams that reach the panel, so each one ends on its own Bragg
  spot.
- **Detector view (right).** The same detector face, seen from the sample
  position.

## Seeing a reflection

- **Rock the crystal.** Drag *omega*. Reflections light up and fade as lattice
  planes pass through the Bragg condition.
- **Drive to a named reflection.** Enter Miller indices, say `2 1 1`, under
  *Drive to a reflection* and press **Find omega**. You get every omega that
  satisfies the Bragg condition with the other three circles held, which is
  what rocking one motor does at a beamline. Choose a solution, press **Drive
  there**, then **Aim detector** to swing the arm onto the diffracted beam.
- **Blind cones.** If no omega reaches a reflection, a limitation of rotating
  about a single axis rather than a bug, the page says so and offers a chi that
  brings it within range.
- **Why there is a pattern at the start.** The crystal opens axis-aligned,
  which is a zone axis, and the beam opens at 18.859 keV, where the wavelength
  is exactly a/9 for the default 5.917 Å cell. Twelve reflections then sit
  exactly on the Ewald sphere. Detune the energy and they go out.

## Goniometer and detector circles

The simulator follows the six-circle convention of
[H. You (1999)](https://doi.org/10.1107/S0021889899001223).

| Circle | Moves | Rotation |
|---|---|---|
| mu, omega, chi, phi | Sample | Nested goniometer rotations that orient the lattice in space |
| delta | Detector | Swings the arm through the vertical arc |
| gamma | Detector | Swings the arm through the horizontal arc |

## Beam and detector

- Set the energy or wavelength.
- **Distance and detector model** set how much of reciprocal space you collect.
  A shorter distance or a larger panel buys angular range at the cost of
  spatial resolution.
- **Preview bin** trades detector pixels for frame rate. Raise it while
  dragging a motor; set it to 1 for a full-resolution frame once the setting is
  worth keeping.

## Sample and mosaicity

- **Structure.** Here you can select a crystallographic structure. Structure
  factors are available for the crystal lattices offered.
- **Peak width** is the size of a Bragg peak in reciprocal space. Widening it
  keeps reflections lit further from the exact condition, which is what mosaic
  spread does to a real crystal.

## Orientation and geometry of the crystal

- **Manual alignment.** The rx, ry and rz sliders tilt the crystal on its
  mount.
- **Automated alignment.** *Point a crystal direction* puts a chosen direction
  along a laboratory axis: send (1 1 1) along the beam, then bring a second
  direction toward the vertical to remove the rotation that is still free about
  the first.
- **Transmission and reflection.** In reflection geometry the sample carries a
  surface normal: reflections pointing into the bulk are blocked and the angle
  of incidence is reported. Tick *Rays that miss* to see which reflections
  never reach the panel, and why.
"""


PAGE = PAGE.replace("__BUTTON_CSS__", BUTTON_CSS)
GUIDE = GUIDE.replace("__BUTTON_CSS__", BUTTON_CSS)


def sync_tree(src, dst, report):
    """Copy src -> dst, reporting what actually changed."""
    if os.path.isfile(src):
        changed = not (os.path.exists(dst) and filecmp.cmp(src, dst, shallow=False))
        if changed:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            report.append(os.path.relpath(dst))
        return
    for name in sorted(os.listdir(src)):
        sync_tree(os.path.join(src, name), os.path.join(dst, name), report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site", help="path to the al-folio site repo")
    ap.add_argument("--nav", action="store_true",
                    help="show the tab in the site navigation")
    ap.add_argument("--page-only", action="store_true",
                    help="only rewrite _pages/diffraction.md")
    args = ap.parse_args()

    site = os.path.abspath(args.site)
    if not os.path.isdir(os.path.join(site, "_pages")):
        sys.exit(f"{site} does not look like an al-folio site (no _pages/)")

    changed = []
    if not args.page_only:
        target = os.path.join(site, SUBDIR)
        for item in COPY:
            src = os.path.join(WEB, item)
            if not os.path.exists(src):
                sys.exit(f"missing {src}")
            sync_tree(src, os.path.join(target, item), changed)

    # The site's Prettier workflow runs `npx prettier . --check` over the whole
    # repo and ships no ignore file. Neither the vendored Three.js build nor the
    # generated lookup tables are hand-written source, so exclude both rather
    # than reformat them. (The check is already failing on plenty of the site's
    # own files; this at least keeps the app from adding to it.)
    ignore_path = os.path.join(site, ".prettierignore")
    entries = [
        ("# vendored third-party build, not ours to format",
         "assets/diffraction/lib/"),
        ("# generated lookup tables, see tools/export_web_data.py",
         "assets/diffraction/data/"),
    ]
    lines = []
    if os.path.exists(ignore_path):
        lines = open(ignore_path, encoding="utf-8").read().splitlines()
    have = {ln.strip() for ln in lines}
    missing = [(c, e) for c, e in entries if e not in have]
    if missing:
        with open(ignore_path, "a", encoding="utf-8", newline="\n") as fh:
            if lines and lines[-1].strip():
                fh.write("\n")
            for comment, entry in missing:
                fh.write(comment + "\n")
                fh.write(entry + "\n")
        changed.append(".prettierignore")

    # The guide is its own page rather than a panel on the game page, so
    # opening it does not push the app down the screen.
    pages = [("diffraction.md", PAGE.replace("{nav}", "true" if args.nav else "false")),
             ("diffraction-guide.md", GUIDE)]
    for name, text in pages:
        page_path = os.path.join(site, "_pages", name)
        old = (open(page_path, encoding="utf-8").read()
               if os.path.exists(page_path) else None)
        if old != text:
            with open(page_path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            changed.append(os.path.relpath(page_path))

    total = 0
    for root, _, files in os.walk(os.path.join(site, SUBDIR)):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)

    print(f"site:        {site}")
    print(f"app files:   {os.path.join(SUBDIR)}  ({total / 1024:.0f} kB on disk)")
    print(f"pages:       _pages/diffraction.md  (nav: {'true' if args.nav else 'false'})")
    print(f"             _pages/diffraction-guide.md  -> /diffraction/guide/")
    if changed:
        print(f"changed {len(changed)} file(s):")
        for c in changed[:20]:
            print(f"  {c}")
        if len(changed) > 20:
            print(f"  ... and {len(changed) - 20} more")
    else:
        print("nothing changed; already in sync")
    print("\nNothing was committed. Review, then commit and push yourself.")


if __name__ == "__main__":
    main()
