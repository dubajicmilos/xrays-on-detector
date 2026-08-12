"""Sync the Diffraction Game into an al-folio Jekyll site.

    python tools/deploy_to_site.py <path-to-site-repo> [--nav] [--page-only]

Copies web/ into <site>/assets/diffraction/ and writes <site>/_pages/diffraction.md.
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

PAGE = """---
layout: page
permalink: /diffraction/
title: game of diffraction
nav: {nav}
nav_order: 6
description: An interactive six-circle diffractometer. Drive the motors and watch the Bragg spots land on the detector.
_styles: >
  .game-shell {{
    width: 100vw;
    margin-left: calc(50% - 50vw);
    height: min(86vh, 940px);
    min-height: 520px;
    border: 0;
  }}
  .game-shell iframe {{ width: 100%; height: 100%; border: 0; display: block; }}
  .game-note {{ font-size: 0.85rem; opacity: 0.75; margin-top: 0.6rem; }}
  .game-launch {{
    display: inline-block;
    margin: 0.1rem 0 1.1rem;
    padding: 0.55rem 1.15rem;
    border: 1px solid var(--global-theme-color);
    border-radius: 6px;
    color: var(--global-theme-color);
    font-weight: 600;
    text-decoration: none;
  }}
  .game-launch:hover {{
    background: var(--global-theme-color);
    color: var(--global-bg-color);
    text-decoration: none;
  }}
---

<a class="game-launch" href="{{{{ '/assets/diffraction/index.html' | relative_url }}}}"
   target="_blank" rel="noopener">Open full screen &#8599;</a>

Mount a crystal on a six-circle diffractometer, drive the motors, and watch which
reflections light up and where they land on the detector. Everything is computed
in your browser as you drag.

It opens on a pattern: the crystal starts axis-aligned, which is a zone axis, and
the beam starts at 18.859 keV, where the wavelength is exactly a/9 for the default
5.917 Å cell. Twelve reflections then sit exactly on the Ewald sphere. Nudge the
energy and they go out, which is the Bragg condition made visible.

<div class="game-shell">
  <iframe src="{{{{ '/assets/diffraction/index.html' | relative_url }}}}"
          title="Game of Diffraction"
          loading="lazy"
          allow="fullscreen"></iframe>
</div>

<p class="game-note">Works best on a desktop browser.</p>

## How to use it

**The three panels.** Controls on the left, the instrument in the middle, the
detector image on the right. The middle panel is the machine seen from upstream,
looking downstream along the beam; the right panel is that same detector face,
seen from the sample. The cyan rays are the diffracted beams that reach the
panel, so each one ends on its own spot.

**Try this first.** Drag **omega** and watch reflections sweep on and off the
sphere. Every spot that lights up is a lattice plane that has just come into the
Bragg condition, and the pattern goes dark between them because Bragg is a
condition, not a suggestion.

**Drive to a named reflection.** Type `2 1 1` under _Drive to a reflection_ and
press **Find omega**. You get every omega that puts that reflection on the sphere
with the other three circles held, which is what rocking one motor does at a
beamline. Pick a solution, press **Drive there**, then **Aim detector** to swing
the arm onto it. If a reflection cannot be reached by omega alone it says so, and
offers a chi that brings it into reach: that is the blind cone of a single-axis
rotation, and moving an outer circle is the real fix.

**The circles.** `mu`, `omega`, `chi`, `phi` move the crystal; `delta` and
`gamma` move the detector arm. In the You (1999) convention used here `delta`
swings the arm vertically and `gamma` horizontally, so watch the panel move in
the 3D view while the spots stay put in space and slide across the panel.

**Beam and detector.** Energy and wavelength are two views of one number, so
either box drives the other. Distance and detector model change how much of
reciprocal space you catch: a shorter distance or a larger panel buys angular
range at the cost of resolution. _Preview bin_ trades detector pixels for frame
rate; set it to 1 for a full-resolution frame once you like a setting.

**Sample.** Pick a bundled structure for real |F(hkl)| from tabulated form
factors, or a bare lattice, where every reflection is allowed and only a
Debye-Waller falloff dims the high-angle ones. _Peak width_ is the reciprocal
space size of a Bragg peak: widen it and reflections stay lit further from the
exact condition, which is what mosaic spread does to a real crystal.

**Orientation.** The rx/ry/rz sliders tilt the crystal on its mount. _Point a
crystal direction somewhere_ does the alignment for you: put `1 1 1` along the
beam, then a second direction toward the vertical, and you have set the crystal
the way you would on the floor.

**Reflection geometry.** Switch from _Transmission_ to _Reflection_ and the
sample becomes a surface with a normal. Reflections pointing into the sample are
blocked, and the incidence angle is reported; tick _show blocked/missed rays_ to
see which reflections were lost and why.

**Reading the frame.** Hover the detector image for pixel, 2θ, |Q| and d. Log
scale is on by default, since a diffraction pattern spans orders of magnitude.

### What is modelled, and what is not

Kinematic single scattering: each reflection carries |F(hkl)|² times its
excitation and a polarization factor, spread as a Gaussian on the panel. No
absorption, no extinction, no multiple scattering, no anomalous dispersion, no
detector point-spread or noise. Peak positions are exact geometry; relative
intensities are a good guide rather than a structure refinement.
"""


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

    page_path = os.path.join(site, "_pages", "diffraction.md")
    text = PAGE.format(nav="true" if args.nav else "false")
    old = open(page_path, encoding="utf-8").read() if os.path.exists(page_path) else None
    if old != text:
        with open(page_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        changed.append(os.path.relpath(page_path))

    total = 0
    for root, _, files in os.walk(os.path.join(site, SUBDIR)):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)

    print(f"site:        {site}")
    print(f"app files:   {os.path.join(SUBDIR)}  ({total / 1024:.0f} kB on disk)")
    print(f"page:        _pages/diffraction.md  (nav: {'true' if args.nav else 'false'})")
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
