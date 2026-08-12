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
  .game-launch {
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
  .game-guide { display: inline-block; vertical-align: top; }
  .game-guide[open] { display: block; }
  .game-guide > summary { list-style: none; }
  .game-guide > summary::-webkit-details-marker { display: none; }
  .game-guide > summary::marker { content: ""; }
  .game-guide-body {
    margin: 0 0 1.6rem;
    padding: 0.2rem 0 0.6rem;
    border-top: 1px solid var(--global-divider-color);
  }
  .game-guide-body h3 { margin: 1.5rem 0 0.5rem; font-size: 1.1rem; }
  .game-guide-body li { margin-bottom: 0.35rem; }
  .game-guide-body table { width: 100%; margin: 0.6rem 0 1rem; }
  .game-guide-body td, .game-guide-body th { padding: 0.3rem 0.6rem; }
---

<div class="game-buttons">
<a class="game-launch" href="{{ '/assets/diffraction/index.html' | relative_url }}"
   target="_blank" rel="noopener">Open full screen &#8599;</a>
<details class="game-guide">
  <summary class="game-launch">How to use</summary>
  <div class="game-guide-body">

    <h3>The workspace</h3>
    <ul>
      <li><strong>Control panel (left).</strong> Beamline parameters, sample
        structure and orientation, and the motor positions.</li>
      <li><strong>Instrument view (centre).</strong> The diffractometer in three
        dimensions, seen from upstream looking downstream along the beam. The
        cyan rays are the diffracted beams that reach the panel, so each one
        ends on its own Bragg spot.</li>
      <li><strong>Detector view (right).</strong> The same detector face, seen
        from the sample position.</li>
    </ul>

    <h3>Seeing a reflection</h3>
    <ul>
      <li><strong>Rock the crystal.</strong> Drag <em>omega</em>. Reflections
        light up and fade as lattice planes pass through the Bragg condition,
        and the pattern goes dark between them because Bragg is a condition,
        not a suggestion.</li>
      <li><strong>Drive to a named reflection.</strong> Enter Miller indices,
        say <code>2 1 1</code>, under <em>Drive to a reflection</em> and press
        <strong>Find omega</strong>. You get every omega that satisfies the
        Bragg condition with the other three circles held, which is what
        rocking one motor does at a beamline. Choose a solution, press
        <strong>Drive there</strong>, then <strong>Aim detector</strong> to
        swing the arm onto the diffracted beam.</li>
      <li><strong>Blind cones.</strong> If no omega reaches a reflection, a
        limitation of rotating about a single axis rather than a bug, the page
        says so and offers a chi that brings it within range.</li>
      <li><strong>Why there is a pattern at the start.</strong> The crystal
        opens axis-aligned, which is a zone axis, and the beam opens at
        18.859 keV, where the wavelength is exactly a/9 for the default
        5.917 &#197; cell. Twelve reflections then sit exactly on the Ewald
        sphere. Detune the energy and they go out.</li>
    </ul>

    <h3>Goniometer and detector circles</h3>
    <p>The simulator follows the six-circle convention of You (1999).</p>
    <table>
      <thead><tr><th>Circle</th><th>Moves</th><th>Rotation</th></tr></thead>
      <tbody>
        <tr><td>mu, omega, chi, phi</td><td>Sample</td>
            <td>Nested goniometer rotations that orient the lattice in space</td></tr>
        <tr><td>delta</td><td>Detector</td>
            <td>Swings the arm through the vertical arc</td></tr>
        <tr><td>gamma</td><td>Detector</td>
            <td>Swings the arm through the horizontal arc</td></tr>
      </tbody>
    </table>
    <p>Moving delta or gamma carries the panel with it in the instrument view
      while the diffracted beams stay fixed in the laboratory frame, so the
      spots slide across the detector face rather than travelling with it.</p>

    <h3>Beam and detector</h3>
    <ul>
      <li><strong>Energy and wavelength</strong> are one quantity seen two
        ways, so editing either updates the other.</li>
      <li><strong>Distance and detector model</strong> set how much of
        reciprocal space you collect. A shorter distance or a larger panel buys
        angular range at the cost of spatial resolution.</li>
      <li><strong>Preview bin</strong> trades detector pixels for frame rate.
        Raise it while dragging a motor; set it to 1 for a full-resolution
        frame once the setting is worth keeping.</li>
    </ul>

    <h3>Sample and mosaicity</h3>
    <ul>
      <li><strong>Structure.</strong> A bundled crystal structure gives real
        |F(hkl)| from tabulated atomic form factors. A bare lattice allows
        every reflection and dims the high-angle ones through a Debye-Waller
        falloff alone.</li>
      <li><strong>Peak width</strong> is the size of a Bragg peak in reciprocal
        space. Widening it keeps reflections lit further from the exact
        condition, which is what mosaic spread does to a real crystal.</li>
    </ul>

    <h3>Orientation and geometry</h3>
    <ul>
      <li><strong>Manual alignment.</strong> The rx, ry and rz sliders tilt the
        crystal on its mount.</li>
      <li><strong>Automated alignment.</strong> <em>Point a crystal direction</em>
        puts a chosen direction along a laboratory axis: send (1 1 1) along the
        beam, then bring a second direction toward the vertical to remove the
        rotation that is still free about the first.</li>
      <li><strong>Transmission and reflection.</strong> In reflection geometry
        the sample carries a surface normal: reflections pointing into the bulk
        are blocked and the angle of incidence is reported. Tick <em>Rays that
        miss</em> to see which reflections never reach the panel, and why.</li>
    </ul>

    <h3>Reading the frame</h3>
    <ul>
      <li>Hover anywhere on the detector image for the pixel coordinates, the
        scattering angle 2-theta, the scattering vector |Q| and the lattice
        spacing d.</li>
      <li>Log scale is on by default, since a diffraction pattern spans several
        orders of magnitude in intensity.</li>
    </ul>

    <h3>What is modelled, and what is not</h3>
    <p>Kinematic single scattering: each reflection carries |F(hkl)|&#178;
      times its excitation and a polarization factor, spread as a Gaussian on
      the panel. There is no absorption, extinction, multiple scattering,
      anomalous dispersion, detector point-spread or noise. Peak positions are
      exact geometry; relative intensities are a good guide rather than a
      structure refinement.</p>

  </div>
</details>
</div>

Try the game, made using
[xrays_on_detector](https://github.com/dubajicmilos/xrays-on-detector), an
open-source package for six-circle diffraction. Every reflection, every spot
position and every intensity on this page is computed in your browser as you
drag; nothing is precomputed and there is no server doing the work.

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
    text = PAGE.replace("{nav}", "true" if args.nav else "false")
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
