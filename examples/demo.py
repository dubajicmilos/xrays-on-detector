"""Demo: single-crystal diffraction image of cubic CsPbBr3 on a six-circle setup.

Shows two things the package is for:
  1. "given a CIF, diffractometer angles and a detector, what image do we see?"
  2. how the image changes when the detector arm (nu, delta) is moved.

Run:  python demo.py
"""
import csv
import os

import numpy as np

# make the package importable when run from examples/
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xrays_on_detector import Crystal, Detector, simulate_frame

HERE = os.path.dirname(os.path.abspath(__file__))


def build_cspbbr3_cif(path, a=5.87):
    """Cubic perovskite CsPbBr3 (Pm-3m) written as an explicit-atom CIF."""
    from ase import Atoms
    atoms = Atoms(
        "CsPbBr3",
        scaled_positions=[
            (0.0, 0.0, 0.0),      # Cs
            (0.5, 0.5, 0.5),      # Pb
            (0.5, 0.5, 0.0),      # Br
            (0.5, 0.0, 0.5),      # Br
            (0.0, 0.5, 0.5),      # Br
        ],
        cell=[a, a, a, 90, 90, 90],
        pbc=True,
    )
    atoms.write(path)
    return path


def save_table(table, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "h", "k", "l", "fast_px", "slow_px", "eps", "two_theta_deg", "intensity"])
        w.writeheader()
        for row in sorted(table, key=lambda r: -r["intensity"]):
            w.writerow(row)


def main():
    cif = build_cspbbr3_cif(os.path.join(HERE, "cspbbr3.cif"))
    crystal = Crystal.from_cif(cif, expand_symmetry=True)
    print(f"CsPbBr3: {crystal.n_atoms} atoms, a={crystal.cell['a']} A")

    wavelength = 0.7                       # Angstrom (monochromatic)
    sigma = 0.04                           # reciprocal-space Gaussian width, 1/A

    # Precompute the reflection list and |F|^2 once, reuse for every frame.
    det0 = Detector(distance=120.0, n_fast=1024, n_slow=1024, pixel_size=0.2)
    hkl = crystal.hkl_within_Qmax(det0.max_Qmax(wavelength))
    Fmag2 = crystal.structure_factor_mag2(hkl)
    print(f"{len(hkl)} reflections within detector Q-range")

    # Pick a crystal orientation that lights up several reflections.
    best = max(
        (simulate_frame(crystal, det0, wavelength, sigma, eta=e,
                        hkl=hkl, Fmag2=Fmag2) for e in np.arange(0, 90, 0.5)),
        key=lambda fr: len(fr.table),
    )
    eta = best.sample_angles["eta"]
    print(f"chosen eta = {eta} deg, {len(best.table)} spots on the panel")
    save_table(best.table, os.path.join(HERE, "reflections.csv"))

    # Same crystal orientation, detector arm swung out in the horizontal plane.
    det1 = Detector(distance=120.0, n_fast=1024, n_slow=1024, pixel_size=0.2,
                    delta=25.0)
    moved = simulate_frame(crystal, det1, wavelength, sigma, eta=eta,
                           hkl=hkl, Fmag2=Fmag2)
    print(f"detector at delta=25 deg: {len(moved.table)} spots on the panel")

    _plot([(best, f"detector on-axis\n(eta={eta} deg)"),
           (moved, "detector arm delta=25 deg\n(same crystal)")],
          os.path.join(HERE, "cspbbr3_frames.png"))


def _plot(frames, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    fig, axes = plt.subplots(1, len(frames), figsize=(12, 6), dpi=110)
    for ax, (fr, title) in zip(np.atleast_1d(axes), frames):
        img = fr.image
        vmax = img.max() if img.max() > 0 else 1.0
        ax.imshow(img, cmap="inferno", origin="upper",
                  norm=PowerNorm(gamma=0.4, vmin=0.0, vmax=vmax))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("fast (px)")
        ax.set_ylabel("slow (px)")
    fig.suptitle("CsPbBr3, lambda=0.7 A, monochromatic, Gaussian peaks")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
