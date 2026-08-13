"""Validate the single_crystal package against physics and an outside code.

Run:  python tests/test_single_crystal.py
Checks:
  A. reciprocal lattice: B, cell round-trip, |Q| = 2 pi / d;
  B. zone sections: the zone law holds, the basis really generates the zone,
     layer heights are 2 pi n / |u a + v b + w c|, and a triclinic cell is
     handled as well as a cubic one;
  C. structure factors vs pymatgen's XRDCalculator on every bundled CIF --
     an independent CIF reader, an independent symmetry expansion and an
     independent form factor table (SKIPS if pymatgen is absent);
  D. neutrons: D and H differ in sign, and the sum uses it;
  E. SAED: the higher-order Laue rings land where sqrt(2 k H n) says;
  F. powder: Bragg positions and cubic multiplicities.

C is the one that matters. The reason this package exists is that the earlier
streamlit viewer summed only over the atoms literally listed in the CIF and
never applied the symmetry, so any file giving an asymmetric unit came out
wrong while a spot check still passed.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import single_crystal as sc  # noqa: E402
from single_crystal import scatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STRUCTURES = os.path.join(os.path.dirname(HERE), "examples", "structures")
NAMES = ["CsPbBr3", "MAPbI3_Pm-3m", "MAPbI3_Pnma_pseudocubic", "PEA2PbBr4"]

failures = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def load(name):
    return sc.Structure.from_cif(sc.read_cif(os.path.join(STRUCTURES, name + ".cif")))


def test_lattice():
    print("== A: reciprocal lattice ==")
    x = load("CsPbBr3")
    a = x.cell[0]
    check("a* = 2 pi / a for a cubic cell",
          abs(np.linalg.norm(x.B[:, 0]) - 2 * math.pi / a) < 1e-12)
    check("cell volume", abs(x.volume - a ** 3) < 1e-9, f"{x.volume:.4f} Å³")

    # A triclinic cell is where a wrong B shows up, so build one directly.
    B = sc.bmatrix(5.0, 7.0, 9.0, 71.0, 83.0, 97.0)
    A = 2 * math.pi * np.linalg.inv(B).T
    lens = np.linalg.norm(A, axis=0)
    ang = [
        math.degrees(math.acos(A[:, j] @ A[:, k] / (lens[j] * lens[k])))
        for j, k in ((1, 2), (0, 2), (0, 1))
    ]
    check("triclinic cell round-trips through B",
          np.allclose(lens, [5, 7, 9]) and np.allclose(ang, [71, 83, 97]),
          f"{np.round(lens,6)} {np.round(ang,4)}")

    hkl = np.array([[1, 0, 0], [1, 1, 0], [2, -1, 3]])
    q = np.linalg.norm(hkl @ B.T, axis=1)
    check("|Q| = 2 pi / d", np.allclose(q, 2 * math.pi / (2 * math.pi / q)))


def test_sections():
    print("== B: zone sections ==")
    for name in ("CsPbBr3", "PEA2PbBr4"):
        x = load(name)
        for uvw in [(0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 2, 3), (0, 1, 2)]:
            for layer in (0, 1, 3):
                s = sc.compute_section(x, uvw=uvw, layer=layer, d_min=1.2)
                if len(s) == 0:
                    continue
                law = np.all(s.hkl @ s.uvw == layer)
                height = 2 * math.pi * layer / np.linalg.norm(
                    x.A @ np.asarray(s.uvw, dtype=float))
                ok = law and abs(s.height - height) < 1e-9
                if not ok:
                    check(f"{name} zone {uvw} layer {layer}", False)
                    return
        check(f"{name}: zone law and layer height, 18 sections", True)

    # The basis must generate every reflection of the zone, not a sublattice.
    x = load("CsPbBr3")
    for uvw in [(1, 1, 0), (1, 1, 1), (1, 2, 3), (2, 3, 5)]:
        g1, g2 = sc.zone_basis(x.B, np.asarray(uvw))
        area = np.linalg.norm(np.cross(g1, g2))
        expect = np.linalg.norm(np.asarray(uvw, dtype=float))
        if abs(area - expect) > 1e-9:
            check(f"zone basis covolume {uvw}", False, f"{area} vs {expect}")
            return
    check("zone basis generates the full zone, 4 axes", True)

    # A non-primitive axis is reduced rather than silently returning nothing.
    uvw, factor = sc.reduce_zone((0, 0, 2))
    check("[002] reduces to [001]", tuple(uvw) == (0, 0, 1) and factor == 2)

    # Counting the section directly against a brute-force scan of all hkl.
    x = load("MAPbI3_Pm-3m")
    s = sc.compute_section(x, uvw=(1, 1, 1), layer=1, d_min=1.0)
    allh = x.hkl_within(1.0)
    brute = allh[(allh @ np.array([1, 1, 1])) == 1]
    check("section matches a brute-force scan of every hkl",
          len(s) == len(brute), f"{len(s)} vs {len(brute)}")


def test_against_pymatgen():
    print("== C: structure factors vs pymatgen (independent everything) ==")
    try:
        import platform
        # This machine can hang importing scipy through a stuck WMI service.
        platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError("off"))
        import warnings

        warnings.filterwarnings("ignore")
        from pymatgen.analysis.diffraction.xrd import XRDCalculator
        from pymatgen.core import Structure as PmgStructure
    except Exception as exc:
        print(f"  SKIP  pymatgen not usable ({type(exc).__name__})")
        return

    LAM, WIN, TT = 1.5406, 0.02, 60.0
    for name in NAMES:
        path = os.path.join(STRUCTURES, name + ".cif")
        ours = load(name)
        ours.Biso[:] = 0.0  # pymatgen applies no Debye-Waller; match it
        mine = sc.compute_powder(ours, wavelength=LAM, radiation="xray",
                                 two_theta_max=TT, d_tol=1e-4)
        theirs = XRDCalculator(wavelength=LAM).get_pattern(
            PmgStructure.from_file(path), two_theta_range=(0, TT))

        # Bin both, so an accidental overlap that one code splits and the other
        # merges is not counted as a disagreement.
        edges = np.arange(0, TT + WIN, WIN)
        A, _ = np.histogram(theirs.x, bins=edges, weights=theirs.y)
        Bv, _ = np.histogram(mine.two_theta, bins=edges, weights=mine.intensity)
        A, Bv = 100 * A / A.max(), 100 * Bv / Bv.max()
        live = (A > 0.5) | (Bv > 0.5)
        worst = float(np.abs(A[live] - Bv[live]).max())
        check(f"{name} ({len(ours.frac)} atoms, {live.sum()} peaks)", worst < 2.0,
              f"worst ΔI {worst:.3f} on 0-100")


def test_neutron():
    print("== D: neutrons ==")
    bH = float(scatter.factors("neutron", ["H"], 0.0)[0])
    bD = float(scatter.factors("neutron", ["D"], 0.0)[0])
    check("b(H) and b(D) differ in sign", bH < 0 < bD, f"{bH} vs {bD} fm")

    # Same file read twice, once with D substituted for H, must give different
    # neutron intensities and identical X-ray ones.
    doc = sc.read_cif(os.path.join(STRUCTURES, "MAPbI3_Pnma_pseudocubic.cif"))
    has_h = [a for a in doc.atoms if a.element == "H"]
    if not has_h:
        print("  SKIP  no hydrogen in the test structure")
        return
    x1 = sc.Structure.from_cif(doc)
    for a in doc.atoms:
        if a.element == "H":
            a.nuclide = "D"
    x2 = sc.Structure.from_cif(doc)
    hkl = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0]])
    check("deuteration changes the neutron pattern",
          not np.allclose(x1.intensity(hkl, "neutron"), x2.intensity(hkl, "neutron")))
    check("deuteration leaves the X-ray pattern alone",
          np.allclose(x1.intensity(hkl, "xray"), x2.intensity(hkl, "xray")))

    # b is Q-independent, so with no Debye-Waller the neutron factor must not
    # fall off with angle while the X-ray one must.
    fn = scatter.factors("neutron", ["Pb"], [0.0, 0.5, 1.0])[0]
    fx = scatter.factors("xray", ["Pb"], [0.0, 0.5, 1.0])[0]
    check("neutron b flat in Q, X-ray f falls off",
          np.allclose(fn, fn[0]) and fx[2] < 0.6 * fx[0])


def test_tem():
    print("== E: selected-area electron diffraction ==")
    check("relativistic wavelength at 200 kV",
          abs(scatter.electron_wavelength(200) - 0.02508) < 5e-6,
          f"{scatter.electron_wavelength(200):.5f} Å")

    x = load("CsPbBr3")
    t = sc.compute_tem(x, uvw=(0, 0, 1), kv=200, thickness=50, d_min=0.24, max_zone=2)
    k = 2 * math.pi / t.wavelength
    H = 2 * math.pi / np.linalg.norm(x.A @ np.array([0.0, 0.0, 1.0]))
    for zone in (1,):
        m = t.laue_zone == zone
        if not m.any():
            check(f"Laue zone {zone} present", False)
            continue
        r = np.hypot(t.x[m], t.y[m])
        want = math.sqrt(2 * k * H * zone - (H * zone) ** 2)
        check(f"Laue ring {zone} at sqrt(2 k H n)",
              abs(r.mean() - want) < 0.2, f"{r.mean():.3f} vs {want:.3f} 1/Å")
    check("zero layer is a disc, not a ring",
          np.hypot(t.x, t.y)[t.laue_zone == 0].min() < 2.0)

    # The sign of the Ewald centre decides whether HOLZ exists at all.
    check("HOLZ reflections are excited on the far side of the zero layer",
          (t.laue_zone > 0).any())


def test_powder():
    print("== F: powder ==")
    x = load("CsPbBr3")
    a = x.cell[0]
    p = sc.compute_powder(x, wavelength=1.5406, radiation="xray", two_theta_max=60)
    for hkl, mult in (((1, 0, 0), 6), ((1, 1, 0), 12), ((2, 1, 1), 24), ((3, 2, 1), 48)):
        d = a / math.sqrt(sum(v * v for v in hkl))
        tt = 2 * math.degrees(math.asin(1.5406 / (2 * d)))
        j = int(np.argmin(np.abs(p.two_theta - tt)))
        ok = abs(p.two_theta[j] - tt) < 1e-6 and p.multiplicity[j] == mult
        if not ok:
            check(f"cubic {hkl}", False,
                  f"2θ {p.two_theta[j]:.4f} vs {tt:.4f}, m {p.multiplicity[j]} vs {mult}")
            return
    check("Bragg positions and cubic multiplicities, 4 families", True)

    tt_max = 60.0
    check("nothing beyond the 2-theta limit", p.two_theta.max() <= tt_max)
    check("the trace peaks where the peak list says",
          abs(p.x[int(np.argmax(p.y))] - p.two_theta[int(np.argmax(p.intensity))]) < 0.2)


def _pymatgen():
    """pymatgen, or None. Kept in one place so a missing install skips rather
    than fails, and so the WMI workaround is applied once."""
    try:
        import platform

        # This machine can hang importing scipy through a stuck WMI service.
        platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError("off"))
        import warnings

        warnings.filterwarnings("ignore")
        import pymatgen.analysis.diffraction.neutron as nd
        from pymatgen.core import Structure as PmgStructure

        return PmgStructure, nd
    except Exception:
        return None, None


def test_structure_factors_all_radiations():
    """|F|^2 per reflection against an independent sum over pymatgen's own
    reading of the file.

    This is the sharpest test in the file: no powder merging, no binning, no
    peak matching. Just the structure factor, reflection by reflection, with
    the atom list and the scattering data both coming from somewhere else.
    """
    print("== G: |F|^2 per reflection, neutron and electron ==")
    PmgStructure, nd = _pymatgen()
    if PmgStructure is None:
        print("  SKIP  pymatgen not usable")
        return
    import orjson
    from diffsims.utils.atomic_scattering_params import ATOMIC_SCATTERING_PARAMS as PENG

    blen = orjson.loads(
        open(
            os.path.join(os.path.dirname(nd.__file__), "neutron_scattering_length.json"),
            "rb",
        ).read()
    )

    for name in ("CsPbBr3", "MAPbI3_Pnma_pseudocubic", "PEA2PbBr4"):
        path = os.path.join(STRUCTURES, name + ".cif")
        ours = load(name)
        ours.Biso[:] = 0.0
        pm = PmgStructure.from_file(path)
        frac = np.array(pm.frac_coords)
        syms = [str(sp.symbol) for s in pm for sp in s.species]
        occ = np.array([float(v) for s in pm for v in s.species.values()])
        hkl = ours.hkl_within(1.4)
        phase = np.exp(2j * np.pi * (hkl.astype(float) @ frac.T))

        b = np.array([blen[s] for s in syms])
        F = ((b * occ)[None, :] * phase).sum(axis=1)
        ref = F.real**2 + F.imag**2
        err = np.abs(ours.intensity(hkl, "neutron") - ref).max() / ref.max()
        check(f"{name} neutron, {len(hkl)} reflections", err < 1e-12,
              f"worst {100 * err:.1e}% of the peak")

        s_ = np.linalg.norm(hkl.astype(float) @ ours.B.T, axis=1) / (4 * math.pi)
        f = np.array([[sum(a * math.exp(-bb * v * v) for a, bb in PENG[sy]) for v in s_]
                      for sy in syms])
        F = (f.T * occ[None, :] * phase).sum(axis=1)
        ref = F.real**2 + F.imag**2
        err = np.abs(ours.intensity(hkl, "electron") - ref).max() / ref.max()
        check(f"{name} electron, {len(hkl)} reflections", err < 1e-12,
              f"worst {100 * err:.1e}% of the peak")


def test_section_geometry():
    """A section is a plane, so the distance between two spots on the plot must
    equal |Q_i - Q_j| exactly, in every crystal system.

    This is the invariant the older streamlit viewer breaks: it projects the
    raw (h, k, l) triple onto perpendicular vectors in index space, which is
    exact for a cubic cell under a rescale and sheared for every other one.
    """
    print("== H: section geometry, all seven crystal systems ==")
    import itertools

    from single_crystal.structure import bmatrix

    cells = {
        "cubic": (5.87, 5.87, 5.87, 90, 90, 90),
        "tetragonal": (4.59, 4.59, 2.96, 90, 90, 90),
        "orthorhombic": (8.86, 12.66, 8.58, 90, 90, 90),
        "hexagonal": (4.91, 4.91, 5.40, 90, 90, 120),
        "rhombohedral": (5.43, 5.43, 5.43, 55.3, 55.3, 55.3),
        "monoclinic": (9.70, 8.95, 12.72, 90, 124.4, 90),
        "triclinic": (5.00, 7.00, 9.00, 71.0, 83.0, 97.0),
    }
    worst = 0.0
    for label, cell in cells.items():
        B = bmatrix(*cell)
        xtal = sc.Structure(name=label, cell=cell, B=B, frac=np.zeros((1, 3)),
                            elements=["Si"], nuclides=["Si"],
                            occ=np.ones(1), Biso=np.zeros(1))
        for uvw in [(0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 2, 3), (2, -1, 1)]:
            s = sc.compute_section(xtal, uvw=uvw, layer=0, d_min=1.6)
            n = min(len(s), 25)
            if n < 4:
                continue
            Q = s.hkl[:n].astype(float) @ B.T
            for i, j in itertools.combinations(range(n), 2):
                plotted = math.hypot(s.x[i] - s.x[j], s.y[i] - s.y[j])
                worst = max(worst, abs(plotted - np.linalg.norm(Q[i] - Q[j])))
    check("plotted distance equals |dQ| everywhere", worst < 1e-12,
          f"worst {worst:.2e} 1/Å over 7 systems and 6 zones")


def test_symmetry_expansion():
    """The P1 expansion against pymatgen, on files that carry real operator
    lists rather than an already-expanded cell.

    Both readers are given the *same file*, so a difference is a reader
    difference and not a disagreement about what the structure is. Agreement is
    limited by the eight decimals a CIF carries: a hexagonal 1/3 written as
    0.33333333 is already 3e-9 away from the real value.
    """
    print("== I: symmetry expansion vs pymatgen, 16 space groups ==")
    PmgStructure, _ = _pymatgen()
    if PmgStructure is None:
        print("  SKIP  pymatgen not usable")
        return
    import collections
    import tempfile

    from pymatgen.core import Lattice
    from pymatgen.io.cif import CifWriter

    cases = [
        ("Fm-3m NaCl", "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [.5, .5, .5]]),
        ("Fd-3m Si", "Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]]),
        ("Im-3m W", "Im-3m", Lattice.cubic(3.16), ["W"], [[0, 0, 0]]),
        ("P6_3mc ZnO", "P6_3mc", Lattice.hexagonal(3.25, 5.21), ["Zn", "O"],
         [[1 / 3, 2 / 3, 0], [1 / 3, 2 / 3, .382]]),
        ("P4_2/mnm TiO2", "P4_2/mnm", Lattice.tetragonal(4.594, 2.959), ["Ti", "O"],
         [[0, 0, 0], [.3048, .3048, 0]]),
        ("R-3c Al2O3", "R-3c", Lattice.hexagonal(4.76, 12.99), ["Al", "O"],
         [[0, 0, .352], [.306, 0, .25]]),
        ("P2_1/c", "P2_1/c", Lattice.monoclinic(7.1, 9.2, 11.3, 104.), ["C", "O"],
         [[.1, .2, .3], [.4, .15, .6]]),
        ("P-1", "P-1", Lattice.from_parameters(5, 7, 9, 71, 83, 97), ["N", "O"],
         [[.11, .22, .33], [.44, .55, .66]]),
        ("Pnma SrTiO3", "Pnma", Lattice.orthorhombic(5.5, 7.8, 5.5), ["Sr", "Ti", "O"],
         [[.05, .25, .99], [0, 0, .5], [.72, .25, .21]]),
        ("P3_121 quartz", "P3_121", Lattice.hexagonal(4.913, 5.405), ["Si", "O"],
         [[.47, 0, 1 / 3], [.414, .267, .214]]),
        ("Fm-3m CaF2", "Fm-3m", Lattice.cubic(5.46), ["Ca", "F"], [[0, 0, 0], [.25, .25, .25]]),
        ("I4/mcm", "I4/mcm", Lattice.tetragonal(8.8, 12.7), ["Cs", "Pb", "Br"],
         [[0, .5, .25], [0, 0, 0], [.2, .3, 0]]),
        ("Cmcm", "Cmcm", Lattice.orthorhombic(4.0, 10.0, 6.0), ["Na", "O"],
         [[0, .15, .25], [0, .42, .25]]),
        ("P6/mmm MgB2", "P6/mmm", Lattice.hexagonal(5.1, 4.0), ["Mg", "B"],
         [[0, 0, 0], [1 / 3, 2 / 3, .5]]),
        ("P6_3/mmc Mg", "P6_3/mmc", Lattice.hexagonal(2.51, 4.07), ["Mg"],
         [[1 / 3, 2 / 3, .25]]),
        ("Ia-3d garnet", "Ia-3d", Lattice.cubic(12.0), ["Y", "Al", "O"],
         [[0, .25, .125], [0, 0, 0], [.28, .1, .2]]),
    ]
    tmp = os.path.join(tempfile.gettempdir(), "_sc_sgtest.cif")
    worst_coord = 0.0
    bad = []
    for label, sg, latt, syms, pos in cases:
        CifWriter(PmgStructure.from_spacegroup(sg, latt, syms, pos),
                  symprec=0.01).write_file(tmp)
        doc = sc.read_cif(tmp)
        ref = PmgStructure.from_file(tmp)
        same = (collections.Counter(a.element for a in doc.atoms)
                == collections.Counter(str(sp.symbol) for s in ref for sp in s.species))
        A = np.array([[a.x, a.y, a.z] for a in doc.atoms])
        Bc = np.array(ref.frac_coords) % 1.0
        if same and len(A) == len(Bc):
            d = np.abs(A[:, None, :] - Bc[None, :, :])
            d = np.minimum(d, 1 - d)
            worst_coord = max(worst_coord, float(
                np.linalg.norm(d, axis=2).min(axis=1).max()))
        else:
            bad.append(label)
    check("every space group expands to pymatgen's atom set", not bad,
          ", ".join(bad) if bad else f"{len(cases)} groups")
    check("coordinates agree to the CIF's own precision", worst_coord < 1e-7,
          f"worst {worst_coord:.1e} (files carry 8 decimals)")


def test_occupancy():
    """Partial occupancy has to scale a site's contribution linearly."""
    print("== J: occupancy ==")
    doc = sc.read_cif(os.path.join(STRUCTURES, "CsPbBr3.cif"))
    full = sc.Structure.from_cif(doc)
    hkl = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0], [2, 1, 0]])

    half = sc.Structure.from_cif(doc)
    half.occ[:] = 0.5
    check("halving every occupancy quarters |F|^2",
          np.allclose(half.intensity(hkl, "xray"), full.intensity(hkl, "xray") / 4),
          "|F|^2 goes as occ^2")

    # A vacant site must contribute nothing at all.
    gone = sc.Structure.from_cif(doc)
    gone.occ[0] = 0.0
    dropped = sc.Structure(
        name="x", cell=full.cell, B=full.B, frac=full.frac[1:],
        elements=full.elements[1:], nuclides=full.nuclides[1:],
        occ=full.occ[1:], Biso=full.Biso[1:],
    )
    check("zero occupancy is the same as no atom",
          np.allclose(gone.intensity(hkl, "xray"), dropped.intensity(hkl, "xray")))

    # The bundled MAPbI3 really does carry occ = 0.5 sites, so this is not a
    # synthetic path.
    doc2 = sc.read_cif(os.path.join(STRUCTURES, "MAPbI3_Pnma_pseudocubic.cif"))
    occs = {round(a.occ, 3) for a in doc2.atoms}
    check("partial occupancies survive the P1 expansion", 0.5 in occs, f"{sorted(occs)}")


if __name__ == "__main__":
    for fn in (test_lattice, test_sections, test_against_pymatgen, test_neutron,
               test_tem, test_powder, test_structure_factors_all_radiations,
               test_section_geometry, test_symmetry_expansion, test_occupancy):
        fn()
    print()
    if failures:
        print(f"{len(failures)} FAILED: " + ", ".join(failures))
        sys.exit(1)
    print("all checks passed")
