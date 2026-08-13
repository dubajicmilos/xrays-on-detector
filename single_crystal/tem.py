"""Selected-area electron diffraction down a zone axis.

A flat section shows one layer of the reciprocal lattice exactly. A real
electron pattern does not: the Ewald sphere is very large but not flat, so it
cuts the zero layer over a disc and then clips the layers above it into rings.
Those rings are the higher-order Laue zones, and they are the reason a
selected-area pattern is worth simulating separately from a precession section.

At 200 kV the electron wavelength is 0.0251 Angstrom, so k = 2 pi / lambda is
250 1/Angstrom against reciprocal lattice spacings of order 1. The sphere is
nearly flat over the zero layer, which is why a section is a good picture of it
near the middle and a poor one at the edge.

Thickness enters through the relrod: a slab of thickness t turns each
reciprocal lattice point into a rod of length ~2/t along the beam, and a
reflection with excitation error s_g contributes

    I = |F|^2 sinc^2(pi t s_g)

which is the kinematic result. Dynamical scattering, which really governs
electron intensities in a thick crystal, is out of scope here and is stated as
such in the app rather than being quietly implied.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .scatter import electron_wavelength
from .section import TWO_PI, layer_step, reduce_zone, zone_basis


def d_min_for_zone(structure, uvw, zone: int, kv: float = 200.0) -> float:
    """Smallest d that still contains Laue ring `zone`, in Angstrom.

    The ring of zone n lies at sqrt(2 k H n - (H n)^2), where H is the layer
    spacing along the beam. A little headroom is added so the ring is inside
    the limit rather than exactly on it.
    """
    uvw, _ = reduce_zone(uvw)
    if zone <= 0:
        return float("inf")
    k = TWO_PI / electron_wavelength(kv)
    H = TWO_PI / np.linalg.norm(structure.A @ uvw.astype(float))
    r2 = 2 * k * H * zone - (H * zone) ** 2
    if r2 <= 0:
        return float("inf")
    return float(TWO_PI / (math.sqrt(r2) * 1.06))


@dataclass
class TEMPattern:
    hkl: np.ndarray
    x: np.ndarray  # 1/Angstrom, perpendicular to the beam
    y: np.ndarray
    q: np.ndarray
    d: np.ndarray
    intensity: np.ndarray  # |F|^2 times the relrod shape factor
    s_g: np.ndarray  # excitation error, 1/Angstrom
    laue_zone: np.ndarray  # 0 for ZOLZ, 1 for FOLZ, ...
    uvw: np.ndarray
    g1: np.ndarray
    g2: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    kv: float
    wavelength: float
    thickness: float
    q_max: float  # resolution limit actually used, 1/Angstrom
    zone_radii: list  # where each Laue ring falls, 1/Angstrom

    def __len__(self) -> int:
        return len(self.hkl)

    def zones_visible(self) -> int:
        """How many Laue zones fall inside the resolution limit.

        The rings sit at sqrt(2 k H n) and that is far out: for a 6 Angstrom
        cell at 200 kV the first one is near 23 1/Angstrom, which is d = 0.27
        Angstrom. Asking for HOLZ at a comfortable d_min quietly returns none,
        so the app reads this and says where the ring actually is.
        """
        return sum(1 for r in self.zone_radii if r is not None and r <= self.q_max)


def compute_tem(
    structure,
    uvw=(0, 0, 1),
    kv: float = 200.0,
    thickness: float = 50.0,
    d_min: float = 0.5,
    max_zone: int = 0,
    s_max: float | None = None,
) -> TEMPattern:
    """Kinematic SAED pattern with the beam along [uvw].

    thickness : specimen thickness in Angstrom, setting the relrod length.
    max_zone  : how many Laue zones above the zero layer to include. Zero by
                default, which gives the zero-layer pattern an experiment
                normally shows.
    s_max     : excitation errors beyond this are dropped; defaults to the
                first zero of the relrod, 1/thickness, which is where a
                reflection stops contributing.

    Asking for Laue zones needs a much smaller d_min than the zero layer does.
    The first ring sits near sqrt(2 k H), which for a 6 Angstrom cell at 200 kV
    is 23 1/Angstrom, or d = 0.27 Angstrom. d_min_for_zone() works that out, so
    a caller can widen the limit instead of quietly getting no rings.
    """
    uvw, _factor = reduce_zone(uvw)
    lam = electron_wavelength(kv)
    k = TWO_PI / lam
    if s_max is None:
        s_max = 1.0 / max(thickness, 1e-6)

    B = structure.B
    g1, g2 = zone_basis(B, uvw)
    p = layer_step(uvw)

    normal = structure.A @ uvw.astype(float)
    n_hat = normal / np.linalg.norm(normal)
    q_max = TWO_PI / d_min

    # Where each Laue ring falls: the layer at height H n meets the sphere on a
    # circle of radius sqrt(2 k H n - (H n)^2). Computed up front so the app can
    # say "the first ring is at 23 1/Angstrom" instead of just showing nothing.
    H = TWO_PI / np.linalg.norm(normal)
    zone_radii = []
    for n in range(0, max_zone + 1):
        r2 = 2 * k * H * n - (H * n) ** 2
        zone_radii.append(math.sqrt(r2) if r2 >= 0 else None)

    e1 = B @ g1.astype(float)
    e2 = B @ g2.astype(float)
    G = np.array([[e1 @ e1, e1 @ e2], [e2 @ e1, e2 @ e2]])
    Ginv = np.linalg.inv(G)

    hkl_all = []
    for layer in range(0, max_zone + 1):
        origin = B @ (layer * p).astype(float)
        o_in = origin - np.dot(origin, n_hat) * n_hat
        R = q_max + np.linalg.norm(o_in)
        a_max = int(math.ceil(R * (abs(Ginv[0, 0]) * np.linalg.norm(e1)
                                   + abs(Ginv[0, 1]) * np.linalg.norm(e2)))) + 1
        b_max = int(math.ceil(R * (abs(Ginv[1, 0]) * np.linalg.norm(e1)
                                   + abs(Ginv[1, 1]) * np.linalg.norm(e2)))) + 1
        aa, bb = np.meshgrid(
            np.arange(-a_max, a_max + 1), np.arange(-b_max, b_max + 1), indexing="ij"
        )
        hkl_all.append(
            layer * p[None, :]
            + aa.ravel()[:, None] * g1[None, :]
            + bb.ravel()[:, None] * g2[None, :]
        )
    hkl = np.unique(np.concatenate(hkl_all), axis=0)
    hkl = hkl[np.any(hkl != 0, axis=1)]

    Q = hkl @ B.T
    qn = np.linalg.norm(Q, axis=1)
    keep = (qn <= q_max) & (qn > 0)
    hkl, Q, qn = hkl[keep], Q[keep], qn[keep]

    # Ewald sphere of radius k, passing through the origin, centred at -k_i.
    # The beam runs antiparallel to the zone axis -- [uvw] points from the
    # specimen towards the gun, the usual electron-microscopy convention -- so
    # the centre is at +k n_hat and a point is on the sphere when
    # |Q - k n_hat| = k. The signed shortfall is the excitation error.
    #
    # The sign matters and is not cosmetic. The sphere curves away from the
    # beam, so the layers it clips are the ones on the far side: solving
    # |Q - k n_hat| = k for Q = (r, 0, z) gives z = (r^2 + z^2) / 2k > 0. Get
    # this backwards and every higher-order Laue zone silently disappears,
    # leaving a pattern that looks like a plain section.
    centre_to_q = Q - k * n_hat[None, :]
    s_g = k - np.linalg.norm(centre_to_q, axis=1)

    keep = np.abs(s_g) <= s_max
    hkl, Q, qn, s_g = hkl[keep], Q[keep], qn[keep], s_g[keep]
    if len(hkl) == 0:
        empty = np.zeros(0)
        return TEMPattern(
            hkl=np.zeros((0, 3), dtype=int), x=empty, y=empty, q=empty, d=empty,
            intensity=empty, s_g=empty, laue_zone=np.zeros(0, dtype=int),
            uvw=uvw, g1=g1, g2=g2, x_axis=np.zeros(3), y_axis=np.zeros(3),
            kv=kv, wavelength=lam, thickness=thickness, q_max=q_max,
            zone_radii=zone_radii,
        )

    x_axis = e1 / np.linalg.norm(e1)
    y_axis = np.cross(n_hat, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # sinc^2 relrod. np.sinc is sin(pi x)/(pi x), so the argument is t s_g.
    shape = np.sinc(thickness * s_g) ** 2
    intensity = structure.intensity(hkl, "electron") * shape

    return TEMPattern(
        hkl=hkl.astype(int),
        x=Q @ x_axis,
        y=Q @ y_axis,
        q=qn,
        d=TWO_PI / qn,
        intensity=intensity,
        s_g=s_g,
        laue_zone=(hkl @ uvw).astype(int),
        uvw=uvw,
        g1=g1,
        g2=g2,
        x_axis=x_axis,
        y_axis=y_axis,
        kv=kv,
        wavelength=lam,
        thickness=thickness,
        q_max=q_max,
        zone_radii=zone_radii,
    )
