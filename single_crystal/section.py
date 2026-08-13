"""Plane sections of the weighted reciprocal lattice.

A section is named by a zone axis [uvw] and a layer index n, and holds every
reflection obeying the zone law

    h u + k v + l w = n

which is the plane of reciprocal space normal to the *direct* lattice vector
u a + v b + w c. That is the plane a precession camera records, and it is the
only reading of "the [uvw] cut" that has lattice rows in it for a general cell:
the plane perpendicular to the *reciprocal* vector (uvw) generally contains no
lattice points at all unless the cell is orthogonal.

    [001] n=0  ->  the hk0 plane
    [100] n=3  ->  the 3kl plane
    [110] n=0  ->  a diagonal cut, h + k = 0
    [111] n=1  ->  h + k + l = 1

Plot coordinates are Cartesian, in 1/Angstrom, obtained through Q = B @ hkl.
Projecting the raw (h, k, l) triple instead -- as the earlier streamlit viewer
did -- is only correct for a cubic cell and shears every other one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TWO_PI = 2 * math.pi


def _gcd3(u: int, v: int, w: int) -> int:
    return math.gcd(math.gcd(abs(u), abs(v)), abs(w))


def _bezout(a: int, b: int) -> tuple[int, int, int]:
    """(g, s, t) with a s + b t = g = gcd(a, b)."""
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return old_r, old_s, old_t


def layer_step(uvw) -> np.ndarray:
    """An integer p with p . [uvw] = 1, so layer n starts at n p.

    Requires a primitive zone axis; reduce_zone() guarantees that.
    """
    u, v, w = (int(round(x)) for x in uvw)
    g, s, t = _bezout(u, v)
    g2, m, n = _bezout(g, w)
    if abs(g2) != 1:
        raise ValueError(f"zone axis {u} {v} {w} is not primitive")
    p = np.array([s * m, t * m, n], dtype=int) * (1 if g2 == 1 else -1)
    if int(p @ np.array([u, v, w])) != 1:
        raise ValueError(f"failed to solve the zone law for {u} {v} {w}")
    return p


def reduce_zone(uvw) -> tuple[np.ndarray, int]:
    """Primitive direction and the common factor divided out.

    [002] is the same direction as [001] but its zone law only ever produces
    even n, so a caller asking for layer 1 of [002] means layer 1 of [001].
    Reducing here makes every integer layer reachable and is reported, not
    silent.
    """
    uvw = np.array([int(round(x)) for x in uvw], dtype=int)
    if not uvw.any():
        raise ValueError("the zone axis cannot be 0 0 0")
    g = _gcd3(*uvw)
    return uvw // g, g


def zone_basis(B: np.ndarray, uvw) -> tuple[np.ndarray, np.ndarray]:
    """Two integer hkl spanning the reflections of the zero layer.

    The pair achieving the two shortest |Q| is returned, which for a rank-2
    lattice is guaranteed to be a basis and gives the axes a crystallographer
    would draw. Search is exhaustive over a box that must contain them: for a
    primitive [uvw] the kernel always has vectors with entries no larger than
    max|u, v, w|.
    """
    uvw = np.asarray(uvw, dtype=int)
    R = int(max(abs(uvw))) + 1
    rng = np.arange(-R, R + 1)
    grid = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), axis=-1).reshape(-1, 3)
    grid = grid[(grid @ uvw) == 0]
    grid = grid[np.any(grid != 0, axis=1)]
    if len(grid) < 2:
        raise ValueError(f"no reflections lie in the zone {uvw}")

    qlen = np.linalg.norm(grid @ B.T, axis=1)
    order = np.argsort(qlen, kind="stable")
    g1 = grid[order[0]]
    g2 = None
    for i in order[1:]:
        # linearly independent, i.e. not a multiple of g1
        if np.any(np.cross(g1, grid[i]) != 0):
            g2 = grid[i]
            break
    if g2 is None:
        raise ValueError(f"the zone {uvw} spans only one direction")

    # A primitive zone axis has a kernel of covolume |uvw| in index space, so
    # this is an exact check that the pair really generates every reflection of
    # the zone rather than a sublattice of them.
    area = np.linalg.norm(np.cross(g1, g2))
    expect = np.linalg.norm(uvw.astype(float))
    if abs(area - expect) > 1e-6 * max(1.0, expect):
        raise ValueError(
            f"zone basis {g1} {g2} spans a sublattice of zone {uvw} "
            f"(covolume {area:.6f}, expected {expect:.6f})"
        )

    # Both signs of a basis vector are equally valid, and the search returns
    # whichever the sort happened to reach first. Fix them so the axis a user
    # reads off the plot is [100] and not [-100], then orient the pair so that
    # g1 x g2 runs along the zone axis rather than against it. Without this the
    # picture mirrors itself as the zone changes.
    def canonical(g):
        for value in g:
            if value:
                return -g if value < 0 else g
        return g

    g1, g2 = canonical(g1), canonical(g2)
    if np.dot(np.cross(g1, g2), uvw) < 0:
        g2 = -g2
    return g1, g2


@dataclass
class Section:
    """Reflections of one zone-axis layer, with plot coordinates.

    hkl        (N, 3) integer indices
    x, y       (N,) plot coordinates in 1/Angstrom
    q          (N,) |Q| in 1/Angstrom
    d          (N,) d-spacing in Angstrom
    intensity  (N,) |F|^2, raw
    """

    hkl: np.ndarray
    x: np.ndarray
    y: np.ndarray
    q: np.ndarray
    d: np.ndarray
    intensity: np.ndarray
    uvw: np.ndarray
    layer: int
    zone_factor: int
    g1: np.ndarray
    g2: np.ndarray
    x_axis: np.ndarray  # Cartesian unit vector of the plot x axis
    y_axis: np.ndarray
    normal: np.ndarray  # plane normal, the direct vector u a + v b + w c
    height: float  # out-of-plane offset of this layer, 1/Angstrom
    radiation: str

    def __len__(self) -> int:
        return len(self.hkl)

    @property
    def zone_law(self) -> str:
        u, v, w = self.uvw
        terms = []
        for coeff, name in ((u, "h"), (v, "k"), (w, "l")):
            if coeff == 0:
                continue
            if coeff == 1:
                terms.append(f"+ {name}")
            elif coeff == -1:
                terms.append(f"- {name}")
            else:
                terms.append(f"{'+' if coeff > 0 else '-'} {abs(coeff)}{name}")
        expr = " ".join(terms).lstrip("+ ").strip()
        return f"{expr} = {self.layer}"

    def brightest(self, k: int = 1):
        order = np.argsort(self.intensity)[::-1]
        return self.hkl[order[:k]], self.intensity[order[:k]]


def compute_section(
    structure,
    uvw=(0, 0, 1),
    layer: int = 0,
    d_min: float = 0.7,
    radiation: str = "xray",
    q_max: float | None = None,
) -> Section:
    """Every reflection of the [uvw] zone at the given layer, out to d_min.

    q_max overrides the d_min limit when given, in 1/Angstrom (2*pi
    convention). The 000 beam is not included; it is not a reflection.
    """
    B = structure.B
    uvw, factor = reduce_zone(uvw)
    layer = int(layer)
    g1, g2 = zone_basis(B, uvw)
    p = layer_step(uvw)

    if q_max is None:
        if d_min <= 0:
            raise ValueError("d_min must be positive")
        q_max = TWO_PI / d_min

    # Plane normal: the direct lattice vector u a + v b + w c. Q . normal is
    # 2 pi n on layer n, which is what fixes the layer spacing.
    normal = structure.A @ uvw.astype(float)
    n_hat = normal / np.linalg.norm(normal)
    height = TWO_PI * layer / np.linalg.norm(normal)
    if abs(height) > q_max:
        # The whole layer lies outside the resolution limit.
        empty = np.zeros(0)
        return Section(
            hkl=np.zeros((0, 3), dtype=int), x=empty, y=empty, q=empty, d=empty,
            intensity=empty, uvw=uvw, layer=layer, zone_factor=factor,
            g1=g1, g2=g2, x_axis=np.zeros(3), y_axis=np.zeros(3),
            normal=n_hat, height=height, radiation=radiation,
        )
    r_max = math.sqrt(max(q_max * q_max - height * height, 0.0))

    e1 = B @ g1.astype(float)
    e2 = B @ g2.astype(float)
    origin = B @ (layer * p).astype(float)
    o_in = origin - np.dot(origin, n_hat) * n_hat

    # How far to run the two in-plane indices. Solving the Gram system bounds
    # them exactly rather than guessing a box and hoping.
    G = np.array([[e1 @ e1, e1 @ e2], [e2 @ e1, e2 @ e2]])
    Ginv = np.linalg.inv(G)
    R = r_max + np.linalg.norm(o_in)
    a_max = int(math.ceil(R * (abs(Ginv[0, 0]) * np.linalg.norm(e1)
                               + abs(Ginv[0, 1]) * np.linalg.norm(e2)))) + 1
    b_max = int(math.ceil(R * (abs(Ginv[1, 0]) * np.linalg.norm(e1)
                               + abs(Ginv[1, 1]) * np.linalg.norm(e2)))) + 1

    A = np.arange(-a_max, a_max + 1)
    Bc = np.arange(-b_max, b_max + 1)
    aa, bb = np.meshgrid(A, Bc, indexing="ij")
    hkl = (
        layer * p[None, :]
        + aa.ravel()[:, None] * g1[None, :]
        + bb.ravel()[:, None] * g2[None, :]
    )
    hkl = hkl[np.any(hkl != 0, axis=1)]  # drop 000, which is the direct beam

    Q = hkl @ B.T
    qn = np.linalg.norm(Q, axis=1)
    keep = qn <= q_max
    hkl, Q, qn = hkl[keep], Q[keep], qn[keep]
    if len(hkl) == 0:
        empty = np.zeros(0)
        return Section(
            hkl=np.zeros((0, 3), dtype=int), x=empty, y=empty, q=empty, d=empty,
            intensity=empty, uvw=uvw, layer=layer, zone_factor=factor,
            g1=g1, g2=g2, x_axis=np.zeros(3), y_axis=np.zeros(3),
            normal=n_hat, height=height, radiation=radiation,
        )

    # Plot axes: x along the shortest reciprocal row of the zone, y completing
    # a right-handed set with the plane normal, so the picture does not flip
    # when the layer or the zone changes sign.
    x_axis = e1 / np.linalg.norm(e1)
    y_axis = np.cross(n_hat, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    intensity = structure.intensity(hkl, radiation)
    return Section(
        hkl=hkl.astype(int),
        x=Q @ x_axis,
        y=Q @ y_axis,
        q=qn,
        d=TWO_PI / qn,
        intensity=intensity,
        uvw=uvw,
        layer=layer,
        zone_factor=factor,
        g1=g1,
        g2=g2,
        x_axis=x_axis,
        y_axis=y_axis,
        normal=n_hat,
        height=height,
        radiation=radiation,
    )
