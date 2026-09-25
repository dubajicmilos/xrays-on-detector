"""Project a measured reciprocal-space volume S(q) back onto an area detector.

This is the inverse of :mod:`xrays_on_detector.reconstruct`. That module maps
every detector pixel of a rotation series into a voxel grid over (h, k, l); this
one takes such a grid - an rspace3d / CrysAlisPro HDF5 volume - and asks the
opposite question: with the crystal at these angles and the detector there,
which part of the measured S(q) does the panel cut, and what does it record?

The map is the same one, read backwards. For a pixel whose outgoing unit
direction is khat,

    Q_lab = k (khat - beam) ,     k = 2*pi / lambda        (2*pi convention)
    hkl   = (Z U B)^-1 Q_lab

where ``Z U B`` is the product the forward model uses to place a reflection
(sample circles, crystal orientation, reciprocal matrix). The pixel value is the
volume interpolated at that (h, k, l): the detector cuts the Ewald sphere
through the measured data, so Bragg peaks, superlattice peaks and diffuse
scattering all appear exactly where the sphere passes through them.

No polarisation or obliquity factor is applied. The volume already holds
measured intensity; multiplying it by a Thomson factor would add a distortion,
not remove one, and whether the reconstruction that produced the file already
corrected for those is a property of that file, not of this code.

Orientation
-----------
The file's ``UB`` is used only for the **cell metric**, never as an orientation.
A CrysAlisPro UB is expressed in CrysAlisPro's own frame, which differs from the
lab frame here by a fixed rotation that the file does not record (see
RECONSTRUCTION_UNIVERSAL.md). So the volume arrives as a lattice with U = I and
the crystal is oriented with the usual controls.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import BEAM
from .reconstruct import _reciprocal_cell

# Roughly 200 M voxels, i.e. 800 MB as float32, is the default budget; anything
# larger is block-averaged on the way in rather than silently swallowing RAM.
DEFAULT_MAX_VOXELS = 200_000_000


@dataclass
class SqVolume:
    """A measured S(q) on a regular (H, K, L) grid, ready to be sampled.

    Attributes
    ----------
    data : (nH, nK, nL) ndarray
        Intensity, indexed ``data[iH, iK, iL]``. Unmeasured voxels (NaN in the
        file) are replaced by zero on load; the fraction is kept in `nan_frac`.
    H, K, L : 1D ndarrays
        Voxel-centre coordinates in r.l.u. Each axis is uniform.
    recip : (3, 3) ndarray
        Columns a*, b*, c* in 1/d (the file's UB divided by its wavelength).
    cell : dict
        Direct cell a, b, c (Angstrom), alpha, beta, gamma (degrees).
    wavelength : float or None
        The wavelength the data were measured at (Angstrom).
    bin_factor : int
        Block-averaging applied on load; 1 means the file's own sampling.
    """

    data: np.ndarray
    H: np.ndarray
    K: np.ndarray
    L: np.ndarray
    recip: np.ndarray
    cell: dict
    wavelength: float | None = None
    name: str = "S(q) volume"
    bin_factor: int = 1
    nan_frac: float = 0.0

    # -- loading ----------------------------------------------------------

    @classmethod
    def from_h5(cls, path: str, *, max_voxels: int = DEFAULT_MAX_VOXELS,
                progress=None) -> "SqVolume":
        """Read an rspace3d / CrysAlisPro reconstruction.

        Expects ``/data``, ``/H``, ``/K``, ``/L``, ``/UB`` and a ``wavelength``
        attribute, which is the layout :func:`reconstruct.save_rspace3d_h5`
        writes and the rsp_viewer reads.

        A volume larger than `max_voxels` is block-averaged by an integer factor
        while it is read, so a multi-gigabyte raw reconstruction still opens.
        `progress` is an optional callable taking (slabs_done, slabs_total).
        """
        import os

        import h5py

        with h5py.File(path, "r") as f:
            for key in ("data", "H", "K", "L", "UB"):
                if key not in f:
                    raise KeyError(
                        f"{path}: no /{key}. This does not look like an rspace3d "
                        "volume (expected /data, /H, /K, /L and /UB)."
                    )
            dset = f["data"]
            H, K, L = (np.asarray(f[a][...], float).ravel() for a in ("H", "K", "L"))
            if dset.shape != (H.size, K.size, L.size):
                raise ValueError(
                    f"{path}: /data has shape {dset.shape} but the axes are "
                    f"({H.size}, {K.size}, {L.size}); the index order of this "
                    "file is not the (H, K, L) one this reader assumes."
                )
            wl = f.attrs.get("wavelength", None)
            if not wl:
                raise ValueError(
                    f"{path}: the 'wavelength' attribute is missing or zero, so "
                    "the UB cannot be put into 1/d units. rspace3d writes it; a "
                    "file without it is incomplete."
                )
            wl = float(wl)
            ub = np.asarray(f["UB"][...], float)      # CrysAlis, lambda-scaled
            for axis, name in ((H, "H"), (K, "K"), (L, "L")):
                _check_uniform(axis, name, path)
            factor = _bin_factor(dset.shape, max_voxels)
            data = _read_binned(dset, factor, progress)

        if factor > 1:
            H, K, L = (_bin_axis(a, factor) for a in (H, K, L))

        nan = ~np.isfinite(data)
        nan_frac = float(nan.mean())
        if nan_frac:
            data = np.where(nan, 0.0, data)

        recip = ub / wl                               # columns a*, b*, c* in 1/d
        return cls(
            data=np.ascontiguousarray(data, np.float32),
            H=H, K=K, L=L,
            recip=recip,
            cell=_reciprocal_cell(recip),
            wavelength=wl,
            name=os.path.basename(path),
            bin_factor=factor,
            nan_frac=nan_frac,
        )

    # -- sampling ---------------------------------------------------------

    def sample(self, hkl: np.ndarray) -> tuple:
        """Trilinear interpolation at fractional (h, k, l).

        Parameters
        ----------
        hkl : (..., 3) ndarray

        Returns
        -------
        values : (...) ndarray
            Interpolated intensity; zero wherever the point is outside the grid.
        inside : (...) bool ndarray
            Where the point actually fell inside the measured volume, so a
            caller can tell "no data here" from "measured zero here".
        """
        hkl = np.asarray(hkl, float)
        shape = hkl.shape[:-1]
        pts = hkl.reshape(-1, 3)

        idx = np.empty_like(pts)
        inside = np.ones(len(pts), bool)
        for d, axis in enumerate((self.H, self.K, self.L)):
            step = axis[1] - axis[0]
            u = (pts[:, d] - axis[0]) / step
            inside &= (u >= 0.0) & (u <= axis.size - 1)
            idx[:, d] = u

        idx = idx[inside]
        i0 = np.floor(idx).astype(np.intp)
        # Clamp so a point sitting exactly on the last voxel centre keeps its
        # upper corner in range; that corner carries zero weight anyway.
        top = np.array(self.data.shape, np.intp) - 2
        i0 = np.clip(i0, 0, np.maximum(top, 0))
        frac = idx - i0

        acc = np.zeros(len(i0), np.float64)
        for corner in range(8):
            cx, cy, cz = (corner >> 2) & 1, (corner >> 1) & 1, corner & 1
            w = (np.where(cx, frac[:, 0], 1.0 - frac[:, 0])
                 * np.where(cy, frac[:, 1], 1.0 - frac[:, 1])
                 * np.where(cz, frac[:, 2], 1.0 - frac[:, 2]))
            acc += w * self.data[i0[:, 0] + cx, i0[:, 1] + cy, i0[:, 2] + cz]

        out = np.zeros(len(pts), np.float64)
        out[inside] = acc
        return out.reshape(shape), inside.reshape(shape)

    # -- reporting --------------------------------------------------------

    def hkl_range(self) -> tuple:
        """((h_lo, h_hi), (k_lo, k_hi), (l_lo, l_hi)) of the measured grid."""
        return tuple((float(a[0]), float(a[-1])) for a in (self.H, self.K, self.L))

    def q_max(self) -> float:
        """Largest |Q| (2*pi convention, 1/Angstrom) inside the grid's box."""
        (h0, h1), (k0, k1), (l0, l1) = self.hkl_range()
        corners = np.array([[h, k, l] for h in (h0, h1)
                            for k in (k0, k1) for l in (l0, l1)])
        B = 2.0 * np.pi * self.recip
        return float(np.max(np.linalg.norm(corners @ B.T, axis=1)))

    def describe(self) -> str:
        (h0, h1), (k0, k1), (l0, l1) = self.hkl_range()
        c = self.cell
        n0, n1, n2 = self.data.shape
        return (f"{self.name}\n"
                f"{n0} x {n1} x {n2} voxels"
                + (f" (binned {self.bin_factor}x)" if self.bin_factor > 1 else "")
                + f"\nH {h0:+.2f}..{h1:+.2f}  K {k0:+.2f}..{k1:+.2f}  "
                f"L {l0:+.2f}..{l1:+.2f} r.l.u.\n"
                f"a={c['a']:.4f} b={c['b']:.4f} c={c['c']:.4f} A  "
                f"al={c['alpha']:.2f} be={c['beta']:.2f} ga={c['gamma']:.2f}\n"
                f"measured at lambda = {self.wavelength:.4f} A")


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------


def _check_uniform(axis: np.ndarray, name: str, path: str) -> None:
    if axis.size < 2:
        raise ValueError(f"{path}: axis /{name} has {axis.size} point(s)")
    d = np.diff(axis)
    if not np.allclose(d, d[0], rtol=1e-6, atol=0.0):
        raise ValueError(
            f"{path}: axis /{name} is not uniformly spaced (steps "
            f"{d.min():.6g}..{d.max():.6g}); this reader interpolates on a "
            "regular grid only."
        )


def _bin_factor(shape, max_voxels: int) -> int:
    """Smallest integer block size that brings `shape` under `max_voxels`."""
    n = float(shape[0]) * shape[1] * shape[2]
    if not max_voxels or n <= max_voxels:
        return 1
    return int(np.ceil((n / max_voxels) ** (1.0 / 3.0)))


def _bin_axis(axis: np.ndarray, factor: int) -> np.ndarray:
    n = (axis.size // factor) * factor
    return axis[:n].reshape(-1, factor).mean(axis=1)


def _read_binned(dset, factor: int, progress=None,
                 max_block_bytes: int = 512 << 20) -> np.ndarray:
    """Read an h5 dataset, block-averaging by `factor` on all three axes.

    Read block by block along the first axis, so a volume far larger than the
    result never has to exist in memory at once. The block spans a few HDF5
    chunk rows: these files are gzip-compressed with chunks tens of rows deep,
    so reading one output row at a time would decompress every chunk over and
    over (measured: 5 minutes instead of 20 seconds on a 5.9 GB volume).
    """
    if factor == 1:
        data = np.asarray(dset[...], np.float32)
        if progress is not None:
            progress(1, 1)
        return data

    n0, n1, n2 = dset.shape
    m0, m1, m2 = n0 // factor, n1 // factor, n2 // factor
    if min(m0, m1, m2) < 2:
        raise ValueError(
            f"binning by {factor} would leave a {m0}x{m1}x{m2} grid; raise "
            "max_voxels or use a smaller reconstruction."
        )

    chunk0 = dset.chunks[0] if dset.chunks else factor
    by_chunk = int(np.ceil(2.0 * chunk0 / factor))       # ~1.5x re-read at worst
    by_memory = max(1, max_block_bytes // (factor * n1 * n2 * 4))
    rows = int(max(1, min(by_chunk, by_memory)))

    out = np.empty((m0, m1, m2), np.float32)
    for start in range(0, m0, rows):
        stop = min(start + rows, m0)
        block = np.asarray(dset[start * factor:stop * factor,
                                :m1 * factor, :m2 * factor], np.float32)
        out[start:stop] = block.reshape(
            stop - start, factor, m1, factor, m2, factor).mean(axis=(1, 3, 5))
        if progress is not None:
            progress(stop, m0)
    return out


# --------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------


def pixel_khat(detector, oversample: int = 1) -> np.ndarray:
    """Outgoing unit directions for every pixel, in image order.

    Returns (n_slow, n_fast, 3), or (n_slow, n_fast, oversample**2, 3) when
    `oversample` > 1. Row 0 is the top of the displayed frame, i.e. the largest
    slow coordinate, matching :func:`xrays_on_detector.render.render`.
    """
    centre, _, e_fast, e_slow, _ = detector.frame()
    ps = detector.pixel_size

    off = ((np.arange(oversample) + 0.5) / oversample - 0.5 if oversample > 1
           else np.zeros(1))
    cols = np.arange(detector.n_fast) - detector.beam_center_fast
    rows = (detector.n_slow - 1 - np.arange(detector.n_slow)
            - detector.beam_center_slow)
    u = ((cols[:, None] + off[None, :]).ravel() * ps)          # (n_fast * os,)
    v = ((rows[:, None] + off[None, :]).ravel() * ps)          # (n_slow * os,)

    r = (centre[None, None, :]
         + v[:, None, None] * e_slow[None, None, :]
         + u[None, :, None] * e_fast[None, None, :])
    r /= np.linalg.norm(r, axis=2, keepdims=True)
    if oversample > 1:
        r = r.reshape(detector.n_slow, oversample,
                      detector.n_fast, oversample, 3)
        r = r.transpose(0, 2, 1, 3, 4).reshape(
            detector.n_slow, detector.n_fast, oversample ** 2, 3)
    return r


def fill_distance(volume: SqVolume, detector, wavelength: float) -> float:
    """Distance at which the measured data would span the panel's shorter side.

    A reconstruction covers a box in (h, k, l), so data reliably stops at the
    largest |Q| sphere that fits inside that box, which is one 2theta ring. This
    returns the sample-detector distance that puts that ring on the edge of the
    panel's shorter dimension. It is a suggestion, not geometry: it assumes the
    arm is on the beam axis, and for a strongly non-orthogonal cell the inscribed
    sphere is only approximated by the per-axis half-widths.

    The number matters because short wavelengths make the Ewald sphere flat: at
    72 keV a volume reaching 6 r.l.u. subtends about 10 degrees in 2theta, so at
    a typical 200 mm it lands in a small central disc and most of the panel has
    no data at all.
    """
    B = 2.0 * np.pi * volume.recip
    q_edge = min(min(abs(a[0]), abs(a[-1])) * float(np.linalg.norm(B[:, i]))
                 for i, a in enumerate((volume.H, volume.K, volume.L)))
    s = q_edge / (2.0 * (2.0 * np.pi / wavelength))
    if s >= 1.0:
        return float("inf")
    two_theta = 2.0 * np.arcsin(s)
    if two_theta <= 0.0:
        return float("inf")
    half_panel = 0.5 * min(detector.n_fast, detector.n_slow) * detector.pixel_size
    return float(half_panel / np.tan(two_theta))


def project_volume(detector, volume: SqVolume, wavelength: float,
                   ZUB: np.ndarray, *, oversample: int = 1) -> tuple:
    """Cut the Ewald sphere through `volume` and read it out on `detector`.

    Parameters
    ----------
    detector : Detector (or LabDetector) carrying the arm angles and distance.
    volume : SqVolume
    wavelength : float
        Incident wavelength in Angstrom. It need not be the one the volume was
        measured at: a different wavelength simply cuts a differently curved
        sphere through the same S(q).
    ZUB : (3, 3) ndarray
        The forward map hkl -> Q_lab in the 2*pi convention, i.e. the sample
        matrix times the orientation matrix times the reciprocal matrix.
    oversample : int
        Sample each pixel on an `oversample` x `oversample` sub-grid and
        average, for a panel whose pixels are coarser than the voxels.

    Returns
    -------
    image : (n_slow, n_fast) float64 ndarray
        Row 0 at the top, as :func:`render.render` writes it.
    coverage : float
        Fraction of pixel samples that fell inside the measured volume. The
        rest read zero because there is no data there, not because nothing
        scattered.
    """
    k = 2.0 * np.pi / wavelength
    khat = pixel_khat(detector, oversample)
    Q = k * (khat - BEAM)

    M = np.linalg.inv(np.asarray(ZUB, float))          # Q_lab -> hkl
    hkl = Q @ M.T

    values, inside = volume.sample(hkl)
    image = values.mean(axis=2) if oversample > 1 else values
    return image, float(inside.mean())
