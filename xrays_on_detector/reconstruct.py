"""Reconstruct a 3D reciprocal-space volume from raw rotation frames.

This is the inverse of the realframe forward model: every detector pixel is
mapped to a reciprocal-space (h,k,l) coordinate and its intensity accumulated
into a voxel grid over all frames. The map is

    hkl = (R_n . UB)^-1 . r_lab ,   R_n = R_osc(sense*(phi_n - phi0)) . R0 ,

where r_lab is the pixel's scattering vector (fixed by the detector) and R0 is
the crystal orientation for the reference frame (from realframe.index_frame).
R0 is precisely the rotation linking the CrysAlisPro UB frame to the detector
lab frame, which is what makes Bragg peaks land at integer hkl.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .realframe import _axis_rot


@dataclass
class Volume:
    data: np.ndarray        # (n,n,n) mean intensity, NaN where unmeasured
    H: np.ndarray           # 1D axis (same for K, L; cubic grid)
    UB: np.ndarray          # columns a*,b*,c* in 1/d (CrysAlis UB / lambda)
    counts: np.ndarray      # (n,n,n) number of pixels per voxel
    wavelength: float | None = None   # Angstrom; needed for a viewer-compatible h5


def reconstruct_volume(frames, phis, UB, R0, detector, *, phi0,
                       osc_axis=(0, 1, 0), sense=1,
                       hkl_range=(-6.0, 6.0), step=0.05,
                       read_fn=None, hot=None, progress=None, corr=None) -> Volume:
    """Accumulate raw frames into a reciprocal-space voxel grid.

    Parameters
    ----------
    frames : sequence of CBF paths (or whatever read_fn accepts)
    phis   : per-frame phi angle (deg), same length as frames
    UB     : (3,3) columns a*,b*,c* in 1/d (CrysAlis UB / lambda)
    R0     : (3,3) reference-frame orientation from index_frame
    detector : FlatDetector
    phi0   : phi of the reference frame that R0 was found on
    hkl_range, step : cubic grid extent and voxel size (r.l.u.)
    read_fn : path -> (ny,nx) int array; defaults to fabio
    hot    : optional upper intensity clip (ignore pixels above this)
    corr   : optional (Npix,) per-pixel intensity multiplier (corrections.pixel_
             corrections), in the same ravel order as the pixels; None = raw counts
    """
    if read_fn is None:
        import fabio

        def read_fn(p):
            return fabio.open(p).data

    ny, nx = detector.shape
    ys, xs = np.mgrid[0:ny, 0:nx]
    px = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    r_lab = detector.scattering_vectors(px).astype(np.float32)   # (Npix,3), fixed

    lo, hi = hkl_range
    n = int(round((hi - lo) / step))
    nvox = n ** 3
    ssum = np.zeros(nvox, np.float64)
    scount = np.zeros(nvox, np.int64)
    UBinv = np.linalg.inv(UB)

    for i, (path, phi) in enumerate(zip(frames, phis)):
        Rn = _axis_rot(osc_axis, sense * (phi - phi0)) @ R0
        Mn = (UBinv @ Rn.T).astype(np.float32)                   # hkl = r_lab @ Mn.T
        hkl = r_lab @ Mn.T
        img = read_fn(path).ravel()

        vi = np.floor((hkl - lo) / step).astype(np.int64)
        inb = ((img >= 0) & (vi[:, 0] >= 0) & (vi[:, 0] < n)
               & (vi[:, 1] >= 0) & (vi[:, 1] < n)
               & (vi[:, 2] >= 0) & (vi[:, 2] < n))
        if hot is not None:
            inb &= img < hot
        flat = vi[inb, 0] * n * n + vi[inb, 1] * n + vi[inb, 2]
        w = img[inb].astype(np.float64)
        if corr is not None:
            w = w * corr[inb]
        ssum += np.bincount(flat, weights=w, minlength=nvox)
        scount += np.bincount(flat, minlength=nvox)
        if progress and (i % progress == 0):
            print(f"  frame {i+1}/{len(frames)} (phi={phi:.1f})", flush=True)

    with np.errstate(invalid="ignore"):
        vol = np.where(scount > 0, ssum / np.maximum(scount, 1), np.nan)
    axis = lo + (np.arange(n) + 0.5) * step
    return Volume(vol.reshape(n, n, n).astype(np.float32),
                  axis, np.asarray(UB), scount.reshape(n, n, n),
                  wavelength=detector.wavelength)


def _reciprocal_cell(recip):
    """Direct cell dict (a,b,c,alpha,beta,gamma) from reciprocal vectors
    (columns a*,b*,c* in 1/d)."""
    Gstar = recip.T @ recip
    G = np.linalg.inv(Gstar)
    a, b, c = np.sqrt(np.diag(G))

    def ang(i, j, x, y):
        return float(np.degrees(np.arccos(np.clip(G[i, j] / (x * y), -1, 1))))

    return dict(a=float(a), b=float(b), c=float(c),
                alpha=ang(1, 2, b, c), beta=ang(0, 2, a, c), gamma=ang(0, 1, a, b))


def _plane_M_inv_HK(recip):
    """2x2 Cartesian(A^-1)->Miller transform for the HK plane, matching
    rspace3d.compute_plane_M_inv (raw reciprocal vectors, no fixed-axis
    projection; handles shear for non-orthogonal cells)."""
    v1, v2 = recip[:, 0], recip[:, 1]                 # a*, b*
    e_x = v1 / np.linalg.norm(v1)
    v2p = v2 - (v2 @ e_x) * e_x
    e_y = v2p / np.linalg.norm(v2p)
    M = np.array([[v1 @ e_x, v2 @ e_x], [v1 @ e_y, v2 @ e_y]])
    return np.linalg.inv(M)


def save_rspace3d_h5(path, vol: Volume, *, source_folder=None):
    """Save in the rspace3d HDF5 layout, compatible with the rsp_viewer.

    Writes /data, /H, /K, /L, /UB, /M_inv and wavelength/cell_*/s/plane_type
    attrs. Two things the viewer needs that a bare /data+/H+/K+/L+/UB file lacks:
    (1) the `wavelength` attribute - the viewer computes M_inv as UB/wavelength,
        so a missing wavelength defaults to 0 -> NaN M_inv -> an empty (all-zero)
        plane; (2) UB in the CrysAlisPro lambda-scaled convention (= vol.UB *
        wavelength), because vol.UB is already in 1/d and the viewer divides by
        wavelength again. `s` is the Cartesian (A^-1) step per voxel.
    """
    import h5py

    if vol.wavelength is None:
        raise ValueError("Volume.wavelength is required for a viewer-compatible file")
    wl = float(vol.wavelength)
    recip = np.asarray(vol.UB, float)                 # columns a*,b*,c* in 1/d
    ub_crysalis = recip * wl                          # lambda-scaled (viewer convention)
    step = float(vol.H[1] - vol.H[0])
    s = float(np.linalg.norm(recip[:, 0]) * step)     # A^-1 per voxel
    cell = _reciprocal_cell(recip)
    M_inv = _plane_M_inv_HK(recip)

    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=vol.data, compression="gzip", compression_opts=4)
        f.create_dataset("H", data=vol.H)
        f.create_dataset("K", data=vol.H)
        f.create_dataset("L", data=vol.H)
        f.create_dataset("UB", data=ub_crysalis)
        f.create_dataset("M_inv", data=M_inv)
        f.attrs["plane_type"] = "HK"
        f.attrs["wavelength"] = wl
        f.attrs["s"] = s
        f.attrs["bin_xy"] = 1
        f.attrs["bin_z"] = 1
        for k, v in cell.items():
            f.attrs[f"cell_{k}"] = v
        if source_folder:
            f.attrs["source_folder"] = str(source_folder)
        f.attrs["reconstructed_by"] = "xrays_on_detector.reconstruct"
