"""GPU reciprocal-space reconstruction (CuPy) - the fast path for reconstruct_volume.

Same pixel -> hkl map and voxel grid as reconstruct.reconstruct_volume, but the
per-frame histogram (the CPU bottleneck: two np.bincount over ~4.5M pixels x N
frames) runs as a float64 scatter-add on the GPU, and disk reads are prefetched on
background threads so I/O overlaps compute. Optional per-pixel intensity
corrections (see corrections.pixel_corrections) are folded into the weights.

Design notes / Meerkat-inspired choices:
  * one persistent float64 sum + int32 count accumulator on the GPU (no per-frame
    nvox-sized temporaries); scatter_add does not support int64 accumulators.
  * count normalisation (sum/count) is what turns raw per-pixel photon counts into
    a mean voxel intensity and supplies the geometric Lorentz correction, so the
    only per-pixel corrections applied are photometric (solid angle, polarisation).
  * the pixel scattering vectors r_lab and the correction map are frame-independent
    and uploaded to the GPU once.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .realframe import _axis_rot
from .reconstruct import Volume
from .corrections import pixel_corrections


def _prefetch(paths, read_fn, ahead):
    """Yield read_fn(path) in order, keeping up to `ahead` reads in flight."""
    if ahead <= 0:
        for p in paths:
            yield read_fn(p)
        return
    with ThreadPoolExecutor(max_workers=min(ahead, 8)) as ex:
        q = deque()
        i = 0
        for _ in range(min(ahead, len(paths))):
            q.append(ex.submit(read_fn, paths[i])); i += 1
        while q:
            img = q.popleft().result()
            if i < len(paths):
                q.append(ex.submit(read_fn, paths[i])); i += 1
            yield img


def _resolve_corrections(corrections, detector, r_lab):
    """None/False -> None; True -> defaults; dict -> kwargs; array -> validated."""
    if corrections is None or corrections is False:
        return None
    if corrections is True:
        return pixel_corrections(detector, r_lab)
    if isinstance(corrections, dict):
        return pixel_corrections(detector, r_lab, **corrections)
    arr = np.asarray(corrections, np.float64).ravel()
    if arr.shape[0] != r_lab.shape[0]:
        raise ValueError(f"corrections length {arr.shape[0]} != n_pixels {r_lab.shape[0]}")
    return arr


def reconstruct_volume_gpu(frames, phis, UB, R0, detector, *, phi0,
                           osc_axis=(0, 1, 0), sense=1,
                           hkl_range=(-6.0, 6.0), step=0.025,
                           read_fn=None, hot=None, progress=None,
                           corrections=None, device=0, prefetch=3) -> Volume:
    """GPU version of reconstruct_volume. Returns a host-side Volume.

    Parameters mirror reconstruct.reconstruct_volume, plus:

    corrections : None/False (raw counts), True (solid-angle + polarisation with
        default synchrotron parameters), a dict forwarded to pixel_corrections, or
        a precomputed (Npix,) multiplier in the detector's ravel order.
    device   : CUDA device index (RTX 3090 = 0).
    prefetch : frames to read ahead on background threads (0 = synchronous read).
    """
    import cupy as cp
    import cupyx

    if read_fn is None:
        import fabio

        def read_fn(p):
            return fabio.open(p).data

    cp.cuda.Device(device).use()

    frames = list(frames)
    phis = list(phis)
    ny, nx = detector.shape
    ys, xs = np.mgrid[0:ny, 0:nx]
    px = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    r_lab = detector.scattering_vectors(px).astype(np.float32)   # (Npix,3), fixed

    lo, hi = hkl_range
    n = int(round((hi - lo) / step))
    nvox = n ** 3
    n2 = n * n

    corr = _resolve_corrections(corrections, detector, r_lab)
    r_lab_g = cp.asarray(r_lab)
    corr_g = None if corr is None else cp.asarray(corr)          # float64 (Npix,)

    ssum = cp.zeros(nvox, cp.float64)
    scount = cp.zeros(nvox, cp.int32)
    UBinv = np.linalg.inv(UB)

    for i, (phi, img_host) in enumerate(zip(phis, _prefetch(frames, read_fn, prefetch))):
        Rn = _axis_rot(osc_axis, sense * (phi - phi0)) @ R0
        Mn = cp.asarray((UBinv @ Rn.T).astype(np.float32))       # hkl = r_lab @ Mn.T
        img = cp.asarray(img_host.ravel())
        # int32 voxel indices (grid <= 480 so flat max ~1.1e8 fits int32);
        # build flat once and compact with the mask a single time (not per column).
        vi = cp.floor((r_lab_g @ Mn.T - lo) / step).astype(cp.int32)
        h0, k0, l0 = vi[:, 0], vi[:, 1], vi[:, 2]
        inb = ((img >= 0)
               & (h0 >= 0) & (h0 < n) & (k0 >= 0) & (k0 < n) & (l0 >= 0) & (l0 < n))
        if hot is not None:
            inb &= img < hot
        flat = (h0 * n2 + k0 * n + l0)[inb]
        w = img[inb].astype(cp.float64)
        if corr_g is not None:
            w = w * corr_g[inb]
        cupyx.scatter_add(ssum, flat, w)
        cupyx.scatter_add(scount, flat, 1)                       # scalar broadcast
        if progress and (i % progress == 0):
            print(f"  frame {i+1}/{len(frames)} (phi={phi:.1f})", flush=True)

    ssum_h = cp.asnumpy(ssum)
    scount_h = cp.asnumpy(scount).astype(np.int64)
    del ssum, scount, r_lab_g, corr_g
    cp.get_default_memory_pool().free_all_blocks()

    with np.errstate(invalid="ignore"):
        vol = np.where(scount_h > 0, ssum_h / np.maximum(scount_h, 1), np.nan)
    axis = lo + (np.arange(n) + 0.5) * step
    return Volume(vol.reshape(n, n, n).astype(np.float32),
                  axis, np.asarray(UB), scount_h.reshape(n, n, n),
                  wavelength=detector.wavelength)
