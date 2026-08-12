"""A small software 3D renderer for the virtual diffractometer scene.

Everything is drawn with QPainter: the scene is a few hundred convex primitives,
so a painter's-algorithm renderer (depth sort, near-plane clip, flat shading) is
both sufficient and dependency-free. The one trick worth naming is that the live
detector image is mapped onto the detector face with ``QTransform.quadToQuad``,
which is a genuine projective map, so the pattern stays correctly foreshortened
as the arm swings.

Lab frame: +x up, +y along the beam, +z horizontal.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin, tan

import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import (QBrush, QColor, QFont, QImage, QLinearGradient,
                         QPainterPath, QPen, QPolygonF, QTransform)

UP = np.array([1.0, 0.0, 0.0])      # lab vertical
BEAM = np.array([0.0, 1.0, 0.0])    # lab beam direction
LIGHT = np.array([0.45, 0.55, 0.70])
LIGHT = LIGHT / np.linalg.norm(LIGHT)

BG_TOP = QColor(16, 20, 34)
BG_BOTTOM = QColor(6, 8, 16)

COL_MU = QColor(232, 93, 117)
COL_ETA = QColor(240, 190, 70)
COL_CHI = QColor(95, 200, 140)
COL_PHI = QColor(90, 160, 255)
COL_DET = QColor(150, 160, 180)
COL_BEAM = QColor(255, 232, 150)
COL_RAY = QColor(120, 230, 255)
COL_MISS = QColor(130, 150, 185)      # near the sphere but off the panel
COL_BLOCK = QColor(235, 110, 95)      # stopped by the sample surface
COL_SAMPLE = QColor(220, 225, 240)
COL_SURF = QColor(120, 140, 190)
COL_A = QColor(255, 110, 110)
COL_B = QColor(140, 235, 140)
COL_C = QColor(130, 175, 255)


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------


@dataclass
class Camera:
    target: np.ndarray = None
    distance: float = 900.0
    # Start upstream of the sample, so the detector's sensitive face is toward
    # the camera rather than its casing, and the pattern reads the same way up
    # as in the 2D view. Same starting view as the browser build.
    azimuth: float = 138.0     # degrees, about the vertical
    elevation: float = 19.0    # degrees above the horizontal
    fov: float = 42.0
    near: float = 1.0

    def __post_init__(self):
        if self.target is None:
            self.target = np.zeros(3)

    def eye(self) -> np.ndarray:
        el, az = radians(self.elevation), radians(self.azimuth)
        d = np.array([sin(el), cos(el) * cos(az), cos(el) * sin(az)])
        return self.target + self.distance * d

    def basis(self):
        eye = self.eye()
        fwd = self.target - eye
        fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(fwd, UP)
        n = np.linalg.norm(right)
        if n < 1e-9:                      # looking straight down the vertical
            right = np.array([0.0, 0.0, 1.0])
        else:
            right = right / n
        up = np.cross(right, fwd)
        return eye, right, up, fwd

    def to_view(self, pts: np.ndarray) -> np.ndarray:
        """World -> view coordinates (x right, y up, z forward)."""
        eye, right, up, fwd = self.basis()
        rel = np.atleast_2d(np.asarray(pts, float)) - eye
        return np.stack([rel @ right, rel @ up, rel @ fwd], axis=1)

    def project_view(self, view: np.ndarray, w: int, h: int) -> np.ndarray:
        f = (h / 2.0) / tan(radians(self.fov) / 2.0)
        z = np.maximum(view[:, 2], 1e-6)
        return np.stack([w / 2.0 + f * view[:, 0] / z,
                         h / 2.0 - f * view[:, 1] / z], axis=1)

    def orbit(self, daz: float, dele: float) -> None:
        self.azimuth = (self.azimuth + daz) % 360.0
        self.elevation = float(np.clip(self.elevation + dele, -89.0, 89.0))

    def zoom(self, factor: float) -> None:
        self.distance = float(np.clip(self.distance * factor, 40.0, 20000.0))

    def pan(self, dx: float, dy: float) -> None:
        _, right, up, _ = self.basis()
        scale = self.distance * 0.0015
        self.target = self.target - right * dx * scale + up * dy * scale


# --------------------------------------------------------------------------
# Near-plane clipping
# --------------------------------------------------------------------------


def clip_polygon_view(view: np.ndarray, near: float) -> np.ndarray | None:
    """Sutherland-Hodgman clip of a view-space polygon against z = near."""
    n = len(view)
    out = []
    for i in range(n):
        a, b = view[i], view[(i + 1) % n]
        ain, bin_ = a[2] >= near, b[2] >= near
        if ain:
            out.append(a)
        if ain != bin_:
            t = (near - a[2]) / (b[2] - a[2])
            out.append(a + t * (b - a))
    if len(out) < 3:
        return None
    return np.array(out)


def clip_segment_view(a: np.ndarray, b: np.ndarray, near: float):
    ain, bin_ = a[2] >= near, b[2] >= near
    if ain and bin_:
        return a, b
    if not ain and not bin_:
        return None
    t = (near - a[2]) / (b[2] - a[2])
    p = a + t * (b - a)
    return (a, p) if ain else (p, b)


# --------------------------------------------------------------------------
# Draw list
# --------------------------------------------------------------------------


class DrawList:
    """Collects primitives and paints them back-to-front."""

    def __init__(self, camera: Camera, w: int, h: int):
        self.cam = camera
        self.w, self.h = w, h
        self.items = []   # (depth, callable(painter))

    # -- geometry helpers ------------------------------------------------

    def _screen(self, view: np.ndarray) -> np.ndarray:
        return self.cam.project_view(view, self.w, self.h)

    def add_polygon(self, pts3: np.ndarray, color: QColor, *, normal=None,
                    pen: QPen | None = None, shade: bool = True,
                    alpha: int | None = None):
        view = self.cam.to_view(pts3)
        clipped = clip_polygon_view(view, self.cam.near)
        if clipped is None:
            return
        depth = float(np.mean(clipped[:, 2]))
        scr = self._screen(clipped)
        c = QColor(color)
        if shade and normal is not None:
            lam = 0.35 + 0.65 * max(0.0, abs(float(np.dot(normal, LIGHT))))
            c = QColor(int(c.red() * lam), int(c.green() * lam),
                       int(c.blue() * lam))
        if alpha is not None:
            c.setAlpha(alpha)
        poly = QPolygonF([QPointF(float(x), float(y)) for x, y in scr])

        def paint(p, poly=poly, c=c, pen=pen):
            p.setBrush(QBrush(c))
            p.setPen(pen if pen is not None else Qt.PenStyle.NoPen)
            p.drawPolygon(poly)

        self.items.append((depth, paint))

    def add_line(self, a3, b3, color: QColor, width: float = 1.5,
                 style=Qt.PenStyle.SolidLine, depth_bias: float = 0.0):
        view = self.cam.to_view(np.array([a3, b3], float))
        seg = clip_segment_view(view[0], view[1], self.cam.near)
        if seg is None:
            return
        a, b = seg
        depth = 0.5 * (a[2] + b[2]) + depth_bias
        scr = self._screen(np.array([a, b]))
        p0 = QPointF(float(scr[0, 0]), float(scr[0, 1]))
        p1 = QPointF(float(scr[1, 0]), float(scr[1, 1]))
        pen = QPen(color, width)
        pen.setStyle(style)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        def paint(p, p0=p0, p1=p1, pen=pen):
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawLine(p0, p1)

        self.items.append((depth, paint))

    def add_polyline(self, pts3, color: QColor, width: float = 1.5,
                     style=Qt.PenStyle.SolidLine):
        pts3 = np.asarray(pts3, float)
        for i in range(len(pts3) - 1):
            self.add_line(pts3[i], pts3[i + 1], color, width, style)

    def add_textured_quad(self, corners3: np.ndarray, img: QImage,
                          border: QColor | None = None):
        """Map `img` onto the quad. Corners must be given in order
        (top-left, top-right, bottom-right, bottom-left) of the image."""
        view = self.cam.to_view(corners3)
        if np.any(view[:, 2] < self.cam.near):
            return False
        scr = self._screen(view)
        depth = float(np.mean(view[:, 2]))
        src = QPolygonF([QPointF(0.0, 0.0), QPointF(img.width(), 0.0),
                         QPointF(img.width(), img.height()),
                         QPointF(0.0, img.height())])
        dst = QPolygonF([QPointF(float(x), float(y)) for x, y in scr])
        t = QTransform()
        if not QTransform.quadToQuad(src, dst, t):
            return False

        clip = QPainterPath()
        clip.addPolygon(dst)
        clip.closeSubpath()

        def paint(p, t=t, img=img, dst=dst, border=border, clip=clip):
            p.save()
            p.setClipPath(clip)
            p.setTransform(t, True)
            p.drawImage(0, 0, img)
            p.restore()
            if border is not None:
                p.setPen(QPen(border, 1.6))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPolygon(dst)

        self.items.append((depth, paint))
        return True

    def add_label(self, pos3, text: str, color: QColor, dx: float = 6.0,
                  dy: float = -6.0, size: int = 9, bold: bool = False):
        view = self.cam.to_view(np.array([pos3], float))[0]
        if view[2] < self.cam.near:
            return
        scr = self._screen(np.array([view]))[0]
        font = QFont("Segoe UI", size)
        font.setBold(bold)

        def paint(p, scr=scr, text=text, color=color, font=font):
            p.setPen(QPen(color))
            p.setFont(font)
            p.drawText(QPointF(float(scr[0] + dx), float(scr[1] + dy)), text)

        self.items.append((view[2] - 1e6, paint))   # labels last (on top)

    # -- output ----------------------------------------------------------

    def paint(self, painter):
        for _, fn in sorted(self.items, key=lambda it: -it[0]):
            fn(painter)


# --------------------------------------------------------------------------
# Mesh helpers
# --------------------------------------------------------------------------


def perp_basis(axis: np.ndarray):
    axis = axis / np.linalg.norm(axis)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(axis, ref))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, ref)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(axis, e1)


def band_quads(axis, radius, half_width, n=72, t0=0.0, t1=360.0, R=None,
               centre=None):
    """A cylindrical band (a goniometer circle) as quads with outward normals."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    e1, e2 = perp_basis(axis)
    ts = np.radians(np.linspace(t0, t1, n + 1))
    if centre is None:
        centre = np.zeros(3)
    quads = []
    for i in range(n):
        out = []
        for t in (ts[i], ts[i + 1]):
            r = cos(t) * e1 + sin(t) * e2
            out.append(r)
        n_mid = (out[0] + out[1])
        n_mid /= np.linalg.norm(n_mid)
        pts = np.array([
            centre + radius * out[0] + half_width * axis,
            centre + radius * out[1] + half_width * axis,
            centre + radius * out[1] - half_width * axis,
            centre + radius * out[0] - half_width * axis,
        ])
        if R is not None:
            pts = pts @ R.T
            n_mid = R @ n_mid
        quads.append((pts, n_mid))
    return quads


def box_quads(centre, e1, e2, e3, h1, h2, h3, R=None):
    """An axis-aligned-in-its-own-frame box as 6 quads with normals."""
    centre = np.asarray(centre, float)
    axes = [(e1, h1), (e2, h2), (e3, h3)]
    quads = []
    for i in range(3):
        a, ha = axes[i]
        b, hb = axes[(i + 1) % 3]
        c, hc = axes[(i + 2) % 3]
        for s in (+1, -1):
            n = s * np.asarray(a, float)
            pts = np.array([
                centre + s * ha * a + hb * b + hc * c,
                centre + s * ha * a - hb * b + hc * c,
                centre + s * ha * a - hb * b - hc * c,
                centre + s * ha * a + hb * b - hc * c,
            ])
            if R is not None:
                pts = pts @ R.T
                n = R @ n
            quads.append((pts, n))
    return quads


# --------------------------------------------------------------------------
# Colour map for the detector image
# --------------------------------------------------------------------------


def build_lut(name: str = "inferno") -> np.ndarray:
    try:
        import matplotlib
        cmap = matplotlib.colormaps[name]
        lut = (np.asarray([cmap(i / 255.0)[:3] for i in range(256)]) * 255)
        return lut.astype(np.uint8)
    except Exception:
        g = np.linspace(0, 255, 256).astype(np.uint8)
        return np.stack([g, g, g], axis=1)


_LUT_CACHE = {}


def get_lut(name: str) -> np.ndarray:
    if name not in _LUT_CACHE:
        _LUT_CACHE[name] = build_lut(name)
    return _LUT_CACHE[name]


def image_to_qimage(arr: np.ndarray, *, log: bool = True, gain: float = 1.0,
                    cmap: str = "inferno") -> QImage:
    """Colour-map a detector image to an RGB QImage (row 0 = top)."""
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return QImage(1, 1, QImage.Format.Format_RGB888)
    vmax = float(a.max())
    if vmax <= 0:
        norm = np.zeros_like(a)
    elif log:
        norm = np.log1p(a * (gain * 500.0 / vmax)) / np.log1p(gain * 500.0)
    else:
        norm = np.clip(a * gain / vmax, 0.0, 1.0)
    idx = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    rgb = np.ascontiguousarray(get_lut(cmap)[idx])
    h, w = idx.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


# --------------------------------------------------------------------------
# The scene
# --------------------------------------------------------------------------


def draw_background(painter, w, h):
    grad = QLinearGradient(0, 0, 0, h)
    grad.setColorAt(0.0, BG_TOP)
    grad.setColorAt(1.0, BG_BOTTOM)
    painter.fillRect(0, 0, w, h, QBrush(grad))


def build_scene(dl: DrawList, inst, shot, det_image: QImage | None,
                *, show_rings=True, show_rays=True, show_grid=True,
                show_axes=True, show_missed=False, max_rays=140,
                max_missed=600):
    """Populate the draw list from the instrument state and the last shot."""
    from ..geometry import sample_matrix

    det = shot.detector
    centre, normal, e_fast, e_slow, arm = det.frame()
    half_w = 0.5 * det.n_fast * det.pixel_size
    half_h = 0.5 * det.n_slow * det.pixel_size
    scale = max(inst.distance, 120.0)

    # -- ground grid ----------------------------------------------------
    if show_grid:
        y0 = -0.42 * scale
        span = 1.5 * scale
        step = span / 8.0
        faint = QColor(60, 72, 100, 110)
        for i in range(-8, 9):
            u = i * step
            dl.add_line([y0, -span, u], [y0, span, u], faint, 1.0)
            dl.add_line([y0, u, -span], [y0, u, span], faint, 1.0)

    # -- goniometer circles ---------------------------------------------
    if show_rings:
        MU = sample_matrix(inst.mu, 0, 0, 0)
        MUE = sample_matrix(inst.mu, inst.eta, 0, 0)
        MUEC = sample_matrix(inst.mu, inst.eta, inst.chi, 0)
        r0 = 0.46 * scale
        rings = [
            ("mu", np.array([1.0, 0.0, 0.0]), r0 * 1.00, np.eye(3), COL_MU, (0, 360)),
            ("omega", np.array([0.0, 0.0, -1.0]), r0 * 0.80, MU, COL_ETA, (0, 360)),
            ("chi", np.array([0.0, 1.0, 0.0]), r0 * 0.62, MUE, COL_CHI, (-105, 105)),
            ("phi", np.array([0.0, 0.0, -1.0]), r0 * 0.42, MUEC, COL_PHI, (0, 360)),
        ]
        for name, axis, radius, R, col, (t0, t1) in rings:
            hw = max(2.5, 0.016 * scale)
            for pts, n in band_quads(axis, radius, hw, n=64, t0=t0, t1=t1, R=R):
                dl.add_polygon(pts, col, normal=n)
            # axis stub, so the rotation axis of each circle is visible
            a_lab = R @ axis
            dl.add_line(-a_lab * radius * 1.18, a_lab * radius * 1.18,
                        QColor(col.red(), col.green(), col.blue(), 150), 1.2,
                        Qt.PenStyle.DashLine)
            dl.add_label(a_lab * radius * 1.22, name, col, size=9, bold=True)

    # -- sample ----------------------------------------------------------
    Z = inst.sample_M()
    ZU = Z @ inst.U
    s = max(9.0, 0.045 * scale)
    if inst.mode == "reflection":
        n_lab = inst.surface_normal_lab()
        e1, e2 = perp_basis(n_lab)
        plate = 2.6 * s
        for pts, n in box_quads(np.zeros(3), e1, e2, n_lab, plate, plate, 0.18 * s):
            dl.add_polygon(pts, COL_SURF, normal=n)
        dl.add_line(np.zeros(3), n_lab * plate * 1.5, QColor(255, 255, 255, 190),
                    1.6, Qt.PenStyle.DashLine)
        dl.add_label(n_lab * plate * 1.55, "surface normal",
                     QColor(210, 220, 240), size=8)
    else:
        for pts, n in box_quads(np.zeros(3), ZU[:, 0], ZU[:, 1], ZU[:, 2],
                                s, s * 0.75, s * 0.6):
            dl.add_polygon(pts, COL_SAMPLE, normal=n)

    # -- incident beam ---------------------------------------------------
    b0 = np.array([0.0, -1.35 * scale, 0.0])
    dl.add_line(b0, np.zeros(3), COL_BEAM, 3.0)
    for f in (0.55, 0.75, 0.95):
        p = b0 * (1 - f)
        dl.add_line(p, p + np.array([0.0, 0.05 * scale, 0.0]),
                    QColor(255, 245, 200), 5.0)
    dl.add_label(b0 * 0.75, "beam", COL_BEAM, size=9, bold=True)

    # direct beam continuation to the detector plane
    denom = float(np.dot(BEAM, normal))
    if abs(denom) > 1e-9:
        t = float(np.dot(centre, normal)) / denom
        if t > 0:
            dl.add_line(np.zeros(3), BEAM * t, QColor(255, 232, 150, 90), 1.2,
                        Qt.PenStyle.DotLine)

    # -- detector arm and panel ------------------------------------------
    dl.add_line(np.zeros(3), centre, QColor(120, 132, 158), 2.2)

    corners = np.array([
        centre - half_w * e_fast + half_h * e_slow,   # top-left
        centre + half_w * e_fast + half_h * e_slow,   # top-right
        centre + half_w * e_fast - half_h * e_slow,   # bottom-right
        centre - half_w * e_fast - half_h * e_slow,   # bottom-left
    ])
    # casing behind the sensitive area: further from the sample along the arm,
    # so it never comes between the sample and the face it is meant to back
    back = centre + arm * (0.05 * scale)
    back_corners = np.array([
        back - half_w * 1.06 * e_fast + half_h * 1.06 * e_slow,
        back + half_w * 1.06 * e_fast + half_h * 1.06 * e_slow,
        back + half_w * 1.06 * e_fast - half_h * 1.06 * e_slow,
        back - half_w * 1.06 * e_fast - half_h * 1.06 * e_slow,
    ])
    for i in range(4):
        side = np.array([corners[i], corners[(i + 1) % 4],
                         back_corners[(i + 1) % 4], back_corners[i]])
        nrm = np.cross(side[1] - side[0], side[2] - side[0])
        nn = np.linalg.norm(nrm)
        dl.add_polygon(side, COL_DET.darker(150),
                       normal=nrm / nn if nn > 1e-9 else None)
    dl.add_polygon(back_corners, COL_DET.darker(190), normal=arm)

    placed = False
    if det_image is not None:
        placed = dl.add_textured_quad(corners, det_image,
                                      border=QColor(190, 200, 220, 200))
    if not placed:
        dl.add_polygon(corners, QColor(24, 26, 34), normal=normal,
                       pen=QPen(QColor(190, 200, 220, 200), 1.4), shade=False)
    dl.add_label(corners[0], "detector", QColor(200, 210, 230), size=9, bold=True)

    # -- rays that miss the panel, and rays the surface stopped -----------
    # Drawn first so the ones that actually land stay on top. These answer
    # "why am I not seeing that reflection": it is either off the panel or,
    # in reflection geometry, pointing into the sample.
    miss_len = 1.55 * max(inst.distance, 60.0)
    if show_missed:
        if len(shot.refl.khat):
            _, _, inside, _ = det.project(shot.refl.khat)
            outside = np.nonzero(~inside)[0][:max_missed]
            col = QColor(COL_MISS)
            col.setAlpha(170)
            tip = QColor(COL_MISS)
            for i in outside:
                end = shot.refl.khat[i] * miss_len
                dl.add_line(np.zeros(3), end, col, 1.1)
                dl.add_line(end * 0.965, end, tip, 3.0)   # a head, so it reads
        if shot.blocked is not None and len(shot.blocked.khat):
            col = QColor(COL_BLOCK)
            col.setAlpha(175)
            for kh in shot.blocked.khat[:max_missed]:
                end = kh * miss_len * 0.5
                dl.add_line(np.zeros(3), end, col, 1.1)
                dl.add_line(end * 0.93, end, COL_BLOCK, 3.0)

    # -- diffracted rays that land on the detector -------------------------
    if show_rays and shot.table:
        rows = sorted(shot.table, key=lambda r: -r["intensity"])[:max_rays]
        imax = max(r["intensity"] for r in rows) or 1.0
        for r in rows:
            u = (r["fast_px"] - det.beam_center_fast) * det.pixel_size
            v = (r["slow_px"] - det.beam_center_slow) * det.pixel_size
            hit = centre + u * e_fast + v * e_slow
            w = float(np.clip(r["intensity"] / imax, 0.0, 1.0)) ** 0.35
            col = QColor(COL_RAY)
            col.setAlpha(int(40 + 190 * w))
            dl.add_line(np.zeros(3), hit, col, 0.8 + 1.8 * w)

    # -- lab frame and crystal axes, standing on the floor side by side ----
    floor = -0.42 * scale                     # same height as the grid
    L = 0.42 * scale
    # Well upstream and to either side, where the detector arm rarely reaches.
    lab_org = np.array([floor, -1.05 * scale, -0.64 * scale])
    cry_org = lab_org + np.array([0.0, 0.0, 1.28 * scale])

    def triad(org, axes, title):
        # Reference overlays, not physical objects: drawn over the scene with a
        # depth bias so the detector cannot half-hide them while their labels
        # (which are always on top) still show.
        for vec, col, name in axes:
            tip = org + vec * L
            dl.add_line(org, tip, col, 3.4, depth_bias=-1e6)
            dl.add_line(org + vec * L * 0.86, tip, col, 6.5, depth_bias=-1e6)
            dl.add_label(tip, name, col, size=12, bold=True)
        dl.add_label(org, title, QColor(165, 178, 205), dx=-10, dy=20, size=10,
                     bold=True)

    triad(lab_org,
          ((UP, QColor(235, 120, 140), "x  up"),
           (BEAM, QColor(240, 205, 110), "y  beam"),
           (np.array([0.0, 0.0, 1.0]), QColor(120, 200, 255), "z")),
          "lab")

    if show_axes:
        units, _ = inst.direct_axes_lab()
        triad(cry_org,
              ((units[:, 0], COL_A, "a"),
               (units[:, 1], COL_B, "b"),
               (units[:, 2], COL_C, "c")),
              "crystal")
