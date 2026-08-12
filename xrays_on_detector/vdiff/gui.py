"""PyQt6 front end for the virtual six-circle diffractometer."""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np
from PyQt6.QtCore import QPointF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                             QFileDialog, QFormLayout, QFrame, QGridLayout,
                             QGroupBox, QHBoxLayout, QLabel, QMainWindow,
                             QMessageBox, QProgressDialog, QPushButton,
                             QRadioButton, QScrollArea, QSizePolicy, QSlider,
                             QSpinBox, QSplitter, QVBoxLayout, QWidget)

from .instrument import Instrument, LatticeCrystal, euler_matrix
from .presets import CELLS, DETECTORS
from .scene3d import (COL_BLOCK, COL_MISS, COL_RAY, Camera, DrawList,
                      build_scene, draw_background, image_to_qimage)

HC_KEV_A = 12.398419843320026     # h*c in keV.Angstrom

STYLE = """
QWidget { background: #12151f; color: #d6dcec; font-family: 'Segoe UI'; font-size: 11px; }
QGroupBox { border: 1px solid #2a3145; border-radius: 6px; margin-top: 12px; padding-top: 8px;
            font-weight: 600; color: #9fb0d0; }
QGroupBox::title { subcontrol-origin: margin; left: 9px; padding: 0 4px; }
QPushButton { background: #1e2536; border: 1px solid #38415c; border-radius: 4px; padding: 5px 10px; }
QPushButton:hover { background: #29334a; border-color: #4d5b80; }
QPushButton:pressed { background: #161c29; }
QPushButton:disabled { color: #5a6076; border-color: #262c3c; }
QPushButton:checked { background: #2f5d46; border-color: #57a882; color: #b9f0d4; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background: #1a2030; border: 1px solid #333b52; border-radius: 4px; padding: 3px 6px; }
QComboBox QAbstractItemView { background: #1a2030; selection-background-color: #2f3b58; }
QSlider::groove:horizontal { height: 4px; background: #262d40; border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; margin: -5px 0; border-radius: 6px; background: #6f8ec9; }
QSlider::sub-page:horizontal { background: #3d4c6e; border-radius: 2px; }
QRadioButton::indicator, QCheckBox::indicator { width: 12px; height: 12px;
    border: 1px solid #4a5573; background: #161b28; }
QRadioButton::indicator { border-radius: 7px; }
QCheckBox::indicator { border-radius: 3px; }
QRadioButton::indicator:checked, QCheckBox::indicator:checked {
    background: #6f9ee0; border-color: #9fc2f0; }
QTableWidget { background: #151a27; gridline-color: #262d40; border: 1px solid #2a3145; }
QHeaderView::section { background: #1b2231; border: 0; padding: 3px; color: #93a3c4; }
QLabel#hdr { color: #7f8fb5; font-weight: 600; letter-spacing: 0.5px; }
QLabel#val { color: #8fe3c0; font-family: 'Consolas'; }
QLabel#warn { color: #f0a05a; }
"""


# --------------------------------------------------------------------------
# Small widgets
# --------------------------------------------------------------------------


class MotorRow(QWidget):
    """One circle: name, slider, spin box and a continuous-rotation toggle."""

    changed = pyqtSignal(str, float)
    spin_toggled = pyqtSignal(str, bool)

    def __init__(self, name: str, label: str, lo: float, hi: float,
                 colour: str, spinnable: bool = True, parent=None):
        super().__init__(parent)
        self.name = name
        self._guard = False
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 1, 0, 1)
        lay.setSpacing(6)

        tag = QLabel(label)
        tag.setFixedWidth(46)
        tag.setStyleSheet(f"color: {colour}; font-weight: 700;")
        lay.addWidget(tag)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(lo * 100), int(hi * 100))
        self.slider.setValue(0)
        lay.addWidget(self.slider, 1)

        self.spin = QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setDecimals(2)
        self.spin.setSingleStep(0.1)
        self.spin.setFixedWidth(74)
        self.spin.setToolTip("degrees")
        lay.addWidget(self.spin)

        self.run = None
        if spinnable:
            self.run = QPushButton("▶")
            self.run.setCheckable(True)
            self.run.setFixedWidth(26)
            self.run.setToolTip(f"rotate {label} continuously "
                                "(several circles can run at once)")
            self.run.toggled.connect(self._on_run)
            lay.addWidget(self.run)

        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def _on_run(self, on):
        self.run.setText("■" if on else "▶")
        self.spin_toggled.emit(self.name, on)

    def spinning(self) -> bool:
        return self.run is not None and self.run.isChecked()

    def stop(self):
        if self.run is not None and self.run.isChecked():
            self.run.setChecked(False)

    def _from_slider(self, v):
        if self._guard:
            return
        self._guard = True
        self.spin.setValue(v / 100.0)
        self._guard = False
        self.changed.emit(self.name, v / 100.0)

    def _from_spin(self, v):
        if self._guard:
            return
        self._guard = True
        self.slider.setValue(int(round(v * 100)))
        self._guard = False
        self.changed.emit(self.name, float(v))

    def value(self) -> float:
        return self.spin.value()

    def set_value(self, v: float) -> None:
        self._guard = True
        self.spin.setValue(v)
        self.slider.setValue(int(round(v * 100)))
        self._guard = False


class View3D(QWidget):
    """Orbit camera view of the instrument."""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.win = window
        self.cam = Camera()
        self.setMinimumSize(360, 300)
        self.setMouseTracking(True)
        self._last = None
        self._button = None
        self.show_rings = True
        self.show_rays = True
        self.show_grid = True
        self.show_axes = True
        self.show_missed = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        w, h = self.width(), self.height()
        draw_background(p, w, h)
        shot = self.win.shot
        if shot is not None:
            dl = DrawList(self.cam, w, h)
            try:
                build_scene(dl, self.win.inst, shot, self.win.det_qimage,
                            show_rings=self.show_rings, show_rays=self.show_rays,
                            show_grid=self.show_grid, show_axes=self.show_axes,
                            show_missed=self.show_missed)
                dl.paint(p)
                self._legend(p, shot)
            except Exception:
                # An exception escaping paintEvent makes PyQt abort the whole
                # process, so report it and keep the window alive instead.
                traceback.print_exc()
        p.setPen(QPen(QColor(110, 125, 155)))
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(10, h - 10,
                   "drag orbit   .   right-drag pan   .   wheel zoom")
        p.end()

    def _legend(self, p, shot):
        rows = [(COL_RAY, f"on the detector ({len(shot.table)})")]
        if self.show_missed:
            n_off = max(len(shot.refl.hkl) - len(shot.table), 0)
            rows.append((COL_MISS, f"misses the panel ({n_off})"))
            if shot.blocked is not None:
                rows.append((COL_BLOCK,
                             f"into the sample ({len(shot.blocked.hkl)})"))
        p.setFont(QFont("Segoe UI", 8))
        for i, (col, text) in enumerate(rows):
            y = 16 + 14 * i
            p.setPen(QPen(col, 2))
            p.drawLine(12, y, 30, y)
            p.setPen(QPen(QColor(165, 178, 205)))
            p.drawText(36, y + 4, text)

    def mousePressEvent(self, ev):
        self._last = ev.position()
        self._button = ev.button()

    def mouseMoveEvent(self, ev):
        if self._last is None:
            return
        d = ev.position() - self._last
        self._last = ev.position()
        if self._button == Qt.MouseButton.LeftButton:
            self.cam.orbit(-d.x() * 0.35, d.y() * 0.35)
        elif self._button == Qt.MouseButton.RightButton:
            self.cam.pan(d.x(), d.y())
        self.update()

    def mouseReleaseEvent(self, ev):
        self._last = None
        self._button = None

    def wheelEvent(self, ev):
        self.cam.zoom(0.88 if ev.angleDelta().y() > 0 else 1.0 / 0.88)
        self.update()


class DetectorView(QWidget):
    """The simulated frame, with a crosshair at the beam centre."""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.win = window
        self.setMinimumSize(280, 260)
        self.setMouseTracking(True)
        self.show_labels = True
        self.hover = ""
        self._rect = None

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor(9, 11, 18))
        img = self.win.det_qimage
        shot = self.win.shot
        if img is None or shot is None:
            p.end()
            return

        w, h = self.width(), self.height() - 20
        iw, ih = img.width(), img.height()
        scale = min(w / iw, h / ih)
        dw, dh = iw * scale, ih * scale
        x0, y0 = (w - dw) / 2.0, (h - dh) / 2.0
        self._rect = (x0, y0, dw, dh, scale)
        p.drawImage(int(x0), int(y0),
                    img.scaled(int(dw), int(dh),
                               Qt.AspectRatioMode.IgnoreAspectRatio,
                               Qt.TransformationMode.SmoothTransformation))
        p.setPen(QPen(QColor(90, 105, 140), 1))
        p.drawRect(int(x0), int(y0), int(dw), int(dh))

        det = shot.detector
        cx = x0 + det.beam_center_fast * scale
        cy = y0 + (det.n_slow - 1 - det.beam_center_slow) * scale
        p.setPen(QPen(QColor(120, 200, 255, 130), 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(cx, y0), QPointF(cx, y0 + dh))
        p.drawLine(QPointF(x0, cy), QPointF(x0 + dw, cy))

        if self.show_labels and shot.table:
            rows = sorted(shot.table, key=lambda r: -r["intensity"])[:22]
            p.setFont(QFont("Consolas", 8))
            for r in rows:
                sx = x0 + r["fast_px"] * scale
                sy = y0 + (det.n_slow - 1 - r["slow_px"]) * scale
                p.setPen(QPen(QColor(140, 240, 210, 190)))
                p.drawEllipse(QPointF(sx, sy), 5.5, 5.5)
                p.drawText(QPointF(sx + 7, sy - 5),
                           f"{r['h']} {r['k']} {r['l']}")

        p.setPen(QPen(QColor(130, 145, 175)))
        p.setFont(QFont("Consolas", 8))
        p.drawText(6, self.height() - 6, self.hover or
                   f"{det.n_fast} x {det.n_slow} px  "
                   f"({det.pixel_size * 1000:.0f} um bins)")
        p.end()

    def mouseMoveEvent(self, ev):
        shot = self.win.shot
        if shot is None or self._rect is None:
            return
        x0, y0, dw, dh, scale = self._rect
        fx = (ev.position().x() - x0) / scale
        fy = (ev.position().y() - y0) / scale
        det = shot.detector
        if 0 <= fx < det.n_fast and 0 <= fy < det.n_slow:
            slow = det.n_slow - 1 - fy
            centre, normal, e_fast, e_slow, arm = det.frame()
            u = (fx - det.beam_center_fast) * det.pixel_size
            v = (slow - det.beam_center_slow) * det.pixel_size
            r = centre + u * e_fast + v * e_slow
            khat = r / np.linalg.norm(r)
            tt = np.degrees(np.arccos(np.clip(khat[1], -1, 1)))
            q = 4 * np.pi / self.win.inst.wavelength * np.sin(np.radians(tt) / 2)
            d = (2 * np.pi / q) if q > 1e-9 else float("inf")
            self.hover = (f"px ({fx:6.1f}, {fy:6.1f})   2theta {tt:6.2f} deg   "
                          f"|Q| {q:5.3f} 1/A   d {d:6.3f} A")
        else:
            self.hover = ""
        self.update()


# --------------------------------------------------------------------------
# CIF loading worker
# --------------------------------------------------------------------------


class CifWorker(QThread):
    progress = pyqtSignal(int, int)
    done = pyqtSignal(object, int, str)

    def __init__(self, path: str, inst: Instrument):
        super().__init__()
        self.path = path
        self.inst = inst

    def run(self):
        try:
            from ..crystal import Crystal
            crystal = Crystal.from_cif(self.path)
            probe = Instrument(**{k: getattr(self.inst, k) for k in
                                  ("wavelength", "distance", "n_fast", "n_slow",
                                   "pixel_size", "delta", "gamma")})
            probe.crystal = crystal
            n = probe.build_reflection_list(progress=self.progress.emit)
            self.done.emit((crystal, probe.hkl, probe.Fmag2), n, "")
        except Exception as exc:
            self.done.emit(None, 0, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Virtual six-circle diffractometer")
        self.resize(1640, 960)
        self.inst = Instrument()
        self.shot = None
        self.det_qimage = None
        self._sim_pending = False
        self._anim = None
        self._U_base = np.eye(3)     # orientation the free-rotation sliders start from

        self.view3d = View3D(self)
        self.detview = DetectorView(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self.view3d)
        splitter.addWidget(self.detview)
        splitter.setSizes([440, 700, 500])
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        self.setCentralWidget(splitter)

        self.status = self.statusBar()
        self._sim_timer = QTimer(self)
        self._sim_timer.setSingleShot(True)
        self._sim_timer.setInterval(25)
        self._sim_timer.timeout.connect(self._run_simulation)

        self._scan_timer = QTimer(self)
        self._scan_timer.timeout.connect(self._scan_step)

        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(16)
        self._anim_timer.timeout.connect(self._anim_step)

        self._apply_cell_preset(0)
        self.request_sim(rebuild=True)

    # -- control panel ---------------------------------------------------

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)

        # ---- beam and detector
        g = QGroupBox("Beam and detector")
        f = QFormLayout(g)
        f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.sp_wl = QDoubleSpinBox()
        self.sp_wl.setRange(0.05, 6.0)
        self.sp_wl.setDecimals(4)
        self.sp_wl.setSingleStep(0.01)
        self.sp_wl.setValue(self.inst.wavelength)
        self.sp_wl.setSuffix(" A")
        self.sp_wl.valueChanged.connect(self._on_wavelength)
        f.addRow("Wavelength", self.sp_wl)

        self.sp_en = QDoubleSpinBox()
        self.sp_en.setRange(2.0, 250.0)
        self.sp_en.setDecimals(3)
        self.sp_en.setValue(HC_KEV_A / self.inst.wavelength)
        self.sp_en.setSuffix(" keV")
        self.sp_en.valueChanged.connect(self._on_energy)
        f.addRow("Energy", self.sp_en)

        self.cb_det = QComboBox()
        for d in DETECTORS:
            self.cb_det.addItem(d.label(), d)
        self.cb_det.setCurrentIndex(3)
        self.cb_det.currentIndexChanged.connect(self._on_detector)
        f.addRow("Detector", self.cb_det)

        self.sp_px = QDoubleSpinBox()
        self.sp_px.setRange(0.005, 2.0)
        self.sp_px.setDecimals(4)
        self.sp_px.setSingleStep(0.001)
        self.sp_px.setValue(self.inst.pixel_size)
        self.sp_px.setSuffix(" mm")
        self.sp_px.valueChanged.connect(self._on_pixel)
        f.addRow("Pixel size", self.sp_px)

        self.sp_nf = QSpinBox()
        self.sp_nf.setRange(16, 8192)
        self.sp_nf.setValue(self.inst.n_fast)
        self.sp_nf.valueChanged.connect(self._on_npix)
        self.sp_ns = QSpinBox()
        self.sp_ns.setRange(16, 8192)
        self.sp_ns.setValue(self.inst.n_slow)
        self.sp_ns.valueChanged.connect(self._on_npix)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.sp_nf)
        rl.addWidget(QLabel("x"))
        rl.addWidget(self.sp_ns)
        f.addRow("Pixels (h x v)", row)

        self.sp_dist = QDoubleSpinBox()
        self.sp_dist.setRange(20.0, 5000.0)
        self.sp_dist.setDecimals(1)
        self.sp_dist.setValue(self.inst.distance)
        self.sp_dist.setSuffix(" mm")
        self.sp_dist.valueChanged.connect(self._on_distance)
        f.addRow("Distance", self.sp_dist)

        self.sp_bin = QSpinBox()
        self.sp_bin.setRange(1, 16)
        self.sp_bin.setValue(self.inst.preview_bin)
        self.sp_bin.valueChanged.connect(self._on_bin)
        f.addRow("Preview binning", self.sp_bin)
        v.addWidget(g)

        # ---- geometry mode
        g = QGroupBox("Scattering geometry")
        gl = QVBoxLayout(g)
        self.rb_trans = QRadioButton("Transmission  (single-crystal rotation)")
        self.rb_refl = QRadioButton("Reflection  (surface / CDI, one peak)")
        self.rb_trans.setChecked(True)
        self.rb_trans.toggled.connect(self._on_mode)
        gl.addWidget(self.rb_trans)
        gl.addWidget(self.rb_refl)

        self.w_surface = QWidget()
        sf = QFormLayout(self.w_surface)
        sf.setContentsMargins(14, 2, 0, 0)
        self.sp_sh = QSpinBox(); self.sp_sh.setRange(-9, 9); self.sp_sh.setValue(0)
        self.sp_sk = QSpinBox(); self.sp_sk.setRange(-9, 9); self.sp_sk.setValue(0)
        self.sp_sl = QSpinBox(); self.sp_sl.setRange(-9, 9); self.sp_sl.setValue(1)
        hkr = QWidget()
        hl = QHBoxLayout(hkr)
        hl.setContentsMargins(0, 0, 0, 0)
        for s in (self.sp_sh, self.sp_sk, self.sp_sl):
            s.valueChanged.connect(self._on_surface)
            hl.addWidget(s)
        sf.addRow("Surface (hkl)", hkr)
        self.btn_flat = QPushButton("Mount surface normal vertical")
        self.btn_flat.clicked.connect(self._mount_flat)
        sf.addRow(self.btn_flat)
        self.lb_alpha = QLabel("-")
        self.lb_alpha.setObjectName("val")
        sf.addRow("Incidence alpha", self.lb_alpha)
        self.w_surface.setVisible(False)
        gl.addWidget(self.w_surface)
        v.addWidget(g)

        # ---- sample
        g = QGroupBox("Sample")
        f = QFormLayout(g)
        self.cb_cell = QComboBox()
        for c in CELLS:
            self.cb_cell.addItem(c.label(), c)
        self.cb_cell.currentIndexChanged.connect(self._apply_cell_preset)
        f.addRow("Lattice", self.cb_cell)

        self.btn_cif = QPushButton("Load CIF ...")
        self.btn_cif.clicked.connect(self._load_cif)
        f.addRow(self.btn_cif)

        self.lb_cell = QLabel("-")
        self.lb_cell.setObjectName("val")
        self.lb_cell.setWordWrap(True)
        f.addRow("Cell", self.lb_cell)

        self.sp_sigma = QDoubleSpinBox()
        self.sp_sigma.setRange(0.0005, 0.5)
        self.sp_sigma.setDecimals(4)
        self.sp_sigma.setSingleStep(0.002)
        self.sp_sigma.setValue(self.inst.sigma)
        self.sp_sigma.setSuffix(" 1/A")
        self.sp_sigma.valueChanged.connect(self._on_sigma)
        f.addRow("Peak width", self.sp_sigma)

        v.addWidget(g)

        # ---- crystal orientation
        g = QGroupBox("Crystal orientation")
        gl = QVBoxLayout(g)
        gl.setSpacing(3)
        gl.addWidget(self._hdr("Free rotation about the lab axes"))
        self.rot = {}
        for name, label, col in (("rx", "rx", "#ff6e6e"),
                                 ("ry", "ry", "#8ceb8c"),
                                 ("rz", "rz", "#82afff")):
            row = MotorRow(name, label, -180, 180, col, spinnable=False)
            row.changed.connect(self._on_mount)
            self.rot[name] = row
            gl.addWidget(row)

        gl.addWidget(self._hdr("Point a crystal direction somewhere"))
        self.al_idx, self.al_kind, self.al_target = self._align_row(
            gl, "Put", (1, 1, 0), "beam  +y", self._align_primary)
        self.al2_idx, self.al2_kind, self.al2_target = self._align_row(
            gl, "then", (0, 0, 1), "vertical  +x", self._align_secondary)

        self.lb_align = QLabel("")
        self.lb_align.setObjectName("val")
        self.lb_align.setWordWrap(True)
        gl.addWidget(self.lb_align)

        btn_reset = QPushButton("Reset orientation (U = I)")
        btn_reset.clicked.connect(self._reset_orientation)
        gl.addWidget(btn_reset)

        ub_head = QWidget()
        uh = QHBoxLayout(ub_head)
        uh.setContentsMargins(0, 4, 0, 0)
        uh.addWidget(QLabel("UB"))
        self.cb_ubconv = QComboBox()
        self.cb_ubconv.addItem("2*pi   |a*| = 2pi/a", "2pi")
        self.cb_ubconv.addItem("1/d    |a*| = 1/a", "1/d")
        self.cb_ubconv.addItem("lambda-scaled (CrysAlis)", "lambda")
        self.cb_ubconv.currentIndexChanged.connect(self._refresh_ub)
        uh.addWidget(self.cb_ubconv, 1)
        btn_copy = QPushButton("Copy")
        btn_copy.setFixedWidth(52)
        btn_copy.clicked.connect(self._copy_ub)
        uh.addWidget(btn_copy)
        gl.addWidget(ub_head)

        self.lb_ub = QLabel("")
        self.lb_ub.setStyleSheet(
            "font-family: Consolas; color: #8fe3c0; background: #10141f;"
            "border: 1px solid #262d40; border-radius: 4px; padding: 5px;")
        gl.addWidget(self.lb_ub)
        v.addWidget(g)

        # ---- motors
        g = QGroupBox("Motors")
        gl = QVBoxLayout(g)
        gl.setSpacing(2)
        self.motors = {}
        specs = [("mu", "mu", -180, 180, "#e85d75"),
                 ("eta", "omega", -180, 180, "#f0be46"),
                 ("chi", "chi", -180, 180, "#5fc88c"),
                 ("phi", "phi", -180, 180, "#5aa0ff"),
                 ("delta", "delta", -100, 160, "#c9d3ea"),
                 ("gamma", "gamma", -100, 160, "#c9d3ea")]
        for name, label, lo, hi, col in specs:
            row = MotorRow(name, label, lo, hi, col)
            row.changed.connect(self._on_motor)
            row.spin_toggled.connect(self._on_spin)
            self.motors[name] = row
            gl.addWidget(row)

        sc = QWidget()
        sl = QHBoxLayout(sc)
        sl.setContentsMargins(0, 4, 0, 0)
        sl.addWidget(QLabel("speed"))
        self.sp_step = QDoubleSpinBox()
        self.sp_step.setRange(-20.0, 20.0)
        self.sp_step.setDecimals(2)
        self.sp_step.setValue(0.5)
        self.sp_step.setSuffix(" deg/frame")
        self.sp_step.setToolTip("Applies to every circle that is running. "
                                "Negative reverses the direction.")
        btn_stop = QPushButton("Stop all")
        btn_stop.clicked.connect(self._stop_all_spins)
        btn_zero = QPushButton("Zero all")
        btn_zero.clicked.connect(self._zero_motors)
        sl.addWidget(self.sp_step, 1)
        sl.addWidget(btn_stop)
        sl.addWidget(btn_zero)
        gl.addWidget(sc)
        v.addWidget(g)

        # ---- drive to a reflection
        g = QGroupBox("Drive to a reflection")
        gl = QVBoxLayout(g)
        hk = QWidget()
        hl = QHBoxLayout(hk)
        hl.setContentsMargins(0, 0, 0, 0)
        self.sp_h = QSpinBox(); self.sp_k = QSpinBox(); self.sp_l = QSpinBox()
        for s, dv in ((self.sp_h, 1), (self.sp_k, 1), (self.sp_l, 1)):
            s.setRange(-30, 30)
            s.setValue(dv)
            hl.addWidget(s)
        btn_find = QPushButton("Find omega")
        btn_find.clicked.connect(self._find_omega)
        hl.addWidget(btn_find)
        gl.addWidget(hk)

        self.cb_solutions = QComboBox()
        gl.addWidget(self.cb_solutions)
        self.lb_reach = QLabel("")
        self.lb_reach.setWordWrap(True)
        self.lb_reach.setObjectName("val")
        gl.addWidget(self.lb_reach)
        br = QWidget()
        bl = QHBoxLayout(br)
        bl.setContentsMargins(0, 0, 0, 0)
        btn_go = QPushButton("Drive there")
        btn_go.clicked.connect(self._drive_to_solution)
        btn_aim = QPushButton("Aim detector")
        btn_aim.clicked.connect(self._aim_detector)
        self.btn_chi = QPushButton("Move chi into reach")
        self.btn_chi.setEnabled(False)
        self.btn_chi.clicked.connect(self._move_chi_into_reach)
        bl.addWidget(btn_go)
        bl.addWidget(btn_aim)
        gl.addWidget(br)
        gl.addWidget(self.btn_chi)
        v.addWidget(g)

        # ---- display
        g = QGroupBox("Display")
        gl = QGridLayout(g)
        self.ck_rings = QCheckBox("Circles"); self.ck_rings.setChecked(True)
        self.ck_rays = QCheckBox("Rays"); self.ck_rays.setChecked(True)
        self.ck_grid = QCheckBox("Floor"); self.ck_grid.setChecked(True)
        self.ck_axes = QCheckBox("Crystal a b c"); self.ck_axes.setChecked(True)
        self.ck_missed = QCheckBox("Rays that miss")
        self.ck_missed.setToolTip(
            "Also draw reflections that are on the Ewald sphere but do not land "
            "on the panel, and (in reflection mode) those pointing into the "
            "sample. Shows why a reflection is not being recorded.")
        self.ck_labels = QCheckBox("hkl labels"); self.ck_labels.setChecked(True)
        self.ck_log = QCheckBox("Log scale"); self.ck_log.setChecked(True)
        for i, ck in enumerate((self.ck_rings, self.ck_rays, self.ck_grid,
                                self.ck_axes, self.ck_missed, self.ck_labels,
                                self.ck_log)):
            ck.toggled.connect(self._on_display)
            gl.addWidget(ck, i // 2, i % 2)
        self.cb_cmap = QComboBox()
        self.cb_cmap.addItems(["inferno", "viridis", "magma", "turbo", "gray"])
        self.cb_cmap.currentTextChanged.connect(self._on_display)
        gl.addWidget(self.cb_cmap, 4, 0, 1, 2)
        btn_full = QPushButton("Render at full resolution")
        btn_full.clicked.connect(self._render_full)
        gl.addWidget(btn_full, 5, 0, 1, 2)
        v.addWidget(g)

        v.addStretch(1)
        self.lb_info = QLabel("-")
        self.lb_info.setObjectName("val")
        self.lb_info.setWordWrap(True)
        v.addWidget(self.lb_info)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        # Size the column from what the layout actually asks for, plus room for
        # the vertical scroll bar, so nothing is ever clipped on a small window.
        bar = self.style().pixelMetric(
            self.style().PixelMetric.PM_ScrollBarExtent)
        scroll.setMinimumWidth(panel.sizeHint().width() + bar + 8)
        return scroll

    # -- orientation helpers ----------------------------------------------

    @staticmethod
    def _hdr(text: str) -> QLabel:
        lb = QLabel(text)
        lb.setObjectName("hdr")
        return lb

    def _align_row(self, layout, prefix, default_idx, default_target, callback):
        """One 'put [h k l] along <lab direction>' row."""
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(3)
        tag = QLabel(prefix)
        tag.setFixedWidth(28)
        h.addWidget(tag)

        spins = []
        for d in default_idx:
            s = QSpinBox()
            s.setRange(-20, 20)
            s.setValue(d)
            s.setFixedWidth(40)
            s.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
            s.setAlignment(Qt.AlignmentFlag.AlignCenter)
            spins.append(s)
            h.addWidget(s)

        kind = QComboBox()
        kind.addItem("(hkl)", "hkl")
        kind.addItem("[uvw]", "uvw")
        kind.setFixedWidth(60)
        kind.setToolTip("(hkl) is the plane normal, [uvw] the real-space "
                        "direction. They differ unless the lattice is cubic.")
        h.addWidget(kind)

        h.addWidget(QLabel("along"))
        target = QComboBox()
        for label, vec in (("beam  +y", (0.0, 1.0, 0.0)),
                           ("vertical  +x", (1.0, 0.0, 0.0)),
                           ("horizontal  +z", (0.0, 0.0, 1.0)),
                           ("upstream  -y", (0.0, -1.0, 0.0)),
                           ("detector arm", None)):
            target.addItem(label, vec)
        target.setCurrentText(default_target)
        h.addWidget(target, 1)

        btn = QPushButton("Go")
        btn.setFixedWidth(38)
        btn.clicked.connect(callback)
        h.addWidget(btn)

        layout.addWidget(w)
        return spins, kind, target

    def _target_vector(self, combo) -> np.ndarray:
        vec = combo.currentData()
        if vec is None:                      # detector arm
            return self.inst.detector_obj(1).frame()[4]
        return np.array(vec, dtype=float)

    @staticmethod
    def _indices(spins):
        return tuple(s.value() for s in spins)

    def _align_primary(self):
        idx = self._indices(self.al_idx)
        if not any(idx):
            self.lb_align.setText("Give a non-zero direction.")
            return
        ok = self.inst.align_in_lab(idx, self._target_vector(self.al_target),
                                    kind=self.al_kind.currentData())
        kind = self.al_kind.currentText()
        self.lb_align.setText(
            f"{kind} {idx} now along {self.al_target.currentText().strip()} "
            f"at these motor positions." if ok else "Alignment failed.")
        self._sync_rotation_boxes()
        self.request_sim()

    def _align_secondary(self):
        idx = self._indices(self.al2_idx)
        if not any(idx):
            self.lb_align.setText("Give a non-zero direction.")
            return
        axis = self._target_vector(self.al_target)     # keep the primary fixed
        ok = self.inst.align_secondary_in_lab(
            idx, self._target_vector(self.al2_target), axis,
            kind=self.al2_kind.currentData())
        self.lb_align.setText(
            f"Spun about {self.al_target.currentText().strip()} to bring "
            f"{self.al2_kind.currentText()} {idx} as close as possible to "
            f"{self.al2_target.currentText().strip()}."
            if ok else
            "That direction is parallel to the primary axis, so spinning about "
            "it changes nothing. Pick a different one.")
        self._sync_rotation_boxes()
        self.request_sim()

    def _reset_orientation(self):
        self._cancel_anim()
        self.inst.U = np.eye(3)
        self._sync_rotation_boxes()
        self.lb_align.setText("U reset to the identity: crystal axes on the "
                              "phi frame.")
        self.request_sim()

    def _sync_rotation_boxes(self):
        """Make the current orientation the new zero for the free-rotation
        sliders, so dragging them turns the crystal *from* an alignment instead
        of throwing that alignment away."""
        self._U_base = self.inst.U.copy()
        for row in self.rot.values():
            row.set_value(0.0)

    def _refresh_ub(self, *_):
        conv = self.cb_ubconv.currentData()
        try:
            ub = self.inst.UB(conv)
        except Exception as exc:
            self.lb_ub.setText(str(exc))
            return
        self._ub_text = "\n".join(
            "  ".join(f"{v:+.6f}" for v in row) for row in ub)
        self.lb_ub.setText(self._ub_text)

    def _copy_ub(self):
        QApplication.clipboard().setText(getattr(self, "_ub_text", ""))
        self.status.showMessage("UB copied to the clipboard", 2500)

    # -- state changes ----------------------------------------------------

    def request_sim(self, rebuild: bool = False):
        if rebuild:
            self._sim_pending = True
        self._sim_timer.start()

    def _on_wavelength(self, v):
        self.inst.wavelength = float(v)
        self.sp_en.blockSignals(True)
        self.sp_en.setValue(HC_KEV_A / v)
        self.sp_en.blockSignals(False)
        self.request_sim(rebuild=True)

    def _on_energy(self, v):
        self.sp_wl.setValue(HC_KEV_A / float(v))

    def _on_detector(self, i):
        d = self.cb_det.itemData(i)
        for sp, val in ((self.sp_px, d.pixel_size), (self.sp_nf, d.n_fast),
                        (self.sp_ns, d.n_slow)):
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)
        self.inst.pixel_size = d.pixel_size
        self.inst.n_fast = d.n_fast
        self.inst.n_slow = d.n_slow
        self.request_sim(rebuild=True)

    def _on_pixel(self, v):
        self.inst.pixel_size = float(v)
        self.request_sim(rebuild=True)

    def _on_npix(self, _):
        self.inst.n_fast = self.sp_nf.value()
        self.inst.n_slow = self.sp_ns.value()
        self.request_sim(rebuild=True)

    def _on_distance(self, v):
        self.inst.distance = float(v)
        self.view3d.cam.distance = max(self.view3d.cam.distance, 2.2 * float(v))
        self.request_sim(rebuild=True)

    def _on_bin(self, v):
        self.inst.preview_bin = int(v)
        self.request_sim()

    def _on_sigma(self, v):
        self.inst.sigma = float(v)
        self.request_sim()

    def _on_mode(self, _):
        self.inst.mode = "transmission" if self.rb_trans.isChecked() else "reflection"
        self.w_surface.setVisible(self.inst.mode == "reflection")
        self.request_sim()

    def _on_surface(self, _):
        self.inst.surface_hkl = (self.sp_sh.value(), self.sp_sk.value(),
                                 self.sp_sl.value())
        self.request_sim()

    def _on_mount(self, _name, _value):
        self._cancel_anim()
        R = euler_matrix(self.rot["rx"].value(), self.rot["ry"].value(),
                         self.rot["rz"].value())
        self.inst.U = R @ self._U_base
        self.request_sim()

    def _cancel_anim(self):
        """Stop a running 'drive there' move.

        Anything the user does by hand wins over an in-flight move, otherwise
        the animation keeps writing angles underneath them.
        """
        if self._anim is not None:
            self._anim = None
            self._anim_timer.stop()

    def _on_motor(self, name, value):
        self._cancel_anim()
        setattr(self.inst, name, float(value))
        self.request_sim()

    def _on_display(self, *_):
        self.view3d.show_rings = self.ck_rings.isChecked()
        self.view3d.show_rays = self.ck_rays.isChecked()
        self.view3d.show_grid = self.ck_grid.isChecked()
        self.view3d.show_axes = self.ck_axes.isChecked()
        self.view3d.show_missed = self.ck_missed.isChecked()
        self.detview.show_labels = self.ck_labels.isChecked()
        self._refresh_image()
        self.view3d.update()
        self.detview.update()

    def _mount_flat(self):
        """Mount the crystal with the chosen surface facing up.

        Aligned in the phi frame, not the lab, so the surface is horizontal when
        every circle reads zero. That is what a flat mount means, and it makes
        omega the incidence angle.
        """
        self._cancel_anim()
        self.inst.align_in_lab(self.inst.surface_hkl,
                               np.array([1.0, 0.0, 0.0]), frame="phi")
        self._sync_rotation_boxes()
        self.request_sim()

    def _zero_motors(self):
        self._cancel_anim()
        self._stop_all_spins()
        for row in self.motors.values():
            row.set_value(0.0)
            setattr(self.inst, row.name, 0.0)
        self.request_sim()

    def _apply_cell_preset(self, i):
        c = self.cb_cell.itemData(i)
        if c is None:
            return
        self.inst.crystal = LatticeCrystal.from_cell(c.a, c.b, c.c, c.alpha,
                                                     c.beta, c.gamma, name=c.name)
        self.request_sim(rebuild=True)

    # -- simulation -------------------------------------------------------

    def _run_simulation(self):
        try:
            if self._sim_pending or self.inst.hkl is None:
                n = self.inst.build_reflection_list()
                self._sim_pending = False
            self.shot = self.inst.shoot()
            self._refresh_image()
            self._refresh_readouts()
        except Exception as exc:
            traceback.print_exc()
            self.status.showMessage(f"simulation failed: {type(exc).__name__}: {exc}")
            return
        self.view3d.update()
        self.detview.update()

    def _refresh_image(self):
        if self.shot is None:
            return
        self.det_qimage = image_to_qimage(
            self.shot.image, log=self.ck_log.isChecked(),
            cmap=self.cb_cmap.currentText())

    def _refresh_readouts(self):
        inst, shot = self.inst, self.shot
        cell = inst.crystal.cell
        name = getattr(inst.crystal, "name", "from CIF")
        self.lb_cell.setText(
            f"{name}\na={cell['a']:.4f} b={cell['b']:.4f} c={cell['c']:.4f} A\n"
            f"al={cell['alpha']:.2f} be={cell['beta']:.2f} ga={cell['gamma']:.2f}")
        if inst.mode == "reflection":
            a = shot.alpha_deg
            self.lb_alpha.setText(f"{a:+.3f} deg" + ("" if a > 0 else "   BEAM BELOW SURFACE"))
            self.lb_alpha.setStyleSheet("color:#8fe3c0" if a > 0 else "color:#f0725a")
        self._refresh_ub()
        d_min = 2 * np.pi / inst.q_max() if inst.q_max() > 0 else float("inf")
        n_blocked = 0 if shot.blocked is None else len(shot.blocked.hkl)
        self.lb_info.setText(
            f"{len(inst.hkl)} hkl in range   d_min {d_min:.3f} A   "
            f"|Q|max {inst.q_max():.3f} 1/A\n"
            f"{shot.n_near} near the Ewald sphere   "
            f"{len(shot.table)} on the detector"
            + (f"   {n_blocked} blocked by surface" if n_blocked else ""))
        self.status.showMessage(
            f"mu {inst.mu:.2f}  omega {inst.eta:.2f}  chi {inst.chi:.2f}  "
            f"phi {inst.phi:.2f}   |   delta {inst.delta:.2f}  gamma {inst.gamma:.2f}"
            f"   |   lambda {inst.wavelength:.4f} A "
            f"({HC_KEV_A / inst.wavelength:.3f} keV)")

    def _render_full(self):
        if self.shot is None:
            return
        self.status.showMessage("rendering at full resolution ...")
        QApplication.processEvents()
        try:
            self.shot = self.inst.shoot(bin_factor=1)
            self._refresh_image()
            self._refresh_readouts()
            self.view3d.update()
            self.detview.update()
        except Exception as exc:
            QMessageBox.warning(self, "Render failed", str(exc))

    # -- scanning and driving ---------------------------------------------

    def _on_spin(self, _name, on):
        """Any number of circles can run at once; one timer drives them all."""
        if on:
            self._cancel_anim()
        if any(r.spinning() for r in self.motors.values()):
            if not self._scan_timer.isActive():
                self._scan_timer.start(40)
        else:
            self._scan_timer.stop()

    def _stop_all_spins(self):
        for row in self.motors.values():
            row.stop()

    def _scan_step(self):
        step = self.sp_step.value()
        moved = False
        for name, row in self.motors.items():
            if not row.spinning():
                continue
            lo, hi = row.spin.minimum(), row.spin.maximum()
            v = row.value() + step
            if v > hi:
                v = lo + (v - hi)          # wrap, so it keeps turning
            elif v < lo:
                v = hi - (lo - v)
            row.set_value(v)
            setattr(self.inst, name, v)
            moved = True
        if moved:
            self.request_sim()

    def _current_hkl(self):
        return (self.sp_h.value(), self.sp_k.value(), self.sp_l.value())

    def _find_omega(self):
        hkl = self._current_hkl()
        self.cb_solutions.clear()
        self.btn_chi.setEnabled(False)
        r = self.inst.eta_reach(hkl)

        if not r["in_limiting_sphere"]:
            self.cb_solutions.addItem("outside the limiting sphere")
            self.lb_reach.setText(
                f"|Q| = {r['Q']:.3f} > 2k = {4 * np.pi / self.inst.wavelength:.3f} "
                f"1/A. No geometry reaches this reflection at this wavelength; "
                f"you need a shorter one.")
            return

        if not r["feasible"]:
            # Blind cone: rocking one circle cannot change the component of Q
            # along that circle's axis, so this hkl never reaches the sphere.
            self.cb_solutions.addItem("no omega solution at this chi / phi / mu")
            chi = self.inst.suggest_chi(hkl)
            self.lb_reach.setText(
                f"Blind cone. Rocking omega holds the part of Q along the omega "
                f"axis fixed, so k_i.Q can only sweep "
                f"[{r['lo']:+.4f}, {r['hi']:+.4f}] while Bragg needs "
                f"{r['required']:+.4f} (short by {r['shortfall']:.4f}). "
                + (f"chi = {chi:+.2f} brings it into reach."
                   if chi is not None else
                   "No chi alone fixes it; move phi or mu too."))
            if chi is not None:
                self._chi_target = chi
                self.btn_chi.setEnabled(True)
                self.btn_chi.setText(f"Move chi to {chi:+.2f}")
            return

        try:
            sols = self.inst.solve_eta(hkl)
        except Exception as exc:
            QMessageBox.warning(self, "Solver failed", str(exc))
            return
        self.lb_reach.setText(
            f"|Q| = {r['Q']:.4f} 1/A,  d = {2 * np.pi / r['Q']:.4f} A,  "
            f"{len(sols)} omega solution(s).")
        for e in sols:
            d, g = self._detector_for_eta(hkl, e)
            self.cb_solutions.addItem(
                f"omega = {e:+8.3f}    ->  delta {d:+7.2f}  gamma {g:+7.2f}",
                (e, d, g))

    def _move_chi_into_reach(self):
        target = getattr(self, "_chi_target", None)
        if target is None:
            return
        self._animate_to({"chi": target})
        QTimer.singleShot(700, self._find_omega)

    def _detector_for_eta(self, hkl, eta):
        saved = self.inst.eta
        self.inst.eta = eta
        try:
            return self.inst.aim_detector_at(hkl)
        finally:
            self.inst.eta = saved

    def _drive_to_solution(self):
        data = self.cb_solutions.currentData()
        if data is None:
            return
        eta, d, g = data
        self._animate_to({"eta": eta, "delta": d, "gamma": g})

    def _aim_detector(self):
        d, g = self.inst.aim_detector_at(self._current_hkl())
        self._animate_to({"delta": d, "gamma": g})

    def _animate_to(self, targets: dict, steps: int = 24):
        start = {k: getattr(self.inst, k) for k in targets}
        self._anim = (start, dict(targets), 0, steps)
        self._anim_timer.start()

    def _anim_step(self):
        if self._anim is None:
            self._anim_timer.stop()
            return
        start, targets, i, steps = self._anim
        i += 1
        t = i / steps
        t = t * t * (3 - 2 * t)                      # smoothstep
        for k, tv in targets.items():
            v = start[k] + (tv - start[k]) * t
            self.motors[k].set_value(v)
            setattr(self.inst, k, v)
        self.request_sim()
        self._anim = (start, targets, i, steps)
        if i >= steps:
            self._anim = None
            self._anim_timer.stop()

    # -- CIF ---------------------------------------------------------------

    def _load_cif(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load a CIF", os.getcwd(), "CIF files (*.cif);;All files (*)")
        if not path:
            return
        dlg = QProgressDialog("Expanding symmetry and computing |F(hkl)| ...",
                              "Cancel", 0, 100, self)
        dlg.setWindowTitle("Loading CIF")
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)

        self._worker = CifWorker(path, self.inst)
        self._worker.progress.connect(
            lambda done, total: dlg.setValue(int(100 * done / max(total, 1))))
        self._worker.done.connect(lambda res, n, err: self._cif_done(res, n, err, dlg))
        self._worker.start()

    def _cif_done(self, res, n, err, dlg):
        dlg.close()
        if err:
            QMessageBox.critical(self, "CIF load failed", err)
            return
        crystal, hkl, Fmag2 = res
        self.inst.crystal = crystal
        self.inst.hkl = hkl
        self.inst.Fmag2 = Fmag2
        self.cb_cell.blockSignals(True)
        self.cb_cell.setCurrentIndex(-1)
        self.cb_cell.blockSignals(False)
        self.status.showMessage(
            f"loaded CIF: {crystal.n_atoms} atoms, {n} reflections in range")
        self.request_sim()


def _install_excepthook():
    """Report unhandled exceptions instead of letting PyQt abort the process.

    PyQt terminates the application when an exception escapes a slot or a
    virtual method. For an interactive tool that means one bad click closes the
    window with no explanation, so route them to the console and a dialog and
    carry on.
    """
    def hook(exc_type, exc, tb):
        traceback.print_exception(exc_type, exc, tb)
        try:
            QMessageBox.critical(
                None, "Unhandled error",
                f"{exc_type.__name__}: {exc}\n\n"
                "The full traceback is on the console. The app is still "
                "running, but this action did not complete.")
        except Exception:
            pass

    sys.excepthook = hook


def main():
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(STYLE)
    _install_excepthook()
    win = MainWindow()
    win.show()
    return app.exec()
