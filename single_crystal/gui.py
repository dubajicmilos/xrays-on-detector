"""Desktop single-crystal diffraction viewer.

    python -m single_crystal [structure.cif]

Open any CIF and look at its diffraction pattern three ways:

  Section   an undistorted plane of the weighted reciprocal lattice, named by
            a zone axis [uvw] and a layer, which is what a precession camera
            records;
  SAED      the same zone through a curved Ewald sphere, so the higher-order
            Laue zones appear;
  Powder    the multiplicity-summed 2-theta trace.

Any of the three, for X-rays, neutrons or electrons.

The palette is the Game of Diffraction's, since this is its sibling; the
browser build of the same model is web/sc/.
"""
from __future__ import annotations

import os
import sys

import numpy as np

try:
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QAction, QActionGroup, QKeySequence
    from PyQt6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
        QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
        QSlider, QSpinBox, QVBoxLayout, QWidget,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"the desktop viewer needs PyQt6 ({exc}).\n"
        "    pip install PyQt6 matplotlib"
    ) from exc

try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"the desktop viewer needs matplotlib ({exc}).\n"
        "    pip install PyQt6 matplotlib"
    ) from exc

from . import display as disp
from .cif import CifError, read_cif
from .powder import compute_powder
from .scatter import RADIATION_LABEL, RADIATIONS, UNITS, electron_wavelength
from .section import compute_section
from .structure import Structure
from .tem import compute_tem, d_min_for_zone

BG = "#0b0e18"
PANEL = "#12151f"
PLOT_BG = "#090b12"
TEXT = "#d6dcec"
DIM = "#8794b0"
ACCENT = "#6f9ee0"
GOOD = "#8fe3c0"
WARN = "#f0a05a"
BAD = "#f0725a"

STYLE = f"""
QWidget {{ background: {PANEL}; color: {TEXT}; font-family: 'Segoe UI'; font-size: 11px; }}
QMainWindow, QMenuBar {{ background: {BG}; }}
QMenuBar::item:selected {{ background: #29334a; }}
QMenu {{ background: #1a2030; border: 1px solid #333b52; }}
QMenu::item:selected {{ background: #2f3b58; }}
QGroupBox {{ border: 1px solid #2a3145; border-radius: 6px; margin-top: 12px;
            padding-top: 8px; font-weight: 600; color: #9fb0d0; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 9px; padding: 0 4px; }}
QPushButton {{ background: #1e2536; border: 1px solid #38415c; border-radius: 4px;
               padding: 5px 10px; }}
QPushButton:hover {{ background: #29334a; border-color: #4d5b80; }}
QPushButton:pressed {{ background: #161c29; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: #1a2030; border: 1px solid #333b52; border-radius: 4px; padding: 3px 6px; }}
QComboBox QAbstractItemView {{ background: #1a2030; selection-background-color: #2f3b58; }}
QSlider::groove:horizontal {{ height: 4px; background: #262d40; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 12px; margin: -5px 0; border-radius: 6px;
                              background: #6f8ec9; }}
QSlider::sub-page:horizontal {{ background: #3d4c6e; border-radius: 2px; }}
QCheckBox::indicator {{ width: 12px; height: 12px; border: 1px solid #4a5573;
                        background: #161b28; border-radius: 3px; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: #9fc2f0; }}
QLabel#hdr {{ color: #7f8fb5; font-weight: 600; letter-spacing: 0.5px; }}
QLabel#val {{ color: {GOOD}; font-family: 'Consolas'; }}
QLabel#warn {{ color: {WARN}; }}
QStatusBar {{ background: {BG}; color: {DIM}; }}
"""

MODES = ("section", "saed", "powder")
MODE_LABEL = {
    "section": "Reciprocal-lattice section",
    "saed": "Electron diffraction (SAED)",
    "powder": "Powder pattern",
}
COLORMAPS = ("inferno", "viridis", "magma", "turbo", "gray")


def _row(*widgets, stretch_at=None):
    box = QWidget()
    lay = QHBoxLayout(box)
    lay.setContentsMargins(0, 1, 0, 1)
    lay.setSpacing(6)
    for i, w in enumerate(widgets):
        if isinstance(w, str):
            lab = QLabel(w)
            lab.setFixedWidth(74)
            lab.setStyleSheet(f"color: {DIM};")
            lay.addWidget(lab)
        else:
            lay.addWidget(w, 1 if i == stretch_at else 0)
    return box


class Viewer(QMainWindow):
    def __init__(self, cif_path: str | None = None):
        super().__init__()
        self.setWindowTitle("Single-Crystal Diffraction")
        self.resize(1320, 860)

        self.structure: Structure | None = None
        self.result = None
        self.mode = "section"
        self.radiation = "xray"
        self._pending = QTimer(self)
        self._pending.setSingleShot(True)
        self._pending.timeout.connect(self.recompute)

        self._build_menus()
        self._build_ui()
        self.statusBar().showMessage("Open a CIF to begin  ·  File ▸ Open CIF")

        if cif_path:
            self.load(cif_path)

    # -- construction -----------------------------------------------------

    def _build_menus(self):
        m_file = self.menuBar().addMenu("&File")
        act = QAction("&Open CIF…", self)
        act.setShortcut(QKeySequence.StandardKey.Open)
        act.triggered.connect(self.on_open)
        m_file.addAction(act)
        m_file.addSeparator()
        for label, slot in (
            ("Save &image…", self.on_save_png),
            ("Export &reflection table…", self.on_export_table),
        ):
            a = QAction(label, self)
            a.triggered.connect(slot)
            m_file.addAction(a)
        m_file.addSeparator()
        a = QAction("&Quit", self)
        a.setShortcut(QKeySequence.StandardKey.Quit)
        a.triggered.connect(self.close)
        m_file.addAction(a)

        m_sim = self.menuBar().addMenu("&Simulate")
        grp = QActionGroup(self)
        grp.setExclusive(True)
        self.rad_actions = {}
        for rad in RADIATIONS:
            a = QAction(RADIATION_LABEL[rad], self, checkable=True)
            a.setChecked(rad == self.radiation)
            a.triggered.connect(lambda _c, r=rad: self.set_radiation(r))
            grp.addAction(a)
            m_sim.addAction(a)
            self.rad_actions[rad] = a
        m_sim.addSeparator()
        grp2 = QActionGroup(self)
        grp2.setExclusive(True)
        for mode in MODES:
            a = QAction(MODE_LABEL[mode], self, checkable=True)
            a.setChecked(mode == self.mode)
            a.triggered.connect(lambda _c, m=mode: self.set_mode(m))
            grp2.addAction(a)
            m_sim.addAction(a)

        m_help = self.menuBar().addMenu("&Help")
        a = QAction("&About", self)
        a.triggered.connect(self.on_about)
        m_help.addAction(a)

    def _build_ui(self):
        central = QWidget()
        outer = QHBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        panel = QWidget()
        panel.setFixedWidth(280)
        side = QVBoxLayout(panel)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(7)

        # structure
        g = QGroupBox("Structure")
        v = QVBoxLayout(g)
        self.lb_struct = QLabel("no file loaded")
        self.lb_struct.setObjectName("val")
        self.lb_struct.setWordWrap(True)
        v.addWidget(self.lb_struct)
        btn = QPushButton("Open CIF…")
        btn.clicked.connect(self.on_open)
        v.addWidget(btn)
        side.addWidget(g)

        # geometry
        self.g_zone = QGroupBox("Zone axis and layer")
        v = QVBoxLayout(self.g_zone)
        self.sp_u, self.sp_v, self.sp_w = (self._int_spin(x) for x in (0, 0, 1))
        v.addWidget(_row("[u v w]", self.sp_u, self.sp_v, self.sp_w))
        self.sp_layer = self._int_spin(0, lo=-99, hi=99)
        v.addWidget(_row("layer n", self.sp_layer))
        self.lb_law = QLabel("")
        self.lb_law.setObjectName("val")
        v.addWidget(self.lb_law)
        quick = QWidget()
        qh = QHBoxLayout(quick)
        qh.setContentsMargins(0, 2, 0, 0)
        qh.setSpacing(4)
        for u, vv, w in ((0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1)):
            b = QPushButton(f"{u}{vv}{w}")
            b.setFixedWidth(40)
            b.clicked.connect(lambda _c, a=(u, vv, w): self.set_zone(a))
            qh.addWidget(b)
        v.addWidget(quick)
        side.addWidget(self.g_zone)

        # radiation-specific
        self.g_beam = QGroupBox("Beam")
        v = QVBoxLayout(self.g_beam)
        self.sp_dmin = self._float_spin(0.70, 0.05, 20.0, 0.05, " Å")
        v.addWidget(_row("d min", self.sp_dmin))
        self.sp_wl = self._float_spin(1.5406, 0.05, 30.0, 0.01, " Å")
        self.row_wl = _row("λ", self.sp_wl)
        v.addWidget(self.row_wl)
        self.sp_kv = self._float_spin(200.0, 20.0, 1000.0, 10.0, " kV")
        self.row_kv = _row("voltage", self.sp_kv)
        v.addWidget(self.row_kv)
        self.sp_thick = self._float_spin(50.0, 1.0, 2000.0, 5.0, " Å")
        self.row_thick = _row("thickness", self.sp_thick)
        v.addWidget(self.row_thick)
        self.sp_zones = self._int_spin(0, lo=0, hi=6)
        self.row_zones = _row("Laue zones", self.sp_zones)
        v.addWidget(self.row_zones)
        self.sp_ttmax = self._float_spin(120.0, 0.5, 179.0, 5.0, "°")
        self.row_ttmax = _row("2θ max", self.sp_ttmax)
        v.addWidget(self.row_ttmax)
        side.addWidget(self.g_beam)

        # display
        g = QGroupBox("Display")
        v = QVBoxLayout(g)
        self.sl_gain = QSlider(Qt.Orientation.Horizontal)
        self.sl_gain.setRange(0, 60)  # 0..6 decades, tenths
        self.sl_gain.setValue(0)
        self.lb_gain = QLabel("×1")
        self.lb_gain.setObjectName("val")
        self.lb_gain.setFixedWidth(52)
        v.addWidget(_row("contrast", self.sl_gain, self.lb_gain, stretch_at=1))
        self.sl_size = QSlider(Qt.Orientation.Horizontal)
        self.sl_size.setRange(10, 600)
        self.sl_size.setValue(180)
        v.addWidget(_row("spot size", self.sl_size, stretch_at=1))
        self.cb_log = QCheckBox("log intensity")
        self.cb_log.setChecked(True)
        self.cb_labels = QCheckBox("hkl labels")
        self.cb_labels.setChecked(True)
        v.addWidget(_row(self.cb_log, self.cb_labels))
        self.sl_thresh = QSlider(Qt.Orientation.Horizontal)
        self.sl_thresh.setRange(0, 100)
        self.sl_thresh.setValue(35)
        v.addWidget(_row("label above", self.sl_thresh, stretch_at=1))
        self.cm = QComboBox()
        self.cm.addItems(COLORMAPS)
        v.addWidget(_row("colours", self.cm, stretch_at=1))
        side.addWidget(g)

        self.lb_info = QLabel("")
        self.lb_info.setObjectName("val")
        self.lb_info.setWordWrap(True)
        side.addWidget(self.lb_info)
        side.addStretch(1)

        credit = QLabel("© 2026 Miloš Dubajić · MIT")
        credit.setStyleSheet("color: #5d6884; font-size: 10px;")
        side.addWidget(credit)

        outer.addWidget(panel)

        self.fig = Figure(figsize=(8, 8), facecolor=BG)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        outer.addWidget(self.canvas, 1)
        self.setCentralWidget(central)

        for w in (self.sp_u, self.sp_v, self.sp_w, self.sp_layer):
            w.valueChanged.connect(self.request)
        self.sp_zones.valueChanged.connect(self.on_zones_changed)
        for w in (self.sp_dmin, self.sp_wl, self.sp_kv, self.sp_thick, self.sp_ttmax):
            w.valueChanged.connect(self.request)
        for w in (self.sl_gain, self.sl_size, self.sl_thresh):
            w.valueChanged.connect(self.redraw_only)
        for w in (self.cb_log, self.cb_labels):
            w.toggled.connect(self.redraw_only)
        self.cm.currentTextChanged.connect(self.redraw_only)
        self._sync_mode_rows()

    def _int_spin(self, value, lo=-30, hi=30):
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(value)
        s.setFixedWidth(56)
        return s

    def _float_spin(self, value, lo, hi, step, suffix=""):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(4 if step < 0.05 else 2)
        s.setSingleStep(step)
        s.setValue(value)
        s.setSuffix(suffix)
        return s

    # -- state ------------------------------------------------------------

    def set_zone(self, uvw):
        for spin, value in zip((self.sp_u, self.sp_v, self.sp_w), uvw):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.request()

    #: A sensible 2-theta limit per radiation. Electrons need a much smaller
    #: one: at 0.025 Å even a few degrees reaches further into reciprocal
    #: space than 120° of Cu K-alpha does.
    TT_MAX = {"xray": 120.0, "neutron": 120.0, "electron": 4.0}

    def set_radiation(self, rad):
        previous = self.radiation
        self.radiation = rad
        if self.mode == "powder" and rad != previous:
            self.sp_ttmax.blockSignals(True)
            self.sp_ttmax.setValue(self.TT_MAX[rad])
            self.sp_ttmax.blockSignals(False)
        self._sync_mode_rows()
        self.request()

    def set_mode(self, mode):
        self.mode = mode
        # SAED is electron diffraction by definition -- compute_tem uses the
        # electron form factors whatever the menu says -- so switch the menu
        # rather than let it report a radiation that is not being used.
        if mode == "saed" and self.radiation != "electron":
            self.radiation = "electron"
            self.rad_actions["electron"].setChecked(True)
        if mode == "powder":
            self.sp_ttmax.blockSignals(True)
            self.sp_ttmax.setValue(self.TT_MAX[self.radiation])
            self.sp_ttmax.blockSignals(False)
        self._sync_mode_rows()
        self.request()

    def _sync_mode_rows(self):
        for rad, action in self.rad_actions.items():
            action.setEnabled(self.mode != "saed" or rad == "electron")
        self.g_zone.setVisible(self.mode in ("section", "saed"))
        self.row_wl.setVisible(self.mode == "powder" and self.radiation != "electron")
        self.row_kv.setVisible(
            self.mode == "saed" or (self.mode == "powder" and self.radiation == "electron")
        )
        self.row_thick.setVisible(self.mode == "saed")
        self.row_zones.setVisible(self.mode == "saed")
        self.row_ttmax.setVisible(self.mode == "powder")
        self.sp_layer.setEnabled(self.mode == "section")

    def on_zones_changed(self, zones: int):
        """Asking for a Laue ring is asking for the resolution that shows it.

        The rings sit far outside any comfortable d min, so leaving the limit
        alone would answer the request with an unchanged picture. The limit is
        widened to suit and the status bar says so; it is never narrowed, so a
        d min the user chose by hand survives.
        """
        if self.structure is not None and zones > 0:
            uvw = (self.sp_u.value(), self.sp_v.value(), self.sp_w.value())
            try:
                need = d_min_for_zone(self.structure, uvw, zones, self.sp_kv.value())
            except ValueError:
                need = float("inf")
            if need < self.sp_dmin.value():
                self.sp_dmin.blockSignals(True)
                self.sp_dmin.setValue(max(need, self.sp_dmin.minimum()))
                self.sp_dmin.blockSignals(False)
                self.statusBar().showMessage(
                    f"d min widened to {need:.3f} Å to reach Laue zone {zones}")
        self.request()

    def request(self):
        self._pending.start(60)

    # -- loading ----------------------------------------------------------

    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CIF", "", "Crystallographic Information File (*.cif);;All files (*)"
        )
        if path:
            self.load(path)

    def load(self, path):
        try:
            doc = read_cif(path)
            self.structure = Structure.from_cif(doc)
        except (CifError, ValueError, OSError) as exc:
            QMessageBox.warning(self, "Cannot read this CIF", str(exc))
            self.statusBar().showMessage(f"failed: {exc}")
            return
        self.lb_struct.setText(self.structure.describe())
        note = ""
        if doc.blocks_in_file > 1:
            note = f"  ·  {doc.blocks_in_file} structures in the file, using the first"
        self.statusBar().showMessage(f"loaded {os.path.basename(path)}{note}")
        self.recompute()

    # -- compute and draw -------------------------------------------------

    def recompute(self):
        if self.structure is None:
            return
        uvw = (self.sp_u.value(), self.sp_v.value(), self.sp_w.value())
        try:
            if self.mode == "section":
                self.result = compute_section(
                    self.structure, uvw=uvw, layer=self.sp_layer.value(),
                    d_min=self.sp_dmin.value(), radiation=self.radiation,
                )
                self.lb_law.setText(f"zone law   {self.result.zone_law}")
            elif self.mode == "saed":
                self.result = compute_tem(
                    self.structure, uvw=uvw, kv=self.sp_kv.value(),
                    thickness=self.sp_thick.value(), d_min=self.sp_dmin.value(),
                    max_zone=self.sp_zones.value(),
                )
                self.lb_law.setText("")
            else:
                lam = (
                    electron_wavelength(self.sp_kv.value())
                    if self.radiation == "electron"
                    else self.sp_wl.value()
                )
                self.result = compute_powder(
                    self.structure, wavelength=lam, radiation=self.radiation,
                    two_theta_max=self.sp_ttmax.value(),
                )
        except (ValueError, KeyError) as exc:
            self.result = None
            self.ax.clear()
            self.ax.set_facecolor(PLOT_BG)
            self.ax.text(0.5, 0.5, str(exc), color=BAD, ha="center", va="center",
                         wrap=True, transform=self.ax.transAxes)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw_idle()
            self.statusBar().showMessage(str(exc))
            return
        self.redraw_only()

    def redraw_only(self):
        gain = 10.0 ** (self.sl_gain.value() / 10.0)
        self.lb_gain.setText(f"×{gain:.0f}" if gain < 1e4 else f"×1e{gain:.0e}"[-4:])
        if self.result is None:
            return
        if self.mode == "powder":
            self._draw_powder()
        else:
            self._draw_spots(gain)
        self.canvas.draw_idle()

    def _draw_spots(self, gain):
        r = self.result
        ax = self.ax
        ax.clear()
        ax.set_facecolor(PLOT_BG)
        ax.set_aspect("equal")

        if len(r) == 0:
            ax.text(0.5, 0.5, "no reflections in this section\n"
                              "try a lower d min or a different layer",
                    color=WARN, ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            self.statusBar().showMessage("0 reflections")
            return

        value = disp.stretch(r.intensity, gain=gain, log=self.cb_log.isChecked())
        keep = value > 1e-3
        x, y, value = r.x[keep], r.y[keep], value[keep]
        hkl = r.hkl[keep]

        sizes = disp.spot_sizes(value, scale=self.sl_size.value())
        cmap = matplotlib.colormaps[self.cm.currentText()]
        ax.scatter(x, y, s=sizes, c=value, cmap=cmap, vmin=0, vmax=1,
                   linewidths=0, zorder=3)
        # The direct beam, which is not a reflection but locates the origin.
        ax.scatter([0], [0], s=42, facecolors="none", edgecolors=DIM,
                   linewidths=1.0, zorder=4)

        if self.cb_labels.isChecked():
            thr = self.sl_thresh.value() / 100.0
            shown = 0
            for xi, yi, vi, h in zip(x, y, value, hkl):
                if vi < thr or shown > 400:
                    continue
                ax.annotate(disp.format_hkl(h), (xi, yi), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=6.5, color=DIM,
                            zorder=5)
                shown += 1

        lim = max(np.abs(x).max(), np.abs(y).max()) * 1.12 + 0.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        for spine in ax.spines.values():
            spine.set_color("#2a3145")
        ax.tick_params(colors=DIM, labelsize=8)
        ax.set_xlabel("Å⁻¹", color=DIM, fontsize=8)

        self._draw_axes_key(ax, r, lim)
        self._status_for_spots(r, keep)

    def _draw_axes_key(self, ax, r, lim):
        """The two in-plane reciprocal directions, drawn as a corner key.

        This is what tells you which way the picture is oriented; without it a
        section of a low-symmetry cell is unreadable.
        """
        f = 0.20 * lim
        for g, colour, dx in ((r.g1, ACCENT, 1.0), (r.g2, GOOD, 0.0)):
            vec = self.structure.B @ np.asarray(g, dtype=float)
            gx, gy = float(vec @ r.x_axis), float(vec @ r.y_axis)
            n = np.hypot(gx, gy)
            if n < 1e-9:
                continue
            gx, gy = gx / n * f, gy / n * f
            ax.annotate("", xy=(-lim * 0.80 + gx, -lim * 0.80 + gy),
                        xytext=(-lim * 0.80, -lim * 0.80),
                        arrowprops=dict(arrowstyle="->", color=colour, lw=1.2),
                        zorder=6)
            ax.text(-lim * 0.80 + gx * 1.28, -lim * 0.80 + gy * 1.28,
                    disp.format_hkl(g), color=colour, fontsize=7,
                    ha="center", va="center", zorder=6)

        # Scale bar, so a distance on screen means something.
        step = disp.nice_step(2 * lim)
        x0, y0 = lim * 0.94 - step, -lim * 0.92
        ax.plot([x0, x0 + step], [y0, y0], color=DIM, lw=1.4, zorder=6)
        ax.text(x0 + step / 2, y0 + lim * 0.03, f"{step:g} Å⁻¹", color=DIM,
                fontsize=7, ha="center", zorder=6)

    def _status_for_spots(self, r, keep):
        unit = UNITS[getattr(r, "radiation", "electron")]
        head = f"{RADIATION_LABEL[self.radiation]}  ·  {int(keep.sum())} of {len(r)} shown"
        if self.mode == "section":
            self.lb_info.setText(
                f"zone [{' '.join(map(str, r.uvw))}]   layer {r.layer}\n"
                f"{r.zone_law}\n"
                f"layer height {r.height:+.4f} Å⁻¹\n"
                f"axes {disp.format_hkl(r.g1)} and {disp.format_hkl(r.g2)}\n"
                f"|F|² in {unit}²"
            )
            extra = ""
            if r.zone_factor > 1:
                extra = f"  ·  zone reduced by {r.zone_factor}"
            self.statusBar().showMessage(head + extra)
        else:
            rings = ", ".join(
                "—" if v is None else f"{v:.2f}" for v in r.zone_radii
            )
            visible = r.zones_visible()
            self.lb_info.setText(
                f"beam ∥ [{' '.join(map(str, r.uvw))}]\n"
                f"λ {r.wavelength:.5f} Å at {r.kv:.0f} kV\n"
                f"thickness {r.thickness:.0f} Å\n"
                f"Laue rings at {rings} Å⁻¹\n"
                f"{visible} of {len(r.zone_radii)} inside d min"
            )
            note = ""
            if visible < 2 and len(r.zone_radii) > 1 and r.zone_radii[1]:
                need = 2 * np.pi / r.zone_radii[1]
                note = f"  ·  first Laue ring needs d min ≤ {need:.3f} Å"
            self.statusBar().showMessage(head + note)

    def _draw_powder(self):
        p = self.result
        ax = self.ax
        ax.clear()
        ax.set_facecolor(PLOT_BG)
        ax.set_aspect("auto")
        ax.plot(p.x, p.y, color=ACCENT, lw=1.1)
        ax.fill_between(p.x, 0, p.y, color=ACCENT, alpha=0.16)

        if self.cb_labels.isChecked():
            thr = max(self.sl_thresh.value(), 1)
            for tt, I, h in zip(p.two_theta, p.intensity, p.hkl):
                if I < thr:
                    continue
                ax.annotate(disp.format_hkl(h), (tt, I), textcoords="offset points",
                            xytext=(0, 4), ha="center", fontsize=6.5, color=DIM)

        ax.set_xlim(0, p.x.max())
        ax.set_ylim(0, 108)
        ax.set_xlabel("2θ  (degrees)", color=DIM, fontsize=9)
        ax.set_ylabel("intensity", color=DIM, fontsize=9)
        ax.tick_params(colors=DIM, labelsize=8)
        ax.grid(color="#1c2334", lw=0.6)
        for spine in ax.spines.values():
            spine.set_color("#2a3145")

        self.lb_info.setText(
            f"{RADIATION_LABEL[self.radiation]}\n"
            f"λ {p.wavelength:.5f} Å\n"
            f"{len(p)} peaks to {p.x.max():.0f}°\n"
            f"strongest {p.hkl[int(np.argmax(p.intensity))]}"
        )
        self.statusBar().showMessage(
            f"{RADIATION_LABEL[self.radiation]}  ·  {len(p)} peaks  ·  "
            f"Lorentz" + (" and polarisation applied" if self.radiation == "xray"
                          else " applied, no polarisation term")
        )

    # -- export -----------------------------------------------------------

    def _stem(self):
        base = self.structure.name if self.structure else "pattern"
        if self.mode == "powder":
            return f"{base}_powder_{self.radiation}"
        uvw = "".join(str(s.value()) for s in (self.sp_u, self.sp_v, self.sp_w))
        if self.mode == "saed":
            return f"{base}_saed_{uvw}"
        return f"{base}_zone{uvw}_n{self.sp_layer.value()}_{self.radiation}"

    def on_save_png(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save image", self._stem() + ".png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if not path:
            return
        self.fig.savefig(path, dpi=200, facecolor=BG)
        self.statusBar().showMessage(f"wrote {path}")

    def on_export_table(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export reflections", self._stem() + ".txt",
            "Text (*.txt);;CSV (*.csv)")
        if not path:
            return
        r = self.result
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"# {self.structure.name}\n")
            fh.write(f"# radiation      {RADIATION_LABEL[self.radiation]}\n")
            fh.write(f"# |F|^2 unit     {UNITS[self.radiation]}^2\n")
            if self.mode == "powder":
                fh.write(f"# wavelength     {r.wavelength:.6f} A\n")
                fh.write("#\n#   2theta        d      I(rel)  mult   h   k   l\n")
                for i in range(len(r)):
                    h, k, l = r.hkl[i]
                    fh.write(f"{r.two_theta[i]:10.4f} {r.d[i]:9.5f} "
                             f"{r.intensity[i]:9.3f} {r.multiplicity[i]:5d} "
                             f"{h:4d}{k:4d}{l:4d}\n")
            else:
                fh.write(f"# zone axis      {tuple(int(v) for v in r.uvw)}\n")
                if self.mode == "section":
                    fh.write(f"# layer          {r.layer}   ({r.zone_law})\n")
                else:
                    fh.write(f"# voltage        {r.kv:.1f} kV, lambda "
                             f"{r.wavelength:.6f} A\n")
                    fh.write(f"# thickness      {r.thickness:.1f} A\n")
                fh.write("#\n#   h   k   l        d        |Q|          I"
                         + ("          s_g  zone\n" if self.mode == "saed" else "\n"))
                order = np.argsort(-r.intensity)
                for i in order:
                    h, k, l = r.hkl[i]
                    line = (f"{h:5d}{k:4d}{l:4d} {r.d[i]:9.5f} {r.q[i]:10.5f} "
                            f"{r.intensity[i]:12.5g}")
                    if self.mode == "saed":
                        line += f" {r.s_g[i]:12.6f} {int(r.laue_zone[i]):5d}"
                    fh.write(line + "\n")
        self.statusBar().showMessage(f"wrote {len(r)} reflections to {path}")

    def on_about(self):
        QMessageBox.about(
            self, "Single-Crystal Diffraction",
            "<b>Single-Crystal Diffraction</b><br>"
            "Kinematic patterns from any CIF, for X-rays, neutrons and electrons."
            "<br><br>The symmetry in the file is expanded to P1 before the "
            "structure factor sum, so a CIF giving only an asymmetric unit is "
            "handled correctly.<br><br>"
            "© 2026 Miloš Dubajić · MIT<br>"
            "<a href='https://github.com/dubajicmilos/xrays-on-detector'>"
            "github.com/dubajicmilos/xrays-on-detector</a>")


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    path = None
    for a in argv[1:]:
        if not a.startswith("-"):
            path = a
    app = QApplication(argv)
    app.setStyleSheet(STYLE)
    win = Viewer(path)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
