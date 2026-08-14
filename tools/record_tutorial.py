"""Record a narrated walkthrough of either browser app.

    python tools/devserver.py 8777          # in another shell
    python tools/record_tutorial.py game        [--out FILE]
    python tools/record_tutorial.py single      [--out FILE]

Playwright drives the real page with a real mouse, so the controls respond as
they would for a visitor. The pointer is drawn by an injected overlay: a
headless browser composites no system cursor, so without it the clicks look
like the page moving on its own.

ffmpeg then pads a strip under the frame and burns the narration in over the
range each step actually took. The ranges are measured during the run rather
than assumed, because a step that types into four fields takes far longer than
one that ticks a box.

Needs playwright (with `playwright install chromium`), pillow and ffmpeg. None
of them are dependencies of the project itself; this is a developer tool.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 1280, 720
STRIP = 132

# The pointer overlay. A headless browser draws no cursor into the video, and
# the click ring is what makes a press legible when the control it lands on
# does not visibly change.
CURSOR_JS = r"""
(() => {
  const mount = () => {
    if (document.getElementById('__cursor')) return;
    const c = document.createElement('div');
    c.id = '__cursor';
    c.style.cssText =
      'position:fixed;left:0;top:0;width:26px;height:26px;z-index:2147483647;' +
      'pointer-events:none;will-change:transform;transform:translate(-100px,-100px)';
    c.innerHTML =
      '<svg width="26" height="26" viewBox="0 0 26 26">' +
      '<path d="M4 2 L4 20 L9 15.5 L12.2 22.5 L15.4 21 L12.2 14.2 L19 14 Z" ' +
      'fill="#ffffff" stroke="#101018" stroke-width="1.6" stroke-linejoin="round"/></svg>';
    document.documentElement.appendChild(c);

    const ring = document.createElement('div');
    ring.id = '__ring';
    ring.style.cssText =
      'position:fixed;left:0;top:0;width:34px;height:34px;margin:-17px 0 0 -17px;' +
      'border-radius:50%;border:2px solid #6f9ee0;opacity:0;z-index:2147483646;' +
      'pointer-events:none;will-change:transform,opacity';
    document.documentElement.appendChild(ring);

    addEventListener('mousemove', (e) => {
      c.style.transform = `translate(${e.clientX}px,${e.clientY}px)`;
    }, true);
    addEventListener('mousedown', (e) => {
      ring.style.transition = 'none';
      ring.style.transform = `translate(${e.clientX}px,${e.clientY}px) scale(0.5)`;
      ring.style.opacity = '1';
      requestAnimationFrame(() => {
        ring.style.transition = 'transform .45s ease-out, opacity .45s ease-out';
        ring.style.transform = `translate(${e.clientX}px,${e.clientY}px) scale(1.5)`;
        ring.style.opacity = '0';
      });
    }, true);
  };
  if (document.documentElement) mount();
  document.addEventListener('DOMContentLoaded', mount);
})();
"""


# --------------------------------------------------------------------- driving


class Driver:
    """Mouse and keyboard helpers that move visibly rather than teleporting."""

    def __init__(self, page):
        self.page = page

    def _box(self, sel):
        el = self.page.locator(sel).first
        # A short timeout: a control that is not there is a script to fix, not
        # something to spend thirty seconds per step waiting for.
        el.scroll_into_view_if_needed(timeout=4000)
        self.page.wait_for_timeout(90)
        box = el.bounding_box()
        if box is None:
            raise RuntimeError(f"{sel} has no box (hidden?)")
        return box

    def point(self, sel, steps=14, fx=0.5, fy=0.5):
        b = self._box(sel)
        x = b["x"] + b["width"] * fx
        y = b["y"] + b["height"] * fy
        self.page.mouse.move(x, y, steps=steps)
        self.page.wait_for_timeout(90)
        return x, y

    def click(self, sel, **kw):
        self.point(sel, **kw)
        self.page.mouse.down()
        self.page.wait_for_timeout(60)
        self.page.mouse.up()
        self.page.wait_for_timeout(150)

    def type_into(self, sel, text, delay=45):
        """Click a field, clear it, type. Number inputs fire on every
        keystroke, so the picture updates as the digits land."""
        self.click(sel)
        self.page.keyboard.press("Control+A")
        self.page.keyboard.type(str(text), delay=delay)
        self.page.wait_for_timeout(200)

    def choose(self, sel, value=None, index=None, label=None):
        self.point(sel)
        if index is not None:
            self.page.select_option(sel, index=index)
        elif label is not None:
            self.page.select_option(sel, label=label)
        else:
            self.page.select_option(sel, value)
        self.page.wait_for_timeout(240)

    def check(self, sel, on=True):
        el = self.page.locator(sel).first
        if el.is_checked() != on:
            self.click(sel)

    def drag_range(self, sel, to, steps=26):
        """Drag a range input from where its handle is to a target value."""
        b = self._box(sel)
        el = self.page.locator(sel).first
        lo = float(el.get_attribute("min") or 0)
        hi = float(el.get_attribute("max") or 100)
        cur = float(el.input_value())
        pad = 7  # the thumb cannot reach the very ends of the track
        span = b["width"] - 2 * pad

        def x_of(v):
            return b["x"] + pad + span * (v - lo) / (hi - lo)

        y = b["y"] + b["height"] / 2
        self.page.mouse.move(x_of(cur), y, steps=12)
        self.page.wait_for_timeout(90)
        self.page.mouse.down()
        self.page.mouse.move(x_of(to), y, steps=steps)
        self.page.wait_for_timeout(110)
        self.page.mouse.up()
        self.page.wait_for_timeout(170)

    def orbit(self, dx, dy, steps=30):
        b = self._box("#stage")
        cx, cy = b["x"] + b["width"] / 2, b["y"] + b["height"] / 2
        self.page.mouse.move(cx, cy, steps=16)
        self.page.mouse.down()
        self.page.mouse.move(cx + dx, cy + dy, steps=steps)
        self.page.mouse.up()
        self.page.wait_for_timeout(250)

    def wheel(self, sel, dy, times=3):
        self.point(sel)
        for _ in range(times):
            self.page.mouse.wheel(0, dy)
            self.page.wait_for_timeout(140)
        self.page.wait_for_timeout(200)

    def hover_detector(self, fx=0.42, fy=0.46):
        self.point("#detCanvas", fx=fx, fy=fy, steps=26)
        self.page.wait_for_timeout(500)


MOTOR = "#motorRows .motor:nth-child({}) input[type=range]"
MOTOR_NUM = "#motorRows .motor:nth-child({}) input[type=number]"
MOTOR_RUN = "#motorRows .motor:nth-child({}) button"
ROT = "#rotRows .motor:nth-child({}) input[type=range]"
MU, OMEGA, CHI, PHI, DELTA, GAMMA = 1, 2, 3, 4, 5, 6


def game_steps(d: Driver):
    """The Game of Diffraction, every control group in the panel."""
    p = d.page
    yield ("The Game of Diffraction: a six-circle diffractometer in the browser.\n"
           "Three panes — the controls, the instrument in 3D, and the detector face.",
           lambda: p.wait_for_timeout(2600))

    yield ("The centre pane is the instrument seen from upstream, looking along the\n"
           "beam. Drag to orbit it. The cyan rays are the diffracted beams.",
           lambda: (d.orbit(150, 60), d.orbit(-90, -30)))

    yield ("Right-drag pans, and the wheel zooms.",
           lambda: d.wheel("#stage", -220, 3))

    # -- beam and detector
    yield ("Beam and detector. Wavelength and energy are locked together — type\n"
           "one and the other follows. Detune the beam and the detector empties.",
           lambda: d.type_into("#wl", "0.70"))

    yield ("It opens at 19.010 keV for a reason: there λ is exactly a/9 for this\n"
           "cell, so twelve reflections sit precisely on the Ewald sphere.",
           lambda: d.type_into("#energy", "19.010"))

    yield ("Pick a real detector. The preset sets pixel size and pixel count\n"
           "together; both stay editable underneath.",
           lambda: d.choose("#detPreset", index=3))

    yield ("Distance decides how much of reciprocal space you collect. Far away\n"
           "catches little; closer buys angular range and costs spatial resolution.",
           lambda: (d.type_into("#distance", "300"), p.wait_for_timeout(900),
                    d.type_into("#distance", "120")))

    yield ("Preview bin trades detector pixels for frame rate. Raise it while\n"
           "dragging a motor, then set it back to 1 for a full-resolution frame.",
           lambda: (d.type_into("#bin", "4"), p.wait_for_timeout(600),
                    d.type_into("#bin", "1")))

    # -- sample
    #
    # Peak width comes first and is then left wide for the rest of the tour. At
    # the default only reflections landing exactly on the sphere light up, so
    # every later step would play against a nearly empty detector.
    yield ("Peak width is the size of a Bragg peak in reciprocal space. Widening it\n"
           "keeps reflections lit further from the exact condition, as mosaic spread does.",
           lambda: d.type_into("#sigma", "0.025"))

    yield ("The sample. Four real structures are bundled, and the panel reports the\n"
           "cell it read — here a 144-atom perovskite, so the pattern is dense.",
           lambda: d.choose("#structure", index=2))

    yield ("Load CIF reads your own file in the browser — it never leaves your\n"
           "machine. The reader applies the symmetry operators the file lists.",
           lambda: (d.point("#cifPick"), p.wait_for_timeout(1400)))

    # -- motors
    yield ("The six circles. Four move the sample — mu, omega, chi, phi — and two\n"
           "swing the detector arm: delta and gamma.",
           lambda: (d.point(MOTOR.format(MU)), p.wait_for_timeout(900)))

    yield ("Drag omega to rock the crystal. Reflections light up and fade as lattice\n"
           "planes pass through the Bragg condition.",
           lambda: (d.drag_range(MOTOR.format(OMEGA), 26),
                    d.drag_range(MOTOR.format(OMEGA), -14)))

    yield ("Chi and phi reach the reflections a single rotation cannot.",
           lambda: (d.drag_range(MOTOR.format(CHI), 22),
                    d.drag_range(MOTOR.format(PHI), 35)))

    yield ("Delta and gamma point the detector arm, so the panel follows the\n"
           "diffracted beam rather than the direct one.",
           lambda: (d.drag_range(MOTOR.format(DELTA), 18),
                    d.drag_range(MOTOR.format(GAMMA), 12)))

    yield ("Every circle has a run button for continuous rotation, and Speed sets\n"
           "how fast. Stop all halts them; Zero all returns to the datum.",
           lambda: (d.click(MOTOR_RUN.format(OMEGA)), p.wait_for_timeout(2600),
                    d.click("#stopAll"), p.wait_for_timeout(400), d.click("#zeroAll")))

    # -- orientation
    yield ("Crystal orientation. rx, ry and rz tilt the crystal on its mount,\n"
           "which is the free rotation a real sample has after it is glued down.",
           lambda: (d.drag_range(ROT.format(1), 30), d.drag_range(ROT.format(3), -40)))

    yield ("Better: point a crystal direction where you want it. Send (1 1 1)\n"
           "along the beam and press Go.",
           lambda: (d.type_into("#a1h", "1"), d.type_into("#a1k", "1"),
                    d.type_into("#a1l", "1"), d.click("#a1go")))

    yield ("One direction leaves a rotation still free about it. The second block\n"
           "spends it: bring (0 0 1) as near the vertical as it will go.",
           lambda: d.click("#a2go"))

    yield ("The UB matrix is shown in three conventions, and Copy puts it on the\n"
           "clipboard for CrysAlis or your own code. Reset returns U to the identity.",
           lambda: (d.choose("#ubConv", value="1/d"), p.wait_for_timeout(700),
                    d.choose("#ubConv", value="lambda"), p.wait_for_timeout(700),
                    d.click("#ubCopy"), p.wait_for_timeout(500), d.click("#resetU")))

    # -- drive to a reflection
    yield ("Drive to a reflection. Type Miller indices and press Find omega:\n"
           "you get every omega that satisfies Bragg with the other circles held.",
           lambda: (d.type_into("#dh", "2"), d.type_into("#dk", "1"),
                    d.type_into("#dl", "1"), d.click("#findOmega")))

    yield ("Pick a solution, drive there, then aim the detector to swing the arm\n"
           "onto the diffracted beam. The spot lands on the beam centre.",
           lambda: (d.click("#driveThere"), p.wait_for_timeout(900), d.click("#aimDet")))

    yield ("If no omega reaches a reflection — a limit of rotating about one axis,\n"
           "not a bug — the page says so and offers a chi that brings it into range.",
           lambda: (d.type_into("#dh", "9"), d.type_into("#dk", "9"),
                    d.type_into("#dl", "9"), d.click("#findOmega"),
                    p.wait_for_timeout(1400)))

    # -- geometry
    yield ("Scattering geometry. In reflection the sample carries a surface normal,\n"
           "so every reflection pointing into the bulk is blocked — the panel empties.",
           lambda: (d.click("#resetU"),
                    d.click('input[name="mode"][value="reflection"]'),
                    d.click("#mountFlat"), p.wait_for_timeout(900)))

    yield ("The incidence angle alpha says why: at zero the beam runs along the\n"
           "surface. Rock omega to lift it and the pattern comes back.",
           lambda: (d.drag_range(MOTOR.format(OMEGA), 12),
                    p.wait_for_timeout(1200),
                    d.click('input[name="mode"][value="transmission"]')))

    # -- display
    yield ("Display. Circles, rays, floor and the crystal a b c axes can each be\n"
           "turned off to clear the 3D view.",
           lambda: (d.click("#showRings"), d.click("#showFloor"),
                    p.wait_for_timeout(700), d.click("#showAxes"),
                    p.wait_for_timeout(700), d.click("#showRings"),
                    d.click("#showFloor"), d.click("#showAxes")))

    yield ("Rays that miss draws the reflections that never reach the panel, which\n"
           "is how you see why one is absent rather than merely that it is.",
           lambda: (d.click("#showMissed"), p.wait_for_timeout(1800),
                    d.click("#showMissed")))

    yield ("hkl labels name each spot. Log scale and the colour map decide how the\n"
           "detector image is stretched and coloured.",
           lambda: (d.click("#showLabels"), p.wait_for_timeout(900),
                    d.click("#showLabels"), d.choose("#cmap", index=1),
                    p.wait_for_timeout(600), d.choose("#cmap", index=3)))

    yield ("Contrast is the one to reach for. It brings up reflections far weaker\n"
           "than the brightest, over six decades.",
           lambda: (d.drag_range("#gain", 3.4), p.wait_for_timeout(1200),
                    d.drag_range("#gain", 0.8)))

    yield ("The detector pane zooms independently, and hovering a pixel reports its\n"
           "position, |Q| and d-spacing.",
           lambda: (d.click("#detZoomIn"), d.click("#detZoomIn"),
                    d.hover_detector(0.45, 0.45), d.click("#detZoomReset")))

    yield ("Everything here runs in your browser, and the physics is the same\n"
           "package that drives the desktop simulator. Source on GitHub.",
           lambda: p.wait_for_timeout(2600))


def single_steps(d: Driver):
    """The single-crystal CIF viewer."""
    p = d.page
    yield ("Single-Crystal Diffraction: kinematic patterns from any CIF.\n"
           "It opens on CsPbBr3, the [001] section — the hk0 plane.",
           lambda: p.wait_for_timeout(2400))

    yield ("Pick any zone axis. [110] is a diagonal cut: every reflection with\n"
           "h + k = 0. The arrows bottom-left name the in-plane directions.",
           lambda: d.click('.quickzone button[data-zone="1,1,0"]'))

    yield ("Layers step along the zone axis. [100] at layer 3 is the 3kl section,\n"
           "not the 0kl one — the zone law reads h = 3.",
           lambda: (d.click('.quickzone button[data-zone="1,0,0"]'),
                    d.type_into("#layer", "3"), p.wait_for_timeout(900),
                    d.type_into("#layer", "0")))

    yield ("Or type any indices you like, including a general direction such as [123].",
           lambda: (d.type_into("#zu", "1"), d.type_into("#zv", "2"),
                    d.type_into("#zw", "3"), p.wait_for_timeout(900),
                    d.click('.quickzone button[data-zone="0,0,1"]')))

    yield ("Switch the radiation. Neutrons scatter off nuclei, so there is no\n"
           "form-factor falloff: high-angle reflections stay strong.",
           lambda: d.choose("#radiation", value="neutron"))

    yield ("Electrons sit between the two. d min sets how far out the pattern runs.",
           lambda: (d.choose("#radiation", value="electron"),
                    d.type_into("#dmin", "0.6")))

    yield ("Contrast is a six-decade log stretch, and the spot size and labels are\n"
           "yours to set.",
           lambda: (d.drag_range("#gain", 26), p.wait_for_timeout(900),
                    d.drag_range("#spotSize", 16), d.drag_range("#gain", 0)))

    yield ("Electron diffraction takes the same zone through a curved Ewald sphere,\n"
           "with a sinc² relrod, so spots fade towards the edge.",
           lambda: (d.choose("#mode", value="saed"), p.wait_for_timeout(900)))

    yield ("Ask for a Laue zone and d min widens itself to reach it: the first ring\n"
           "sits near 23 Å⁻¹, far outside any comfortable limit.",
           lambda: (d.type_into("#zones", "1"), p.wait_for_timeout(1200)))

    yield ("The powder pattern sums reflections sharing a d-spacing, so multiplicity\n"
           "is counted rather than looked up.",
           lambda: (d.choose("#mode", value="powder"),
                    d.choose("#radiation", value="xray")))

    yield ("Upload your own CIF. The symmetry is expanded to P1 before the structure\n"
           "factors are summed, so an asymmetric unit is handled correctly.",
           lambda: (d.choose("#mode", value="section"), upload_rutile(p),
                    p.wait_for_timeout(1600)))

    yield ("Rutile, from a file listing 2 sites and 16 operators. The h00 row shows\n"
           "only 200, 400, 600 — the P4₂/mnm absences, drawn correctly.",
           lambda: p.wait_for_timeout(2600))

    yield ("Save the picture, or export every reflection with d, |Q| and |F|².\n"
           "Nothing you upload leaves your machine.",
           lambda: (d.point("#savePng"), p.wait_for_timeout(1000),
                    d.point("#saveTable"), p.wait_for_timeout(1400)))


RUTILE = """# generated using pymatgen
data_TiO2
_symmetry_space_group_name_H-M   P4_2/mnm
_cell_length_a   4.59370000
_cell_length_b   4.59370000
_cell_length_c   2.95870000
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   136
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
  2  '-x, -y, -z'
  3  '-y+1/2, x+1/2, z+1/2'
  4  'y+1/2, -x+1/2, -z+1/2'
  5  '-x, -y, z'
  6  'x, y, -z'
  7  'y+1/2, -x+1/2, z+1/2'
  8  '-y+1/2, x+1/2, -z+1/2'
  9  'x+1/2, -y+1/2, -z+1/2'
  10  '-x+1/2, y+1/2, z+1/2'
  11  '-y, -x, -z'
  12  'y, x, z'
  13  '-x+1/2, y+1/2, -z+1/2'
  14  'x+1/2, -y+1/2, z+1/2'
  15  'y, x, -z'
  16  '-y, -x, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Ti  Ti0  0.00000000  0.00000000  0.00000000  1
  O  O1  0.19522000  0.80478000  0.50000000  1
"""


def upload_rutile(page):
    import tempfile

    path = os.path.join(tempfile.gettempdir(), "rutile_tutorial.cif")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(RUTILE)
    page.set_input_files("#cifFile", path)


# ------------------------------------------------------------------- captions


def caption_png(text, index, out):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (W, STRIP), "#12151f")
    dr = ImageDraw.Draw(img)
    dr.line([(0, 0), (W, 0)], fill="#2a3145", width=2)
    font = None
    for name in ("segoeui.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            font = ImageFont.truetype(name, 21)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    lines = text.split("\n")
    y = (STRIP - len(lines) * 30) // 2
    for line in lines:
        dr.text(((W - dr.textlength(line, font=font)) / 2, y), line,
                font=font, fill="#d6dcec")
        y += 30
    path = os.path.join(out, f"cap{index:03d}.png")
    img.save(path)
    return path


# ----------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("app", choices=["game", "single"])
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--out", default=None)
    ap.add_argument("--crf", type=int, default=26)
    # Recording and encoding are separable because each can outlast a shell's
    # patience on a long walkthrough, and because a re-worded caption should
    # not mean driving the whole app again.
    ap.add_argument("--record-only", action="store_true")
    ap.add_argument("--encode-only", action="store_true")
    # The full six-circle walkthrough runs well past ten minutes, so it is
    # recorded in slices and the finished parts joined. A slice that goes wrong
    # is also re-recorded on its own rather than starting the whole take again.
    ap.add_argument("--from", dest="lo", type=int, default=0)
    ap.add_argument("--to", dest="hi", type=int, default=10**6)
    ap.add_argument("--part", type=int, default=None,
                    help="tag for this slice's output files")
    ap.add_argument("--join", nargs="*", default=None,
                    help="concatenate the named part files into --out")
    args = ap.parse_args()

    out = os.path.join(ROOT, "tools", "_video")
    os.makedirs(out, exist_ok=True)
    stem = "game_of_diffraction" if args.app == "game" else "single_crystal"
    tag = "" if args.part is None else f"_p{args.part}"
    target = args.out or os.path.join(out, f"{stem}{tag}_tutorial.mp4")
    spans_path = os.path.join(out, f"{stem}{tag}_spans.json")

    if args.join is not None:
        # The parts share an encoder and a size, so the concat demuxer can
        # stream-copy them and nothing is re-compressed.
        listing = os.path.join(out, "join.txt")
        with open(listing, "w", encoding="utf-8") as fh:
            for part in args.join:
                path = part if os.path.isabs(part) else os.path.join(out, part)
                if not os.path.isfile(path):
                    raise SystemExit(f"missing part: {path}")
                fh.write(f"file '{path.replace(chr(92), '/')}'\n")
        r = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listing,
             "-c", "copy", "-movflags", "+faststart", target],
            capture_output=True, text=True)
        if r.returncode:
            print(r.stderr[-2500:])
            return 1
        print(f"wrote {target}  ({os.path.getsize(target) / 1e6:.2f} MB)")
        return 0

    if args.encode_only:
        import json

        with open(spans_path, encoding="utf-8") as fh:
            saved = json.load(fh)
        return encode(saved["video"], [tuple(s) for s in saved["spans"]],
                      target, out, args.crf)

    from playwright.sync_api import sync_playwright

    url = f"http://localhost:{args.port}/" + ("" if args.app == "game" else "sc/")

    spans = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--force-device-scale-factor=1"])
        ctx = browser.new_context(
            viewport={"width": W, "height": H},
            record_video_dir=out,
            record_video_size={"width": W, "height": H},
            device_scale_factor=1,
        )
        ctx.add_init_script(CURSOR_JS)
        page = ctx.new_page()
        t0 = time.monotonic()
        page.goto(url)
        page.wait_for_selector("#boot", state="detached", timeout=30000)
        page.wait_for_timeout(900)
        page.mouse.move(W * 0.62, H * 0.5)

        d = Driver(page)
        steps = list(game_steps(d) if args.app == "game" else single_steps(d))
        print(f"{len(steps)} steps defined; recording {args.lo}.."
              f"{min(args.hi, len(steps)) - 1}", flush=True)
        for i, (text, action) in enumerate(steps):
            if not (args.lo <= i < args.hi):
                continue
            start = time.monotonic() - t0
            try:
                action()
            except Exception as exc:  # a step that cannot run must not lose the take
                print(f"  step {i} ({text.splitlines()[0][:40]}...): {exc}")
            # Hold long enough to read the caption. Some steps are over in a
            # click and some type into four fields, so the pad is whatever is
            # left of a dwell set by the length of the text, not a constant.
            want = 1.7 + len(text) / 21.0
            spent = (time.monotonic() - t0) - start
            if spent < want:
                page.wait_for_timeout(int((want - spent) * 1000))
            spans.append((start, time.monotonic() - t0, text))
            print(f"  {i:2d}  {start:6.1f}-{spans[-1][1]:6.1f}s  "
                  f"{text.splitlines()[0][:56]}", flush=True)

        page.wait_for_timeout(500)
        video = page.video.path()
        ctx.close()
        browser.close()

    import json

    with open(spans_path, "w", encoding="utf-8") as fh:
        json.dump({"video": video, "spans": spans}, fh)
    print(f"\nrecorded {spans[-1][1]:.0f} s -> {video}", flush=True)
    if args.record_only:
        print(f"now: python tools/record_tutorial.py {args.app} --encode-only")
        return 0
    return encode(video, spans, target, out, args.crf)


def encode(video, spans, target, out, crf):
    caps = [caption_png(t, i, out) for i, (_, _, t) in enumerate(spans)]
    inputs = ["-i", video]
    for c in caps:
        inputs += ["-i", c]
    filters = [f"[0:v]pad={W}:{H + STRIP}:0:0:color=#12151f[bg]"]
    prev = "bg"
    for i, (a, b, _) in enumerate(spans):
        nxt = f"v{i}"
        filters.append(
            f"[{prev}][{i + 1}:v]overlay=0:{H}:"
            f"enable='between(t,{a:.2f},{b:.2f})'[{nxt}]"
        )
        prev = nxt
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters),
           "-map", f"[{prev}]", "-c:v", "libx264", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-crf", str(crf),
           "-movflags", "+faststart", target]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-3000:])
        return 1
    for c in caps:
        os.remove(c)
    print(f"wrote {target}  ({os.path.getsize(target) / 1e6:.2f} MB, "
          f"{spans[-1][1]:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
