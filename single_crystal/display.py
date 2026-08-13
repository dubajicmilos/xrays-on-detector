"""How an intensity becomes something you can see.

Shared by the desktop app and, in ported form, by web/sc/js/render.js, so the
two show the same picture for the same numbers.

A diffraction pattern spans many decades. Plotted linearly, one reflection is
white and the rest are black; that is true to the data and useless to look at.
The stretch below maps intensity onto a fixed number of decades below the
strongest reflection, with a gain that slides the window up and down. It is
the same treatment the Game of Diffraction applies to its detector image.
"""
from __future__ import annotations

import math

import numpy as np

#: How many decades below the brightest reflection stay visible.
DECADES = 6.0


def stretch(intensity, gain: float = 1.0, log: bool = True, i_max: float | None = None):
    """Map intensity to 0..1 for display.

    gain multiplies the normalised intensity before the stretch, so raising it
    brings weak reflections up; the brightest simply saturates.

    i_max pins the normalisation to a chosen reference instead of the strongest
    reflection present. Pass the strongest reflection of the whole structure
    when stepping through layers, or an upper layer will be renormalised to its
    own weak maximum and look as bright as the zero layer.
    """
    I = np.asarray(intensity, dtype=float)
    if I.size == 0:
        return I
    top = float(I.max()) if i_max is None else float(i_max)
    if not (top > 0):
        return np.zeros_like(I)

    v = np.clip(I / top * gain, 0.0, 1.0)
    if not log:
        return v
    floor = 10.0 ** (-DECADES)
    v = np.clip(v, floor, 1.0)
    return (np.log10(v) + DECADES) / DECADES


def spot_sizes(display, scale: float = 180.0, floor: float = 3.0):
    """Marker area for a scatter plot, from the stretched value.

    Area rather than radius tracks the eye's reading of a spot, and a floor
    keeps a just-visible reflection from vanishing into a subpixel dot.
    """
    d = np.asarray(display, dtype=float)
    return floor + scale * d


def format_hkl(hkl, bar: bool = True) -> str:
    """(1, -1, 0) -> '11̅0'. Negative indices take an overbar, as they are
    written everywhere else.

    U+0305 attaches to the character before it, so a two-digit index needs one
    after every digit: -11 written as '11̅' bars only the second digit and reads
    as the pair (1, -1). Barring both gives an unbroken line over '11'.
    """
    out = []
    for v in hkl:
        v = int(v)
        digits = str(abs(v))
        if v < 0 and bar:
            out.append("".join(d + "̅" for d in digits))
        else:
            out.append(str(v))
    # Indices of ten and above need a separator or 1 12 0 reads as 1 1 2 0.
    sep = " " if any(abs(int(v)) > 9 for v in hkl) else ""
    return sep.join(out)


def nice_step(span: float) -> float:
    """A round number near span/6, for a scale bar or a grid."""
    if span <= 0:
        return 1.0
    raw = span / 6.0
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= m * mag:
            return m * mag
    return 10.0 * mag
