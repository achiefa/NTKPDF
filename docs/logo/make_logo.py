"""docs.logo.make_logo -- generate the ntkpdf logo set.

The mark is a PDF-shaped hill (slow rise at small x, sharp fall as x -> 1, as on
a log-x axis) built from five bands, one per NTK mode, coloured by rank with
viridis (rank 1 darkest, at the bottom), as in the paper's NTK figures. A line
tangent to the hill, touching from outside, stands for the frozen-NTK
linearisation. Lettering is Latin Modern (the OpenType Computer Modern), outlined
into paths so the SVGs need no fonts.

Stylised, not literal: the real eigenfunctions q^(r)(x) change sign, so the mode
expansion is not a stack of positive layers.

Usage (needs numpy, matplotlib, fontTools and a TeX Live with the lm fonts;
rsvg-convert and ImageMagick for the raster exports):

    python docs/logo/make_logo.py            # writes next to this script
"""
import os
import shutil
import subprocess

import numpy as np
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from matplotlib import colormaps

HERE = os.path.dirname(os.path.abspath(__file__))

INK = {"light": "#1b1b1f", "dark": "#ececf1"}
DOT = "#e07b24"           # the current point (orange of the paper's tangent plane)
WORD = "ntkpdf"           # set as \texttt{ntkpdf}, as in the paper
TAGLINE = "the proton structure, one mode at a time"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _hex(t):
    r, g, b, _ = colormaps["viridis"](t)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def _smooth(pts, move=True):
    """Catmull-Rom spline through pts, as cubic Beziers."""
    p = np.asarray(pts, float)
    n = len(p)
    d = [("M" if move else "L") + f"{p[0, 0]:.1f},{p[0, 1]:.1f}"]
    for i in range(n - 1):
        p0, p1, p2, p3 = p[max(i - 1, 0)], p[i], p[i + 1], p[min(i + 2, n - 1)]
        c1, c2 = p1 + (p2 - p0) / 6, p2 - (p3 - p1) / 6
        d.append(f"C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}")
    return " ".join(d)


def _font(name):
    path = subprocess.run(["kpsewhich", name], capture_output=True, text=True).stdout.strip()
    if not path:
        raise FileNotFoundError(f"{name} not found by kpsewhich (needs TeX Live's lm fonts)")
    return TTFont(path)


def _text(text, font, size):
    """Outline `text` with its baseline at y=0. Returns (path data, advance width)."""
    gs, cmap = font.getGlyphSet(), font.getBestCmap()
    s = size / font["head"].unitsPerEm
    out, x = [], 0.0
    for ch in text:
        g = cmap[ord(ch)]
        pen = SVGPathPen(gs)
        gs[g].draw(TransformPen(pen, (s, 0, 0, -s, x, 0)))
        out.append(pen.getCommands())
        x += font["hmtx"][g][0] * s
    return " ".join(out), x


def _svg(w, h, body, bg=None):
    rect = f'<rect width="{w:.0f}" height="{h:.0f}" fill="{bg}"/>' if bg else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" '
            f'width="{w:.0f}" height="{h:.0f}">{rect}{body}</svg>\n')


def _beta(u, a, b):
    y = u**a * (1 - u) ** b
    return y / y.max()


# ---------------------------------------------------------------------------
# The mark (512 x 512 user units)
# ---------------------------------------------------------------------------
X0, X1, BASE, HEIGHT = 60, 452, 396, 300
GAP = 7  # transparent gap between bands


def _stack(u):
    """Cumulative mode contributions; leading mode carries the bulk."""
    comps = [(0.36, _beta(u, 1.8, 1.35)), (0.24, _beta(u, 3.3, 1.3)), (0.17, _beta(u, 6.5, 1.7)),
             (0.13, _beta(u, 12.0, 2.6)), (0.10, _beta(u, 22.0, 4.2))]
    s = np.cumsum([w * c for w, c in comps], axis=0)
    return s / s[-1].max()


def _tangent_point(half=150):
    """Steepest point of the rising flank whose tangent segment stays outside the hill."""
    u = np.linspace(0, 1, 4001)
    xs, ys = X0 + u * (X1 - X0), BASE - _stack(u)[-1] * HEIGHT
    m_all = np.gradient(ys, xs)
    ipk = int(np.argmin(ys))
    for i in range(int(np.argmin(m_all[:ipk])), ipk):
        m = m_all[i]
        dx = half / np.hypot(1, m)
        sel = (xs >= xs[i] - dx) & (xs <= xs[i] + dx)
        if np.all(ys[i] + m * (xs[sel] - xs[i]) <= ys[sel] + 0.25):
            return xs[i], ys[i], m, dx
    raise RuntimeError("no outside tangent found")


def mark(ink, tangent=True, uid="m"):
    """SVG body of the mark. `ink` colours the axis and tangent ('currentColor' allowed)."""
    u = np.linspace(0, 1, 41)
    S = _stack(u)
    xs = X0 + u * (X1 - X0)
    bands, prev = [], np.zeros_like(u)
    for top, t in zip(S, np.linspace(0.0, 0.9, len(S))):
        d = (_smooth(np.c_[xs, BASE - top * HEIGHT]) + " "
             + _smooth(np.c_[xs, BASE - prev * HEIGHT][::-1], move=False) + " Z")
        bands.append(f'<path d="{d}" fill="{_hex(t)}"/>')
        prev = top
    seps = "".join(f'<path d="{_smooth(np.c_[xs, BASE - s * HEIGHT])}" fill="none" stroke="#000" '
                   f'stroke-width="{GAP}"/>' for s in S[:-1])
    axis = (f'<path d="M{X0 - 18},{BASE + 15} H{X1 + 18}" stroke="{ink}" stroke-width="10" '
            f'stroke-linecap="round"/>')
    full = '<rect width="512" height="512" fill="#fff"/>'
    defs = f'<mask id="{uid}-bands" maskUnits="userSpaceOnUse" x="0" y="0" width="512" height="512">{full}{seps}</mask>'
    inner = f'<g mask="url(#{uid}-bands)">{"".join(bands)}</g>{axis}'
    if not tangent:
        return f"<defs>{defs}</defs>{inner}"
    px, py, m, dx = _tangent_point()
    line = (f'<path d="M{px - 0.72 * dx:.1f},{py - 0.72 * m * dx:.1f} L{px + dx:.1f},{py + m * dx:.1f}" '
            f'stroke="{ink}" stroke-width="12" stroke-linecap="round"/>')
    # a transparent ring around the dot, so it reads on any background
    defs += (f'<mask id="{uid}-dot" maskUnits="userSpaceOnUse" x="0" y="0" width="512" height="512">{full}'
             f'<circle cx="{px:.1f}" cy="{py:.1f}" r="29" fill="#000"/></mask>')
    return (f'<defs>{defs}</defs><g mask="url(#{uid}-dot)">{inner}{line}</g>'
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="21" fill="{DOT}"/>')


def lockup(ink, tagline, uid="l"):
    """Mark + wordmark (+ optional tagline). Returns (body, width, height)."""
    H = 220
    word, ww = _text(WORD, _font("lmmonolt10-bold.otf"), 132)
    xt = H + 14
    parts = [f'<g transform="scale({H / 512:.5f})">{mark(ink, uid=uid)}</g>',
             f'<path d="{word}" transform="translate({xt},128)" fill="{ink}"/>']
    width = xt + ww + 20
    if tagline:
        tag, tw = _text(TAGLINE, _font("lmroman10-italic.otf"), 25.5)
        parts.append(f'<path d="{tag}" transform="translate({xt + 4},176)" fill="{ink}" fill-opacity="0.72"/>')
        width = max(width, xt + tw + 24)
    return "".join(parts), width, H


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def _write(name, text):
    with open(os.path.join(HERE, name), "w") as fh:
        fh.write(text)
    return os.path.join(HERE, name)


def write_svgs():
    for theme, ink in INK.items():
        sfx = "" if theme == "light" else "-dark"
        _write(f"ntkpdf-mark{sfx}.svg", _svg(512, 512, mark(ink)))
        _write(f"ntkpdf-icon{sfx}.svg", _svg(512, 512, mark(ink, tangent=False)))
        b, w, h = lockup(ink, False)
        _write(f"ntkpdf-logo{sfx}.svg", _svg(w, h, b))
        b, w, h = lockup(ink, True)
        _write(f"ntkpdf-banner{sfx}.svg", _svg(w, h, b))
    # favicon: plain hill, axis follows the browser's colour scheme
    style = (f'<style>.ink{{color:{INK["light"]}}}@media (prefers-color-scheme: dark)'
             f'{{.ink{{color:{INK["dark"]}}}}}</style>')
    _write("favicon.svg", _svg(512, 512, f'{style}<g class="ink">{mark("currentColor", tangent=False)}</g>'))


def write_rasters():
    rsvg, magick = shutil.which("rsvg-convert"), shutil.which("magick")
    if not (rsvg and magick):
        print("rsvg-convert/magick not found: skipping PNG/ICO exports")
        return
    png = os.path.join(HERE, "png")
    os.makedirs(png, exist_ok=True)
    icon = os.path.join(HERE, "ntkpdf-icon.svg")
    for s in (16, 32, 48, 64, 128, 256, 512, 1024):
        subprocess.run([rsvg, "-w", str(s), "-h", str(s), icon, "-o", f"{png}/ntkpdf-icon-{s}.png"], check=True)
    subprocess.run([magick] + [f"{png}/ntkpdf-icon-{s}.png" for s in (16, 32, 48)]
                   + [os.path.join(HERE, "favicon.ico")], check=True)
    for name in ("ntkpdf-logo", "ntkpdf-logo-dark", "ntkpdf-banner", "ntkpdf-banner-dark"):
        subprocess.run([rsvg, "-h", "440", f"{HERE}/{name}.svg", "-o", f"{png}/{name}.png"], check=True)
    # GitHub social preview, 1280 x 640, banner centred on white
    b, w, h = lockup(INK["light"], True, uid="s")
    sc = 1000 / w
    body = f'<g transform="translate({(1280 - w * sc) / 2:.1f},{(640 - h * sc) / 2:.1f}) scale({sc:.5f})">{b}</g>'
    social = _write("png/_social.svg", _svg(1280, 640, body, bg="#ffffff"))
    subprocess.run([rsvg, social, "-o", f"{png}/ntkpdf-social.png"], check=True)
    os.remove(social)


if __name__ == "__main__":
    write_svgs()
    write_rasters()
    print("wrote", sorted(f for f in os.listdir(HERE) if f != os.path.basename(__file__)))
