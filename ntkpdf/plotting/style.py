"""Shared plotting style for figures.
"""
from collections import namedtuple

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Page geometry  (A4, inner=40mm, outer=25mm → text width = 145mm)
# ---------------------------------------------------------------------------
TEXT_WIDTH_MM = 145.0
TEXT_WIDTH = TEXT_WIDTH_MM / 25.4  # inches

# Common figure sizes (width, height)
SINGLE_FIG = (TEXT_WIDTH, TEXT_WIDTH * 0.618)       # full-width, golden ratio
HALF_FIG = (TEXT_WIDTH / 2, TEXT_WIDTH / 2 * 0.618) # half-width, golden ratio

# ---------------------------------------------------------------------------
# Style definition
# ---------------------------------------------------------------------------
# Colour-blind-safe qualitative palette (Okabe-Ito), ordered for contrast so the
# first few overlaid curves are maximally distinct.
OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# The authoritative NTKPDF rcParams. A flat dict so the style is one transparent
# source of truth (see NTKPDF_STYLE / NTKApp.default_style for how it is applied).
NTKPDF_RCPARAMS = {
    # Text: LaTeX serif, to match the papers
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    # Coherent font-size hierarchy
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 16,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    # Axes & spines
    "axes.linewidth": 0.8,
    "axes.edgecolor": "0.15",
    "axes.grid": False,
    "axes.axisbelow": True,
    "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
    # Ticks: inward, on all four sides, with minor ticks (standard HEP look)
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 5.0,
    "ytick.major.size": 5.0,
    "xtick.minor.size": 3.0,
    "ytick.minor.size": 3.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    # Lines & markers
    "lines.linewidth": 1.6,
    "lines.markersize": 5.0,
    # Legend
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.labelspacing": 0.4,
    "legend.columnspacing": 1.2,
    # Figure & saving
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.02,
    "errorbar.capsize": 2.5,
}

# matplotlib.style.use accepts a list mixing style names and rcParam dicts and
# applies them left-to-right. seaborn-paper is a sane base; NTKPDF_RCPARAMS is the
# authoritative layer on top. This single object is shared by both entry points:
#   - setup() for the API/notebook path (applied at `import ntkpdf`)
#   - NTKApp.default_style for the CLI path (applied by reportengine's init_style)
NTKPDF_STYLE = ["seaborn-v0_8-paper", NTKPDF_RCPARAMS]


# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------
def setup():
    """Apply the NTKPDF matplotlib style globally. Used by the non-App paths
    (notebooks/scripts/API); the CLI applies the same NTKPDF_STYLE via
    NTKApp.default_style."""
    print("Applying NTKPDF plotting style...")
    plt.style.use(NTKPDF_STYLE)

def make_figure(ax_width=SINGLE_FIG[0], ax_height=SINGLE_FIG[1], left=0.9, bottom=0.7, right=0.1, top=0.1):
    """Create a figure with a fixed axes box size (in inches)."""
    fig_width = left + ax_width + right
    fig_height = bottom + ax_height + top
    # Use a bare Figure (not plt.figure) so the figure is never registered in
    # pyplot's global state. reportengine materialises every yielded figure
    # into a list before saving and never closes them, so pyplot-managed
    # figures would accumulate and trigger the "More than 20 figures" warning.
    fig = Figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes([left / fig_width, bottom / fig_height,
                       ax_width / fig_width, ax_height / fig_height])
    return fig, ax

# ---------------------------------------------------------------------------
# Legend Handler
# ---------------------------------------------------------------------------
HandlerSpec = namedtuple("HandlerSpec", ["color", "alpha", "ls", "draw_box"])

class ComposedHandler:
    """Legend handler for plots with uncertainty bands."""

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        ret = []

        line = mlines.Line2D(
            [x0, x0 + width],
            [y0 + height / 2, y0 + height / 2],
            color=orig_handle.color,
            linestyle=orig_handle.ls or "-",
            linewidth=1.5,
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(line)
        ret.append(line)

        if orig_handle.alpha:
          patch = mpatches.Rectangle(
              [x0, y0],
              width,
              height,
              facecolor=orig_handle.color,
              alpha=orig_handle.alpha,
              edgecolor="none",
              transform=handlebox.get_transform(),
          )
          handlebox.add_artist(patch)
          ret.append(patch)
        
        if orig_handle.draw_box:
           box = mpatches.Rectangle(
              [x0, y0],
              width,
              height,
              facecolor="none",
              edgecolor=orig_handle.color,
              linestyle=orig_handle.ls,
              linewidth=1.5,
              transform=handlebox.get_transform(),
          )
           handlebox.add_artist(box)
           ret.append(box)
        return ret


class LineHandler:
    """Legend handler that draws a single line -- no band/box. For line plots
    (e.g. per-replica losses) where the ComposedHandler box is unwanted."""

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        line = mlines.Line2D(
            [x0, x0 + width],
            [y0 + height / 2, y0 + height / 2],
            color=orig_handle.color,
            linestyle=orig_handle.ls or "-",
            linewidth=1.5,
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(line)
        return [line]


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
# Styled, container-agnostic: each draws one curve (or curve + band) for a given
# NTKStats onto an axis, applying the style and *returning* a HandlerSpec for the
# ComposedHandler legend (the caller collects them). Independent of what is on the
# axes (PDFs over x, losses or eigenvalues over epochs, ...), so any provider can
# reuse them instead of duplicating ax.plot. ``stats`` has replicas on axis 0
# (``stats.error_members()`` -> ``(nreplicas, n_xgrid)``) aligned with ``xgrid``.
# Styling is taken via explicit keyword-only args; any extra ``**line_kwargs`` are
# forwarded to ``ax.plot`` (and ``band_kwargs`` to ``ax.fill_between``).
def draw_bounds(ax, stats, xgrid, *,
                color=None, alpha=0.3, linestyle="-", linewidth=1.5,
                band_linestyle="--",
                no_bands=False, show_68=True, show_std=True,
                central_value="mean", band_kwargs=None, **line_kwargs):
    """Draw the central value with std and/or 68% bands; return the HandlerSpec.

    The 68% band edges are drawn with ``band_linestyle`` (dashed by default), and
    that same style is carried in the returned HandlerSpec so the ComposedHandler
    legend box contour matches the band edges.
    """
    band_kwargs = band_kwargs or {}

    if central_value == "mean":
        cv = stats.central_value()
    elif central_value == "median":
        cv = stats.median()
    else:
        raise ValueError(f"Unknown central_value: {central_value}")

    (line,) = ax.plot(xgrid, cv, color=color, linewidth=linewidth, linestyle=linestyle, **line_kwargs)
    color = line.get_color()  # resolve the actual color, even if it was None

    if no_bands:
        return HandlerSpec(color, None, linestyle, False)

    if show_std:
        down, up = stats.errorbarstd()  # (down, up)
        ax.fill_between(xgrid, down, up, color=color, alpha=alpha, **band_kwargs)
    if show_68:
        down68, up68 = stats.errorbar68()
        ax.plot(xgrid, up68, linestyle=band_linestyle, color=color)
        ax.plot(xgrid, down68, linestyle=band_linestyle, color=color)
    return HandlerSpec(color, alpha, band_linestyle, True)


def draw_line(ax, stats, xgrid, *,
              color=None, linestyle="-", linewidth=1.5, alpha=None,
              replicas=None, **line_kwargs):
    """Draw one line per replica (single-member ``stats`` -> a single line);
    return the HandlerSpec."""
    # Resolve a single colour up front so every replica shares it: a bare
    # color=None advances the prop cycle on each ax.plot, giving rainbow lines
    # and a legend swatch (color=None) that matches nothing.
    if color is None:
        color = ax._get_lines.get_next_color()

    members = stats.error_members()
    if replicas is None:
        replicas = range(1, members.shape[0] + 1)

    for r in replicas:
        ax.plot(xgrid, members[r-1], color=color, linestyle=linestyle,
                linewidth=linewidth, alpha=alpha, **line_kwargs)

    return HandlerSpec(color, None, linestyle, False)


def select_draw(provider):
    """Resolve a draw provider given as a name (``"bounds"``/``"line"``) or as a
    callable with the ``(ax, stats, xgrid, **kwargs)`` signature returning a
    ``HandlerSpec`` (or ``None``)."""
    if isinstance(provider, str):
        if provider == "bounds":
            return draw_bounds
        elif provider == "line":
            return draw_line
        raise ValueError(f"Unknown plot provider: {provider}")
    return provider