"""
ntkpdf.plotting.pdfplots_utils.py

Plotting utilities for PDF plots in ntkpdf.
"""

import logging
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from validphys.pdfgrids import XPlottingGrid
from ntkpdf.plotting.style import (
   make_figure,
   HandlerSpec,
   ComposedHandler,
   draw_bounds,
   select_draw,
)

log = logging.getLogger(__name__)

# The styled drawing primitives (draw_bounds/draw_line/select_draw) now live in
# ntkpdf.plotting.style so they can be reused by non-PDF providers (losses, ...).


def _normalise(grids, normalise_to):
    if normalise_to is None:
        return grids

    normvals = grids[normalise_to-1].grid_values.central_value()

    # ``np.errstate`` restores the floating-point error *mode* on exit, so unlike
    # ``np.seterr``/``np.seterrcall`` it does not leak global NumPy state. We
    # silence the (expected) divide-by-zero where the reference grid is ~0 and
    # report it once afterwards instead.
    newgrids = []
    with np.errstate(all="ignore"):
        for grid in grids:
            newvalues = type(grid.grid_values)(grid.grid_values.data / normvals)
            newgrids.append(grid.copy_grid(grid_values=newvalues))

    if any(not np.all(np.isfinite(g.grid_values.data)) for g in newgrids):
        log.warning(
            "Non-finite values encountered while normalising PDF grids "
            "(division by ~zero in the reference grid)."
        )
    return newgrids
    

def plot_grids(grids: list[XPlottingGrid],
               flavours: list[str],
               plot_args: list[dict] = None,
               ylabels: list[str] = None,
               labels: list[str] = None,
               savepath: str = None,
               figname: str = None,
               plot_provider: Callable | str = draw_bounds,
               normalise_to: int = None,
               separate_axis: list[int] = None,
               kwargs_twinx: dict = {},
               legend_outside: bool = False,
               axes: list = None,
               **kwargs,
               ):
    plot_provider = select_draw(plot_provider)

    # Fail loudly on mismatched lengths rather than silently dropping curves
    # (zip truncation) or raising an opaque IndexError mid-plot.
    if labels is not None and len(labels) != len(grids):
        raise ValueError(f"Got {len(labels)} labels for {len(grids)} grids.")
    if plot_args is not None and len(plot_args) != len(grids):
        raise ValueError(f"Got {len(plot_args)} plot_args for {len(grids)} grids.")
    if ylabels is not None and len(ylabels) != len(flavours):
        raise ValueError(f"Got {len(ylabels)} ylabels for {len(flavours)} flavours.")
    # When the caller supplies axes (e.g. a grid of subplots) we draw into them
    # instead of making one Figure per flavour. The shared figure is yielded once
    # at the end rather than per flavour.
    own_fig = axes is None
    if not own_fig and len(axes) < len(flavours):
        raise ValueError(f"Got {len(axes)} axes for {len(flavours)} flavours.")

    has_separate_axis = separate_axis is not None and len(separate_axis) > 0
    if not has_separate_axis:
        ax_idx_map = [0 for _ in range(len(grids))]
    else:
        ax_idx_map = separate_axis

    if plot_args is None:
        plot_args = [{}] * len(grids)

    newgrids = _normalise(grids, normalise_to)

    for flidx, fl in enumerate(flavours):
        if own_fig:
            fig, ax = make_figure()
        else:
            ax = axes[flidx]

        if has_separate_axis:
            ax_twin = ax.twinx()

        handles = []
        for grid, args, ax_idx in zip(newgrids, plot_args, ax_idx_map):
          stats = grid.select_flavour(fl).grid_values
          handle = plot_provider(ax if ax_idx == 0 else ax_twin, stats, grid.xgrid, **args)
          if handle is not None:
              handles.append(handle)

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(ylabels[flidx] if ylabels else rf"${fl}$")
        # With many overlaid curves the in-plot legend hides the data; `legend_outside`
        # anchors it to the right of the axes instead (reportengine saves with
        # bbox_inches='tight', so it is not clipped).
        legend_kw = (
            {"loc": "upper left", "bbox_to_anchor": (1.01, 1.0)}
            if legend_outside
            else {"loc": "best"}
        )
        ax.legend(handles=handles, labels=labels or [""] * len(handles),
                  handler_map={HandlerSpec: ComposedHandler()}, **legend_kw)

        for key, val in kwargs.items():
          if isinstance(val, dict):
             v = val.copy()
             getattr(ax, f"set_{key}")(v.pop("value"), **v)
          else:
            ax.set(**{key: kwargs[key]})
        
        if has_separate_axis:
          for key, val in kwargs_twinx.items():
            if isinstance(val, dict):
              v = val.copy()
              getattr(ax_twin, f"set_{key}")(v.pop("value"), **v)
            else:
              ax_twin.set(**{key: kwargs_twinx[key]})

        if own_fig and savepath is not None:
          savepath.mkdir(parents=True, exist_ok=True)
          # Compute outside the f-string: backslashes in an f-string expression
          # are a SyntaxError before Python 3.12.
          clean_flavour = flavours[flidx].replace("\\", "")
          filename = f"pdf_{clean_flavour}"
          if figname is not None:
              filename += "_" + figname
          filename += ".pdf"
          fig.savefig(savepath / filename)
          plt.close(fig)

        if own_fig:
            yield fig

    if not own_fig:
        yield axes[0].figure
