from functools import partial
from typing import Optional

from colibri.ntk.plotntk import (
    ntk_plot_provider,
    draw_band,
    iter_by_rank,
    iter_by_fit,
    _figuregen
)

# =============================================================================
# Convenience functions feature eigenvalues
# =============================================================================

@_figuregen
def plot_feature_eigvals_by_rank(
    h_val_grids_by_fit,
    rank_indices: Optional[list] = None,
    error_type: str = "mean",
    title_fn = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
):
    """Plot eigenvalues, one figure per rank showing all fits."""
    yield from ntk_plot_provider(
        h_val_grids_by_fit,
        rank_indices,
        draw_fn=partial(draw_band, error_type=error_type),
        iterator_fn=iter_by_rank,
        title_fn=title_fn if title_fn is not None else lambda rank_index: rf"$h^{{({rank_index})}}$",
        name_fn=lambda rank_index: f"h_{rank_index}",
        ylabel_fn=lambda rank_index: rf"$h^{{({rank_index})}}$",
        xscale=xscale,
        yscale=yscale,
        ymin=ymin,
        ymax=ymax,
    )

@_figuregen
def plot_feature_eigvals_by_fit(
    h_val_grids_by_fit,
    rank_indices: Optional[list] = None,
    error_type: str = "mean",
    title_fn = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
):
    """Plot eigenvalues, one figure per fit showing multiple ranks."""
    
    yield from ntk_plot_provider(
        h_val_grids_by_fit,
        rank_indices,
        draw_fn=partial(draw_band, error_type=error_type),
        iterator_fn=iter_by_fit,
        title_fn=title_fn if title_fn is not None else lambda grid: grid.label,
        name_fn=lambda grid: f"h_{grid.label}",
        ylabel_fn=lambda _: r"$\textrm{Feature eigenvalues}$",
        xscale=xscale,
        yscale=yscale,
        ymin=ymin,
        ymax=ymax,
    )