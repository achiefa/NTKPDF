from functools import partial
from typing import Optional

import numpy as np

from colibri.ntk.plotntk import (
    ntk_plot_provider,
    draw_band,
    iter_by_rank,
    iter_by_fit,
    _figuregen
)


def _draw_one_replica(ax, xgrid, stats, label, replica_index, handles=None, labels=None):
    """Draw a single replica's trajectory (1-based ``replica_index``) as one line.

    Matches the ``draw_fn`` signature colibri's ``ntk_plot_provider`` expects, but
    selects one replica from ``stats`` (``(nreplicas, n_epochs)``) instead of a band
    over all of them. Used with ``custom_handler=None`` so the legend is built from
    the ``ax.plot`` label (like colibri's own ``draw_replicas``).
    """
    data = np.asarray(stats.data)
    if not (1 <= replica_index <= data.shape[0]):
        raise ValueError(
            f"replica_index {replica_index} out of range [1, {data.shape[0]}]"
        )
    color = ax._get_lines.get_next_color()
    y = data[replica_index - 1]
    ax.plot(xgrid, y, color=color, linewidth=1.5, label=label)
    return np.atleast_2d(y)

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


# =============================================================================
# Single-replica trajectories (one figure per replica, from Replicaspecs)
# =============================================================================
# These reuse colibri's ``ntk_plot_provider`` + ``iter_by_fit`` but swap the band
# draw for ``_draw_one_replica`` (colibri's ``draw_replicas`` always overlays all
# replicas plus a mean, so it can't isolate one). They depend on the same
# all-replica grids as the ensemble plots (cached), and pick the chosen replica's
# row -- no per-replica recomputation.

@_figuregen
def plot_feature_eigvals_replica_by_fit(
    h_val_grids_by_fit,
    replica_index: int,
    rank_indices: Optional[list] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
):
    """Single-replica feature eigenvalues h^(k) vs epoch: one figure per fit, the
    ranks in the group as individual lines for replica ``replica_index`` (1-based,
    from ``Replicaspecs``). The per-replica counterpart of
    :func:`plot_feature_eigvals_by_fit`."""
    yield from ntk_plot_provider(
        h_val_grids_by_fit,
        rank_indices,
        draw_fn=partial(_draw_one_replica, replica_index=replica_index),
        iterator_fn=iter_by_fit,
        custom_handler=None,
        title_fn=lambda grid: rf"{grid.label} ($\rm replica\ {replica_index}$)",
        name_fn=lambda grid: f"h_replica_{replica_index}_{grid.label}",
        ylabel_fn=lambda _: r"$\textrm{Feature eigenvalues}$",
        xscale=xscale,
        yscale=yscale,
        ymin=ymin,
        ymax=ymax,
    )


@_figuregen
def plot_eigvals_replica_by_fit(
    eigval_grids_by_fit,
    replica_index: int,
    rank_indices: Optional[list] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
):
    """Single-replica NTK eigenvalues lambda^(k) vs epoch: one figure per fit, the
    ranks in the group as individual lines for replica ``replica_index`` (1-based,
    from ``Replicaspecs``). The per-replica counterpart of colibri's
    ``plot_eigvals_by_fit``; ``eigval_grids_by_fit`` is colibri's collect over fits
    of the (all-replica) eigenvalue grids."""
    yield from ntk_plot_provider(
        eigval_grids_by_fit,
        rank_indices,
        draw_fn=partial(_draw_one_replica, replica_index=replica_index),
        iterator_fn=iter_by_fit,
        custom_handler=None,
        title_fn=lambda grid: rf"{grid.label} ($\rm replica\ {replica_index}$)",
        name_fn=lambda grid: f"eigvals_replica_{replica_index}_{grid.label}",
        ylabel_fn=lambda _: r"$\textrm{NTK eigenvalues}$",
        xscale=xscale,
        yscale=yscale,
        ymin=ymin,
        ymax=ymax,
    )