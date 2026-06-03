"""
ntkpdf.plotting.loss_providers

Report plot providers for the training/validation loss curves.
"""

from typing import Optional

from reportengine.figure import figuregen

from ntkpdf.plotting.style import (
    make_figure,
    draw_line,
    HandlerSpec,
    LineHandler,
)


@figuregen
def plot_losses_all_replicas(
    loss_function_grid,
    replica_indices: Optional[list] = None,
    xscale: str = "linear",
    yscale: str = "linear",
):
    """One figure per replica showing its training and validation loss vs epoch.

    ``@figuregen`` (rather than ``@figure``) because a fit can have many replicas
    and this yields them lazily, one figure each. Each figure is yielded as a
    ``(fig, suffix)`` tuple so reportengine gives it a distinct filename, and the
    replica is also labelled as the axes title.

    ``replica_indices`` is an optional list of 1-based replica ids to plot
    (default: all). It is intentionally *not* named ``replicas``/``replica_ids``:
    those are existing config resources (the replica collect dimension and the
    nreplicas/weight-loading selector), so reusing either name would collide with
    a production rule or silently change what the loss grid computes.
    """
    tr_grid, vl_grid = loss_function_grid

    if replica_indices is None:
        replica_indices = range(1, tr_grid.nreplicas + 1)

    for replica in replica_indices:
        fig, ax = make_figure()
        handles, labels = [], []

        # get_plotting_data takes a 0-based row; `replica` is the 1-based id.
        tr_stats = tr_grid.get_plotting_data(replica_idx=replica - 1)  # NTKStats (1, n_epochs)
        vl_stats = vl_grid.get_plotting_data(replica_idx=replica - 1)

        handles.append(draw_line(ax, tr_stats, tr_grid.xgrid))
        labels.append(r"$\rm Training$")
        handles.append(draw_line(ax, vl_stats, vl_grid.xgrid))
        labels.append(r"$\rm Validation$")

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel(tr_grid.xlabel)
        ax.set_ylabel(r"$\mathcal{L}$")
        ax.set_title(rf"$\rm Replica\ {replica}$")
        ax.legend(handles=handles, labels=labels, handler_map={HandlerSpec: LineHandler()})

        yield fig, f"replica_{replica}"
