"""
ntkpdf.plotting.loss_providers

Report plot providers for the training/validation loss curves.
"""

from typing import Optional

from reportengine.figure import figure, figuregen

from ntkpdf.plotting.style import (
    make_figure,
    draw_bounds,
    draw_line,
    ComposedHandler,
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


@figure
def plot_losses_ensemble(
    loss_function_grid,
    central_value: str = "mean",
    show_std: bool = True,
    show_68: bool = False,
    xscale: str = "linear",
    yscale: str = "linear",
):
    """Ensemble training and validation loss vs epoch in a single figure:
    the central value over all replicas with an uncertainty band.

    Unlike ``plot_losses_all_replicas`` (one curve per replica), this collapses
    the whole ensemble into a central value + band. ``get_plotting_data()`` with
    no ``replica_idx`` returns the full ``(nreplicas, n_epochs)`` ensemble, which
    feeds straight into the shared ``draw_bounds`` primitive -- so the band kind
    is whatever ``draw_bounds`` supports:

    - ``central_value``: ``"mean"`` or ``"median"``.
    - ``show_std``: shade the central value +- 1 standard deviation.
    - ``show_68``: draw the 68% interval edges (dashed).

    These are exposed as parameters so the runcard (``analysis.yaml``) chooses the
    convention; the default is central value +- 1 std (``show_std`` on, ``show_68``
    off). ``@figure`` (not ``@figuregen``) because it is a single plot.
    """
    tr_grid, vl_grid = loss_function_grid

    fig, ax = make_figure()
    handles, labels = [], []

    # replica_idx=None -> full ensemble (nreplicas, n_epochs), as draw_bounds wants.
    tr_stats = tr_grid.get_plotting_data()
    vl_stats = vl_grid.get_plotting_data()

    band_kw = dict(central_value=central_value, show_std=show_std, show_68=show_68)
    handles.append(draw_bounds(ax, tr_stats, tr_grid.xgrid, **band_kw))
    labels.append(r"$\rm Training$")
    handles.append(draw_bounds(ax, vl_stats, vl_grid.xgrid, **band_kw))
    labels.append(r"$\rm Validation$")

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(tr_grid.xlabel)
    ax.set_ylabel(r"$\mathcal{L}$")
    ax.legend(handles=handles, labels=labels, handler_map={HandlerSpec: ComposedHandler()})
    return fig
