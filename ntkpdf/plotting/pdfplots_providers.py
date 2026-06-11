"""
This module contains plot providers for PDF plots.
"""

from copy import copy
from typing import Optional

from reportengine.figure import figuregen

from ntkpdf.plotting.pdfplots_utils import plot_grids

# Default flavours to plot, paired with their y-axis labels so the two cannot
# desync. Shared by all plot providers; override `flavours` in a runcard and (if
# you also override `ylabels`) keep the lengths matching -- ``plot_grids`` checks.
_DEFAULT_FLAVOUR_YLABELS = (
    (r"\Sigma", r"$\Sigma$"),
    ("g", r"$g$"),
    ("V", r"$V$"),
    ("V3", r"$V_3$"),
    ("V8", r"$V_8$"),
    ("T3", r"$T_3$"),
    ("T8", r"$T_8$"),
    ("T15", r"$T_{15}$"),
)
DEFAULT_FLAVOURS = [fl for fl, _ in _DEFAULT_FLAVOUR_YLABELS]
DEFAULT_YLABELS = [yl for _, yl in _DEFAULT_FLAVOUR_YLABELS]


def _resolve_flavours(flavours, ylabels):
    """Fill in the paired flavour/ylabel defaults. If the caller overrides
    `flavours` but not `ylabels`, leave `ylabels` as ``None`` so ``plot_grids``
    derives ``$<fl>$`` per flavour (never a length mismatch)."""
    if flavours is None:
        flavours = DEFAULT_FLAVOURS
        if ylabels is None:
            ylabels = DEFAULT_YLABELS
    return flavours, ylabels


@figuregen
def plot_pdfs_at_initialisation(
    init_grids_by_name,
    selectors: list,
    labels: list,
    flavours: Optional[list] = None,
    ylabels: Optional[list] = None,
    plot_args=None,
    plot_provider="bounds",
    normalise_to=None,
    xscale='linear',
    yscale='linear',
    xlim=None,
    ylim=None
    ):
    flavours, ylabels = _resolve_flavours(flavours, ylabels)
    # ``init_grids_by_name`` is the shared {selector_name: grid} mapping built
    # once per fit/nreplicas; this Selectors group just picks the curves to show.
    grids = [init_grids_by_name[name] for name in selectors]

    yield from plot_grids(grids,
                          flavours=flavours,
                          ylabels=ylabels,
                          labels=labels,
                          plot_args=plot_args,
                          plot_provider=plot_provider,
                          normalise_to=normalise_to,
                          xscale=xscale,
                          yscale=yscale,
                          xlim=xlim,
                          ylim=ylim
                          )


@figuregen
def compare_fits_at_init(
    collected_init_grids,
    labels: list,
    selector: str = "with_prefactors",
    flavours: list = None,
    ylabels: list = None,
    plot_args=None,
    plot_provider="bounds",
    normalise_to=None,
    xscale='linear',
    yscale='linear',
    xlim=None,
    ylim=None
    ):
    flavours, ylabels = _resolve_flavours(flavours, ylabels)
    # ``collected_init_grids`` is the per-fit {selector_name: grid} mapping
    # collected over the ``fits`` dimension. For a cross-fit comparison we pick
    # the same ``selector`` from each fit and overlay one curve per fit; the
    # per-fit base models were each built once by produce_init_grids_by_name.
    grids = [fit_grids[selector] for fit_grids in collected_init_grids]

    yield from plot_grids(grids,
                          flavours=flavours,
                          ylabels=ylabels,
                          labels=labels,
                          plot_args=plot_args,
                          plot_provider=plot_provider,
                          normalise_to=normalise_to,
                          xscale=xscale,
                          yscale=yscale,
                          xlim=xlim,
                          ylim=ylim
                          )


@figuregen
def plot_features_at_epoch(
    feature_grids_at_epoch,
    rank_indices: list,
    flavours: Optional[list] = None,
    ylabels: Optional[list] = None,
    plot_args=None,
    normalise_to=None,
    xscale='log',
    xlim=None,
    ylim=None,
    legend_outside=False,
    ):
    """One figure per flavour of the feature columns ``q^(k)`` at one training
    epoch, overlaying the ranks of the current ``Rankspecs`` group as ensemble
    bands.

    Deliberately exposes **no** ``yscale``: this provider runs inside ``{@with
    Rankspecs@}``, whose entries carry ``yscale: log`` (meant for the positive
    eigenvalue trajectories). Features are PDF-like and signed, so a log y-axis is
    wrong; the y-axis is forced linear and only ``xscale`` (default log, the natural
    choice for the x-grid) is configurable.

    The epoch is bound from ``Epochspecs`` (the same epochs as the PDF-at-epoch
    page) through ``feature_grids_at_epoch``; ``rank_indices`` comes from
    ``Rankspecs``. A feature is a vector over ``(flavour, x)``, so it plots exactly
    like a PDF -- reusing ``plot_grids`` with the ``bounds`` primitive for the
    per-replica band. Note: at a given rank, replicas whose NTK cut is below that
    rank carry a zero-padded (absent) feature and are excluded from the band (see
    ``feature_grids_at_epoch``), so different ranks may average over different
    replica counts.
    """
    flavours, ylabels = _resolve_flavours(flavours, ylabels)
    labels = [rf"$q^{{({k})}}$" for k in rank_indices]

    yield from plot_grids(feature_grids_at_epoch,
                          flavours=flavours,
                          ylabels=ylabels,
                          labels=labels,
                          plot_args=plot_args,
                          plot_provider="bounds",
                          normalise_to=normalise_to,
                          xscale=xscale,
                          yscale="linear",
                          xlim=xlim,
                          ylim=ylim,
                          legend_outside=legend_outside,
                          )


@figuregen
def plot_features_at_epoch_single_replica(
    feature_grids_at_epoch,
    replica_index,
    rank_indices: list,
    flavours: Optional[list] = None,
    ylabels: Optional[list] = None,
    plot_args=None,
    normalise_to=None,
    xscale='log',
    xlim=None,
    ylim=None,
    legend_outside=False,
    ):
    """Single-replica version of :func:`plot_features_at_epoch`: the feature columns
    ``q^(k)`` of one replica (``replica_index``, 1-based, from ``Replicaspecs``)
    drawn as individual lines rather than an ensemble band.

    ``feature_grids_at_epoch`` resolves *with* ``replica_index`` at this call site,
    so each returned grid already holds just that replica; ``plot_grids`` with the
    ``line`` primitive then draws one curve per rank. Like the ensemble version it
    exposes no ``yscale`` (forced linear; see :func:`plot_features_at_epoch`).
    """
    flavours, ylabels = _resolve_flavours(flavours, ylabels)
    labels = [rf"$q^{{({k})}}$" for k in rank_indices]

    yield from plot_grids(feature_grids_at_epoch,
                          flavours=flavours,
                          ylabels=ylabels,
                          labels=labels,
                          plot_args=plot_args,
                          plot_provider="line",
                          normalise_to=normalise_to,
                          xscale=xscale,
                          yscale="linear",
                          xlim=xlim,
                          ylim=ylim,
                          legend_outside=legend_outside,
                          )


@figuregen
def plot_pdfs_at_epoch(
    pdf_grid_at_epoch,
    fakepdf_grid,
    epoch,
    flavours: Optional[list] = None,
    ylabels: Optional[list] = None,
    plot_args=None,
    plot_provider="bounds",
    normalise_to=None,
    xscale='linear',
    yscale='linear',
    xlim=None,
    ylim=None,
    show_central_fakepdf=False
    ):
    """One figure per flavour of the PDF at the given training ``epoch``.

    ``epoch`` is bound from the ``Epochspecs`` namespace (one entry per epoch the
    user selected on the command line), so the report renders this once per epoch.

    The closure-test underlying-law PDF is overlaid only when the user opted in
    (``show_fakepdf``): otherwise ``fakepdf_grid`` is ``None`` and just the fitted
    PDF is shown. When present the underlying law is the second grid (index 1), so
    ``normalise_to: 1`` in the runcard gives a ratio-to-truth plot.
    """
    flavours, ylabels = _resolve_flavours(flavours, ylabels)

    grids = [pdf_grid_at_epoch]
    labels = [rf"$\rm Epoch\ {epoch}$"]
    if fakepdf_grid is not None:
        grids.append(fakepdf_grid)
        labels.append(r"$\rm Underlying\ law$")
        if show_central_fakepdf:
            plot_args = [None, {"linestyle": "solid", "color": "black"}]

    yield from plot_grids(grids,
                          flavours=flavours,
                          ylabels=ylabels,
                          labels=labels,
                          plot_args=plot_args,
                          plot_provider=plot_provider,
                          normalise_to=normalise_to,
                          xscale=xscale,
                          yscale=yscale,
                          xlim=xlim,
                          ylim=ylim
                          )
    
@figuregen
def plot_pdfs_all_epochs(
    pdf_grid_all_epochs,
    all_epochs,
    fakepdf_grid,
    flavours: Optional[list] = None,
    ylabels: Optional[list] = None,
    labels: Optional[list] = None,
    plot_args=None,
    plot_provider="bounds",
    normalise_to=None,
    xscale='linear',
    yscale='linear',
    xlim=None,
    ylim=None,
    show_central_fakepdf=False
    ):
    """One figure per flavour overlaying the PDF at every user-selected epoch.

    Both the grids (``pdf_grid_all_epochs``) and their labels (``all_epochs``) are
    collected over the ``Epochspecs`` namespace, i.e. the ``--epochs`` the user
    passed on the command line -- the same epochs as the per-epoch snapshot page,
    *not* ``custom_epochs`` (which only limits the trajectory grids).

    The closure-test underlying-law PDF is overlaid only when the user opted in
    (``show_fakepdf``): otherwise ``fakepdf_grid`` is ``None`` and just the fitted
    PDF is shown. When present the underlying law is the second grid (index 1), so
    ``normalise_to: 1`` in the runcard gives a ratio-to-truth plot.
    """
    flavours, ylabels = _resolve_flavours(flavours, ylabels)

    if labels is None:
        labels = [rf"Epoch {epoch}" for epoch in all_epochs]
    
    # Make a deep copy of the list of grids
    grids = copy(pdf_grid_all_epochs)
    if fakepdf_grid is not None:
        grids.append(fakepdf_grid)
        labels.append(r"$\rm Underlying\ law$")
        if show_central_fakepdf:
            plot_args = [None, {"linestyle": "solid", "color": "black"}]

    yield from plot_grids(grids,
                          flavours=flavours,
                          ylabels=ylabels,
                          labels=labels,
                          plot_args=plot_args,
                          plot_provider=plot_provider,
                          normalise_to=normalise_to,
                          xscale=xscale,
                          yscale=yscale,
                          xlim=xlim,
                          ylim=ylim
                          )
