"""
This module contains plot providers for PDF plots.
"""

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
