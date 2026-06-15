import logging
import re
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

from ntkpdf.utils import peak_rss_gb

log = logging.getLogger(__name__)


def _slug(text):
    """Filename-safe slug for a title fragment."""
    return re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_") or "x"


def _grouped_trajectory_plots(
    grids_by_fit,
    Rankspecs,
    *,
    ylabel,
    name_kind,
    PDFscalespecs=None,
    Replicaspecs=None,
    error_type="mean",
    legend_outside=False,
    ymin=None,
    ymax=None,
):
    """Shared engine for the per-(fit, name) eigenvalue trajectory pages.

    The grid collect (``grids_by_fit``) is resolved **once** -- the calling provider
    is invoked a single time per (fit, name), *outside* any ``{@with ...@}`` loop --
    and the rank groups (``Rankspecs``), x-scales (``PDFscalespecs``) and replicas
    (``Replicaspecs``) are iterated here in Python. So the heavy (e.g. ~400-epoch)
    grid is built once instead of re-collected at every presentation leaf, which is
    what exhausted memory before (reportengine re-makes a ``collect`` -- and its whole
    subtree -- at every request site, and never frees the results).

    ``Replicaspecs is None`` -> ensemble bands; a list (possibly empty) -> single-
    replica line plots, one set per selected replica (empty list -> nothing).
    """
    scalespecs = PDFscalespecs or [{}]
    if Replicaspecs is None:
        replicas = [None]                                       # ensemble
    else:
        replicas = [r["replica_index"] for r in Replicaspecs]   # [] -> no figures

    for ridx in replicas:
        if ridx is None:
            draw_fn = partial(draw_band, error_type=error_type)
            handler_kw = {}                                      # ntk_plot_provider band default
            rep_tag, rep_title = "", ""
        else:
            draw_fn = partial(_draw_one_replica, replica_index=ridx)
            handler_kw = {"custom_handler": None}                # single line -> plain legend
            rep_tag, rep_title = f"replica_{ridx}_", rf" ($\rm replica\ {ridx}$)"

        for sspec in scalespecs:
            xscale = sspec.get("xscale")
            xtitle = sspec.get("Xscaletitle", "")
            for rspec in Rankspecs:
                rtitle = rspec.get("rank_title", "")
                yield from ntk_plot_provider(
                    grids_by_fit,
                    rspec["rank_indices"],
                    draw_fn=draw_fn,
                    iterator_fn=iter_by_fit,
                    title_fn=(lambda grid, rt=rtitle, xt=xtitle, rp=rep_title:
                              f"{grid.label}{rp} -- {rt}" + (f" ({xt})" if xt else "")),
                    name_fn=(lambda grid, rt=rtitle, xt=xtitle, rp=rep_tag, nk=name_kind:
                             f"{nk}_{rp}{grid.label}_{_slug(rt)}" + (f"_{_slug(xt)}" if xt else "")),
                    ylabel_fn=lambda _: ylabel,
                    xscale=xscale,
                    yscale=rspec.get("yscale"),
                    ymin=ymin,
                    ymax=ymax,
                    legend_outside=legend_outside,
                    **handler_kw,
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
    legend_outside: bool = False,
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
        legend_outside=legend_outside,
    )


# =============================================================================
# {@with@}-friendly leaf providers: take the SINGLE (fit-anchored) grid, not the
# *_grids_by_fit collect. Because the grid is one shared node, putting these inside
# {@with Rankspecs@}/{@with PDFscalespecs@} slices a grid that is built once, instead
# of the collect rebuilding the (1000-epoch) grid per leaf. Single current fit only
# (use the *_grouped providers for multi-fit overlays).
# =============================================================================

@_figuregen
def plot_feature_eigvals_rank(
    h_val_grid,
    rank_indices: Optional[list] = None,
    error_type: str = "mean",
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    legend_outside: bool = False,
):
    """Ensemble feature eigenvalues h^(k) vs epoch for one rank group (one figure).

    ``{@with@}`` counterpart of ``plot_feature_eigvals_grouped``: takes the single
    ``h_val_grid`` (fit-anchored, built once) so the template can drive the rank/scale
    faceting and section headings while the grid is sliced, not rebuilt.
    """
    yield from ntk_plot_provider(
        [h_val_grid],
        rank_indices,
        draw_fn=partial(draw_band, error_type=error_type),
        iterator_fn=iter_by_fit,
        title_fn=lambda grid: grid.label,
        name_fn=lambda grid: f"h_{grid.label}",
        ylabel_fn=lambda _: _H_YLABEL,
        xscale=xscale,
        yscale=yscale,
        ymin=ymin,
        ymax=ymax,
        legend_outside=legend_outside,
    )


@_figuregen
def plot_feature_eigvals_replica_rank(
    h_val_grid,
    replica_index: int,
    rank_indices: Optional[list] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    legend_outside: bool = False,
):
    """Single-replica feature eigenvalues h^(k) vs epoch for one rank group (one figure).
    ``{@with@}`` counterpart of ``plot_feature_eigvals_replicas_grouped`` (single grid)."""
    yield from ntk_plot_provider(
        [h_val_grid],
        rank_indices,
        draw_fn=partial(_draw_one_replica, replica_index=replica_index),
        iterator_fn=iter_by_fit,
        custom_handler=None,
        title_fn=lambda grid: rf"{grid.label} ($\rm replica\ {replica_index}$)",
        name_fn=lambda grid: f"h_replica_{replica_index}_{grid.label}",
        ylabel_fn=lambda _: _H_YLABEL,
        xscale=xscale,
        yscale=yscale,
        ymin=ymin,
        ymax=ymax,
        legend_outside=legend_outside,
    )


@_figuregen
def plot_eigvals_rank(
    eigenvalue_grid,
    rank_indices: Optional[list] = None,
    error_type: str = "mean",
    yscale: Optional[str] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    legend_outside: bool = False,
):
    """Ensemble NTK eigenvalues lambda^(k) vs epoch for one rank group (one figure).
    ``{@with@}`` counterpart of ``plot_eigvals_grouped`` (single ``eigenvalue_grid``)."""
    yield from ntk_plot_provider(
        [eigenvalue_grid],
        rank_indices,
        draw_fn=partial(draw_band, error_type=error_type),
        iterator_fn=iter_by_fit,
        title_fn=lambda grid: grid.label,
        name_fn=lambda grid: f"eigvals_{grid.label}",
        ylabel_fn=lambda _: _LAMBDA_YLABEL,
        yscale=yscale,
        ymin=ymin,
        ymax=ymax,
        legend_outside=legend_outside,
    )


@_figuregen
def plot_eigvals_replica_rank(
    eigenvalue_grid,
    replica_index: int,
    rank_indices: Optional[list] = None,
    yscale: Optional[str] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    legend_outside: bool = False,
):
    """Single-replica NTK eigenvalues lambda^(k) vs epoch for one rank group (one figure).
    ``{@with@}`` counterpart of ``plot_eigvals_replicas_grouped`` (single grid)."""
    yield from ntk_plot_provider(
        [eigenvalue_grid],
        rank_indices,
        draw_fn=partial(_draw_one_replica, replica_index=replica_index),
        iterator_fn=iter_by_fit,
        custom_handler=None,
        title_fn=lambda grid: rf"{grid.label} ($\rm replica\ {replica_index}$)",
        name_fn=lambda grid: f"eigvals_replica_{replica_index}_{grid.label}",
        ylabel_fn=lambda _: _LAMBDA_YLABEL,
        yscale=yscale,
        ymin=ymin,
        ymax=ymax,
        legend_outside=legend_outside,
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
    legend_outside: bool = False,
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
        legend_outside=legend_outside,
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
    legend_outside: bool = False,
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
        legend_outside=legend_outside,
    )


# =============================================================================
# Grouped providers: resolve the grid ONCE per (fit, name), loop ranks/scales/
# replicas internally. The report calls these directly (no {@with ...@} loops),
# which is what keeps the heavy NTK grid from being rebuilt at every leaf.
# =============================================================================

_LAMBDA_YLABEL = r"$\textrm{NTK eigenvalues}$"
_H_YLABEL = r"$\textrm{Feature eigenvalues}$"


@_figuregen
def plot_eigvals_grouped(
    eigval_grids_by_fit,
    Rankspecs,
    error_type: str = "mean",
    legend_outside: bool = False,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
):
    """Ensemble NTK eigenvalues lambda^(k) vs epoch -- one figure per ``Rankspecs``
    group. The eigenvalue grid is resolved once here (call this once per fit/name)."""
    log.info("[plot] NTK eigenvalues (ensemble): %d rank group(s) | peak RSS %.1f GB",
             len(Rankspecs), peak_rss_gb())
    yield from _grouped_trajectory_plots(
        eigval_grids_by_fit, Rankspecs,
        ylabel=_LAMBDA_YLABEL, name_kind="eigvals",
        error_type=error_type, legend_outside=legend_outside, ymin=ymin, ymax=ymax,
    )


@_figuregen
def plot_eigvals_replicas_grouped(
    eigval_grids_by_fit,
    Rankspecs,
    Replicaspecs,
    legend_outside: bool = False,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
):
    """Single-replica NTK eigenvalues -- one figure per (selected replica, Rankspecs
    group). Empty ``Replicaspecs`` -> nothing. Grid resolved once."""
    log.info("[plot] NTK eigenvalues (single replicas): %d replica(s) x %d rank "
             "group(s) | peak RSS %.1f GB", len(Replicaspecs), len(Rankspecs), peak_rss_gb())
    yield from _grouped_trajectory_plots(
        eigval_grids_by_fit, Rankspecs, Replicaspecs=Replicaspecs,
        ylabel=_LAMBDA_YLABEL, name_kind="eigvals",
        legend_outside=legend_outside, ymin=ymin, ymax=ymax,
    )


@_figuregen
def plot_feature_eigvals_grouped(
    h_val_grids_by_fit,
    Rankspecs,
    PDFscalespecs=None,
    error_type: str = "mean",
    legend_outside: bool = False,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
):
    """Ensemble feature eigenvalues h^(k) vs epoch -- one figure per (x-scale,
    Rankspecs group). The h grid is resolved once here (call this once per
    fit/name)."""
    log.info("[plot] feature eigenvalues h (ensemble): %d rank group(s) | peak RSS %.1f GB",
             len(Rankspecs), peak_rss_gb())
    yield from _grouped_trajectory_plots(
        h_val_grids_by_fit, Rankspecs, PDFscalespecs=PDFscalespecs,
        ylabel=_H_YLABEL, name_kind="h",
        error_type=error_type, legend_outside=legend_outside, ymin=ymin, ymax=ymax,
    )


@_figuregen
def plot_feature_eigvals_replicas_grouped(
    h_val_grids_by_fit,
    Rankspecs,
    Replicaspecs,
    PDFscalespecs=None,
    legend_outside: bool = False,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
):
    """Single-replica feature eigenvalues -- one figure per (selected replica,
    x-scale, Rankspecs group). Empty ``Replicaspecs`` -> nothing. Grid resolved
    once."""
    log.info("[plot] feature eigenvalues h (single replicas): %d replica(s) x %d rank "
             "group(s) | peak RSS %.1f GB", len(Replicaspecs), len(Rankspecs), peak_rss_gb())
    yield from _grouped_trajectory_plots(
        h_val_grids_by_fit, Rankspecs, PDFscalespecs=PDFscalespecs,
        Replicaspecs=Replicaspecs, ylabel=_H_YLABEL, name_kind="h",
        legend_outside=legend_outside, ymin=ymin, ymax=ymax,
    )
