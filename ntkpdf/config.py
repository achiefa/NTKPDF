"""ntkpdf.config.py

This module contains the configuration and environment classes for the ntkpdf app.

Whatever depends on how a fit stores its results (n3fit or colibri) is delegated to the
fit's reader, ``fit_kind`` (see :mod:`ntkpdf.fits`).
"""

import logging
import numpy as np

from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral

from colibri.config import colibriConfig
from colibri.ntk import NTKStats
from reportengine.namespaces import NSList
from reportengine.configparser import ConfigError
from validphys.core import PDF, FitSpec
from validphys.pdfgrids import XPlottingGrid, xplotting_grid

from ntkpdf.data_theory import _build_ntk_fast_kernel_arrays
from ntkpdf.fits import load_fit
from ntkpdf.fits.common import TrainValData, sort_replicas
from ntkpdf.utils import EVOL_LIST, XGRID, plotting_grid_from_ntkstat

log = logging.getLogger(__name__)

NO_PREFACTORS = frozenset({"remove_prefactors": True}.items())
NN = frozenset({"exclude_layers":('impose_msr',), "remove_prefactors": True}.items())
NO_SMR = frozenset({"exclude_layers":('impose_msr',)}.items())
FULL_MODEL = frozenset({})
# Model selector name -> colibri NTK ``kwargs`` (the options of ``grid_values_func``).
SELECTOR_KWARGS = {"model": FULL_MODEL, "no_prefactors": NO_PREFACTORS, "nn": NN, "no_smr": NO_SMR}

def layer_kwargs(layers: Iterable[int], cumulative: bool = True, base: frozenset = NN) -> frozenset:
    """Build the colibri NTK ``kwargs`` frozenset for a layer-restricted NTK on top of
    a base selector. ``layers`` are 1-based NN (Dense) layer indices. With
    ``cumulative`` the gradient is restricted to layers 1..max(layers) (the cumulative
    kernel Theta<=l); otherwise to exactly ``layers`` (e.g. the single-layer Theta_l)."""
    grad = tuple(range(1, max(layers) + 1)) if cumulative else tuple(layers)
    return frozenset({**dict(base), "grad_layers": grad}.items())

def _evolution_grid(pdfs: NTKStats) -> XPlottingGrid:
    """Plotting grid (evolution basis, Q = 1.65 GeV) of PDFs on ``EVOL_INDEX``."""
    return plotting_grid_from_ntkstat(pdfs.reshape((len(EVOL_LIST), len(XGRID))), XGRID, "evolution", EVOL_LIST)

#----------------------------------------
# Config class
#########################################
class ntkConfig(colibriConfig):
    """Configuration class for ntkpdf, inheriting from colibriConfig."""

    def produce_fit_kind(self, fit: FitSpec):
        """The reader of ``fit``, an :class:`~ntkpdf.fits.n3fit.N3Fit` or a
        :class:`~ntkpdf.fits.colibri.ColibriFit`."""
        return load_fit(fit)

    def produce_kwargs(self, fit_kind, name: str) -> frozenset:
        """The colibri NTK ``kwargs`` of the model selector ``name``, which must be one
        of the fit's (a colibri model has no prefactors or network layers)."""
        if name not in fit_kind.selectors:
            raise ConfigError(
                f"Model selector '{name}' does not apply to fit {fit_kind.fit.name}; "
                f"choose from {fit_kind.selectors}."
            )
        return SELECTOR_KWARGS[name]

    def produce_wreplica_paths(self, fit: FitSpec) -> NSList:
        """Returns the paths to the weight replicas for the specified fit."""
        path = fit.path / "fit_replicas"
        sorted_replicas = sort_replicas(path.glob("replica_*"))
        return NSList(sorted_replicas, nskey="wreplica_path")

    def parse_model_selectors(self, model_selectors: list) -> list:
        """Global list of model selectors. Each entry is a mapping with a
        ``name`` and the modifier flags (e.g. ``no_prefactors``). This is no
        longer a collect dimension: the per-selector grids are built together in
        :meth:`produce_init_grids_by_name`, and the presentation layer selects
        them by ``name``."""
        return list(model_selectors)

    def produce_replica_paths(self, fit: FitSpec) -> NSList:
        """Returns the paths to the weight replicas for the specified fit."""
        path = fit.path / "nnfit"
        sorted_replicas = sort_replicas(path.glob("replica_*"))
        return NSList(sorted_replicas, nskey="replica_path")

    def produce_replicas(self, fit: FitSpec) -> NSList:
        """Returns the list of replica indices for the specified fit."""
        path = fit.path / "nnfit"
        sorted_replicas = sort_replicas(path.glob("replica_*"))
        replica_indices = [int(r.stem.split("_")[-1]) for r in sorted_replicas]
        return NSList(replica_indices, nskey="replica")

    def produce_nreplicas(
        self, fit: FitSpec, replica_ids: Sequence[int] | None = None, custom_nreplicas: int | None = None
    ) -> int:
        """Returns the number of replicas for the specified fit."""
        if custom_nreplicas is not None:
            return custom_nreplicas

        path = fit.path / "fit_replicas"
        fit_nreplicas = len(list(path.glob("replica_*")))
        if replica_ids is not None:
            nreplicas = len(replica_ids)
            if nreplicas > fit_nreplicas:
                raise ValueError(f"Number of specified replica_ids ({nreplicas}) exceeds the total number of replicas in the fit ({fit_nreplicas}).")
            return nreplicas
        return fit_nreplicas

    def produce_fit_weights(
        self, fit: FitSpec, epoch: int | Iterable[int], replica_ids: Sequence[int] | None = None
    ) -> list[np.ndarray]:
        """Load the fit weights for the given fit and epoch(s), as flat parameter
        vectors. ``epoch`` is either a single int (broadcast to all replicas) or one int
        per replica."""
        path = fit.path / "fit_replicas"
        if replica_ids is None:
            rf_paths = sort_replicas(path.glob("replica_*"))
        else:
            rf_paths = [path / f"replica_{rid}" for rid in replica_ids]

        if isinstance(epoch, Integral):
            epoch_by_rep = [epoch] * len(rf_paths)
        elif isinstance(epoch, Iterable):
            epoch_by_rep = list(epoch)
            if len(epoch_by_rep) != len(rf_paths):
                raise ConfigError(
                    f"Got {len(epoch_by_rep)} epochs for {len(rf_paths)} replicas."
                )
        else:
            raise ConfigError(
                f"epoch must be an int or an iterable of ints, got {type(epoch)}."
            )

        return [
            np.load(rf / "parameters" / f"params_{e}.npz")["params"]
            for rf, e in zip(rf_paths, epoch_by_rep)
        ]

    def produce_stored_epochs(self, fit_kind) -> NSList:
        """The epochs whose parameters the fit saved (for an n3fit fit, without the best
        epoch)."""
        return NSList(fit_kind.stored_epochs(), nskey="epoch")

    def produce_epochs(self, stored_epochs: NSList, custom_epochs: Sequence[int] | None = None) -> NSList:
        """
        Get the list of epochs to evaluate, either from the stored epochs or from a custom list provided by the user.
        """
        if custom_epochs is not None:
            return NSList(custom_epochs, nskey="epoch")
        return stored_epochs

    def produce_learning_rate(self, fit_kind) -> float:
        return fit_kind.learning_rate()

    def produce_pdf_ensemble(self, fit_kind, nreplicas: int, replica_ids: Sequence[int] | None = None):
        """The fit's PDF model for ``nreplicas`` replicas, which evaluates the PDFs of
        all replicas at once (see ``N3FitEnsemble``/``ColibriEnsemble``).

        Built **once** per fit and replica count and shared by ``pdf_at_init``,
        ``pdf_at_epoch``, ``init_grids_by_name`` and ``loss_function_grid``. This holds
        only while ``fit`` is a direct input: a ``from_:`` would pin this production to
        the deepest namespace and rebuild the model in every presentation loop (see
        ``ntkanalysis.complete_mapping``).
        """
        return fit_kind.ensemble(nreplicas, replica_ids)

    def produce_init_grids_by_name(
        self, pdf_ensemble, model_selectors: Sequence[Mapping]
    ) -> dict[str, XPlottingGrid]:
        """The PDF plotting grids at initialisation for every model selector, as a
        ``{selector_name: XPlottingGrid}`` mapping.

        This production depends only on the fit, the replica count and the global
        ``model_selectors`` list -- not on the ``Selectors``/``PDFscalespecs``
        presentation namespaces, which pick their grids from it by name. A
        reportengine ``collect`` over ``model_selectors`` would be re-made at every
        request site, and so would the model.
        """
        return {selector["name"]: _evolution_grid(pdf_ensemble.at_init(selector)) for selector in model_selectors}

    def produce_pdf_at_init(self, pdf_ensemble, model_selector: Mapping | None = None) -> NTKStats:
        """The PDFs at initialisation, in the NTKStats format."""
        return pdf_ensemble.at_init(model_selector)

    def produce_pdf_at_epoch(
        self, pdf_ensemble, fit_weights: Sequence[np.ndarray], model_selector: Mapping | None = None
    ) -> NTKStats:
        """The PDFs at the epoch of ``fit_weights``, in the NTKStats format."""
        return pdf_ensemble.at(fit_weights, model_selector)

    def produce_pdf_grid_at_epoch(self, pdf_at_epoch: NTKStats) -> XPlottingGrid:
        """PDF plotting grid (``XPlottingGrid``) at a given epoch."""
        return _evolution_grid(pdf_at_epoch)

    def produce_fakepdf(self, fit_kind) -> PDF:
        """The closure-test underlying-law PDF of the fit, parsed into a validphys PDF;
        raises if the fit is not a closure test."""
        return self.parse_pdf(fit_kind.fakepdf())

    def produce_fakepdf_grid(self, fit_kind, show_fakepdf: bool = False) -> XPlottingGrid | None:
        """Plotting grid for the closure-test underlying-law PDF, or ``None`` when
        the user did not opt in via ``show_fakepdf``.

        Gated here (rather than as a plain provider) so that a provider depending
        on ``fakepdf_grid`` does not *force* the fakepdf to be built/parsed: when
        ``show_fakepdf`` is False the underlying law is never looked up, so the
        page also works for fits that are not closure tests. Same Q/xgrid/basis/
        flavours as ``pdf_grid_at_epoch`` so the two overlay and normalise.
        """
        if not show_fakepdf:
            return None
        return xplotting_grid(self.produce_fakepdf(fit_kind), 1.65, XGRID, "evolution", EVOL_LIST)

    def produce_ntk_fast_kernel_arrays(
        self, fit_kind, padding: bool = True, flavour_basis: bool = False
    ):
        """FK tables for the fit's data, built **once** per fit as a production.

        A production (not a provider) because the FK is a per-fit constant: as a
        provider it resolved to a separate node per presentation leaf (~200x) ->
        reloaded + retained -> OOM. The fit's reader builds its ``data`` with the fit's
        own theory and cuts, outside this config's namespaces, so the FK depends on
        ``fit`` alone and never on the global, floating ``data``.
        """
        return _build_ntk_fast_kernel_arrays(fit_kind.data(), padding=padding, flavour_basis=flavour_basis)

    def produce_training_data(self, fit_kind, ntk_fast_kernel_arrays) -> TrainValData:
        """The per-replica pseudodata, FK and inverse covmat of the training and
        validation losses, built once per fit, each kind of fit as it was fitted (see
        ``N3Fit``/``ColibriFit.training_data``)."""
        return fit_kind.training_data(ntk_fast_kernel_arrays)
