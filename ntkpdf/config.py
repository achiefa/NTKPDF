"""ntkpdf.config.py

This module contains the configuration and environment classes for the ntkpdf app.
"""

import json
import logging
import numpy as np

from collections.abc import Iterable
from numbers import Integral

from colibri.config import colibriConfig
from colibri.ntk import NTKStats
from n3fit.model_gen import generate_pdf_model, ReplicaSettings
from reportengine.namespaces import NSList
from reportengine.configparser import ConfigError

from ntkpdf.utils import BestEpochStat, INPUT_GRID, EVOL_INDEX

log = logging.getLogger(__name__)

NO_PREFACTORS = frozenset({"remove_prefactors": True}.items())
NN = frozenset({"exclude_layers":('impose_msr',), "remove_prefactors": True}.items())
NO_SMR = frozenset({"exclude_layers":('impose_msr')}.items()) 
FULL_MODEL = frozenset({})

#----------------------------------------
# Utilities functions
#########################################
def _sort_replicas(replicas):
    return sorted(replicas, key=lambda r: int(r.stem.split("_")[-1]))

def _unpack_params(shapes, weights, prepr):
    """Returns the weight parameters in the format expected by the n3fit model."""
    wight_list = []
    pos = 0
    for shape in shapes:
        size = np.prod(shape)
        param = weights[pos:pos+size]
        param = param.reshape(shape)
        wight_list.append(param)
        pos += size

    params_dict = dict(
        all_NNs=wight_list,
        preprocessing_factor=prepr
    )
    return params_dict

def set_epoch_weights(metamodels, weights, fit_preprocessing, nreplicas):
    """Load one epoch's per-replica ``weights`` (flat vectors) and
    ``fit_preprocessing`` into the shared ``metamodels`` in place, and return it.
    Shared by the per-epoch providers and by the streaming loss loop in
    ``ntkpdf.data_theory`` (kept module-level so the latter need not go through a
    config method)."""
    shapes = [list(v.shape) for v in metamodels.get_replica_weights(0)["all_NNs"]]
    for idx in range(nreplicas):
        parameters = _unpack_params(shapes, weights[idx], fit_preprocessing[idx])
        metamodels.set_replica_weights(parameters, idx)
    return metamodels

def predict_pdf_grid(metamodels, input_grid, vetoes=None):
    """Predict the PDF grid from ``metamodels`` and wrap it as ``NTKStats``."""
    pdf_grids = metamodels.predict({"pdf_input": input_grid.T}, verbose=False)[0]
    if vetoes is not None:
        pdf_grids = pdf_grids[vetoes]
    pdf_grids = pdf_grids.transpose((0, 2, 1)).reshape(pdf_grids.shape[0], -1, 1)
    f = NTKStats(pdf_grids)
    f.set_index(EVOL_INDEX, None)
    return f

def _apply_model_selector(pdf_model, nreplicas, model_selector=None):
    """Set the preprocessing (prefactor) weights of ``pdf_model`` in place
    according to ``model_selector``, and return it.

    The expensive replica networks are built once in
    ``produce_base_metamodels``; this only overwrites the cheap preprocessing
    weights, so different selectors of the same base share the same networks
    instead of being rebuilt. The preprocessing is always fully (re)set to a
    well-defined state -- the original snapshot (``_ntk_base_preprocessing``)
    for the default case, or the neutral exponents for ``no_prefactors`` -- so
    the result is independent of the order in which selectors are applied to the
    shared, mutated base.
    """
    no_prefactors = model_selector is not None and model_selector.get("no_prefactors")
    if no_prefactors:
        log.info("Generating PDF model with no prefactors (i.e. small-x and large-x exponents set to 1 and 0, respectively).")

    for i in range(nreplicas):
        original = pdf_model._ntk_base_preprocessing[i]
        if no_prefactors:
            # Preprocessing weights are ordered [alpha, beta] per flavour and the
            # prefactor x^(1-alpha)*(1-x)^beta is neutralised by alpha=1, beta=0.
            new_prepr = [
                np.full_like(w, 1.0 if j % 2 == 0 else 0.0)
                for j, w in enumerate(original)
            ]
        else:
            new_prepr = [w.copy() for w in original]

        replica_weights = pdf_model.get_replica_weights(i)
        replica_weights["preprocessing_factor"] = new_prepr
        pdf_model.set_replica_weights(replica_weights, i)

    return pdf_model

#----------------------------------------
# Config class
#########################################
class ntkConfig(colibriConfig):
    """Configuration class for ntkpdf, inheriting from colibriConfig."""

    def produce_kwargs(self, name):
        if name == "model":
            return FULL_MODEL
        elif name == "no_prefactors":
            return NO_PREFACTORS
        elif name == "nn":
            return NN
        elif name == "no_smr":
            return NO_SMR
        else:
            raise ConfigError(f"Unknown model selector name: {name}")
            

    def produce_input_grid(self):
        return INPUT_GRID

    def produce_wreplica_paths(self, fit):
        """Returns the paths to the weight replicas for the specified fit."""
        path = fit.path / "fit_replicas"
        sorted_replicas = _sort_replicas(path.glob("replica_*"))
        return NSList(sorted_replicas, nskey="wreplica_path")
    
    def parse_model_selectors(self, model_selectors: list):
        """Global list of model selectors. Each entry is a mapping with a
        ``name`` and the modifier flags (e.g. ``no_prefactors``). This is no
        longer a collect dimension: the per-selector grids are built together in
        :meth:`produce_init_grids_by_name`, and the presentation layer selects
        them by ``name``."""
        return list(model_selectors)
    
    def produce_replica_paths(self, fit):
        """Returns the paths to the weight replicas for the specified fit."""
        path = fit.path / "nnfit"
        sorted_replicas = _sort_replicas(path.glob("replica_*"))
        return NSList(sorted_replicas, nskey="replica_path")
    
    def produce_replicas(self, fit):
        """Returns the list of replica indices for the specified fit."""
        path = fit.path / "nnfit"
        sorted_replicas = _sort_replicas(path.glob("replica_*"))
        replica_indices = [int(r.stem.split("_")[-1]) for r in sorted_replicas]
        return NSList(replica_indices, nskey="replica")
   
    def produce_nreplicas(self, fit, replica_ids=None, custom_nreplicas=None):
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
    
    def produce_fit_weights(self, fit, epoch, replica_ids=None):
        """Load the fit weights for the given fit and epoch(s), in the format
        expected by the n3fit model. ``epoch`` is either a single int (broadcast
        to all replicas) or one int per replica."""
        path = fit.path / "fit_replicas"
        if replica_ids is None:
            rf_paths = _sort_replicas(path.glob("replica_*"))
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

    def produce_fit_preprocessing(self, fit, replica_ids=None):
        parameters = []

        # Get weight parameters
        path = fit.path / "nnfit"
        if replica_ids is None:
            rf_paths = _sort_replicas(path.glob("replica_*"))
        else:
            rf_paths = [path / f"replica_{rid}" for rid in replica_ids]
          
        for rf in rf_paths:
          with open(rf / f"{fit.name}.json", "r", encoding="utf-8") as f:
              jf = json.load(f)
              prep = jf['preprocessing']
              prepr_weights = []
              for fl in prep:
                prepr_weights.append(fl["smallx"])
                prepr_weights.append(fl["largex"])
          parameters.append(prepr_weights)
        return parameters

    def produce_best_epochs(self, fit):
        """Get the distribution of best epochs across replicas for a given fit"""
        fitpath = fit.path
        fit_replicas = fitpath / "nnfit"
        best_epochs = []

        replica_paths = list(_sort_replicas(fit_replicas.glob("replica_*")))
        replica_paths.sort(key=lambda f: int(f.stem.split("_")[-1]))

        for rep in replica_paths:
            jsonfile = rep / f"{fit.name}.json"
            with open(jsonfile, "r", encoding="utf-8") as f:
                jf = json.load(f)
                best_epoch = (jf['best_epoch'])
                best_epochs.append(best_epoch)

        stopping_points_stat = BestEpochStat(np.array(best_epochs).reshape(-1, 1))
        return stopping_points_stat
    
    def produce_stored_epochs(self, fit, best_epochs):
        """
        Get the list of stored epochs for a given fit, excluding the best epoch.
        """
        fitpath = fit.path

        # Assume all replicas have the same stored epochs
        fit_replica = fitpath / "fit_replicas/replica_1/parameters"

        # Get best epoch to exclude it from the list of (common stored epochs)
        best_epoch = best_epochs.data[0]
        epochs = []
        for epoch_file in fit_replica.glob("params_*.npz"):
            epoch = int(epoch_file.stem.split("_")[-1])
            # Exclude the best epoch
            if epoch != best_epoch:
              epochs.append(epoch)

        epochs_sorted = sorted(epochs)
        return NSList(epochs_sorted, nskey="epoch")
    
    def produce_epochs(self, stored_epochs, custom_epochs=None):
        """
        Get the list of epochs to evaluate, either from the stored epochs or from a custom list provided by the user.
        """
        if custom_epochs is not None:
            return NSList(custom_epochs, nskey="epoch")
        return stored_epochs
    
    def produce_model_info(self, fit):
        fit_info = fit.as_input()
        model_dict = {
            "basis_info": fit_info["fitting"]["basis"],
            "fitbasis": fit_info["fitting"]["fitbasis"],
            "nodes": fit_info["parameters"]["nodes_per_layer"],
            "activations": fit_info["parameters"]["activation_per_layer"],
            "initializer": fit_info["parameters"]["initializer"],
            "initializer_scale": fit_info["parameters"].get("initializer_scale", 1.0),
            # Read back the init-width exponent the fit used (default 1.0 = standard
            # glorot) so the rebuilt network-at-initialisation -- used as f0 -- matches
            # the fit's actual initialisation, not a default one.
            "initializer_gamma": fit_info["parameters"].get("initializer_gamma", 1.0),
            "layer_type": fit_info["parameters"]["layer_type"],
            "dropout": fit_info["parameters"]["dropout"],
            "impose_sumrule": fit_info.get('sum_rules', "All"),
            "optimizer": fit_info["parameters"]["optimizer"]["optimizer_name"],
            "learning_rate": fit_info["parameters"]["optimizer"]["learning_rate"],
            "clipnorm": fit_info["parameters"]["optimizer"]["clipnorm"],
        }
        return model_dict
    
    def produce_learning_rate(self, model_info):
        return model_info["learning_rate"]
    
    def produce_base_metamodels(self, model_info, nreplicas):
        """Build the raw PDF model, with random initialisation and the fit's
        original (with-prefactor) preprocessing.

        This is deliberately independent of ``model_selector``: the cheap,
        per-selector difference (the preprocessing prefactor) is applied later in
        :meth:`produce_metamodels`.
        """
        replica_settings_list = []
        log.info(f"Building base PDF model for {nreplicas} replicas.")

        # Build the model in double precision so its *weights* are float64 (not
        # merely a float64-typed output of a float32 model). colibri's
        # ``Environment`` has already enabled ``jax_enable_x64`` by the time this
        # runs, so float64 here is genuine; this matches what colibri-n3fit's
        # ``N3FitPDFModel`` forces on the NTK path, keeping the two model-build
        # paths at the same precision. Done here (at build time) rather than at
        # import to avoid toggling JAX state before it initialises.
        import keras

        keras.backend.set_floatx("float64")

        for i in range(nreplicas):
          # Initialize the replica setting for each replica
          replica_settings = ReplicaSettings(
              seed=i,
              nodes=model_info["nodes"],
              activations=model_info["activations"],
              architecture=model_info["layer_type"],
              initializer=model_info["initializer"],
              initializer_scale=model_info["initializer_scale"],
              initializer_gamma=model_info["initializer_gamma"],
              dropout_rate=model_info["dropout"],
              regularizer=None,
              regularizer_args={})
          replica_settings_list.append(replica_settings)

        # Generate pdf model
        pdf_model = generate_pdf_model(
          replicas_settings=replica_settings_list,
          flav_info=model_info["basis_info"],
          fitbasis=model_info["fitbasis"],
          out=14,
          impose_sumrule=model_info["impose_sumrule"],
          scaler=None
        )

        # Snapshot the as-built (with-prefactor) preprocessing weights so that
        # produce_metamodels can deterministically restore them, regardless of
        # the order in which selectors mutate this shared, memoised model.
        pdf_model._ntk_base_preprocessing = [
            pdf_model.get_replica_weights(i)["preprocessing_factor"]
            for i in range(nreplicas)
        ]
        return pdf_model

    def produce_metamodels(self, base_metamodels, nreplicas, model_selector=None):
        """Return the raw PDF model with its preprocessing set according to
        ``model_selector`` (see :func:`_apply_model_selector`). The expensive
        replica networks come from :meth:`produce_base_metamodels`; only the
        cheap prefactor weights are overwritten here.
        """
        return _apply_model_selector(base_metamodels, nreplicas, model_selector)

    def produce_init_grids_by_name(self, base_metamodels, nreplicas, model_selectors):
        """Build the PDF plotting grids at initialisation for every model
        selector, once, and return a ``{selector_name: XPlottingGrid}`` mapping.

        This is a config production that depends only on ``base_metamodels``
        (i.e. on the fit and ``nreplicas``) and the global ``model_selectors``
        list -- it does *not* depend on the ``Selectors``/``PDFscalespecs``
        presentation namespaces. Reportengine therefore resolves it once per fit
        and replica count and shares the result across those namespaces, instead
        of rebuilding the (expensive) base model and re-predicting inside every
        ``{@with ...@}`` loop. A reportengine ``collect`` cannot give this
        sharing: a collect node is re-made at every request site, so collecting
        the grids over ``model_selectors`` from inside the loops rebuilt
        everything per loop iteration.

        The selectors are applied in a plain loop (rather than via a collect)
        precisely so this stays a single shared production. ``base_metamodels``
        is mutated in place by :func:`_apply_model_selector`, which is safe
        because each grid is materialised (predictions evaluated into a
        NumPy-backed ``XPlottingGrid``) before the next selector is applied.
        """
        # Imported lazily: these pull in the keras/jax-backed stack, which must
        # only happen after ``ntkpdf`` has set ``KERAS_BACKEND`` (see
        # ``ntkpdf/__init__.py``).
        from n3fit.vpinterface import EVOL_LIST, N3PDF
        from validphys.pdfgrids import xplotting_grid
        from ntkpdf.utils import XGRID

        grids = {}
        for selector in model_selectors:
            model = _apply_model_selector(base_metamodels, nreplicas, selector)
            grids[selector["name"]] = xplotting_grid(
                N3PDF(model.split_replicas()), 1.65, XGRID, "evolution", EVOL_LIST
            )
        return grids
    
    def produce_pdf_at_init(self, metamodels, input_grid):
        """Return the model at initialisation in the NTKStats format."""
        return predict_pdf_grid(metamodels, input_grid)
    
    def produce_model_at_epoch(self, fit_weights, fit_preprocessing, metamodels, nreplicas, vetoname=None):
        """Get the PDF model at the specified epoch, with the corresponding weights."""
        pdf_models = set_epoch_weights(metamodels, fit_weights, fit_preprocessing, nreplicas)

        vetoes = np.array([True] * nreplicas)
        if vetoname:
            raise NotImplementedError("Vetoes are not implemented yet. Please implement the veto logic in the produce_pdf_model_at_epochs method.")
            # from ntk.veto import collect_vetoes
            # vetoes = collect_vetoes(fitname, epochs, pdf_models, vetoname)["Total"]

        return pdf_models, vetoes

    def produce_pdf_at_epoch(self, model_at_epoch, input_grid):
        pdf_models, vetoes = model_at_epoch
        return predict_pdf_grid(pdf_models, input_grid, vetoes=vetoes)

    def produce_pdf_grid_at_epoch(self, model_at_epoch):
        """PDF plotting grid (``XPlottingGrid``) for the model at a given epoch.

        This is a *production*, not a plain provider, and that is load-bearing.
        ``produce_model_at_epoch`` loads each epoch's weights into the *shared*
        base model **in place** and returns that same object, so the grid is
        only valid while the model still holds this epoch's weights. As a
        ``produce_`` method this runs eagerly at graph-build time, interleaved
        with its own epoch's mutation -- ``model_at_epoch`` is resolved (the
        model is set to this epoch) and the prediction is materialised into a
        detached NumPy grid immediately after, before the next epoch's
        production overwrites the model.

        A deferred provider would instead run at *execution* time, after every
        epoch's production has already mutated the shared model, so the model
        would hold only the last epoch's weights and all collected grids (see
        ``pdf_grid_all_epochs``) would be identical.
        """
        # Lazy imports: these pull in the keras/jax-backed stack, which must only
        # happen after ``ntkpdf`` has set ``KERAS_BACKEND`` (see ``__init__.py``).
        from n3fit.vpinterface import EVOL_LIST, N3PDF
        from validphys.pdfgrids import xplotting_grid
        from ntkpdf.utils import XGRID

        pdf_models, _vetoes = model_at_epoch
        return xplotting_grid(
            N3PDF(pdf_models.split_replicas()), 1.65, XGRID, "evolution", EVOL_LIST
        )

    def produce_fakepdf(self, fit):
        """The closure-test underlying-law PDF (``closuretest: fakepdf``) of the
        fit, parsed into a validphys PDF. Mirrors the notebook's
        ``fit.as_input()['closuretest']['fakepdf']``; raises if the fit is not a
        closure test."""
        try:
            name = fit.as_input()["closuretest"]["fakepdf"]
        except KeyError as e:
            raise ConfigError(
                f"fit {fit.name} has no 'closuretest: fakepdf' (not a closure test); "
                "set show_fakepdf: False (the default) to skip the underlying-law overlay."
            ) from e
        return self.parse_pdf(name)

    def produce_fakepdf_grid(self, fit, show_fakepdf=False):
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
        # Lazy import: keras/jax-backed stack, only after KERAS_BACKEND is set.
        from validphys.pdfgrids import xplotting_grid
        from n3fit.vpinterface import EVOL_LIST
        from ntkpdf.utils import XGRID

        fakepdf = self.produce_fakepdf(fit)
        return xplotting_grid(fakepdf, 1.65, XGRID, "evolution", EVOL_LIST)

    def produce_fitting_covmat_eigensystem(self, fit, fitting_covmat_name):
        """Eigendecomposition of the fitting covariance matrix, loaded **once**.

        This is a *production* (not a plain provider) on purpose: the eigenvector
        matrix is ``(Ndata, Ndata)`` -- hundreds of MB -- and a genuine per-fit
        constant, but the consumers (`fk_diagonal_basis_train_val` etc.) are resolved
        inside the report's nested ``{@with ...@}`` loops. As a provider it was
        recomputed *and retained* once per presentation leaf (the from_: theory/dataset
        taint, see CLAUDE.md), growing memory by ~0.6 GB/leaf -> OOM. As a production it
        is evaluated once at graph-build and shared by every leaf.

        The columns are deliberately left as read from disk: aligning them to the FK
        index needs the ``ntk_fast_kernel_arrays`` provider (unavailable to a
        production), so that cheap, idempotent relabelling is done by the single
        consumer ``fk_diagonal_basis_train_val``.

        Only valid for fits run with ``diagonal_basis=True``.
        """
        if not fit.as_input().get("diagonal_basis", True):
            raise ConfigError(
                f"Fit {fit.name} was not run with diagonal_basis=True; "
                "fitting_covmat_eigensystem expects a diagonal-basis fit."
            )
        import pandas as pd

        eigensystem = pd.read_csv(
            fitting_covmat_name, index_col=[0], header=[0], sep="\t|,", engine="python",
        )
        eigvals = eigensystem.iloc[:, 0].values
        eigvecs = eigensystem.iloc[:, 1:]
        return eigvals, eigvecs