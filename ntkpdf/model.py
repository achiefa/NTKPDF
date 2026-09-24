"""
ntkpdf.model.py

The n3fit PDF model interface used by the NTK machinery, and the loader that
builds it from a fit on disk.

This is ntkpdf's replacement for ``colibri-n3fit``. That package exists to drive
*colibri's fitting machinery* with an n3fit network; ntkpdf never runs a fit, so the
only piece it needs is :meth:`NTKPDFN3Fit.grid_values_func` -- the pure-JAX function of
the flat weight vector that ``jax.jacfwd`` differentiates into the NTK
(``colibri.ntk.ntkutils.compute_ntk``) and ``jax.hessian`` into the loss Hessian
(:mod:`ntkpdf.hessian`).

Two halves:

- :class:`NTKPDFN3Fit` -- a port of ``colibri_n3fit.model.N3FitPDFModel`` (branch
  ``layer_selector``, including its unstaged work), minus the fit-time bits.
- :func:`get_pdf_model` / :func:`install` -- a drop-in replacement for
  ``colibri.utils.get_pdf_model``, rebound over colibri's at import time (see
  :func:`install`). colibri's version loads ``pdf_model.pkl`` and hardcodes
  ``from colibri_n3fit.model import N3FitPDFModel``; ours reads the fit's own
  ``filter.yml`` instead, so no pickle is involved and ``colibri_n3fit`` is never
  imported.
"""

import importlib
import json
import logging
import math

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from jax import lax

from colibri.pdf_model import PDFModel
from n3fit.backends.keras_backend.MetaModel import PREPROCESSING_LAYER_ALL_REPLICAS
from n3fit.model_gen import ReplicaSettings, _pdfNN_layer_generator
from validphys.n3fit_data import replica_nnseed

log = logging.getLogger(__name__)

# Modules that bind ``get_pdf_model`` with ``from colibri.utils import ...``, and so
# hold their own reference that patching ``colibri.utils`` alone would not reach.
_PATCH_TARGETS = (
    "colibri.utils",  # the definition; also covers ntkpdf.hessian, which imports
    #                   inside the function body and so resolves at call time
    "colibri.ntk.ntkutils",  # eigenvalues and eigenvectors per replica
    "colibri.ntk.ntk",  # ntk_ensemble_at_epoch
    "colibri.checks",  # fitting-side, off ntkpdf's path; rebound for uniformity
)


# ----------------------------------------
# The model
#########################################
def _nn_layer_param_ranges(model):
    """Flat-param ``(start, stop)`` index range for each NN ``Dense`` layer of ``model``,
    in input->output order. Ranges index the flat weight vector packed in
    ``grid_values_func`` (i.e. ``model.trainable_variables`` order). Each layer's
    variables (kernel, bias) are contiguous there, so a single range covers the layer.
    """
    groups = {}  # layer key -> [start, stop, layer_name]
    order = []
    pos = 0
    for v in model.trainable_variables:
        size = int(np.prod(v.shape))
        parts = getattr(v, "path", v.name).split("/")
        key = "/".join(parts[:-1])
        layer_name = parts[-2] if len(parts) >= 2 else parts[0]
        if key not in groups:
            groups[key] = [pos, pos + size, layer_name]
            order.append(key)
        else:
            groups[key][1] = pos + size
        pos += size
    return [tuple(groups[k][:2]) for k in order if groups[k][2].startswith("dense")]


class NTKPDFN3Fit(PDFModel):
    """The n3fit PDF network as a colibri :class:`PDFModel`.

    Accepts the same constructor arguments as ``colibri_n3fit``'s ``N3FitPDFModel``, so
    it can be built from the ``_init_args`` dict n3fit writes into ``pdf_model.pkl`` as
    well as from a fit runcard (:func:`_model_init_args`).
    """

    def __init__(
        self,
        replica_index,
        nnseed,
        flav_info: list,
        replica_range_settings: dict = {"min_replica": 1, "max_replica": 1},
        impose_sumrule: bool = True,
        fitbasis: str = "EVOL",
        nodes: list = [25, 20, 8],
        activations: list = ["tanh", "tanh", "linear"],
        initializer_name: str = "glorot_normal",
        **kwargs,
    ):
        """
        Args:
            **kwargs: Arguments for n3fit model configuration
                      (nodes, activations, initializer_name, etc.)

        ``initializer_name`` is accepted (and recorded in ``_init_args``) but not
        forwarded: the NTK always overwrites the network with weights loaded from the
        fit's ``params_<epoch>.npz``, so the initialisation is never observed.
        """
        # Add legacy support for old argument names
        if "layer_type" in kwargs:
            kwargs["architecture"] = kwargs.pop("layer_type")

        # Store all constructor arguments, so a model can be rebuilt from them
        self._init_args = {
            "nnseed": nnseed,
            "flav_info": flav_info,
            "replica_range_settings": replica_range_settings,
            "impose_sumrule": impose_sumrule,
            "fitbasis": fitbasis,
            "nodes": nodes,
            "activations": activations,
            "initializer_name": initializer_name,
            **kwargs,
        }

        self.model_kwargs = kwargs

        # Done here (at build time) rather than at import: toggling JAX state before it
        # initialises proved fragile, see the note in ``ntkpdf/__init__.py``. float64
        # keeps this in step with ``ntkConfig.produce_base_metamodels``, so the two
        # model-build paths agree to the last bit.
        import keras

        keras.backend.set_floatx("float64")
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "highest")

        nnseed_rep = replica_nnseed(replica_index, nnseed)
        architecture = kwargs.pop("architecture", "dense")

        self.n3fit_model = _pdfNN_layer_generator(
            [
                ReplicaSettings(
                    nodes=nodes,
                    activations=activations,
                    seed=nnseed_rep,
                    architecture=architecture,
                )
                for i in range(
                    replica_range_settings["min_replica"],
                    replica_range_settings["max_replica"] + 1,
                )
            ],
            impose_sumrule=impose_sumrule,
            flav_info=flav_info,
            fitbasis=fitbasis,
            **kwargs,
        )

    @property
    def n_parameters(self):
        """Return total number of trainable parameters."""
        return sum(jnp.size(w) for w in self.n3fit_model.trainable_weights)

    @property
    def param_names(self):
        return [f"w_{i}" for i in range(self.n_parameters)]

    def _ntk_submodel(self, exclude_layers):
        """The keras model used for an NTK evaluation, built **once** per
        ``exclude_layers`` config and cached on the instance.

        Previously a fresh ``MetaModel`` was constructed on *every* ``pdf_func``
        call (inside the ``jax.jacfwd`` trace); keras retains that graph state, so
        repeated NTK evaluations (e.g. the same snapshot epochs recomputed per
        report leaf) grew memory without bound. The (sub)model depends only on the
        architecture and ``exclude_layers`` -- not on the per-call weights -- so it
        is built once and reused.
        """
        cache = self.__dict__.setdefault("_ntk_submodel_cache", {})
        if exclude_layers not in cache:
            if exclude_layers:
                from n3fit.backends.keras_backend.MetaModel import MetaModel

                intermediate_output = None
                for layer in reversed(self.n3fit_model.layers):
                    if layer.name not in exclude_layers:
                        intermediate_output = layer.output
                        break
                cache[exclude_layers] = MetaModel(
                    input_tensors=self.n3fit_model.input,
                    output_tensors=intermediate_output,
                )
            else:
                cache[exclude_layers] = self.n3fit_model
        return cache[exclude_layers]

    def _ntk_non_trainable(self, model, remove_prefactors):
        """Non-trainable variable values to feed ``stateless_call``.

        With ``remove_prefactors`` the preprocessing exponents are overridden
        (small-x alpha -> 1, large-x beta -> 0) *functionally* -- without mutating
        the shared model, unlike the old ``set_weights`` side-effect.
        """
        if not remove_prefactors:
            return [jnp.asarray(v) for v in model.non_trainable_variables]

        prep = model.get_layer(PREPROCESSING_LAYER_ALL_REPLICAS)
        prep_index = {id(w): i for i, w in enumerate(prep.weights)}
        non_trainable = []
        for v in model.non_trainable_variables:
            if id(v) in prep_index:
                w = jnp.asarray(v)
                # even index = small-x exponent -> 1; odd = large-x exponent -> 0
                non_trainable.append(
                    jnp.ones_like(w) if prep_index[id(v)] % 2 == 0 else jnp.zeros_like(w)
                )
            else:
                non_trainable.append(jnp.asarray(v))
        return non_trainable

    def grid_values_func(
        self, xgrid, exclude_layers=[], remove_prefactors=False, grad_layers=None
    ):
        """
        Returns a function that computes the PDF values on xgrid.
        Maintains same input/output structure as n3fit_pdf_grid.

        The returned ``pdf_func`` is a **pure** function of the flat weight vector:
        weights flow in as a JAX argument and are applied via keras
        ``stateless_call`` (no ``set_weights`` side-effect, no per-call model
        rebuild), so ``jax.jacfwd(pdf_func)`` traces/compiles cleanly and the model
        is differentiated w.r.t. ``_params`` directly.

        Args:
            xgrid: The grid of x values to compute the PDFs on.
            exclude_layers: List of layer names to exclude from the PDF computation.
            Used in the NTK studies.
            grad_layers: Optional tuple of 1-based NN (Dense) layer indices to keep
            differentiable. The PDF output is unchanged (the full network is still
            evaluated), but the parameters of every other layer are frozen via
            ``stop_gradient`` so ``jax.jacfwd`` sees zero columns there -- yielding the
            layer-restricted NTK contribution. ``None`` differentiates all layers.
        """

        xgrid_tensor = jnp.array(xgrid)[None, :, None].astype(jnp.float64)
        input_dict = {"pdf_input": xgrid_tensor}

        if "xgrid_integration" in self.n3fit_model.x_in:
            xgrid_integration_jax = jnp.array(
                tf.convert_to_tensor(self.n3fit_model.x_in["xgrid_integration"])
            )
            input_dict["xgrid_integration"] = xgrid_integration_jax

        # Built/selected once, outside the trace (see _ntk_submodel).
        model = self._ntk_submodel(tuple(exclude_layers))
        shapes = [v.shape for v in model.trainable_variables]
        non_trainable = self._ntk_non_trainable(model, remove_prefactors)

        # Layer-restricted NTK: freeze the parameters of every Dense layer not in
        # ``grad_layers`` so jacfwd contributes zero columns there (see docstring).
        grad_mask = None
        if grad_layers is not None:
            ranges = _nn_layer_param_ranges(model)
            n_params = sum(math.prod(s) for s in shapes)
            mask = np.zeros(n_params, dtype=bool)
            for l in grad_layers:
                start, stop = ranges[l - 1]
                mask[start:stop] = True
            grad_mask = jnp.asarray(mask)

        def unpack_params(flat_params, shapes):
            """Returns the weight parameters in the format expected by the n3fit model."""
            params_list = []
            pos = 0
            for shape in shapes:
                size = math.prod(shape)
                param = lax.dynamic_slice(flat_params, (pos,), (size,))
                param = param.reshape(shape)
                params_list.append(param)
                pos += size

            return params_list

        def pdf_func(_params):
            if grad_mask is not None:
                _params = jnp.where(grad_mask, _params, lax.stop_gradient(_params))
            trainable = unpack_params(_params, shapes)

            # Pure functional evaluation: no mutation of model state.
            model_output, _ = model.stateless_call(trainable, non_trainable, input_dict)

            pdf_output = jnp.squeeze(model_output, axis=0)

            pdf_array = jnp.transpose(pdf_output, axes=(0, 2, 1))

            pdf_array = jnp.squeeze(pdf_array, axis=0)

            return jnp.array(pdf_array)

        return pdf_func


# ----------------------------------------
# Loading a model from a fit on disk
#########################################
def _resolve_fit(colibri_fit):
    """A validphys ``FitSpec`` from either a fit name or an already-resolved spec.

    colibri's NTK workers pass ``fit.name`` (a string); ``colibri.checks`` passes a
    spec. Uses validphys's loader rather than colibri's ``get_fit_path``, which
    hardcodes ``sys.prefix/share/NNPDF/results``, so the fit directory agrees with the
    ``fit.path`` the rest of ntkpdf uses.
    """
    if hasattr(colibri_fit, "as_input"):
        return colibri_fit

    from validphys.loader import FallbackLoader

    return FallbackLoader().check_fit(str(colibri_fit))


def _model_init_args(fit_input):
    """:class:`NTKPDFN3Fit` constructor arguments read from a fit's ``filter.yml``
    (i.e. ``FitSpec.as_input()``).

    Same field set that n3fit bakes into ``pdf_model.pkl`` at
    ``model_trainer.py``'s ``_init_args``, read from the runcard instead. Compare
    ``ntkConfig.produce_model_info``, which maps the same runcard keys for the PDF-grid
    path (and additionally reads the optimiser and initialiser settings, which only
    matter for a *freshly initialised* network).
    """
    parameters = fit_input["parameters"]
    return {
        "nnseed": fit_input["nnseed"],
        "flav_info": fit_input["fitting"]["basis"],
        "fitbasis": fit_input["fitting"]["fitbasis"],
        "nodes": parameters["nodes_per_layer"],
        "activations": parameters["activation_per_layer"],
        "initializer_name": parameters["initializer"],
        "layer_type": parameters["layer_type"],
        # n3fit's ``performfit`` defaults ``sum_rules`` to True; the trainer passes it
        # straight through to the model as ``impose_sumrule``.
        "impose_sumrule": fit_input.get("sum_rules", True),
    }


def _replica_preprocessing(fit_path, fit_name, replica_idx):
    """The fitted preprocessing exponents of one replica, as keras weights.

    Read from ``nnfit/replica_<r>/<fit>.json``, ordered ``[smallx, largex]`` per
    flavour to match the preprocessing layer's weight order.
    """
    path = fit_path / f"nnfit/replica_{replica_idx}" / f"{fit_name}.json"
    with open(path, "r", encoding="utf-8") as f:
        preprocessing = json.load(f)["preprocessing"]

    weights = []
    for fl in preprocessing:
        weights.append(np.array([[fl["smallx"]]]))
        weights.append(np.array([[fl["largex"]]]))
    return weights


def get_pdf_model(colibri_fit, replica_idx=None):
    """Given a fit, returns the PDF model. Drop-in for ``colibri.utils.get_pdf_model``.

    Builds :class:`NTKPDFN3Fit` from the fit's ``filter.yml`` -- no ``pdf_model.pkl``
    and no ``colibri_n3fit`` import. With ``replica_idx`` the network is loaded with
    that replica's fitted preprocessing exponents; without it the model keeps the
    exponents it was built with. This matters whenever the prefactors are *kept*
    (the ``model``/``no_smr`` selectors); ``remove_prefactors`` overrides them anyway.

    Parameters
    ----------
    colibri_fit : str or validphys.core.FitSpec
        The fit to read.
    replica_idx : int, optional
        1-based replica index whose preprocessing factors should be loaded.

    Returns
    -------
    NTKPDFN3Fit
    """
    fit = _resolve_fit(colibri_fit)

    pdf_model = NTKPDFN3Fit(
        replica_index=1,
        replica_range_settings={"min_replica": 1, "max_replica": 1},
        **_model_init_args(fit.as_input()),
    )

    if replica_idx is not None:
        pdf_model.n3fit_model.get_layer(PREPROCESSING_LAYER_ALL_REPLICAS).set_weights(
            _replica_preprocessing(fit.path, fit.name, replica_idx)
        )

    return pdf_model


def install():
    """Rebind :func:`get_pdf_model` over colibri's, in every module that holds a
    reference to it.

    colibri's NTK providers load the model themselves, by fit *name*, from inside
    worker threads (``ntkutils.py``, ``ntk.py``) -- there is no argument through which
    ntkpdf can hand them a model. Since they bind the name with
    ``from colibri.utils import get_pdf_model``, each holds its own reference and
    patching ``colibri.utils`` alone would reach none of them; all of
    :data:`_PATCH_TARGETS` must be rebound.

    Raises if a target no longer defines ``get_pdf_model`` -- that means colibri moved
    or renamed it and the patch is silently doing nothing. It does *not* catch colibri
    growing a **new** import site; only running with ``colibri_n3fit`` uninstalled does.
    Idempotent.
    """
    for name in _PATCH_TARGETS:
        module = importlib.import_module(name)
        if not hasattr(module, "get_pdf_model"):
            raise RuntimeError(
                f"{name} no longer defines 'get_pdf_model': ntkpdf's model loader "
                "cannot be installed. colibri has moved or renamed it -- update "
                "ntkpdf.model._PATCH_TARGETS."
            )
        module.get_pdf_model = get_pdf_model

    log.debug("Installed ntkpdf.model.get_pdf_model over %s", ", ".join(_PATCH_TARGETS))
