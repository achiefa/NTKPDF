"""
test_config.py

Fit-free unit tests for ``ntkpdf.config``: the model-selector -> kwargs mapping
that decides which model modifier the NTK / eigenvalue computation uses, and the
name-keyed serialisation contract.

These exercise the *real* reportengine resolution path through the API
(``API.kwargs`` -> ``ntkConfig.produce_kwargs``) without building any model or
loading a fit -- ``produce_kwargs`` depends only on ``name``. This is the heart
of the ``Eigenvaluesconfigs`` wiring: each entry there carries only a ``name``
(``model``/``nn``), and that name alone must select the right modifier.
"""

import pytest

from reportengine.configparser import ConfigError

from ntkpdf.api import API
from ntkpdf.config import FULL_MODEL, NN, NO_PREFACTORS, NO_SMR


@pytest.mark.parametrize(
    "name, expected",
    [
        ("model", FULL_MODEL),
        ("nn", NN),
        ("no_prefactors", NO_PREFACTORS),
        ("no_smr", NO_SMR),
    ],
)
def test_kwargs_resolves_to_modifier(name, expected):
    """Each selector ``name`` resolves, through ``produce_kwargs``, to the
    modifier kwargs that get passed down to the NTK computation."""
    assert API.kwargs(name=name) == expected


def test_full_model_imposes_no_modifier():
    """``model`` must impose *no* modifier -- the full, unmodified model."""
    assert dict(API.kwargs(name="model")) == {}


def test_model_and_nn_differ():
    """The two ``Eigenvaluesconfigs`` selectors must map to *different*
    modifiers; otherwise both report variants would compute the same NTK."""
    assert API.kwargs(name="model") != API.kwargs(name="nn")


def test_unknown_selector_raises():
    with pytest.raises(ConfigError):
        API.kwargs(name="not_a_selector")


def test_cache_filename_is_name_keyed():
    """Distinct selector names must serialise to distinct cache files, so one
    config's eigenvalues can never be loaded in place of another's. This is the
    other half of the safety property: both the modifier *and* the on-disk cache
    key derive from the same ``name``, so they cannot desync."""
    from colibri.ntk.ntkutils import generate_filename

    model_file = generate_filename(1, "model")
    nn_file = generate_filename(1, "nn")
    assert model_file != nn_file
    assert "model" in model_file
    assert "nn" in nn_file


@pytest.mark.parametrize("selector", [NN, NO_SMR])
def test_exclude_layers_is_a_tuple_of_layer_names(selector):
    """``exclude_layers`` is matched against whole layer names, so it must be a
    tuple: a bare string would be split into characters and match nothing."""
    assert dict(selector)["exclude_layers"] == ("impose_msr",)


# ----------------------------------------
# Fit discovery and weight loading, on the synthetic fit
#########################################
from pathlib import Path

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from ntkpdf.config import _apply_model_selector, _sort_replicas, layer_kwargs, ntkConfig, set_epoch_weights
from ntkpdf.tests.conftest import N_PARAMS, SYNTHETIC_EPOCHS, SYNTHETIC_NREPLICAS

NO_PREFACTORS_SELECTOR = {"name": "no_prefactors", "no_prefactors": True}


@pytest.mark.parametrize("cumulative, grad", [(True, (1, 2)), (False, (2,))])
def test_layer_kwargs(cumulative, grad):
    kwargs = dict(layer_kwargs((2,), cumulative=cumulative))
    assert kwargs.pop("grad_layers") == grad
    assert kwargs == dict(NN)


@given(st.lists(st.integers(0, 10**6), unique=True))
def test_sort_replicas_is_numeric(indices):
    paths = sorted((Path(f"replica_{i}") for i in indices), key=str)
    assert [int(p.stem.split("_")[-1]) for p in _sort_replicas(paths)] == sorted(indices)


def test_fit_discovery(synthetic_fit):
    config = ntkConfig({})
    replicas = list(range(1, SYNTHETIC_NREPLICAS + 1))
    assert list(config.produce_replicas(synthetic_fit)) == replicas
    assert config.produce_nreplicas(synthetic_fit) == SYNTHETIC_NREPLICAS
    assert config.produce_nreplicas(synthetic_fit, custom_nreplicas=1) == 1
    stored = config.produce_stored_epochs(synthetic_fit, config.produce_best_epochs(synthetic_fit))
    assert list(stored) == list(SYNTHETIC_EPOCHS[:-1])  # the best epoch is excluded
    assert list(config.produce_epochs(stored, custom_epochs=[20])) == [20]
    assert config.produce_epochs(stored) is stored


def test_fit_weights_per_replica_epochs(synthetic_fit):
    config = ntkConfig({})
    epochs = list(SYNTHETIC_EPOCHS)
    weights = config.produce_fit_weights(synthetic_fit, epochs)
    for r, (w, e) in enumerate(zip(weights, epochs), start=1):
        path = synthetic_fit.path / f"fit_replicas/replica_{r}/parameters/params_{e}.npz"
        np.testing.assert_array_equal(w, np.load(path)["params"])
    with pytest.raises(ConfigError):
        config.produce_fit_weights(synthetic_fit, epochs[:1])


def _flat_nn(metamodels, i):
    return np.concatenate([np.ravel(v) for v in metamodels.get_replica_weights(i)["all_NNs"]])


def _prepr(metamodels, i):
    return np.ravel(metamodels.get_replica_weights(i)["preprocessing_factor"])


@pytest.mark.parametrize("selector", [None, NO_PREFACTORS_SELECTOR])
def test_set_epoch_weights_round_trip(synthetic_fit, metamodels, selector):
    config = ntkConfig({})
    weights = config.produce_fit_weights(synthetic_fit, SYNTHETIC_EPOCHS[0])
    prepr = config.produce_fit_preprocessing(synthetic_fit)
    set_epoch_weights(metamodels, weights, prepr, SYNTHETIC_NREPLICAS, selector)
    for i in range(SYNTHETIC_NREPLICAS):
        np.testing.assert_array_equal(_flat_nn(metamodels, i), weights[i])
        expected = prepr[i] if selector is None else np.tile([1.0, 0.0], len(prepr[i]) // 2)
        np.testing.assert_array_equal(_prepr(metamodels, i), expected)


def test_apply_model_selector_restores_base(metamodels):
    _apply_model_selector(metamodels, SYNTHETIC_NREPLICAS, NO_PREFACTORS_SELECTOR)
    _apply_model_selector(metamodels, SYNTHETIC_NREPLICAS, None)
    for i in range(SYNTHETIC_NREPLICAS):
        np.testing.assert_array_equal(_prepr(metamodels, i), np.ravel(metamodels._ntk_base_preprocessing[i]))


def test_both_paths_share_weight_layout(synthetic_fit, metamodels):
    """``set_epoch_weights`` and ``grid_values_func`` unpack the same flat vector
    with different shape lists; they must agree."""
    from ntkpdf.model import get_pdf_model

    ntk_shapes = [tuple(v.shape) for v in get_pdf_model(synthetic_fit).n3fit_model.trainable_variables]
    keras_shapes = [tuple(np.shape(v)) for v in metamodels.get_replica_weights(0)["all_NNs"]]
    assert ntk_shapes == keras_shapes
    assert sum(np.prod(s) for s in ntk_shapes) == N_PARAMS
