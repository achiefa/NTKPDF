"""
test_model.py

Fit-free tests of the n3fit model interface (``ntkpdf.model``) and its agreement
with the keras PDF-grid path (``ntkpdf.config``), on the tiny synthetic fits from
``conftest.synthetic_fit`` (with and without sum rules).
"""

import ast
import importlib
from pathlib import Path

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import jax
from n3fit.backends.keras_backend.MetaModel import PREPROCESSING_LAYER_ALL_REPLICAS

import ntkpdf.model as model_mod
from ntkpdf.config import (
    FULL_MODEL, NN, NO_PREFACTORS, NO_SMR, layer_kwargs, ntkConfig,
    predict_pdf_grid, set_epoch_weights,
)
from ntkpdf.model import _model_init_args, _nn_layer_param_ranges, _replica_preprocessing, get_pdf_model
from ntkpdf.tests.conftest import (
    ACTIVATIONS, BASIS_FL, N_PARAMS, NODES, SYNTHETIC_EPOCHS, SYNTHETIC_NREPLICAS,
    neutral_preprocessing,
)
from ntkpdf.utils import INPUT_GRID, XGRID

EPOCH = SYNTHETIC_EPOCHS[0]


def _has_sum_rules(fit):
    return fit.as_input().get("sum_rules", True)


def _weights(fit, replica, epoch=EPOCH):
    return np.load(fit.path / f"fit_replicas/replica_{replica}/parameters/params_{epoch}.npz")["params"]


def _pdf(model, w, kwargs=FULL_MODEL):
    return np.asarray(model.grid_values_func(XGRID, **dict(kwargs))(w))


def _jacobian(model, w, kwargs=FULL_MODEL):
    return np.asarray(jax.jacfwd(model.grid_values_func(XGRID, **dict(kwargs)))(w))


@pytest.fixture(scope="module")
def ntk_model(synthetic_fit):
    return get_pdf_model(synthetic_fit, 1)


@pytest.fixture(scope="module")
def w(synthetic_fit):
    return _weights(synthetic_fit, 1)


# ----------------------------------------
# Backend and the get_pdf_model monkeypatch
#########################################
def test_keras_backend_is_jax():
    import keras

    assert keras.backend.backend() == "jax"


def test_install_rebinds_every_target():
    import ntkpdf.app  # noqa: F401 -- calls install() at import

    for name in model_mod._PATCH_TARGETS:
        assert importlib.import_module(name).get_pdf_model is get_pdf_model


def test_install_is_idempotent(monkeypatch):
    for name in model_mod._PATCH_TARGETS:
        monkeypatch.setattr(importlib.import_module(name), "get_pdf_model", object())
    model_mod.install()
    model_mod.install()
    for name in model_mod._PATCH_TARGETS:
        assert importlib.import_module(name).get_pdf_model is get_pdf_model


def test_install_raises_on_missing_symbol(monkeypatch):
    import colibri.ntk.ntk

    monkeypatch.delattr(colibri.ntk.ntk, "get_pdf_model")
    with pytest.raises(RuntimeError):
        model_mod.install()


def _resolve_import(module, node):
    if node.level == 0:
        return node.module
    base = module.split(".")[: -node.level]
    return ".".join(base + ([node.module] if node.module else []))


def test_patch_targets_cover_every_colibri_import_site():
    """``install()`` cannot see a *new* ``from colibri.utils import get_pdf_model``
    site in colibri; this static scan does. Only module-level imports bind a name
    that needs rebinding: a function-local one resolves at call time."""
    import colibri

    root = Path(colibri.__file__).parent
    sites = set()
    for path in root.rglob("*.py"):
        rel = path.relative_to(root.parent).with_suffix("")
        if "tests" in rel.parts:
            continue
        module = ".".join(rel.parts)
        for node in ast.parse(path.read_text()).body:
            if (
                isinstance(node, ast.ImportFrom)
                and _resolve_import(module, node) == "colibri.utils"
                and any(a.name == "get_pdf_model" for a in node.names)
            ):
                sites.add(module.removesuffix(".__init__"))
    assert sites | {"colibri.utils"} == set(model_mod._PATCH_TARGETS)


# ----------------------------------------
# Building the model from a fit
#########################################
def test_model_init_args(synthetic_fit):
    assert _model_init_args(synthetic_fit.as_input()) == {
        "nnseed": 1234,
        "flav_info": synthetic_fit.as_input()["fitting"]["basis"],
        "fitbasis": "EVOL",
        "nodes": NODES,
        "activations": ACTIVATIONS,
        "initializer_name": "glorot_normal",
        "layer_type": "dense",
        "impose_sumrule": _has_sum_rules(synthetic_fit),
    }


def test_sum_rule_layer_iff_sum_rules(synthetic_fit, ntk_model, metamodels):
    expected = _has_sum_rules(synthetic_fit)
    assert ("impose_msr" in [l.name for l in ntk_model.n3fit_model.layers]) == expected
    assert ("impose_msr" in [l.name for l in metamodels.layers]) == expected


@pytest.mark.parametrize("replica", range(1, SYNTHETIC_NREPLICAS + 1))
def test_preprocessing_reads_agree(synthetic_fit, replica):
    from_json = np.ravel(_replica_preprocessing(synthetic_fit.path, synthetic_fit.name, replica))
    from_config = ntkConfig({}).produce_fit_preprocessing(synthetic_fit)[replica - 1]
    loaded = get_pdf_model(synthetic_fit, replica).n3fit_model.get_layer(
        PREPROCESSING_LAYER_ALL_REPLICAS
    ).get_weights()
    np.testing.assert_array_equal(from_json, from_config)
    np.testing.assert_array_equal(from_json, np.ravel(loaded))
    assert len(from_json) == 2 * len(BASIS_FL)


# ----------------------------------------
# grid_values_func
#########################################
def test_grid_values_func_is_pure(ntk_model, w):
    before = [np.array(v) for v in ntk_model.n3fit_model.weights]
    first = _pdf(ntk_model, w)
    np.testing.assert_array_equal(first, _pdf(ntk_model, w))
    for b, v in zip(before, ntk_model.n3fit_model.weights):
        np.testing.assert_array_equal(b, np.array(v))


def test_grid_values_func_shapes(ntk_model, w):
    out = _pdf(ntk_model, w)
    assert out.shape == (14, len(XGRID))
    assert out.dtype == np.float64
    assert _jacobian(ntk_model, w).shape == (14, len(XGRID), N_PARAMS)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "keras 3.12 maps 64-bit results of keras.backend.result_type to 32-bit on "
    "non-TF backends (BIT64_TO_BIT32_DTYPE in keras/src/backend/common/dtypes.py); "
    "ops.tanh takes its dtype from it, so it returns float32 even with "
    "floatx=float64 and jax_enable_x64."
))
def test_hidden_activations_are_float64(ntk_model, w):
    dense = next(l for l in ntk_model.n3fit_model.get_layer("all_NNs").get_layer("NN_0").layers
                 if type(l).__name__ == "Dense")
    kernel, bias = [jax.numpy.asarray(v) for v in dense.trainable_variables]
    x = jax.numpy.stack([XGRID, np.log(XGRID)], -1)[None]
    assert dense.stateless_call([kernel, bias], [], x)[0].dtype == np.float64


@settings(deadline=None, max_examples=20)
@given(v=arrays(np.float64, N_PARAMS, elements=st.floats(-1, 1, allow_subnormal=False)))
def test_jacobian_matches_finite_difference(ntk_model, w, v):
    """Checks the autodiff only. The float32 tanh (see above) puts ~eps32
    relative noise on every evaluation, so a small step drowns in it: use a
    Richardson-extrapolated central difference (O(h^4)) at a large step, and allow
    for the noise, which scales with |f| / h. Tolerances are per flavour row: a
    global one is set by the large small-x rows and would never test the valence
    ones."""
    assume(np.abs(v).max() > 0.1)
    f = ntk_model.grid_values_func(XGRID)
    f0, jv = (np.asarray(a) for a in jax.jvp(f, (w,), (v,)))

    def cd(h):
        return (np.asarray(f(w + h * v)) - np.asarray(f(w - h * v))) / (2 * h)

    h = 1e-2
    atol = 1e-3 * np.abs(jv).max(axis=1) + 10 * np.finfo(np.float32).eps * np.abs(f0).max(axis=1) / h
    assert np.all(np.abs((4 * cd(h / 2) - cd(h)) / 3 - jv) <= atol[:, None])


def test_photon_row_is_zero(ntk_model, w):
    """Output is in n3fit ``EVOL_LIST`` order, photon first; it is not fitted."""
    out = _pdf(ntk_model, w)
    assert np.all(out[0] == 0)
    assert np.all(np.any(out[1:] != 0, axis=1))


def test_ntk_submodel_is_cached(ntk_model):
    assert ntk_model._ntk_submodel(()) is ntk_model.n3fit_model
    sub = ntk_model._ntk_submodel(("impose_msr",))
    assert ntk_model._ntk_submodel(("impose_msr",)) is sub


def test_submodel_keeps_weight_order(ntk_model):
    """The flat weight vector is unpacked in ``trainable_variables`` order, so
    excluding the sum-rule layer must not reorder it."""
    full = [v.path for v in ntk_model.n3fit_model.trainable_variables]
    sub = [v.path for v in ntk_model._ntk_submodel(("impose_msr",)).trainable_variables]
    assert sub == full


def test_selector_matrix(synthetic_fit, ntk_model, w):
    out = {k: _pdf(ntk_model, w, kwargs) for k, kwargs in
           {"model": FULL_MODEL, "no_prefactors": NO_PREFACTORS, "nn": NN, "no_smr": NO_SMR}.items()}
    assert not np.allclose(out["model"], out["nn"])
    if not _has_sum_rules(synthetic_fit):
        np.testing.assert_array_equal(out["model"], out["no_smr"])
        np.testing.assert_array_equal(out["nn"], out["no_prefactors"])
    else:
        assert not np.allclose(out["model"], out["no_smr"])
        assert not np.allclose(out["nn"], out["no_prefactors"])
        # The sum rules only rescale each flavour by an x-independent factor.
        rows = np.all(out["no_smr"] != 0, axis=1)
        ratio = out["model"][rows] / out["no_smr"][rows]
        np.testing.assert_allclose(ratio, ratio[:, :1] * np.ones_like(ratio), rtol=1e-12)


def test_remove_prefactors_matches_neutral_weights(synthetic_fit, ntk_model, w):
    before = [np.array(v) for v in ntk_model.n3fit_model.weights]
    stripped = _pdf(ntk_model, w, NO_PREFACTORS)
    for b, v in zip(before, ntk_model.n3fit_model.weights):
        np.testing.assert_array_equal(b, np.array(v))

    neutral = get_pdf_model(synthetic_fit)
    layer = neutral.n3fit_model.get_layer(PREPROCESSING_LAYER_ALL_REPLICAS)
    layer.set_weights([np.full_like(p, e) for p, e in zip(layer.get_weights(), neutral_preprocessing(synthetic_fit))])
    np.testing.assert_array_equal(stripped, _pdf(neutral, w))


# ----------------------------------------
# Layer-restricted kernels
#########################################
def test_layer_param_ranges(ntk_model):
    assert _nn_layer_param_ranges(ntk_model.n3fit_model) == [(0, 12), (12, N_PARAMS)]
    assert ntk_model.n_parameters == N_PARAMS


@pytest.mark.parametrize("layer", [1, 2])
def test_grad_layers_restricts_jacobian(ntk_model, w, layer):
    start, stop = _nn_layer_param_ranges(ntk_model.n3fit_model)[layer - 1]
    full = _jacobian(ntk_model, w)
    restricted = _jacobian(ntk_model, w, {"grad_layers": (layer,)})
    np.testing.assert_array_equal(restricted[..., start:stop], full[..., start:stop])
    assert np.all(np.delete(restricted, np.s_[start:stop], axis=-1) == 0)
    np.testing.assert_array_equal(_pdf(ntk_model, w, {"grad_layers": (layer,)}), _pdf(ntk_model, w))


def _kernel(model, w, kwargs):
    J = _jacobian(model, w, kwargs).reshape(-1, N_PARAMS)
    return J @ J.T


def test_layer_kernels_add_up(ntk_model, w):
    """The layer ranges partition the parameters (``test_layer_param_ranges``), so
    the single-layer kernels sum to the full NN kernel, and each cumulative kernel
    adds exactly its own layer to the previous one."""
    n_layers = len(_nn_layer_param_ranges(ntk_model.n3fit_model))
    single = [_kernel(ntk_model, w, layer_kwargs((l,), cumulative=False)) for l in range(1, n_layers + 1)]
    cumulative = [_kernel(ntk_model, w, layer_kwargs((l,))) for l in range(1, n_layers + 1)]
    nn = _kernel(ntk_model, w, NN)
    tol = dict(rtol=1e-10, atol=1e-12 * np.abs(nn).max())
    np.testing.assert_allclose(sum(single), nn, **tol)
    np.testing.assert_allclose(cumulative[0], single[0], **tol)
    for l in range(1, n_layers):
        np.testing.assert_allclose(cumulative[l] - cumulative[l - 1], single[l], **tol)


# ----------------------------------------
# The two model-build paths
#########################################
@pytest.mark.parametrize("no_prefactors", [False, True])
def test_build_paths_agree(synthetic_fit, metamodels, no_prefactors):
    """The keras grid path (multi-replica ``MetaModel``) and the NTK path
    (single-replica, pure JAX) give the same PDFs at a saved epoch: bit-identical
    without sum rules, within a few ULP with them (the sum-rule integrals)."""
    config = ntkConfig({})
    selector = {"name": "s", "no_prefactors": no_prefactors}
    set_epoch_weights(
        metamodels,
        config.produce_fit_weights(synthetic_fit, EPOCH),
        config.produce_fit_preprocessing(synthetic_fit),
        SYNTHETIC_NREPLICAS,
        selector,
    )
    grid = np.asarray(predict_pdf_grid(metamodels, INPUT_GRID).data).reshape(SYNTHETIC_NREPLICAS, 14, -1)
    for r in range(1, SYNTHETIC_NREPLICAS + 1):
        expected = _pdf(get_pdf_model(synthetic_fit, r), _weights(synthetic_fit, r),
                        NO_PREFACTORS if no_prefactors else FULL_MODEL)
        if _has_sum_rules(synthetic_fit):
            np.testing.assert_allclose(grid[r - 1], expected, rtol=1e-13, atol=0)
        else:
            np.testing.assert_array_equal(grid[r - 1], expected)


def test_colibri_ntk_runs_on_ntkpdf_model(synthetic_fit, tmp_path, monkeypatch, ntk_model, w):
    """End to end through the monkeypatch: colibri's NTK worker loads the model
    *by fit name* via the patched ``get_pdf_model`` and must reproduce the
    eigenvalues of J Jᵀ from ``grid_values_func``."""
    import shutil

    import ntkpdf.app  # noqa: F401 -- calls install() at import
    from colibri.ntk.ntkutils import compute_eigenvalues_for_replica, load_replica_eigenvalues
    from validphys.core import FitSpec

    # A copy, since colibri writes its eigenvalue cache next to the parameters.
    fit = FitSpec(synthetic_fit.name, shutil.copytree(synthetic_fit.path, tmp_path / synthetic_fit.name))
    monkeypatch.setattr(model_mod, "_resolve_fit", lambda name: fit)
    replicas_path = fit.path / "fit_replicas"

    assert compute_eigenvalues_for_replica(fit.name, replicas_path, 1, name="e2e") is not None
    stored = load_replica_eigenvalues(1, replicas_path, "e2e")
    J = _jacobian(ntk_model, w).reshape(-1, N_PARAMS)
    expected = np.sort(np.linalg.eigvalsh(J @ J.T))[::-1]
    got = np.sort(np.asarray(stored["eigenvalues"])[list(stored["epochs"]).index(EPOCH)])[::-1]
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-10 * expected[0])
