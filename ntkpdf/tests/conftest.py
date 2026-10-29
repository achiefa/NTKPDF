"""
conftest.py

Pytest fixtures for the ntkpdf test suite. Mirrors the layout of the upstream
validphys/n3fit suites: the test fit is referenced *by name* (the validphys
loader downloads it on demand, so nothing is committed to the repo) plus the
platform-marker plumbing copied from those suites.
"""

import json
import sys

import numpy as np
import pytest
import yaml

# A small NTK fit (3 replicas, epoch snapshots 100..1000) hosted on the NNPDF
# server; ``FallbackLoader`` downloads it on demand. Used by the end-to-end
# eigenvalue tests. The fit-free config tests do not need it.
FIT = "260607-ac-sequential"


@pytest.fixture(scope="session")
def ntk_fit():
    """The test fit, located -- and downloaded if absent -- via the loader."""
    from validphys.loader import FallbackLoader

    return FallbackLoader().check_fit(FIT)


@pytest.fixture
def clean_eigenvalue_cache(ntk_fit):
    """Delete any ``ntk_eigenvalues_*`` cache files created during a test.

    The eigenvalue providers serialise into ``<fit>/fit_replicas/replica_*/``
    (see colibri's ``produce_replicas_path``), i.e. *inside* the fit directory.
    This fixture snapshots the cache files present before the test and removes
    any that appear during it, so the (possibly shared, downloaded) fit is left
    untouched.
    """
    replicas_dir = ntk_fit.path / "fit_replicas"
    pattern = "replica_*/ntk_eigenvalues_*.npz"
    before = set(replicas_dir.glob(pattern))
    yield
    for created in set(replicas_dir.glob(pattern)) - before:
        created.unlink()


# A tiny on-disk n3fit fit, written to a tmp dir: enough of the layout (filter.yml,
# nnfit/ json, fit_replicas/ npz) for the model-building and discovery code, with
# no loader or network involved.
BASIS_FL = ("sng", "g", "v", "v3", "v8", "t3", "t8", "t15")
NODES = [4, 8]
ACTIVATIONS = ["tanh", "linear"]
SYNTHETIC_EPOCHS = (10, 20)
SYNTHETIC_NREPLICAS = 2
N_PARAMS = 2 * 4 + 4 + 4 * 8 + 8  # (x, log x) -> 4 -> 8


@pytest.fixture(scope="session", autouse=True)
def x64():
    """The keras path relies on colibri's ``Environment`` to enable x64; tests
    that build models directly must not depend on test order for it."""
    import jax

    jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="session", params=[{"sum_rules": False}, {}], ids=["no_sr", "sr"])
def synthetic_fit(request, tmp_path_factory):
    from validphys.core import FitSpec

    name = f"synthetic_{request.param_index}"
    path = tmp_path_factory.mktemp(name) / name
    rng = np.random.default_rng(0)

    runcard = {
        "nnseed": 1234,
        "fitting": {
            "fitbasis": "EVOL",
            "basis": [
                {"fl": fl, "trainable": False, "smallx": [1.0, 1.2], "largex": [2.0, 3.0]}
                for fl in BASIS_FL
            ],
        },
        "parameters": {
            "nodes_per_layer": NODES,
            "activation_per_layer": ACTIVATIONS,
            "initializer": "glorot_normal",
            "layer_type": "dense",
            "dropout": 0.0,
            "optimizer": {"optimizer_name": "SGD", "learning_rate": 1e-6, "clipnorm": 0.0},
        },
        **request.param,
    }
    path.mkdir()
    (path / "filter.yml").write_text(yaml.safe_dump(runcard))

    for r in range(1, SYNTHETIC_NREPLICAS + 1):
        nnfit = path / "nnfit" / f"replica_{r}"
        nnfit.mkdir(parents=True)
        preprocessing = [
            {"fl": fl, "smallx": float(rng.uniform(1.0, 1.2)), "largex": float(rng.uniform(2.0, 3.0)), "trainable": False}
            for fl in BASIS_FL
        ]
        (nnfit / f"{name}.json").write_text(
            json.dumps({"preprocessing": preprocessing, "best_epoch": SYNTHETIC_EPOCHS[-1]})
        )
        params = path / "fit_replicas" / f"replica_{r}" / "parameters"
        params.mkdir(parents=True)
        for e in SYNTHETIC_EPOCHS:
            np.savez(params / f"params_{e}.npz", params=rng.normal(scale=0.5, size=N_PARAMS))

    return FitSpec(name, path)


@pytest.fixture(scope="module")
def metamodels(synthetic_fit):
    """The keras (PDF-grid path) multi-replica model for ``synthetic_fit``."""
    from ntkpdf.config import ntkConfig

    config = ntkConfig({})
    return config.produce_base_metamodels(config.produce_model_info(synthetic_fit), SYNTHETIC_NREPLICAS)


def pytest_runtest_setup(item):
    ALL = {"darwin", "linux"}
    supported_platforms = ALL.intersection(mark.name for mark in item.iter_markers())
    plat = sys.platform
    if supported_platforms and plat not in supported_platforms:
        pytest.skip(f"cannot run on platform {plat}")
