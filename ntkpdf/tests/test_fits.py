"""
test_fits.py

Fit-free tests of ``ntkpdf.fits``: telling n3fit and colibri fits apart, and the model
selectors each kind accepts.
"""

import pytest
import yaml

from reportengine.configparser import ConfigError
from validphys.core import FitSpec

from ntkpdf.config import FULL_MODEL, ntkConfig
from ntkpdf.fits import load_fit
from ntkpdf.fits.colibri import ColibriFit
from ntkpdf.fits.n3fit import N3Fit


def _fit_dir(tmp_path, *contents):
    """A fit folder with a runcard and the given files (``name/`` for a folder)."""
    path = tmp_path / "fit"
    path.mkdir(parents=True)
    (path / "filter.yml").write_text(yaml.safe_dump({"theoryid": 40001000}))
    for name in contents:
        (path / name).mkdir() if name.endswith("/") else (path / name).touch()
    return FitSpec("fit", path)


def test_n3fit_fit(synthetic_fit):
    assert isinstance(load_fit(synthetic_fit), N3Fit)


def test_n3fit_fit_with_a_pickled_model(tmp_path):
    """n3fit fits can carry a ``pdf_model.pkl`` too; ``nnfit/`` decides."""
    assert isinstance(load_fit(_fit_dir(tmp_path, "nnfit/", "pdf_model.pkl")), N3Fit)


def test_colibri_fit(tmp_path):
    assert isinstance(load_fit(_fit_dir(tmp_path, "pdf_model.pkl")), ColibriFit)


def test_neither(tmp_path):
    with pytest.raises(ConfigError):
        load_fit(_fit_dir(tmp_path))


@pytest.mark.parametrize("name", ["nn", "no_prefactors", "no_smr"])
def test_colibri_fit_has_only_the_full_model(tmp_path, name):
    """A colibri model has no prefactors or network layers to select."""
    fit_kind = load_fit(_fit_dir(tmp_path, "pdf_model.pkl"))
    assert ntkConfig({}).produce_kwargs(fit_kind, "model") == FULL_MODEL
    with pytest.raises(ConfigError):
        ntkConfig({}).produce_kwargs(fit_kind, name)


def _colibri_fit(tmp_path, epochs, optimizer_settings=None):
    """A colibri fit's layout, without a model; ``epochs`` maps each replica to the
    epochs it saved."""
    fit = _fit_dir(tmp_path, "pdf_model.pkl")
    optimizer_settings = optimizer_settings or {"optimizer_hyperparams": {"learning_rate": 1e-3}}
    (fit.path / "filter.yml").write_text(yaml.safe_dump({"optimizer_settings": optimizer_settings}))
    for replica, saved in epochs.items():
        params = fit.path / f"fit_replicas/replica_{replica}/parameters"
        params.mkdir(parents=True)
        for epoch in saved:
            (params / f"params_{epoch}.npz").touch()
    return ColibriFit(fit)


def test_colibri_epochs_saved_by_every_replica(tmp_path):
    """A replica that stops early saves fewer epochs."""
    assert _colibri_fit(tmp_path, {1: [0, 50, 100], 2: [0, 50]}).stored_epochs() == [0, 50]


def test_colibri_initial_replicas_are_the_fits_own(tmp_path):
    """colibri seeds the initial parameters with the replica index: the fit's replicas
    (here 2 and 4) come first, then fresh ones."""
    fit = _colibri_fit(tmp_path, {2: [0], 4: [0]})
    assert fit._replicas_and_fresh_ones(1) == [2]
    assert fit._replicas_and_fresh_ones(2) == [2, 4]
    assert fit._replicas_and_fresh_ones(4) == [2, 4, 5, 6]


def test_colibri_optimizer_defaults(tmp_path):
    """colibri's defaults apply (adam), and a schedule has no single learning rate."""
    fit = _colibri_fit(tmp_path, {1: [0]})
    assert fit._optimizer_settings()["optimizer"] == "adam"
    assert fit.learning_rate() == 1e-3
    schedule = {"scheduler": {"name": "exponential_decay", "params": {}}}
    with pytest.raises(ConfigError):
        _colibri_fit(tmp_path / "scheduled", {1: [0]}, schedule).learning_rate()
