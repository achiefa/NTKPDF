"""
test_colibri_fit.py

ntkpdf on a real colibri fit (``conftest.colibri_fit``: colibri_msht, 2 replicas,
10 epochs, every epoch saved).

The central check is that ntkpdf reproduces the training loss colibri logged in
``mc_loss.csv``. colibri saves neither the pseudodata nor the covmat, so that loss
tests their regeneration (seeds, data order, cuts, covmat, FK tables) all at once.
"""

import numpy as np
import pandas as pd
import pytest

from reportengine.configparser import ConfigError

from ntkpdf.api import API
from ntkpdf.model import get_pdf_model
from ntkpdf.tests.conftest import COLIBRI_EPOCHS, COLIBRI_FIT, COLIBRI_REPLICAS
from ntkpdf.utils import XGRID

pytestmark = pytest.mark.fit


def _params(fit, replica, epoch):
    return np.load(fit.path / f"fit_replicas/replica_{replica}/parameters/params_{epoch}.npz")["params"]


def test_model(colibri_fit):
    from colibri_msht.model import FixedParamModel

    assert isinstance(get_pdf_model(colibri_fit, 1), FixedParamModel)
    assert API.stored_epochs(fit=COLIBRI_FIT) == list(range(COLIBRI_EPOCHS + 1))


def test_pdf_at_epoch(colibri_fit):
    epoch = 5
    pdfs = np.asarray(API.pdf_at_epoch(fit=COLIBRI_FIT, epoch=epoch).data).reshape(len(COLIBRI_REPLICAS), 14, -1)
    grid_values = get_pdf_model(colibri_fit).grid_values_func(XGRID)
    for i, r in enumerate(COLIBRI_REPLICAS):
        expected = np.asarray(grid_values(_params(colibri_fit, r, epoch)))
        np.testing.assert_allclose(pdfs[i], expected, rtol=1e-12, atol=1e-14 * np.abs(expected).max())


def test_initialisation_is_epoch_0(colibri_fit):
    """ntkpdf draws the initial parameters as colibri does, so its initial PDFs are the
    fit's at epoch 0 (``params_0``)."""
    at_init = API.pdf_at_init(fit=COLIBRI_FIT)
    at_0 = API.pdf_at_epoch(fit=COLIBRI_FIT, epoch=0)
    np.testing.assert_array_equal(np.asarray(at_init.data), np.asarray(at_0.data))


def test_loss_reproduces_the_fit(colibri_fit):
    """With ``record_every: 1``, row k of ``mc_loss.csv`` is the loss at ``params_k``,
    computed before the update of epoch k. The file prints 6 digits."""
    training, validation = API.loss_function_grid(fit=COLIBRI_FIT)
    assert validation is None  # mc_validation_fraction: 0
    for i, r in enumerate(COLIBRI_REPLICAS):
        logged = pd.read_csv(colibri_fit.path / f"fit_replicas/replica_{r}/mc_loss.csv")["training_loss"]
        ours = [training.get_stat_by_epoch(e).data[i, 0] for e in range(COLIBRI_EPOCHS)]
        np.testing.assert_allclose(ours, logged.to_numpy(), rtol=1e-5)


def test_only_the_full_model(colibri_fit):
    with pytest.raises(ConfigError):
        API.kwargs(fit=COLIBRI_FIT, name="nn")


def test_summary(colibri_fit):
    summary = API.ntk_fit_summary(fit=COLIBRI_FIT)
    assert summary["Model"] == "FixedParamModel"
    assert summary["Parameters"] == str(get_pdf_model(colibri_fit).n_parameters)


def test_ntk_and_h_values(colibri_fit, clean_colibri_eigenvalue_cache):
    eigenvalues = API.eigenvalue_grid(fit=COLIBRI_FIT, name="model")
    h = API.h_val_grid(fit=COLIBRI_FIT, name="model", custom_epochs=[0, COLIBRI_EPOCHS - 1])
    for grid in (eigenvalues, h):
        for epoch in grid.epochs:
            assert np.all(np.isfinite(grid.get_stat_by_epoch(epoch).data))
