"""
test_model_fit.py

The n3fit model interface (``ntkpdf.model``) on the real downloaded test fit:
parameter layout and agreement with the keras PDF-grid path.
"""

import numpy as np
import pytest

from ntkpdf.api import API
from ntkpdf.model import get_pdf_model
from ntkpdf.tests.conftest import FIT
from ntkpdf.utils import XGRID

pytestmark = pytest.mark.fit

REPLICAS = (1, 2, 3)
EPOCHS = (100, 1000)


def _weights(ntk_fit, replica, epoch):
    return np.load(ntk_fit.path / f"fit_replicas/replica_{replica}/parameters/params_{epoch}.npz")["params"]


@pytest.fixture(scope="module")
def models():
    return {r: get_pdf_model(FIT, r) for r in REPLICAS}


def test_n_parameters_matches_saved_weights(ntk_fit, models):
    assert models[1].n_parameters == _weights(ntk_fit, 1, EPOCHS[0]).size == 763


@pytest.mark.parametrize("epoch", EPOCHS)
def test_build_paths_agree(ntk_fit, models, epoch):
    grid = np.asarray(API.pdf_at_epoch(fit=FIT, epoch=epoch).data).reshape(len(REPLICAS), 14, -1)
    for r in REPLICAS:
        expected = np.asarray(models[r].grid_values_func(XGRID)(_weights(ntk_fit, r, epoch)))
        np.testing.assert_array_equal(grid[r - 1], expected)


def test_fit_spec_or_name(ntk_fit, models):
    w = _weights(ntk_fit, 1, EPOCHS[0])
    from_spec = get_pdf_model(ntk_fit, 1).grid_values_func(XGRID)(w)
    np.testing.assert_array_equal(np.asarray(from_spec), np.asarray(models[1].grid_values_func(XGRID)(w)))
