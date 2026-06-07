"""
test_pdfgrids.py

Regression test for the per-epoch PDF grids. ``pdf_grid_at_epoch`` is a config
*production* (``ntkConfig.produce_pdf_grid_at_epoch``), not a plain provider, on
purpose: ``produce_model_at_epoch`` loads each epoch's weights into the *shared*
model in place, so the grid must be materialised at graph-build time, right
after its own epoch's mutation and before the next epoch overwrites the model.

If it were a deferred provider, every epoch's production would run (mutating the
shared model) before any grid was materialised, so the whole
``pdf_grid_all_epochs`` collect would return copies of the *last* epoch. This
test pins each collected grid to its own epoch.

Uses the downloaded test fit (model build + predict); a few seconds.
"""

import numpy as np

from ntkpdf.api import API
from ntkpdf.tests.conftest import FIT

EPOCHS = [100, 1000]


def _data(grid):
    return np.asarray(grid.grid_values.data)


def test_pdf_grid_all_epochs_are_per_epoch():
    """The collect yields one grid per epoch, each matching the single-epoch
    production for *that* epoch (not the last epoch repeated)."""
    grids = API.pdf_grid_all_epochs(fit=FIT, custom_epochs=EPOCHS)

    assert len(grids) == len(EPOCHS)

    for grid, epoch in zip(grids, EPOCHS):
        single = API.pdf_grid_at_epoch(fit=FIT, epoch=epoch)
        np.testing.assert_allclose(_data(grid), _data(single))

    # And the epochs are genuinely distinct, so the per-epoch match above is a
    # real constraint rather than vacuously true.
    assert not np.allclose(_data(grids[0]), _data(grids[-1]))
