"""
ntkpdf.pdfgrids.py

PDF grid utilities for ntkpdf.

The per-selector grids at initialisation are built together in
``ntkConfig.produce_init_grids_by_name`` (see ``ntkpdf/config.py``) so that the
expensive base model is constructed once per fit and replica count and shared
across the report's presentation namespaces, rather than recomputed inside a
``collect`` for every ``Selectors``/``PDFscalespecs`` combination.

For *cross-fit* comparison we do want one grid set per fit, so we collect that
shared production over the ``fits`` dimension. Each element is the
``{selector_name: XPlottingGrid}`` mapping for one fit.
"""

from n3fit.vpinterface import EVOL_LIST, N3PDF
from reportengine import collect
from validphys.pdfgrids import xplotting_grid

from ntkpdf.utils import XGRID

collected_init_grids = collect("init_grids_by_name", ("fits",))


def pdf_grid_at_epoch(model_at_epoch):
    """Plotting grid for the PDF model at a given training epoch.

    ``model_at_epoch`` (config production) is the shared model with that epoch's
    fitted weights loaded; ``epoch`` flows into it through ``fit_weights``.
    """
    pdf_models, _vetoes = model_at_epoch
    return xplotting_grid(
        N3PDF(pdf_models.split_replicas()), 1.65, XGRID, "evolution", EVOL_LIST
    )
