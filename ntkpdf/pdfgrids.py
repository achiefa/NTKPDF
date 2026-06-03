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

from reportengine import collect

collected_init_grids = collect("init_grids_by_name", ("fits",))
