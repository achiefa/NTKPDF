"""
ntkpdf.summary.py

This module contains the code for producing summary reports of NTK analyses.
"""

import pandas as pd

from reportengine import collect
from reportengine.table import table

@table
def ntk_fit_summary(fit_name_with_covmat_label, nreplicas, fit_kind):
  data = {r"$N_{\mathrm{rep}}$": f"{nreplicas}", **fit_kind.summary()}
  return pd.Series(data, index=data.keys(), name=fit_name_with_covmat_label)


collected_ntk_fit_summaries = collect('ntk_fit_summary', ('fits',))

@table
def summarise_ntk_fits(collected_ntk_fit_summaries):
    """Produces a table of basic comparisons between fits, includes
    all the fields used in fit_summary"""
    return pd.concat(collected_ntk_fit_summaries, axis=1)
