"""
ntkpdf.summary.py

This module contains the code for producing summary reports of NTK analyses.
"""
from collections import OrderedDict

import pandas as pd

from reportengine import collect
from reportengine.table import table

@table
def ntk_fit_summary(fit_name_with_covmat_label, nreplicas, model_info):
  data = OrderedDict(
      (
          (r"$N_{\mathrm{rep}}$", f"{nreplicas}"),
          (r"Nodes/layer", f"{model_info['nodes']}"),
          (r"Activations/layer", f"{model_info['activations']}"),
          (r"Layer type", f"{model_info['layer_type']}"),
          (r"Initializer", f"{model_info['initializer']}"),
          (r"Initializer scale", f"{model_info['initializer_scale']}"),
          (r"Dropout", f"{model_info['dropout']}"),
          (r"Optimizer", f"{model_info['optimizer']}"),
          (r"Learning rate", f"{model_info['learning_rate']}"),
          (r"Clipnorm", f"{model_info['clipnorm']}"),
          (r"Impose sum rule", f"{model_info['impose_sumrule']}")
      )
  )

  return pd.Series(data, index=data.keys(), name=fit_name_with_covmat_label)


collected_ntk_fit_summaries = collect('ntk_fit_summary', ('fits', 'fitcontext'))

@table
def summarise_ntk_fits(collected_ntk_fit_summaries):
    """Produces a table of basic comparisons between fits, includes
    all the fields used in fit_summary"""
    return pd.concat(collected_ntk_fit_summaries, axis=1)
