"""
ntk.utils.py

Collection of utilities for the ntk app.
"""

from functools import lru_cache
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from colibri.api import API as ColibriAPI
from colibri.constants import EXPORT_LABELS, flavour_to_evolution, XGRID
from colibri.ntk.ntkutils import NTKStats
from colibri.ntk.eigenvalues import EigenvalueGrid

from validphys.api import API
from validphys.core import MCStats
from validphys.pdfgrids import XPlottingGrid, check_basis, xplotting_grid
from validphys.pdfbases import Basis
from n3fit.vpinterface import EVOL_LIST, N3PDF

from ntkpdf.vpinterface import NTKPDF

# Change "\\Sigma" to "sigma" and "g" to "gluon" in evolution flavours for user-friendliness
IDX_TO_FKFLAV = {idx : fl for idx, fl in enumerate(EVOL_LIST)}
FKFLAV_TO_IDX = {fl : idx for idx, fl in enumerate(EVOL_LIST)}

EVOL_INDEX = pd.MultiIndex.from_tuples(
  [(fl, i + 1) for fl in EVOL_LIST for i in range(len(XGRID))],
  names=["flavour", "x"],
)
def FLAVOUR_INDEX_CONSTRUCTOR(flav_list):
    return pd.MultiIndex.from_tuples(
        [(fl, i + 1) for fl in flav_list for i in range(len(XGRID))],
        names=["flavour", "x"],
    )
FLAVOUR_INDEX = FLAVOUR_INDEX_CONSTRUCTOR(EXPORT_LABELS)

XGRID = np.asarray(XGRID)
INPUT_GRID = XGRID.reshape(1,-1,1)

def _setdefault_runcard(fitinfo):
    for key, item in fitinfo['datacuts'].items():
        fitinfo[key] = item
    fitinfo.setdefault('use_t0', True)
    fitinfo.setdefault('use_cuts', fitinfo['datacuts'].get('use_cuts', 'internal'))
    fitinfo.setdefault('theoryid', fitinfo['theory']['theoryid'])
    return fitinfo

def flav_to_evol_matrix(flavour_list: List[str]) -> np.ndarray:
    """Returns the rotation matrix from flavour basis to evolution basis for the
      specified flavours. The rotation is a matrix R_ij such that:
          f_evol[i] = sum_j R[i,j] * f_phys[j]
      where:
      - i = evolution flavour, ordered as the flavours argument (e.g. ["\\Sigma", "g", "V", "T3", "V3", "T8", "V8"])
      - j = physical flavour, ordered as EXPORT_LABELS = [TBAR, BBAR, CBAR, SBAR, UBAR, DBAR, GLUON, D, U, S, C, B, T, PHT]
    """
    export_labels = EXPORT_LABELS
    num_flav_evol = len(flavour_list)
    num_flav = len(export_labels)
    # Change "sigma" to "\\Sigma" in flavours
    if "sigma" in flavour_list:
        flavour_list = list("\\Sigma" if fl == "sigma" else fl for fl in flavour_list)
    if "gluon" in flavour_list:
        flavour_list = list("g" if fl == "gluon" else fl for fl in flavour_list)

    flavour_to_evolution_matrix = np.zeros((num_flav_evol, num_flav))
    for i, flav_evol in enumerate(flavour_list):
        j = 0
        for flav in export_labels:
            if flav in flavour_to_evolution[flav_evol].keys():
                flavour_to_evolution_matrix[i,j] = flavour_to_evolution[flav_evol][flav]
            else:
                flavour_to_evolution_matrix[i,j] = 0
            j += 1

    R = flavour_to_evolution_matrix
    return R

def get_rot_evol_to_flav(flavours):
    """Returns the inverse rotation from evolution basis to flavour basis:
        f_phys = Rinv @ f_evol
    Requires len(flavours) == len(EXPORT_LABELS) (14) so that R is square.
    """
    R = flav_to_evol_matrix(flavours)
    if R.shape[0] != R.shape[1]:
        raise ValueError(
            f"R is not square ({R.shape}): get_rot_evol_to_flav requires all "
            f"{len(EXPORT_LABELS)} evolution flavours, got {len(flavours)}."
        )
    Rinv = np.linalg.inv(R)
    return Rinv


########################################

class BestEpochStat(MCStats):
    def error_members(self):
        return self.data
    
    def median(self):
        return np.median(self.data, axis=0)

    def central_value(self):
        return np.mean(self.data, axis=0)

def produce_plotting_grid(
    fitname,
    epochs,
    xgrid,
    flavours: Optional[Union[list, tuple]] = None,
    vetoname: Optional[str] = None,
    nrep: Optional[int] = None,
    StatClass: type = MCStats,
) -> tuple[Basis, XPlottingGrid]:
    """Produce the plotting grid for the specified fit and epochs, specifying
    flavours, vetoes and number of replicas to use."""
    fitinfo = API.fit(fit=fitname).as_input()
    theoryID = API.theoryid(**fitinfo['theory'])
    basis = fitinfo['fitting']['fitbasis']
    q0 = theoryID.get_description().get("Q0")
    pdf_models, vetoes = get_pdf_model_at_epochs(fitname, epochs, vetoname=vetoname)

    input_xgrid = xgrid.reshape(1,-1,1)
    out = pdf_models.predict({"pdf_input": input_xgrid.T}, verbose=False).squeeze()[vetoes]
    if nrep is None:
        nrep = out.shape[0]
    
    print(f"Using {nrep} replicas for plotting grid.")
    if flavours is None:
        flavours = list(FKFLAV_TO_IDX.keys())
    nn_outputs = out[:nrep,:,[FKFLAV_TO_IDX[fl] for fl in flavours]]
    nn_outputs = nn_outputs.transpose((0,2,1)) # Shape (nrep, nflavour, ngrid)

    # Add central value to the output
    gv = np.concatenate([np.mean(nn_outputs, axis=0, keepdims=True), nn_outputs], axis=0)

    # Make usable outside reportengine
    checked = check_basis(basis, flavours)
    basis = checked['basis']
    flavours = checked['flavours']
    # Eliminante Q axis
    stats_gv = StatClass(gv)

    res = XPlottingGrid(q0, basis, flavours, xgrid, stats_gv, "log")
    return (basis, res)

# def plotting_grid_from_model(
#     fitname,
#     epochs,
#     xgrid,
#     basis,
#     flavours,
#     vetoname: str = None,
#     replica_index_list: tuple = None
# ):
#     """Construct a plotting grid directly from the PDF model, using
#     the vp-interface."""
#     fitinfo = API.fit(fit=fitname).as_input()
#     theoryID = API.theoryid(**fitinfo['theory'])
#     q0 = theoryID.get_description().get("Q0")
#     pdf_models, vetoes = get_pdf_model_at_epochs(fitname, epochs, vetoname=vetoname)
#     selected_replicas = np.asarray(pdf_models.split_replicas())[vetoes]
#     if replica_index_list is not None:
#         selected_replicas = selected_replicas[replica_index_list]
#     n3pdf = N3PDF(selected_replicas)
#     x_grid = xplotting_grid(n3pdf, q0, xgrid, basis, flavours)
#     return x_grid

def plotting_grid_from_ntkstat(
    stats: NTKStats,
    xgrid: np.ndarray,
    basis: str,
    flavours: list
)-> XPlottingGrid:
    """Construct a plotting grid directly from the NTKStats object."""
    pdf = NTKPDF(stats, "ntkpdf", Q=1.65)
    return xplotting_grid(pdf, 1.65, np.asarray(xgrid), basis, flavours)

def count_dim_imgntk(fitname, 
                     eigval_name, 
                     model_kwargs,
                     label,
                     tols):
    eigenvalues = ColibriAPI.eigenvalue_grid(fit=fitname, 
                                             name=eigval_name,
                                             kwargs=model_kwargs, 
                                             force_recompute=False)

    GridDict = {}
    for tol in tols:
        stat_by_epoch = {}
        for epoch in eigenvalues.epochs:
          stat = []
          for k in range(eigenvalues.nreplicas):
              eigval = eigenvalues.get_stat_by_epoch(epoch).data[k]

              cut = int(np.sum(eigval / eigval[0] > tol))
              if cut == 0:
                  raise ValueError(f"All eigenvalues are below tolerance {tol}.")
              stat.append(cut)
          stat_by_epoch[epoch] = NTKStats(np.array(stat).reshape(-1, 1))
        
        GridDict[f"{eigval_name}_{tol}"] = EigenvalueGrid(label=f"{label} (tol={tol})", epochs=eigenvalues.epochs, eigenvalues_stats=stat_by_epoch)

    return GridDict