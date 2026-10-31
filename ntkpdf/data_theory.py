"""
ntk.data_theory.py

Module that contains the data and theory utilities for the ntk app.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from colibri.utils import closest_indices
from colibri.ntk.ntkutils import NTKStats, NTKGrid
from colibri.constants import EXPORT_LABELS, XGRID

from n3fit.layers.DIS import DIS
from n3fit.layers.observable import compute_float_mask
from reportengine import collect
from validphys.results import data_index

from ntkpdf.utils import (
  flav_to_evol_matrix,
  IDX_TO_FKFLAV,
  EVOL_INDEX,
  FLAVOUR_INDEX,
  peak_rss_gb,
)

log = logging.getLogger(__name__)

FK_INDEX = EVOL_INDEX

#---------------------------------------------------------------
# Utility classes 
##################################################################
class LossGrid(NTKGrid):
    """
    Container for loss function data from a single fit.
    """
    def __init__(self, label: str, epochs: List[int], loss_stats: Dict[int, NTKStats], training: bool = True):
        if not epochs:
            raise ValueError("epochs cannot be empty")
        if not loss_stats:
            raise ValueError("loss_stats cannot be empty")

        self._label = label
        self.epochs = sorted(epochs)
        self._loss_stats = loss_stats
        self._training = training

        first_epoch = self.epochs[0]
        first_stats = self._loss_stats[first_epoch]
        self.nreplicas = first_stats.nreplica
        
    @property
    def label(self) -> str:
        """Human-readable label for this grid (e.g., fit name)."""
        return self._label

    @property
    def n_ranks(self) -> int:
        """Number of eigenvalue ranks available."""
        return 1  # Loss grid has only one "rank" corresponding to the loss value

    @property
    def xgrid(self) -> np.ndarray:
        """X-axis grid for plotting (epochs)."""
        return np.array(self.epochs)

    @property
    def xlabel(self) -> str:
        """Label for x-axis."""
        return r"$\rm Epochs$"
    
    def get_stat_by_epoch(self, epoch: int) -> NTKStats:
        """Get NTKStats for a specific epoch."""
        if epoch not in self._loss_stats:
            raise ValueError(f"Epoch {epoch} not found in loss_stats")
        return self._loss_stats[epoch]

    def get_plotting_data(self, rank_index: int = 0, replica_idx=None) -> NTKStats:
        """Loss trajectory as an ``NTKStats`` of shape
        ``(n_selected_replicas, n_epochs)`` (replicas on axis 0, aligned with
        ``self.xgrid`` = epochs), so it can be fed directly to the shared drawing
        primitives. ``rank_index`` is unused -- a LossGrid has a single value per
        replica. With ``replica_idx`` an int, the result is a single-member
        ``NTKStats`` (one curve); with ``None`` it is the full ensemble.
        """
        if replica_idx is None:
            replica_idx = slice(None)  # Select all replicas if not specified

        # Each epoch's stats is (nreplicas, 1); take the column and stack over
        # epochs. atleast_1d keeps a single-int replica_idx as a length-1 axis.
        data_by_epoch = [
            np.atleast_1d(self._loss_stats[epoch].data[replica_idx, 0]) for epoch in self.epochs
        ]
        combined_data = np.stack(data_by_epoch, axis=1)  # (n_selected_replicas, n_epochs)
        return NTKStats(combined_data)

    def get_plotting_label(self, *args) -> str:
        """
        Get plotting label
        """
        if self._training:
            return "Training"
        return "Validation"

#---------------------------------------------------------------
# Main data/theory utilities meant to be used as providers
##################################################################

def _build_ntk_fast_kernel_arrays(data, padding=True, flavour_basis=False) -> pd.DataFrame:
    """Build the FK tables as a pandas DataFrame. Internal helper.

    Exposed to the report as the production ``ntk_fast_kernel_arrays``
    (``ntkConfig.produce_ntk_fast_kernel_arrays``), **not** as a provider: ``data`` is
    a validphys *production* (``produce_data``) available in every namespace, so a
    *provider* depending on it resolves to a separate node per presentation leaf
    (~200x) -> reloaded + retained -> OOM. As a production this FK is evaluated once at
    graph-build, and so is ``training_data``, which is built from it.

    The per-dataset ``(dataset, id)`` row index uses validphys's ``data_index(data)``
    (dropping its top ``experiment`` level) rather than the ``groups_index`` provider:
    ``groups_index`` hangs off the ``groups_data`` collect; ``data_index`` doesn't, and
    the ``(dataset, id)`` labels are identical (both iterate ``ds.cuts.load()`` per
    dataset, and this loop selects per ``ds.name``).
    """
    groups_index_nogroup = data_index(data).droplevel("experiment")

    lumi_indices = None
    blocks = []
    for ds in data.datasets:
          fkspecs = ds.fkspecs
          if len(fkspecs) > 1:
              raise NotImplementedError("Multiple FK specs per dataset not supported.")
          fkdata = fkspecs[0].load_with_cuts(ds.cuts)
                
          fkarray = fkdata.get_np_fktable() # shape [Ndat, 9, ngrid]

          if padding:
            dis = DIS([fkdata], [fkarray], ds.name)
            mask = dis.gen_mask(fkdata.luminosity_mapping)
            float_mask = compute_float_mask(mask)
            fk_padded = np.asarray(dis.pad_fk(fkarray, float_mask)) # shape [Ndat, ngrid, 14]
            fkarray = fk_padded
            fk_index = FK_INDEX
          else:
            fkarray = fkarray.transpose((0,2,1)) # Shape (Ndata, n_grid, 9)
            if lumi_indices is None:
              lumi_indices = fkdata.luminosity_mapping
              
              # Select flavours from FK_INDEX
              fk_index = FK_INDEX[FK_INDEX.get_level_values("flavour").isin([IDX_TO_FKFLAV[idx] for idx in lumi_indices])]
            else:
              # Check that the luminosity mapping is the same across datasets
              if np.any(fkdata.luminosity_mapping != lumi_indices):
                  raise ValueError("Luminosity mapping differs across datasets, which is not supported.")
            

          # Pad the FK array to have the same xgrid as XGRID
          fk_xgrid = fkdata.xgrid
          non_zero_indices = closest_indices(np.asarray(XGRID), fk_xgrid, atol=1e-8)
          new_fk_arr = np.zeros(
            (fkarray.shape[0], len(XGRID), fkarray.shape[2])
          )
          new_fk_arr[:, non_zero_indices, :] = fkarray # Shape (Nadata, n_grid, n_fl)
          new_fk_arr = new_fk_arr.transpose((0,2,1)) # Shape (Ndata, n_fl, n_grid)
          flat = new_fk_arr.reshape(new_fk_arr.shape[0], -1) # (Ndata, n_fl * n_grid), C-order groups by flavour
          ds_index = groups_index_nogroup[groups_index_nogroup.get_level_values("dataset") == ds.name]
          blocks.append(pd.DataFrame(flat, index=ds_index))

    # Set column index
    for block in blocks:
      block.columns = fk_index

    FK_df = pd.concat(blocks)

    if flavour_basis:
      evol_flavours = FK_df.columns.get_level_values("flavour").unique()
      nfl = len(evol_flavours)
      R = flav_to_evol_matrix(evol_flavours)
      FK_reshaped = FK_df.to_numpy().reshape(FK_df.shape[0], nfl, len(XGRID))
      FK_flav = np.einsum("ifa,fg->iga", FK_reshaped, R)
      FK_flav_flat = FK_flav.reshape(FK_df.shape[0], len(EXPORT_LABELS) * len(XGRID))
      col_index = FLAVOUR_INDEX
      FK_df = pd.DataFrame(FK_flav_flat, index=FK_df.index, columns=col_index)

    return FK_df


def get_train_val_data(training_data):
    """Per-replica training and validation pseudodata, rows labelled ``data_idx``."""
    return training_data.y


def fk_train_val(training_data):
    """Per-replica training and validation FK tables, in the basis the fit used (n3fit:
    the covmat's eigenbasis; colibri: the data points), rows labelled ``data_idx``."""
    return training_data.fk


def cinv_train_val(training_data):
    """Per-replica inverse covmat of the training and validation losses: diagonal for
    an n3fit fit (eigenbasis), dense for a colibri fit."""
    return training_data.cinv


def m_matrix_train_val(fk_train_val, cinv_train_val):
    """
    Per-replica training and validation metric matrices of the fit's loss,

    .. math::

        M_{\\mathrm{tr}}[r] = FK_{\\mathrm{tr}}[r]^{T}\\, C^{-1}_{\\mathrm{tr}}[r]\\, FK_{\\mathrm{tr}}[r],

    and similarly for validation.

    Returns
    -------
    M_tr, M_vl : NTKStats
        ``(n_params, n_params)`` matrices per replica with the FK column index
        on both axes.
    """
    fk_tr, fk_vl = fk_train_val
    cinv_tr, cinv_vl = cinv_train_val
    M_tr = fk_tr.T @ cinv_tr @ fk_tr
    M_vl = fk_vl.T @ cinv_vl @ fk_vl
    return M_tr, M_vl


def loss_function_grid(fit,
                       stored_epochs,
                       nreplicas,
                       wreplica_paths,
                       pdf_ensemble,
                       training_data,
                       model_selector=None):
    """
    Build :class:`LossGrid` classes for the training and validation loss
    functions of a fit over ``stored_epochs``. The validation one is ``None`` when the
    fit has no validation data.

    The epochs are streamed in a plain Python loop rather than collected over an
    ``epochs`` namespace. reportengine retains every node/production result for
    the whole run (``resourcebuilder.set_result`` never frees, and ``produce_``
    resources are evaluated eagerly at build time), so materialising the per-epoch
    PDF grid for ~1000 epochs keeps ~1000 grids alive and exhausts memory. Here
    only the current epoch's grid is alive at a time; only the small per-epoch
    losses persist. Memory is O(grid) instead of O(n_epochs * grid).
    """
    fk_tr, fk_vl = training_data.fk
    cinv_tr, cinv_vl = training_data.cinv
    tr_data, vl_data = training_data.y
    ndata_tr = tr_data.shape[0]
    ndata_vl = vl_data.shape[0]

    stored_epochs = list(stored_epochs)
    n_ep = len(stored_epochs)
    log.info("[loss-grid] building for '%s': predicting over %d epochs | peak RSS %.1f GB",
             fit.label, n_ep, peak_rss_gb())

    tr_loss_stats = {}
    vl_loss_stats = {}
    for i, epoch in enumerate(stored_epochs):
        if i % 25 == 0 or i == n_ep - 1:
            log.info("[loss-grid] epoch %d/%d (epoch=%s) | peak RSS %.1f GB",
                     i + 1, n_ep, epoch, peak_rss_gb())
        # wreplica_paths and the ensemble's replicas are both sorted by replica
        # index, so index idx refers to the same replica in each.
        weights = [
            np.load(wreplica_paths[idx] / "parameters" / f"params_{epoch}.npz")["params"]
            for idx in range(nreplicas)
        ]
        f = pdf_ensemble.at(weights, model_selector)

        res_tr = tr_data - fk_tr @ f
        tr_loss_stats[epoch] = NTKStats((res_tr.T @ cinv_tr @ res_tr).data.reshape(-1, 1) / ndata_tr)
        if ndata_vl:
            res_vl = vl_data - fk_vl @ f
            vl_loss_stats[epoch] = NTKStats((res_vl.T @ cinv_vl @ res_vl).data.reshape(-1, 1) / ndata_vl)
        # `f` and the predicted grid for this epoch are released next iteration.

    tr_loss_grid = LossGrid(label=fit.label, epochs=stored_epochs, loss_stats=tr_loss_stats, training=True)
    vl_loss_grid = None
    if ndata_vl:
        vl_loss_grid = LossGrid(label=fit.label, epochs=stored_epochs, loss_stats=vl_loss_stats, training=False)
    return tr_loss_grid, vl_loss_grid

loss_function_grid_by_fit = collect("loss_function_grid", ("fits",))
    