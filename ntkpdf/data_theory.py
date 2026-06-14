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

from ntkpdf.config import set_epoch_weights, predict_pdf_grid
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
# Utilities not meant to be exposed as providers
##################################################################
def _sorted_eigenmode_index(idx: pd.Index) -> pd.Index:
    """Sort a flat ``"eigenmode k"``/``"data k"`` index ascending by ``k``."""
    return pd.Index(sorted(idx, key=lambda s: int(s.split()[-1])))

def _data_idx(n: int) -> pd.MultiIndex:
    """``data_idx``-named MultiIndex 1..n. Matches the convention used in
    ``cross_validation.ipynb`` so chained ``@`` between FK and Cinv has
    consistent row/column indices."""
    return pd.MultiIndex.from_arrays([[i + 1 for i in range(n)]], names=["data_idx"])


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

def ntk_fast_kernel_arrays(data, groups_index, padding=True, flavour_basis=False) -> pd.DataFrame:
    """
    Return the FK tables as a pandas DataFrame for a given fit name.
    """
    # Sha
    groups_index_nogroup = groups_index.droplevel("group")

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


def get_fit_pseudodata(fit, read_fit_pseudodata, groups_index, psd_type="all"):
    """
    Get the pseudodata for a given fit name using the NTK stats. Returns an
    :class:`NTKStats` of one-column DataFrames, one per replica.

    Handles both fitting conventions:

    * ``diagonal_basis=True`` (modern default): the saved pseudodata is in the
      diagonalised covmat basis, indexed by flat ``"eigenmode k"`` strings;
      it is reindexed in ascending ``k`` so the ordering matches
      :func:`fitting_covmat_eigensystem`.
    * ``diagonal_basis=False`` (legacy): the saved pseudodata has the
      ``(group, dataset, id)`` MultiIndex; it is reindexed against
      ``groups_index`` so the ordering matches :func:`ntk_fast_kernel_arrays`.

    ``psd_type`` is one of ``"all"``, ``"train"``, ``"val"``.
    """
    pseudo_data = read_fit_pseudodata
    diagonal = fit.as_input().get("diagonal_basis", True)

    if not diagonal:
        groups_index_nogroup = groups_index.droplevel("group")

    raw_data = []
    for pdr in pseudo_data:
        if diagonal:
            data = pdr.pseudodata.reindex(_sorted_eigenmode_index(pdr.pseudodata.index))
        else:
            data = pdr.pseudodata.droplevel("group").loc[groups_index_nogroup]
        data.columns = data.columns.str.replace(r'\s+\d+$', '', regex=True)

        if psd_type == "all":
            idx_selector = data.index
        elif psd_type == "train":
            idx_selector = pdr.tr_idx if diagonal else pdr.tr_idx.droplevel("group")
        elif psd_type == "val":
            idx_selector = pdr.val_idx if diagonal else pdr.val_idx.droplevel("group")
        else:
            raise ValueError(f"Invalid psd_type: {psd_type}. Must be one of 'all', 'train', or 'val'.")

        raw_data.append(data.loc[idx_selector])

    return NTKStats(raw_data)


def train_val_masks(fit, read_fit_pseudodata):
    """
    Per-replica boolean train and validation masks, aligned with the full
    eigenmode-sorted pseudodata index.

    Each mask is a single-column DataFrame of length ``Ndata`` with ``True``
    on the points belonging to the corresponding subset. Wrapped in
    :class:`NTKStats` so all replicas stack into a ``(Nrep, Ndata, 1)`` array.

    Returns
    -------
    tr_masks, val_masks : NTKStats
    """
    is_diagonal = fit.as_input().get("diagonal_basis", True)
    if not is_diagonal:
        raise ValueError(
            f"Fit {fit.name} was not run with diagonal_basis=True; "
            "train_val_masks expects a diagonal-basis fit."
        )
    pseudo_data = read_fit_pseudodata

    tr_frames, val_frames = [], []
    for pdr in pseudo_data:
        # Training and validation indices are stacked and therefore
        # the eigenmode index is not ordered by the original data index.
        data_index = _sorted_eigenmode_index(pdr.pseudodata.index)
        tr_mask = data_index.isin(pdr.tr_idx)
        val_mask = ~tr_mask
        tr_frames.append(pd.DataFrame(tr_mask, index=data_index))
        val_frames.append(pd.DataFrame(val_mask, index=data_index))

    return NTKStats(tr_frames), NTKStats(val_frames)


def get_train_val_data(fit, read_fit_pseudodata):
    is_diagonal = fit.as_input().get("diagonal_basis", True)
    if not is_diagonal:
        raise ValueError(
            f"Fit {fit.name} was not run with diagonal_basis=True; "
            "train_val_masks expects a diagonal-basis fit."
        )
    pseudo_data = read_fit_pseudodata
    tr_data_list, val_data_list = [], []
    for pdr in pseudo_data:
        data = pdr.pseudodata.reindex(
            sorted(pdr.pseudodata.index, key=lambda s: int(s.split()[-1]))
        )
        data.columns = data.columns.str.replace(r'\s+\d+$', '', regex=True)
        tr_data_list.append(data.loc[pdr.tr_idx])
        val_data_list.append(data.loc[~data.index.isin(pdr.tr_idx)])
    
    tr_data = NTKStats(tr_data_list)
    val_data = NTKStats(val_data_list)
    tr_data.set_index(_data_idx(tr_data.shape[0]), tr_data._df_columns)
    val_data.set_index(_data_idx(val_data.shape[0]), val_data._df_columns)
    return tr_data, val_data


def fk_diagonal_basis_train_val(fitting_covmat_eigensystem, 
                                ntk_fast_kernel_arrays,
                                train_val_masks):
    """
    Per-replica FK rotated into the diagonal covmat basis and split by
    train/val using each replica's mask.

    For each replica r,

    .. code-block:: text

        FK_diag = eigvecs @ FK
        fk_tr[r] = FK_diag[tr_mask[r]]
        fk_vl[r] = FK_diag[val_mask[r]]

    The data-axis index is then reset to a plain integer ``data_idx`` so that
    the chained product with :func:`cinv_diagonal_basis_train_val` is
    consistent (FK rows carry eigenmode labels, Cinv is built from
    ``np.diag`` and has no labels, and ``NTKStats.__matmul__`` requires the
    inner indices to match).

    Returns
    -------
    fk_tr, fk_vl : NTKStats
        Replica-wise (Ntr, n_params) and (Nval, n_params) blocks. The
        parameter axis carries the FK column index.
    """
    _, eigvecs = fitting_covmat_eigensystem
    FK = ntk_fast_kernel_arrays
    # Align eigenvector columns to the FK index so the matmul rotates FK into the
    # diagonal covmat basis. Done here (not in the production that loads the
    # eigensystem) because the FK index needs the ntk_fast_kernel_arrays provider;
    # this is an in-place, idempotent relabel of the shared (per-fit) eigvecs.
    eigvecs.columns = ntk_fast_kernel_arrays.index
    FK_diag = eigvecs @ FK
    tr_masks, val_masks = train_val_masks

    fk_tr_frames, fk_vl_frames = [], []
    for tr_m, vl_m in zip(tr_masks.frames, val_masks.frames):
        fk_tr_frames.append(FK_diag[tr_m.iloc[:, 0]])
        fk_vl_frames.append(FK_diag[vl_m.iloc[:, 0]])
    fk_tr = NTKStats(fk_tr_frames)
    fk_vl = NTKStats(fk_vl_frames)
    fk_tr.set_index(_data_idx(fk_tr.shape[0]), fk_tr._df_columns)
    fk_vl.set_index(_data_idx(fk_vl.shape[0]), fk_vl._df_columns)
    return fk_tr, fk_vl


def cinv_diagonal_basis_train_val(fitting_covmat_eigensystem, train_val_masks):
    """
    Per-replica diagonal :math:`C^{-1}` in the diagonal covmat basis,
    restricted to each replica's train and validation eigenmodes.

    In the diagonal basis the covariance is :math:`C = \\mathrm{diag}(\\Lambda)`,
    so the restriction is also diagonal: ``Cinv_tr[r] = diag(1/eigvals[tr_mask[r]])``.
    Row/column indices are set to ``data_idx`` so the result chains with
    :func:`fk_diagonal_basis_train_val`.

    Returns
    -------
    cinv_tr, cinv_vl : NTKStats
        ``(Ntr, Ntr)`` and ``(Nval, Nval)`` diagonal matrices per replica.
    """
    eigvals, _ = fitting_covmat_eigensystem
    tr_masks, val_masks = train_val_masks

    cinv_tr_frames, cinv_vl_frames = [], []
    for tr_m, vl_m in zip(tr_masks.frames, val_masks.frames):
        tr_vals = 1.0 / eigvals[tr_m.iloc[:, 0].values]
        vl_vals = 1.0 / eigvals[vl_m.iloc[:, 0].values]
        cinv_tr_frames.append(pd.DataFrame(np.diag(tr_vals)))
        cinv_vl_frames.append(pd.DataFrame(np.diag(vl_vals)))
    cinv_tr = NTKStats(cinv_tr_frames)
    cinv_vl = NTKStats(cinv_vl_frames)
    tr_idx = _data_idx(cinv_tr.shape[0])
    vl_idx = _data_idx(cinv_vl.shape[0])
    cinv_tr.set_index(tr_idx, tr_idx)
    cinv_vl.set_index(vl_idx, vl_idx)
    return cinv_tr, cinv_vl


def m_matrix_train_val(fk_diagonal_basis_train_val, cinv_diagonal_basis_train_val):
    """
    Per-replica training and validation metric matrices in the diagonal-basis
    convention used during the fit:

    .. math::

        M_{\\mathrm{tr}}[r] = (V_{\\mathrm{tr}} FK)^{T}\\,\\mathrm{diag}\\!\\bigl(1/\\Lambda_{\\mathrm{tr}}[r]\\bigr)\\,(V_{\\mathrm{tr}} FK),

    and similarly for validation. ``V`` is the orthonormal eigenbasis of the
    fitting covmat and the subscript selects rows by replica ``r``'s mask.

    Returns
    -------
    M_tr, M_vl : NTKStats
        ``(n_params, n_params)`` matrices per replica with the FK column index
        on both axes.
    """
    fk_tr, fk_vl = fk_diagonal_basis_train_val
    cinv_tr, cinv_vl = cinv_diagonal_basis_train_val
    M_tr = fk_tr.T @ cinv_tr @ fk_tr
    M_vl = fk_vl.T @ cinv_vl @ fk_vl
    return M_tr, M_vl


def loss_function_grid(fit,
                       stored_epochs,
                       nreplicas,
                       wreplica_paths,
                       fit_preprocessing,
                       metamodels,
                       input_grid,
                       cinv_diagonal_basis_train_val,
                       fk_diagonal_basis_train_val,
                       get_train_val_data):
    """
    Build :class:`LossGrid` classes for the training and validation loss
    functions of a fit over ``stored_epochs``.

    The epochs are streamed in a plain Python loop rather than collected over an
    ``epochs`` namespace. reportengine retains every node/production result for
    the whole run (``resourcebuilder.set_result`` never frees, and ``produce_``
    resources are evaluated eagerly at build time), so materialising the per-epoch
    PDF grid for ~1000 epochs keeps ~1000 grids alive and exhausts memory. Here
    only the current epoch's grid is alive at a time; only the small per-epoch
    losses persist. Memory is O(grid) instead of O(n_epochs * grid).
    """
    fk_tr, fk_vl = fk_diagonal_basis_train_val
    cinv_tr, cinv_vl = cinv_diagonal_basis_train_val
    tr_data, vl_data = get_train_val_data
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
        # wreplica_paths and fit_preprocessing are both sorted by replica index,
        # so index idx refers to the same replica in each.
        weights = [
            np.load(wreplica_paths[idx] / "parameters" / f"params_{epoch}.npz")["params"]
            for idx in range(nreplicas)
        ]
        set_epoch_weights(metamodels, weights, fit_preprocessing, nreplicas)
        f = predict_pdf_grid(metamodels, input_grid)

        res_tr = tr_data - fk_tr @ f
        res_vl = vl_data - fk_vl @ f
        loss_tr = (res_tr.T @ cinv_tr @ res_tr).data.reshape(-1, 1) / ndata_tr
        loss_vl = (res_vl.T @ cinv_vl @ res_vl).data.reshape(-1, 1) / ndata_vl

        tr_loss_stats[epoch] = NTKStats(loss_tr)
        vl_loss_stats[epoch] = NTKStats(loss_vl)
        # `f` and the predicted grid for this epoch are released next iteration.

    tr_loss_grid = LossGrid(label=fit.label, epochs=stored_epochs, loss_stats=tr_loss_stats, training=True)
    vl_loss_grid = LossGrid(label=fit.label, epochs=stored_epochs, loss_stats=vl_loss_stats, training=False)
    return tr_loss_grid, vl_loss_grid

loss_function_grid_by_fit = collect("loss_function_grid", ("fits",))
    