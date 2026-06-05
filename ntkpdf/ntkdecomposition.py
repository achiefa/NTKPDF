"""
ntkpdf.ntkdecomposition.py

Module for the decomposition of the PDFs into their NTK components.
"""
from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd

from colibri.constants import XGRID
from colibri.ntk.ntkutils import NTKStats
from colibri.ntk.eigenvalues import EigenvalueGrid
from colibri.ntk.eigenvector import eigenvectors_ensemble_at_epoch
from reportengine import collect
from validphys.pdfgrids import XPlottingGrid

from ntkpdf.utils import plotting_grid_from_ntkstat, EVOL_LIST

log = logging.getLogger(__name__)

@dataclass
class NTKDecomposition:
    """
    NTK decomposition stacked across all replicas.

    Each replica uses its own natural cut and is padded to ``(n, n)`` / ``(n,)``
    independently before stacking, so no information is lost and no common cut
    is forced.  The zero-padded entries do not contribute to evolution operators
    since the corresponding ``h`` and ``hinv`` entries are also zero.

    Attributes
    ----------
    cut : float
        Natural cut index
    P_parallel : NTKStats, shape (Nrep, n, n)
    P_perp : NTKStats, shape (Nrep, n, n)
    Q : NTKStats, shape (Nrep, n, n)   — padded with zero columns beyond each replica's cut
    Qinv : NTKStats, shape (Nrep, n, n) — padded with zero rows beyond each replica's cut
    h : NTKStats, shape (Nrep, n)       — padded with zeros beyond each replica's cut
    hinv : NTKStats, shape (Nrep, n)    — padded with zeros beyond each replica's cut
    """

    cut: np.float32
    P_parallel: pd.DataFrame
    P_perp: pd.DataFrame
    Q: pd.DataFrame
    Qinv: pd.DataFrame
    h: np.ndarray
    hinv: np.ndarray
    Z_perp: pd.DataFrame
    Z_par: pd.DataFrame
    W: np.float32
    Lambda_sqrt: np.ndarray
    H_perp: np.ndarray


@dataclass
class EvolutionOperator:
    """
    Evolution operator ensemble across all replicas.
    """
    cuts: np.array
    P_parallel: NTKStats
    P_perp: NTKStats
    q: NTKStats
    qinv: NTKStats
    h: NTKStats
    hinv: NTKStats
    Fk: pd.DataFrame
    Cinv: pd.DataFrame
    basis: str = "evolution"
    flavours: list = field(default_factory=lambda: list(EVOL_LIST))
    xgrid: np.array = field(default_factory=lambda: np.asarray(XGRID))

    def __post_init__(self):
        self.M = self.Fk.T @ self.Cinv @ self.Fk

    def _time_factor(self, training_time: float):
        return 2 * training_time / self.Cinv.shape[0]

    def u_check(self, training_time: float) -> NTKStats:
        M = self.M
        q = self.q
        P_parallel = self.P_parallel
        h = self.h
        hinv = self.hinv.as_diag()

        qt = q.T
        Qtilde = qt @ M @ P_parallel

        one_minus_exp = h.exp_kernel(self._time_factor(training_time))
        U_check = q @ hinv @ one_minus_exp @ Qtilde
        return U_check
    
    def u_hat(self, training_time: float) -> NTKStats:
        q = self.q
        qinv = self.qinv
        h = self.h

        exp_ht = h.exp_kernel_decay(self._time_factor(training_time))

        u_hat = q @ exp_ht @ qinv
        return u_hat
    
    def v(self, training_time: float) -> NTKStats:
        q = self.q
        h = self.h
        hinv_diag = self.hinv.as_diag()
        fk = self.Fk
        Cinv = self.Cinv

        T_tilde = q.T @ fk.T @ Cinv
        one_minus_exp = h.exp_kernel(self._time_factor(training_time))

        v = q @ hinv_diag @ one_minus_exp @ T_tilde
        return v
    
    def u(self, training_time: float) -> NTKStats:
        return self.u_check(training_time) + self.u_hat(training_time) + self.P_parallel
    
    def __call__(self, training_time, data, f0):
        u = self.u(training_time)
        v = self.v(training_time)
        return u @ f0 + v @ data
    
    def plotting_grid(self, training_time: float, data: NTKStats, f0: NTKStats, basis: str = "evolution", flavours: list = EVOL_LIST) -> XPlottingGrid:
        u = self.u(training_time)
        v = self.v(training_time)
        evolved = u @ f0 + v @ data
        new_shape = (len(self.flavours), len(self.xgrid))

        xplot_grid = plotting_grid_from_ntkstat(evolved.reshape(new_shape), self.xgrid, basis, flavours)
        return xplot_grid
    
    def plotting_grid_u(self, training_time: float, f0: NTKStats, basis: str = "evolution", flavours: list = EVOL_LIST) -> XPlottingGrid:
        u = self.u(training_time)
        evolved = u @ f0
        new_shape = (len(self.flavours), len(self.xgrid))

        xplot_grid = plotting_grid_from_ntkstat(evolved.reshape(new_shape), self.xgrid, basis, flavours)
        return xplot_grid
    
    def plotting_grid_v(self, training_time: float, data: NTKStats, basis: str = "evolution", flavours: list = EVOL_LIST) -> XPlottingGrid:
        v = self.v(training_time)
        evolved = v @ data
        new_shape = (len(self.flavours), len(self.xgrid))

        xplot_grid = plotting_grid_from_ntkstat(evolved.reshape(new_shape), self.xgrid, basis, flavours)
        return xplot_grid


def compute_ntk_decomposition_by_replica_at_epoch(
    eigenvalues,
    eigenvectors,
    M: pd.DataFrame,
    tol: float = 1e-07,
    pad: bool = True,
) -> NTKDecomposition:
    """
    Compute the NTK decomposition needed for evolution operators.

    The results are unpadded rectangular matrices — no padding to (n,n) is needed
    because the evolution formulas work identically with rectangular Q:
    ``U_hat = Q @ diag(exp_ht) @ Qinv`` with Q:(n,cut), Qinv:(cut,n) → (n,n).

    Parameters
    ----------
    eigenvalues : np.ndarray, shape (n,)
        NTK eigenvalues in descending order.
    eigenvectors : np.ndarray, shape (n, n)
        NTK eigenvectors; column ``i`` corresponds to ``eigenvalues[i]``.
    M : np.ndarray, shape (n, n)
        Metric matrix, e.g. ``FK.T @ C_inv @ FK``.
    tol : float
        Relative threshold: eigenvalue ``i`` is perp if ``λ_i / λ_0 > tol``.

    Returns
    -------
    NTKDecomposition
    """
    if eigenvalues[0] <= 0:
        raise ValueError("Largest NTK eigenvalue is non-positive.")

    cut = int(np.sum(eigenvalues / eigenvalues[0] > tol))
    if cut == 0:
        raise ValueError(f"All eigenvalues are below tolerance {tol}.")

    n_eig = eigenvectors.shape[1]
    n_columns = cut
    index_eig = eigenvectors.columns[:n_columns]
    
    Z_perp = eigenvectors.iloc[:, :cut]   # (n, cut)
    Z_par = eigenvectors.iloc[:, cut:]    # (n, n-cut)

    P_parallel = Z_par @ Z_par.T
    P_perp = Z_perp @ Z_perp.T

    Lambda_sqrt = np.sqrt(eigenvalues[:cut])
    Lambda_sqrt_inv = 1.0 / Lambda_sqrt
    
    ZtMZ = Z_perp.T @ M @ Z_perp
    H_perp = Lambda_sqrt[:, None] * ZtMZ * Lambda_sqrt[None, :]

    h, W = np.linalg.eigh(H_perp)
    idx = np.argsort(h)[::-1]
    h, W = h[idx], W[:, idx]
    h_inv = 1.0 / h

    Q = (Z_perp * Lambda_sqrt[None, :]) @ W
    Qinv = W.T @ (Lambda_sqrt_inv[:, None] * Z_perp.T)

    if pad:
      h = np.pad(h, (0, n_eig - cut))
      h_inv = np.pad(h_inv, (0, n_eig - cut))
      n_columns = n_eig
      index_eig = eigenvectors.columns
    

    Q = Q.reindex(columns=range(n_columns), fill_value=0.0)
    Q.columns = index_eig
    Qinv = Qinv.reindex(index=range(n_columns), fill_value=0.0)
    Qinv.index = index_eig

    return NTKDecomposition(
        cut=cut,
        P_parallel=P_parallel,
        P_perp=P_perp,
        Q=Q,
        Qinv=Qinv,
        H_perp=H_perp,
        h=h,
        hinv=h_inv,
        Lambda_sqrt=Lambda_sqrt,
        Z_perp=Z_perp,
        Z_par=Z_par,
        W=W
    )

def compute_ntk_decomposition_ensemble(
    eigenvalues_at_epoch: NTKStats,
    eigenvectors_at_epoch: NTKStats,
    fk_diagonal_basis_train_val, 
    cinv_diagonal_basis_train_val,
    m_matrix_train_val,
    tol: float = 1e-7,
    training: bool = True,
) -> EvolutionOperator:
    """
    Apply ``compute_ntk_decomposition`` across all replicas and stack the results.

    Uses eigenvalues and eigenvectors already computed by the framework's providers
    (``EigenvalueGrid``, ``EigenvectorGrid``).

    Each replica uses its own natural cut and is padded to ``(n, n)`` / ``(n,)``
    before stacking, so no information is lost and no common cut is forced.

    Parameters
    ----------
    eigenvalues_stat : NTKStats, shape (Nrep, n)
        Eigenvalue ensemble, sorted descending per replica.
    eigenvectors_stat : NTKStats, shape (Nrep, n, n)
        Eigenvector ensemble; ``data[r, :, i]`` is the i-th eigenvector of replica r.
    M : np.ndarray, shape (n, n)
        Fixed metric matrix shared across replicas.
    tol : float
        Relative eigenvalue threshold forwarded to ``compute_ntk_decomposition``.
    training: bool
        Whether to apply the time factor to the evolution operator components, which

    Returns
    -------
    NTKDecompositionEnsemble
    """
    idx_tr_val = 0 if training else 1
    M = m_matrix_train_val[0]
    Cinv = cinv_diagonal_basis_train_val[idx_tr_val]
    FK = fk_diagonal_basis_train_val[idx_tr_val]

    if isinstance(M, NTKStats):
      per_replica = [
          compute_ntk_decomposition_by_replica_at_epoch(evals_r, evecs_r, M_r, tol=tol)
          for evals_r, evecs_r, M_r in zip(eigenvalues_at_epoch.data, eigenvectors_at_epoch.frames, M.frames)
      ]
    else:
      per_replica = [
          compute_ntk_decomposition_by_replica_at_epoch(evals_r, evecs_r, M, tol=tol)
          for evals_r, evecs_r in zip(eigenvalues_at_epoch.data, eigenvectors_at_epoch.frames)
      ]

    return EvolutionOperator(
        cuts=np.array([d.cut for d in per_replica]),
        P_parallel=NTKStats([d.P_parallel for d in per_replica]),
        P_perp=NTKStats([d.P_perp for d in per_replica]),
        q=NTKStats([d.Q for d in per_replica]),
        qinv=NTKStats([d.Qinv for d in per_replica]),
        h=NTKStats([d.h for d in per_replica]),
        hinv=NTKStats([d.hinv for d in per_replica]),
        Fk=FK,
        Cinv=Cinv,
    )


# ---------------------------------------------------------------------------
# h_val grid (eigenvalues of H_perp), mirroring colibri's EigenvalueGrid
# ---------------------------------------------------------------------------
class hValGrid(EigenvalueGrid):
    """``h`` values (eigenvalues of the per-replica ``H_perp``) as a function of
    epoch.

    The structure is identical to :class:`EigenvalueGrid` -- ``{epoch -> NTKStats
    (nreplicas, n)}`` with ``xgrid = epochs`` and ``get_plotting_data(rank)``
    returning a rank's trajectory across epochs -- so colibri's NTK plot providers
    (``ntk_plot_provider`` / ``plot_eigvals_by_rank`` / ``plot_eigvals_by_fit``)
    work on it unchanged. Only the legend label changes from ``lambda`` to ``h``.
    """

    def get_plotting_label(self, rank_index: int, **kwargs) -> str:
        return rf"$h^{{({rank_index})}}$"


def _h_perp_eigenvalues(eigenvalues, eigenvectors, M, tol):
    """The ``H_perp`` eigenvalues (``h``) for one replica, padded to the full
    eigenvector count.

    This is only the ``h`` part of :func:`compute_ntk_decomposition_by_replica_at_epoch`,
    done in **numpy**: it skips ``P_parallel``/``P_perp``/``Q``/``Qinv`` (each
    ``(n, n)``) and the eigen*vectors* ``W`` of ``H_perp`` (``eigvalsh`` instead of
    ``eigh``), and avoids pandas label alignment. ``hValGrid`` only needs ``h``, so
    building the whole EvolutionOperator would be the bottleneck.
    """
    eigenvalues = np.asarray(eigenvalues)
    if eigenvalues[0] <= 0:
        raise ValueError("Largest NTK eigenvalue is non-positive.")
    cut = int(np.sum(eigenvalues / eigenvalues[0] > tol))
    if cut == 0:
        raise ValueError(f"All eigenvalues are below tolerance {tol}.")

    evecs = np.asarray(eigenvectors)
    Z_perp = evecs[:, :cut]                       # (n, cut)
    Ls = np.sqrt(eigenvalues[:cut])
    ZtMZ = Z_perp.T @ np.asarray(M) @ Z_perp      # (cut, cut), positional matmul
    H_perp = (Ls[:, None] * ZtMZ) * Ls[None, :]

    h = np.linalg.eigvalsh(H_perp)[::-1]          # descending, eigenvalues only
    return np.pad(h, (0, evecs.shape[1] - cut))


def h_val_grid(
    fit,
    replicas_path,
    epochs,
    eigenvalue_grid,
    m_matrix_train_val,
    kwargs=frozenset(),
    replica_index_list=None,
    tol: float = 1e-7,
    training: bool = True,
) -> hValGrid:
    """Build the :class:`hValGrid` by **streaming** the NTK eigenvectors over epochs.

    The function ``eigenvectors_ensemble_at_epoch`` is called directly instead
    of the natural reportengine route (``collect("h_val_ensemble_at_epoch",
    ("epochs",))`` over the per-epoch ``eigenvectors_at_epoch`` provider). This
    is done in order to **bound memory usage**: reportengine **never frees node
    results** -- every provider/production result is kept in the namespace for the
    whole run -- so collecting the eigenvectors over all epochs keeps *every*
    epoch's ``(nreplicas, n, n)`` array alive simultaneously, and memory explodes.

    The ``h`` values we actually need are tiny (``(nreplicas, n)``). So we bypass
    the per-epoch reportengine node and call the underlying function
    ``eigenvectors_ensemble_at_epoch`` in a plain Python loop: each epoch's
    eigenvectors are a local variable that is dropped (``del``) before the next
    iteration, bounding memory to a single epoch's eigenvectors instead of all of
    them. Only the small ``h`` ensembles accumulate. (Same pattern as the streaming
    ``loss_function_grid`` in ``ntkpdf.data_theory``.)

    Eigenvalues come from the cheap, disk-cached ``eigenvalue_grid`` (epoch-
    independent); only the eigenvectors are recomputed per epoch.
    """
    M = m_matrix_train_val[0 if training else 1]
    M_frames = [np.asarray(frame) for frame in M.frames]

    h_stats = {}
    for epoch in epochs:
        eigenvalues = eigenvalue_grid.get_stat_by_epoch(epoch).data        # (nrep, n)
        # Heavy: recomputes the NTK + eigendecomposition for every replica at this
        # epoch. Kept as a local so it is freed before the next epoch (see above).
        eigenvectors = eigenvectors_ensemble_at_epoch(
            fit,
            replicas_path,
            epoch,
            replica_index_list=replica_index_list,
            kwargs=kwargs,
        )["eigenvectors_data"]                                             # (nrep, n, n)

        h_stats[epoch] = NTKStats(
            np.stack([
                _h_perp_eigenvalues(eigenvalues[r], eigenvectors[r], M_frames[r], tol)
                for r in range(eigenvectors.shape[0])
            ])
        )
        del eigenvectors  # release this epoch's eigenvectors before the next one

    return hValGrid(label=fit.label, epochs=list(epochs), eigenvalues_stats=h_stats)


# Collect h_val grids across fits (mirrors colibri's `eigval_grids_by_fit`), so the
# same colibri plot providers can be called with these grids.
h_val_grids_by_fit = collect("h_val_grid", ("fits",))
