"""
ntkpdf.ntkdecomposition.py

Module for the decomposition of the PDFs into their NTK components.
"""
from dataclasses import dataclass, field
import logging
from typing import Optional

import numpy as np
import pandas as pd

from colibri.constants import XGRID
from colibri.ntk.ntkutils import NTKStats
from colibri.ntk.eigenvalues import EigenvalueGrid
from colibri.ntk.eigenvector import eigenvectors_ensemble_at_epoch
from reportengine import collect
from validphys.pdfgrids import XPlottingGrid

from ntkpdf.utils import plotting_grid_from_ntkstat, peak_rss_gb, EVOL_LIST

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


@dataclass
class FeatureColumns:
    """Memory-light container for just the feature columns ``q`` at one epoch.

    The feature *plots* only ever use ``q`` (and the per-replica ``cuts`` for the
    padding mask); they never touch ``qinv`` / ``P_parallel`` / ``P_perp`` (those
    drive the evolution operator's ``u``/``v`` in the analytical-solution path). So
    the report builds this instead of the full :class:`EvolutionOperator`, holding
    *one* ``(nreplicas, n, n)`` array per (epoch, name) rather than four -- ~4x less
    retained memory. That matters because reportengine never frees, and the report
    keeps one of these alive per snapshot epoch x model config.
    """

    q: NTKStats          # (nreplicas, n, n); column k-1 is feature/rank k
    cuts: np.ndarray     # (nreplicas,) per-replica perp-mode count
    flavours: list = field(default_factory=lambda: list(EVOL_LIST))
    xgrid: np.ndarray = field(default_factory=lambda: np.asarray(XGRID))


def _q_columns_by_replica(eigenvalues, eigenvectors, M, tol):
    """The padded ``q`` columns for one replica -- the ``Q`` part of
    :func:`compute_ntk_decomposition_by_replica_at_epoch`, computed in **numpy**
    (positional, no pandas label alignment) and skipping ``Qinv`` / ``P_parallel`` /
    ``P_perp``. Returns ``(cut, Q)`` with ``Q`` padded to ``(n, n)`` (zeros beyond the
    cut), so it is positionally identical to the full path's ``q``."""
    eigenvalues = np.asarray(eigenvalues)
    if eigenvalues[0] <= 0:
        raise ValueError("Largest NTK eigenvalue is non-positive.")
    cut = int(np.sum(eigenvalues / eigenvalues[0] > tol))
    if cut == 0:
        raise ValueError(f"All eigenvalues are below tolerance {tol}.")

    evecs = np.asarray(eigenvectors)               # (n, n)
    n = evecs.shape[1]
    Z_perp = evecs[:, :cut]                         # (n, cut)
    Ls = np.sqrt(eigenvalues[:cut])
    ZtMZ = Z_perp.T @ np.asarray(M) @ Z_perp        # (cut, cut), positional matmul
    H_perp = (Ls[:, None] * ZtMZ) * Ls[None, :]
    _, W = np.linalg.eigh(H_perp)                   # ascending
    W = W[:, ::-1]                                  # -> descending (matches full path)
    Q = (Z_perp * Ls[None, :]) @ W                  # (n, cut)
    Qpad = np.zeros((n, n), dtype=Q.dtype)
    Qpad[:, :cut] = Q
    return cut, Qpad


def feature_columns_at_epoch(
    eigenvalues_at_epoch: NTKStats,
    eigenvectors_at_epoch: NTKStats,
    m_matrix_train_val,
    tol: float = 1e-7,
    training: bool = True,
) -> FeatureColumns:
    """Compute only the feature columns ``q`` (+ per-replica cuts) at one epoch.

    A memory-light alternative to ``compute_ntk_decomposition_ensemble`` for the
    feature *plots*: produces the same ``q`` as the full ``EvolutionOperator``
    (verified positionally identical) without building/retaining
    ``qinv``/``P_parallel``/``P_perp``. Mirrors the full path's metric handling
    (``M = m_matrix_train_val[0]``; per-replica ``M`` if it is an ``NTKStats``).
    """
    log.info("[features] computing q columns (%d replicas) | peak RSS %.1f GB",
             eigenvalues_at_epoch.nreplica, peak_rss_gb())
    M = m_matrix_train_val[0]
    if isinstance(M, NTKStats):
        per_replica = [
            _q_columns_by_replica(evals_r, evecs_r, M_r, tol)
            for evals_r, evecs_r, M_r in zip(
                eigenvalues_at_epoch.data, eigenvectors_at_epoch.frames, M.frames
            )
        ]
    else:
        per_replica = [
            _q_columns_by_replica(evals_r, evecs_r, M, tol)
            for evals_r, evecs_r in zip(
                eigenvalues_at_epoch.data, eigenvectors_at_epoch.frames
            )
        ]
    cuts = np.array([c for c, _ in per_replica])
    q = NTKStats(np.stack([Q for _, Q in per_replica]))
    return FeatureColumns(q=q, cuts=cuts)


def evolution_operator(compute_ntk_decomposition_ensemble) -> EvolutionOperator:
    """The NTK :class:`EvolutionOperator` linearised at a reference epoch -- a
    friendly named alias for ``compute_ntk_decomposition_ensemble``.

    The **reference time** is the epoch at which the NTK is evaluated; it flows in
    through ``eigenvalues_at_epoch`` / ``eigenvectors_at_epoch`` (both keyed on
    ``epoch``). The returned operator is time-independent -- the *evolution* time
    ``t`` is supplied to its methods (``__call__`` / ``u`` / ``v`` /
    ``plotting_grid*``), e.g. ``t = epoch * learning_rate`` to evolve up to the
    reference time.

    This is a regular **provider**, not a ``produce_`` rule. It is built from other
    *providers* (``eigenvalues_at_epoch`` / ``eigenvectors_at_epoch`` and the
    FK/Cinv/M chain), and a ``produce_`` rule can only depend on config inputs and
    other ``produce_`` productions -- never on the provider DAG -- so a production
    here raises ``InputNotFoundError: A parameter is required: eigenvalues_at_epoch``.
    """
    return compute_ntk_decomposition_ensemble


def _feature_grid_from_columns(col, flavours, xgrid, basis):
    """Wrap an ``(m, n)`` block of ``q`` columns (``m`` replicas, ``n =
    n_flavours * n_xgrid``) as an ``XPlottingGrid`` -- reshape each replica to
    ``(n_flavours, n_xgrid)`` and route through ``plotting_grid_from_ntkstat``
    (exactly as ``EvolutionOperator.plotting_grid``)."""
    stats = NTKStats(col).reshape((len(flavours), len(xgrid)))
    return plotting_grid_from_ntkstat(stats, np.asarray(xgrid), basis, flavours)


def feature_grids_at_epoch(
    feature_columns_at_epoch,
    rank_indices: list,
    replica_index: Optional[int] = None,
    basis: str = "evolution",
) -> list:
    """Per-rank feature grids (the ``q`` columns) at one epoch, as ``XPlottingGrid``s.

    The :class:`EvolutionOperator` built for this epoch by
    ``compute_ntk_decomposition_ensemble`` holds ``q`` as an ``NTKStats``
    ``(nreplicas, n, n)``; column ``k-1`` is feature/rank ``k`` -- a vector over the
    ``(flavour, x)`` feature space (the same space a PDF lives in, ``n =
    n_flavours * n_xgrid``). We slice that column, reshape each replica to
    ``(n_flavours, n_xgrid)`` and wrap it as an ``XPlottingGrid``, so a feature plots
    like a PDF: value vs ``x``, one figure per flavour.

    Grids are built on the model output order (``EVOL_LIST``, evolution basis,
    ``XGRID``) -- identical to ``pdf_grid_at_epoch`` -- so they share a flavour axis
    with the PDF grids and the ``plot_grids`` flavour defaults apply unchanged.

    ``rank_indices`` is 1-based (the ``Rankspecs`` convention, matching colibri's
    ``EigenvalueGrid.get_plotting_data``); one grid is returned per rank, in order.

    **Padding.** ``compute_ntk_decomposition`` pads each replica's ``Q`` with
    *zero* columns beyond that replica's natural cut (``pad=True``), so for rank
    ``k`` any replica with ``cut < k`` carries an all-zero -- i.e. *absent* --
    feature. These are not real features:

    - ``replica_index is None`` (ensemble): such replicas are **masked out** per
      rank via ``EvolutionOperator.cuts``, so the central value/band is taken only
      over replicas that actually have that mode (otherwise the zeros drag it to 0).
    - ``replica_index`` given (single replica, 1-based): that replica's column is
      sliced directly (no masking, so the replica axis still maps 1:1 to ids). If
      the rank exceeds its cut the column is zero; we log a warning rather than
      silently drawing a flat zero.
    """
    fc = feature_columns_at_epoch
    q = fc.q.data                        # (nreplicas, n, n)
    cuts = np.asarray(fc.cuts)           # (nreplicas,) per-replica perp-mode count
    flavours = list(fc.flavours)
    xgrid = np.asarray(fc.xgrid)
    n_ranks = q.shape[2]

    grids = []
    for rank in rank_indices:
        if not (1 <= rank <= n_ranks):
            raise ValueError(f"rank {rank} out of range [1, {n_ranks}]")

        if replica_index is None:
            valid = cuts >= rank                      # drop zero-padded replicas
            if not valid.any():
                # No replica reaches this mode (rank > every replica's cut). Plot it
                # as flat zero -- consistent with the single-replica branch below --
                # rather than aborting the whole report: this is the expected case
                # when Rankspecs request more modes than a (small) fit's NTK provides.
                log.warning(
                    "Feature q^(%d): no replica has this mode (all cuts < %d, max %d); "
                    "plotted as flat zero.", rank, rank, int(cuts.max()),
                )
                col = np.zeros((1, q.shape[1]))       # (1, n) flat zero
            else:
                n_dropped = int((~valid).sum())
                if n_dropped:
                    log.warning(
                        "Feature q^(%d): %d/%d replicas have cut < %d (zero-padded / "
                        "absent feature) and are excluded from the ensemble band; it is "
                        "taken over the remaining %d replica(s).",
                        rank, n_dropped, valid.size, rank, int(valid.sum()),
                    )
                col = q[valid, :, rank - 1]           # (n_valid, n)
        else:
            if not (1 <= replica_index <= q.shape[0]):
                raise ValueError(
                    f"replica_index {replica_index} out of range [1, {q.shape[0]}]"
                )
            if cuts[replica_index - 1] < rank:
                log.warning(
                    "Replica %d has cut %d < rank %d: its q^(%d) feature is "
                    "zero-padded (absent) and will plot as flat zero.",
                    replica_index, int(cuts[replica_index - 1]), rank, rank,
                )
            col = q[replica_index - 1: replica_index, :, rank - 1]   # (1, n)

        grids.append(_feature_grid_from_columns(col, flavours, xgrid, basis))
    return grids


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

    epochs = list(epochs)
    n_ep = len(epochs)
    log.info("[h-grid] building for '%s' (training=%s): streaming eigenvectors over "
             "%d epochs | peak RSS %.1f GB", fit.label, training, n_ep, peak_rss_gb())

    h_stats = {}
    for i, epoch in enumerate(epochs):
        if i % 25 == 0 or i == n_ep - 1:
            log.info("[h-grid] epoch %d/%d (epoch=%s) | peak RSS %.1f GB",
                     i + 1, n_ep, epoch, peak_rss_gb())
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
        del eigenvectors  # drop this epoch's (nrep, n, n) eigenvectors before the next

    return hValGrid(label=fit.label, epochs=list(epochs), eigenvalues_stats=h_stats)


# Collect h_val grids across fits (mirrors colibri's `eigval_grids_by_fit`), so the
# same colibri plot providers can be called with these grids.
h_val_grids_by_fit = collect("h_val_grid", ("fits",))
