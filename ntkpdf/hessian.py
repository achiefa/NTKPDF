"""
ntkpdf.hessian.py

Parameter-space Hessian of the training loss at a given epoch.

The fit minimises, per replica, the diagonal-basis training chi2 normalised by the
number of training points,

.. math::

    L(\\theta) = \\frac{1}{N_{\\rm tr}}\\,(y_{\\rm tr} - FK_{\\rm tr}\\,f(\\theta))^{T}
                 \\,{\\rm diag}(1/\\Lambda_{\\rm tr})\\,(y_{\\rm tr} - FK_{\\rm tr}\\,f(\\theta)),

where ``f(theta)`` is the (flattened, flavour-major) PDF grid produced by the network
with flat weight vector ``theta``. This is exactly the scalar built in
``data_theory.loss_function_grid`` (same per-replica ``FK_tr``/``Cinv_tr``/``y_tr``,
positional matmul), so the Hessian computed here is the curvature of the *actual* loss
the fit descended.

The differentiation reuses colibri's NTK route: ``pdf_model.grid_values_func(XGRID,
**kwargs)`` is a pure-JAX function of the flat weights, so ``jax.hessian`` gives the
full ``(P, P)`` Hessian directly. ``kwargs`` is the same model-selector frozenset used
for the NTK (e.g. ``no_prefactors``/``nn`` or ``layer_kwargs(...)``), i.e. the "model
specification" -- it controls which layers are differentiated and whether the
preprocessing prefactors are neutralised.

Relation to the NTK ``h`` values: the Gauss-Newton part of this Hessian is
``(2/N_tr) Jᵀ M J`` (``M = FKᵀ Cinv FK``), whose non-zero eigenvalues equal
``(2/N_tr)`` times the ``h`` values of :mod:`ntkpdf.ntkdecomposition`. The full Hessian
here additionally contains the second-derivative term, so it also sees negative
curvature (saddles) that the NTK/Gauss-Newton spectrum cannot.
"""

import logging

import numpy as np

from colibri.constants import XGRID
from colibri.ntk.ntkutils import NTKStats

log = logging.getLogger(__name__)


def _loss_hessian_by_replica(pdf_func, params, fk_tr, cinv_diag_tr, y_tr):
    """Full ``(P, P)`` Hessian of the training loss for one replica, via
    ``jax.hessian`` of the pure-JAX ``pdf_func`` (a function of the flat weights
    ``params``). ``cinv_diag_tr`` is the diagonal of ``Cinv_tr`` (the metric is
    diagonal in the eigenbasis). Positional matmul throughout, matching
    ``loss_function_grid``."""
    import jax
    import jax.numpy as jnp

    fk = jnp.asarray(fk_tr)
    cinv_diag = jnp.asarray(cinv_diag_tr)
    y = jnp.asarray(y_tr)
    n_tr = y.shape[0]

    def loss(p):
        f = pdf_func(p).reshape(-1)              # flavour-major, == FK column order
        resid = y - fk @ f
        return jnp.sum(cinv_diag * resid ** 2) / n_tr

    return np.asarray(jax.hessian(loss)(params))


def loss_hessian_at_epoch(
    fit,
    replicas_path,
    epoch,
    fk_diagonal_basis_train_val,
    cinv_diagonal_basis_train_val,
    get_train_val_data,
    kwargs=frozenset(),
    replica_index_list=None,
    training: bool = True,
) -> NTKStats:
    """Per-replica parameter-space Hessian of the (training) loss at ``epoch``.

    Returns an :class:`NTKStats` of shape ``(nreplicas, P, P)`` (``P`` = number of
    trainable network weights). ``kwargs`` is the colibri NTK model-selector frozenset
    (the model specification); ``training=False`` builds the validation-loss Hessian
    instead. Replica ``r`` (1-based, ascending) maps to frame index ``r-1`` of the
    FK/Cinv/data ensembles (built over all postfit replicas in ascending order).
    """
    import jax
    from colibri.utils import get_pdf_model
    from colibri.ntk.ntkutils import get_parameters_all_epochs, get_replica_idx_list

    idx = 0 if training else 1
    fk = fk_diagonal_basis_train_val[idx]
    cinv = cinv_diagonal_basis_train_val[idx]
    data = get_train_val_data[idx]

    if replica_index_list is None:
        replica_index_list = get_replica_idx_list(replicas_path)
    replica_index_list = sorted(replica_index_list)

    log.info("[hessian] fit '%s' epoch %s (training=%s): %d replicas, P from model",
             fit.label, epoch, training, len(replica_index_list))

    hessians = []
    for r in replica_index_list:
        pdf_model = get_pdf_model(fit.name, replica_idx=r)
        pdf_func = pdf_model.grid_values_func(np.asarray(XGRID), **dict(kwargs))
        params = np.load(get_parameters_all_epochs(replicas_path, r)[epoch])["params"]

        f_idx = r - 1
        H = _loss_hessian_by_replica(
            pdf_func,
            params,
            fk.data[f_idx],
            np.diag(cinv.data[f_idx]),
            data.data[f_idx].reshape(-1),
        )
        hessians.append(H)
        jax.clear_caches()  # drop this replica's compiled program (see compute_ntk)

    return NTKStats(np.stack(hessians))


def loss_hessian_eigenvalues_at_epoch(loss_hessian_at_epoch) -> NTKStats:
    """Eigenvalues (descending) of the per-replica loss Hessian at one epoch, as an
    :class:`NTKStats` of shape ``(nreplicas, P)``. Eigenvalues can be negative -- the
    full Hessian is not positive semi-definite away from a minimum."""
    H = loss_hessian_at_epoch
    eigs = np.stack(
        [np.linalg.eigvalsh(H.data[r])[::-1] for r in range(H.nreplica)]
    )
    return NTKStats(eigs)
