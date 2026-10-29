"""
test_ntkdecomposition.py

The NTK decomposition (``ntkpdf.ntkdecomposition``) on small synthetic
eigensystems: orthonormal ``Z``, a spectrum with a few modes above the cut and a
tail below it, and an SPD metric ``M``.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.linalg
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from colibri.ntk.ntkutils import NTKStats
from ntkpdf.ntkdecomposition import (
    _h_perp_eigenvalues, _perp_cut, _q_columns_by_replica,
    compute_ntk_decomposition_by_replica_at_epoch, compute_ntk_decomposition_ensemble,
)

N = 6
CUT = 4
TOL = 1e-8


@st.composite
def eigensystems(draw):
    Z, _ = np.linalg.qr(draw(arrays(np.float64, (N, N), elements=st.floats(-1, 1))))
    head = np.sort(draw(arrays(np.float64, CUT, elements=st.floats(0.1, 10))))[::-1]
    lam = np.concatenate([head, [1e-10 * head[0], 0.0]])
    A = draw(arrays(np.float64, (N, N), elements=st.floats(-1, 1)))
    M = A @ A.T + 0.1 * np.eye(N)
    return lam, pd.DataFrame(Z), M


def test_perp_cut():
    lam = np.array([5, 3, 1, 0.5, 1e-10, 0])
    assert _perp_cut(lam, TOL) == CUT
    with pytest.raises(ValueError):
        _perp_cut(np.array([0.0, 1.0]), TOL)
    with pytest.raises(ValueError):
        _perp_cut(lam, 1.0)


@settings(deadline=None)
@given(eigensystems())
def test_h_are_the_eigenvalues_of_k_m(system):
    lam, Z, M = system
    d = compute_ntk_decomposition_by_replica_at_epoch(lam, Z, M, TOL)
    Zp = Z.values[:, :CUT]
    KM = Zp @ np.diag(lam[:CUT]) @ Zp.T @ M
    top = np.sort(np.linalg.eigvals(KM).real)[::-1][:CUT]
    np.testing.assert_allclose(d.h[:CUT], top, rtol=1e-8)
    np.testing.assert_allclose(_h_perp_eigenvalues(lam, Z, M, TOL), d.h, rtol=1e-12, atol=1e-14 * d.h[0])  # eigvalsh vs eigh


@settings(deadline=None)
@given(eigensystems())
def test_decomposition_identities(system):
    lam, Z, M = system
    d = compute_ntk_decomposition_by_replica_at_epoch(lam, Z, M, TOL)
    P_par, P_perp = d.P_parallel.values, d.P_perp.values
    Q, Qinv = d.Q.values[:, :CUT], d.Qinv.values[:CUT]
    np.testing.assert_allclose(P_par + P_perp, np.eye(N), atol=1e-10)
    np.testing.assert_allclose(P_par @ P_par, P_par, atol=1e-10)
    np.testing.assert_allclose(P_perp @ P_perp, P_perp, atol=1e-10)
    np.testing.assert_allclose(Qinv @ Q, np.eye(CUT), atol=1e-10)
    np.testing.assert_allclose(Q.T @ M @ Q, np.diag(d.h[:CUT]), atol=1e-10 * d.h[0])


@settings(deadline=None)
@given(eigensystems())
def test_q_only_path_matches_full_path(system):
    lam, Z, M = system
    cut, Q = _q_columns_by_replica(lam, Z, M, TOL)
    assert cut == CUT
    np.testing.assert_allclose(Q, compute_ntk_decomposition_by_replica_at_epoch(lam, Z, M, TOL).Q.values, atol=1e-12)


def _gradient_flow(system, extra_rows, t):
    """Exact solution of the linearised training dynamics
    df/dt = -2 K (M f - b),  K = Z_perp diag(lambda_perp) Z_perpᵀ,
    M = Fkᵀ Cinv Fk,  b = Fkᵀ Cinv y,
    i.e. gradient flow on (y - Fk f)ᵀ Cinv (y - Fk f) through the kernel, as
    ``(U, V)`` with f(t) = U f0 + V y. Built as one matrix exponential of the
    affine system, independently of the Q/h machinery under test."""
    lam, Z, M_drawn = system
    # Full column rank by construction, so Z_perpᵀ M Z_perp is positive definite.
    Fk = np.vstack([np.linalg.cholesky(M_drawn).T, extra_rows])
    Zp = Z.values[:, :CUT]
    K = Zp @ np.diag(lam[:CUT]) @ Zp.T
    M = Fk.T @ Fk
    ndata = len(Fk)
    gen = np.zeros((N + ndata, N + ndata))
    gen[:N, :N] = -2 * K @ M
    gen[:N, N:] = 2 * K @ Fk.T
    flow = scipy.linalg.expm(t * gen)
    return Fk, flow[:N, :N], flow[:N, N:]


def _operator(system, Fk):
    lam, Z, _ = system
    Fk, Cinv = pd.DataFrame(Fk), pd.DataFrame(np.eye(len(Fk)))
    M = Fk.T @ Cinv @ Fk
    return compute_ntk_decomposition_ensemble(
        NTKStats(lam[None]), NTKStats([Z]), (Fk, Fk), (Cinv, Cinv), (M, M), TOL
    )


TIMES = (0.0, 0.1, 1.0, 10.0)
EXTRA_ROWS = arrays(np.float64, (2, N), elements=st.floats(-1, 1))


@settings(deadline=None, max_examples=20)
@given(eigensystems(), EXTRA_ROWS)
def test_v_matches_gradient_flow(system, extra_rows):
    for t in TIMES:
        Fk, _, V = _gradient_flow(system, extra_rows, t)
        np.testing.assert_allclose(np.asarray(_operator(system, Fk).v(t).data)[0], V, atol=1e-9)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "EvolutionOperator.u_check has the wrong sign: the perp response to the "
    "parallel part of f0 is -Q h^-1 (1 - e^{-2ht}) Qᵀ M P_par f0 (the same "
    "gradient gives v its + sign), but ntkdecomposition.py adds it."
))
@settings(deadline=None, max_examples=20)
@given(eigensystems(), EXTRA_ROWS)
def test_u_matches_gradient_flow(system, extra_rows):
    for t in TIMES:
        Fk, U, _ = _gradient_flow(system, extra_rows, t)
        np.testing.assert_allclose(np.asarray(_operator(system, Fk).u(t).data)[0], U, atol=1e-9)
