"""
test_ntkdecomposition.py

The NTK decomposition (``ntkpdf.ntkdecomposition``) on small synthetic
eigensystems: orthonormal ``Z``, a spectrum with a few modes above the cut and a
tail below it, and an SPD metric ``M``.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from colibri.ntk.ntkutils import NTKStats
from ntkpdf.ntkdecomposition import (
    _perp_cut, _q_columns_by_replica, compute_ntk_decomposition_by_replica_at_epoch,
    compute_ntk_decomposition_ensemble,
)

N = 6
CUT = 4
TOL = 1e-8


@st.composite
def eigensystems(draw):
    Z, _ = np.linalg.qr(draw(arrays(np.float64, (N, N), elements=st.floats(-1, 1))))
    head = np.sort(draw(arrays(np.float64, CUT, elements=st.floats(0.1, 10))))[::-1]
    assume(np.all(np.diff(head) < -1e-2))
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
def test_h_with_identity_metric_is_the_ntk_spectrum(system):
    lam, Z, _ = system
    d = compute_ntk_decomposition_by_replica_at_epoch(lam, Z, np.eye(N), TOL)
    np.testing.assert_allclose(d.h[:CUT], lam[:CUT], rtol=1e-10)


@settings(deadline=None)
@given(eigensystems())
def test_h_are_the_eigenvalues_of_k_m(system):
    lam, Z, M = system
    d = compute_ntk_decomposition_by_replica_at_epoch(lam, Z, M, TOL)
    Zp = Z.values[:, :CUT]
    KM = Zp @ np.diag(lam[:CUT]) @ Zp.T @ M
    top = np.sort(np.linalg.eigvals(KM).real)[::-1][:CUT]
    np.testing.assert_allclose(d.h[:CUT], top, rtol=1e-8)


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


@settings(deadline=None, max_examples=20)
@given(eigensystems(), arrays(np.float64, (8, N), elements=st.floats(-1, 1)))
def test_evolution_operator_at_t0(system, fk):
    """At t=0 the evolution operator is the identity on f0 and ignores the data."""
    lam, Z, _ = system
    Fk, Cinv = pd.DataFrame(fk), pd.DataFrame(np.eye(len(fk)))
    M = Fk.T @ Cinv @ Fk
    assume(np.linalg.eigvalsh(Z.values[:, :CUT].T @ M.values @ Z.values[:, :CUT]).min() > 1e-6)
    op = compute_ntk_decomposition_ensemble(
        NTKStats(lam[None]), NTKStats([Z]), (Fk, Fk), (Cinv, Cinv), (M, M), TOL
    )
    np.testing.assert_allclose(np.asarray(op.u(0).data)[0], np.eye(N), atol=1e-10)
    np.testing.assert_allclose(np.asarray(op.v(0).data)[0], 0, atol=1e-10)
