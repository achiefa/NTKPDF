"""
test_ntkdecomposition.py

The evolution operator (``ntkpdf.ntkdecomposition``) against the exact solution
of the linearised training dynamics, on small synthetic eigensystems.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.linalg

from colibri.ntk.ntkutils import NTKStats
from ntkpdf.ntkdecomposition import compute_ntk_decomposition_ensemble

N = 6
CUT = 4
TOL = 1e-8
TIMES = (0.0, 0.1, 1.0, 10.0)


def _system(seed):
    rng = np.random.default_rng(seed)
    Z, _ = np.linalg.qr(rng.normal(size=(N, N)))
    lam = np.concatenate([np.sort(rng.uniform(0.1, 10, CUT))[::-1], [1e-10, 0.0]])
    Fk = rng.normal(size=(N + 2, N))
    return lam, Z, Fk


def _gradient_flow(lam, Z, Fk, t):
    """``(U, V)`` with f(t) = U f0 + V y, the exact solution of
    df/dt = -2 K (M f - b),  K = Z_perp diag(lambda_perp) Z_perpᵀ, M = FkᵀFk, b = Fkᵀy:
    gradient flow on |y - Fk f|² through the kernel, as one matrix exponential of
    the affine system."""
    Zp = Z[:, :CUT]
    K = Zp @ np.diag(lam[:CUT]) @ Zp.T
    ndata = len(Fk)
    gen = np.zeros((N + ndata, N + ndata))
    gen[:N, :N] = -2 * K @ Fk.T @ Fk
    gen[:N, N:] = 2 * K @ Fk.T
    flow = scipy.linalg.expm(t * gen)
    return flow[:N, :N], flow[:N, N:]


@pytest.mark.parametrize("seed", range(5))
def test_evolution_operator_matches_gradient_flow(seed):
    lam, Z, Fk = _system(seed)
    Fk_df, Cinv = pd.DataFrame(Fk), pd.DataFrame(np.eye(len(Fk)))
    M = Fk_df.T @ Cinv @ Fk_df
    op = compute_ntk_decomposition_ensemble(
        NTKStats(lam[None]), NTKStats([pd.DataFrame(Z)]), (Fk_df, Fk_df), (Cinv, Cinv), (M, M), TOL
    )
    for t in TIMES:
        U, V = _gradient_flow(lam, Z, Fk, t)
        np.testing.assert_allclose(np.asarray(op.u(t).data)[0], U, atol=1e-9)
        np.testing.assert_allclose(np.asarray(op.v(t).data)[0], V, atol=1e-9)
