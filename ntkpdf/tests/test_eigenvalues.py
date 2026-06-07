"""
test_eigenvalues.py

End-to-end test that the ``Eigenvaluesconfigs`` ``name`` drives the eigenvalue
computation *and* its serialisation through the real reportengine graph: the
``model`` (full) and ``nn`` (no-prefactor / no-MSR) configs must produce
*different* NTK eigenvalues and write to *distinct*, name-keyed cache files.

Together with ``test_config.py`` this closes the loop: ``test_config`` proves
``name`` resolves to the exact modifier kwargs; this proves those resolved
kwargs actually flow into the computation and the on-disk cache.

Slow: it uses the downloaded test fit and computes NTKs. Restricted to a single
replica and the first stored epoch to stay fast, and ``force_recompute=True`` so
it never depends on (or trusts) a pre-existing cache.
"""

import numpy as np

from ntkpdf.api import API
from ntkpdf.tests.conftest import FIT

REPLICA = 1
MAX_EPOCH = 100  # smallest stored snapshot -> a single NTK per config


def _eigenvalues(name):
    """First-epoch eigenvalues for ``name``, recomputed from scratch."""
    res = API.eigenvalues_ensemble(
        fit=FIT,
        name=name,
        replica_index_list=(REPLICA,),
        max_epoch=MAX_EPOCH,
        force_recompute=True,
        max_workers=1,
    )
    epoch = res["epochs"][0]
    return res["eigenvalues_by_epoch"][epoch]


def test_model_and_nn_eigenvalues_differ(clean_eigenvalue_cache):
    """The resolved modifier actually changes the computed NTK: the ``model``
    and ``nn`` eigenvalues must differ."""
    model_eigs = _eigenvalues("model")
    nn_eigs = _eigenvalues("nn")
    assert not np.allclose(model_eigs, nn_eigs)


def test_eigenvalues_serialised_under_name(ntk_fit, clean_eigenvalue_cache):
    """Each config serialises to its own name-keyed cache file inside the fit,
    so the two variations can never collide on disk."""
    from colibri.ntk.ntkutils import generate_filename

    _eigenvalues("model")
    _eigenvalues("nn")

    replica_dir = ntk_fit.path / "fit_replicas" / f"replica_{REPLICA}"
    model_file = replica_dir / generate_filename(REPLICA, "model")
    nn_file = replica_dir / generate_filename(REPLICA, "nn")
    assert model_file.exists()
    assert nn_file.exists()
    assert model_file != nn_file
