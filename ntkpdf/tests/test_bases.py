"""
test_bases.py

The rotations between the n3fit output order (``EVOL_LIST``), the validphys
evolution basis and the flavour basis.
"""

import numpy as np

from ntkpdf.bases import EVOL_IDX_REORDER, R_FROM_EVOL_TO_FLAV, R_FROM_FLAV_TO_EVOL, evolution_basis
from ntkpdf.utils import EVOL_LIST, flav_to_evol_matrix

# validphys labels vs the n3fit names for the same flavours
VP_TO_N3FIT = {"\\Sigma": "sigma", "g": "gluon"}


def test_rotations_are_inverse():
    np.testing.assert_allclose(R_FROM_EVOL_TO_FLAV @ R_FROM_FLAV_TO_EVOL, np.eye(14), atol=1e-12)
    np.testing.assert_allclose(R_FROM_FLAV_TO_EVOL @ R_FROM_EVOL_TO_FLAV, np.eye(14), atol=1e-12)


def test_reorder_maps_evol_list_to_validphys_order():
    expected = [VP_TO_N3FIT.get(l, l) for l in evolution_basis.labels]
    assert list(np.array(EVOL_LIST)[EVOL_IDX_REORDER]) == expected


def test_flav_to_evol_matrix_matches_validphys():
    reordered = np.array(EVOL_LIST)[EVOL_IDX_REORDER].tolist()
    np.testing.assert_array_equal(flav_to_evol_matrix(reordered), R_FROM_FLAV_TO_EVOL)


def test_flavour_name_aliases():
    np.testing.assert_array_equal(flav_to_evol_matrix(["gluon", "sigma"]), flav_to_evol_matrix(["g", "\\Sigma"]))
