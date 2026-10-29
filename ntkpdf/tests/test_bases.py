"""
test_bases.py

The rotations between the n3fit output order (``EVOL_LIST``), the validphys
evolution basis and the flavour basis.
"""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from ntkpdf.bases import EVOL_IDX_REORDER, R_FROM_EVOL_TO_FLAV, R_FROM_FLAV_TO_EVOL, evolution_basis
from ntkpdf.utils import EVOL_LIST, flav_to_evol_matrix


@given(arrays(np.float64, 14, elements=st.floats(-1e3, 1e3)))
def test_rotations_round_trip(f):
    np.testing.assert_allclose(R_FROM_EVOL_TO_FLAV @ (R_FROM_FLAV_TO_EVOL @ f), f, atol=1e-9)
    np.testing.assert_allclose(R_FROM_FLAV_TO_EVOL @ (R_FROM_EVOL_TO_FLAV @ f), f, atol=1e-9)


def test_reorder_maps_evol_list_to_validphys_order():
    assert np.all(np.diff(np.asarray(evolution_basis._to_indexes(EVOL_LIST))[EVOL_IDX_REORDER]) > 0)


def test_flav_to_evol_matrix_matches_validphys():
    reordered = np.array(EVOL_LIST)[EVOL_IDX_REORDER].tolist()
    np.testing.assert_array_equal(flav_to_evol_matrix(reordered), R_FROM_FLAV_TO_EVOL)


def test_flavour_name_aliases():
    np.testing.assert_array_equal(flav_to_evol_matrix(["gluon", "sigma"]), flav_to_evol_matrix(["g", "\\Sigma"]))
