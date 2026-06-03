import numpy as np

from validphys.pdfgrids import check_basis
from n3fit.vpinterface import EVOL_LIST

# Basis evolution has this ordering
#
#  sigma V T3 V3 T8 V8 T15 V15 T24 V24 T35 V35 g photon
#
# while EVOL_LIST (which is also the ordering of the output of the PDF model)
#
#  photon sigma g V T3 V3 T8 V8 T15 V15 T24 V24 T35 V35
#
# You should be able the reorder the output of the PDF model using the following permutation:
#
#   result = result[:, EVOL_IDX_REORDER, :]
#
evolution_basis = check_basis("evolution", EVOL_LIST)['basis']
EVOL_IDX_REORDER = np.argsort(evolution_basis._to_indexes(EVOL_LIST))

# This rotation matrix has the following structure
#           tbar bbar cbar sbar ubar dbar g  d  u  s  c  b  t  ph
#   Sigma     1   1    1    1     1   1   0  1  1  1  1  1  1  0
#   V        -1   -1   -1   -1   -1  -1   0  1  1  1  1  1  1  0
#   T3       1   1    1    1     1   1   0  1  1  1  1  1  1  0
#   V3
#   T8
#   V8
#   T15
#   V15
#   T24
#   V24
#   T35
#   V35
#   g
#   ph
R_FROM_FLAV_TO_EVOL = evolution_basis.from_flavour_mat


R_FROM_EVOL_TO_FLAV = np.linalg.inv(R_FROM_FLAV_TO_EVOL)
R_FROM_EVOL_TO_FLAV[np.abs(R_FROM_EVOL_TO_FLAV) < 1e-12] = 0.0

