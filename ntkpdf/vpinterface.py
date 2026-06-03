"""
NTK interface to validphys
"""

from functools import cached_property
import logging

import numpy as np

from colibri.constants import XGRID
from colibri.ntk.ntkutils import NTKStats

from n3fit.vpinterface import EVOL_LIST

from validphys.core import PDF
from validphys.lhapdfset import LHAPDFSet
from validphys.pdfbases import ALL_FLAVOURS, check_basis
from validphys.core import STAT_TYPES

from ntkpdf.bases import R_FROM_EVOL_TO_FLAV as to_flav

log = logging.getLogger(__name__)

STAT_TYPES["NKT"] = NTKStats

class NTKLHAPDF(LHAPDFSet):
    """Extension of LHAPDFSet using NTKStats"""

    def __init__(self, name, ntkstats: NTKStats, Q=1.65, is_t0=False, xgrid=XGRID):
        
        if not isinstance(ntkstats, NTKStats):
            raise ValueError(f"Expected ntkstats to be an instance of NTKStats, but got {type(ntkstats)}")
        
        # Check that NTKStats has the correct shape
        shape = ntkstats.shape
        if len(shape) != 2:
            raise ValueError(f"Expected NTKStats to have shape (n_flavours, n_xgrid), but got {shape}")
        if shape[0] != len(EVOL_LIST):
            raise ValueError(f"Expected NTKStats to have {len(EVOL_LIST)} flavours, but got {shape[0]}")
        if shape[1] != len(xgrid):
            raise ValueError(f"Expected NTKStats to have {len(xgrid)} xgrid points, but got {shape[1]}")
        
        self._error_type = "replicas"
        self._name = name,
        self._lhapdf_set = ntkstats
        self._flavors = None
        self._fitting_q = Q,
        self._is_t0 = is_t0
        self._xgrid = xgrid
        self.basis = check_basis("evolution", EVOL_LIST)["basis"]

    def __call__(self, xarr, flavours=None, replica=None):
        """Uses the internal data to produce pdf values for the grid.
        The output is always in evolution basis as in the n3fit models.

        Since the NTKStats object already has the grid values, the xarr argument
        is not used.
        """
        if flavours is None:
            flavours = EVOL_LIST

        if replica is None or replica == 0 or self._is_t0:
            if replica == 0 or self._is_t0:
                # We want _only_ the central value
                result = self._lhapdf_set.central_value()
            else:
              result = self._lhapdf_set.data # shape (n_replicas, n_flavours, n_xgrid)
        else:
            result = self._lhapdf_set.data[replica-1, ...]

        if flavours != "n3fit":
            # Ensure that the result has its flavour in the basis-defined order
            ii = self.basis._to_indexes(flavours)
            result = result[:, np.argsort(ii), :]
        return result


    def grid_values(self, flavours, xarr, qmat=None):
        """
        Get the grid values for the given flavours. Note that xarr is not used as
        the NTKStats object already has the grid values.
        """
        # Rotate onto flavour basis
        result = self(xarr)
        flav_result = np.einsum("fe,rex->rfx", to_flav, result)

        # Now drop the indices that are not requested
        requested_flavours = [ALL_FLAVOURS.index(i) for i in flavours]
        flav_result = flav_result[:, requested_flavours, :]

        # If given, insert as many copies of the grid as q values
        ret = np.expand_dims(flav_result, -1)
        if qmat is not None:
            lq = len(qmat)
            ret = ret.repeat(lq, -1)
        return ret
        
    def xfxQ(self, x, Q, n, fl):
        """Return the value of the PDF member for the given value in x"""
        if Q != self._fitting_q:
            log.warning(
                "Querying NTKLHAPDFSet at a value of Q=%f different from %f", Q, self._fitting_q
            )
        return self.grid_values([fl], [x]).squeeze()[n]
    

class NTKPDF(PDF):
    """
    Creates a NTKPDF object, extension of the validphys PDF object to perform calculation
    with a n3fit generated model.

    Parameters
    ----------
        NTKStats: :py:class:`colibri.ntk.ntkutils.NTKStats`
            PDF trained with n3fit, x -> f(x)_{i} where i are the flavours in the evol basis
        name: str
            name of the NTKPDF object
    """

    def __init__(self, ntkstats, name="ntkpdf", Q=1.65):
        
        if not isinstance(ntkstats, NTKStats):
            raise ValueError(f"Expected ntkstats to be an instance of NTKStats, but got {type(ntkstats)}")
        
        self._models = ntkstats

        super().__init__(name)
        self._stats_class = NTKStats
        self._Q = Q
        # Since there is no info file, create a fake `_info` dictionary
        self._info = {"ErrorType": "replicas", "NumMembers": self._models.nreplica}

    @cached_property
    def _lhapdf_set(self):
        return NTKLHAPDF(self.name, self._models, Q=self._Q)

    def load(self):
        """If the function needs an LHAPDF object, return a N3LHAPDFSet"""
        return self._lhapdf_set

    def load_t0(self):
        """Load the central PDF object"""
        return NTKLHAPDF(self.name, self._models, Q=self._Q, is_t0=True)

    def __call__(self, xarr, flavours=None, replica=None):
        """Uses the internal model to produce pdf values for the grid
        The output is on the evolution basis.

        Parameters
        ----------
            xarr: numpy.ndarray
                x-points with shape (xgrid_size,) (size-1 dimensions are removed)
            flavours: list
                list of flavours to output
            replica: int
                replica whose value must be returned (by default return all members)
                replica 0 corresponds to the central value

        Returns
        -------
            numpy.ndarray
                (xgrid_size, flavours) pdf result
        """
        return self._lhapdf_set(xarr, flavours=flavours, replica=replica)