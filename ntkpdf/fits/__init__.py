"""
ntkpdf.fits

ntkpdf reads two kinds of fit, which store their results differently:

- n3fit fits (NNPDF's neural network), see :mod:`ntkpdf.fits.n3fit`;
- colibri fits (any colibri ``PDFModel``, e.g. colibri_msht's MSHT parametrisation),
  see :mod:`ntkpdf.fits.colibri`.

:func:`load_fit` tells them apart; the rest of ntkpdf only talks to the object it
returns, whose members are the same for both kinds.
"""

from reportengine.configparser import ConfigError

from ntkpdf.fits.colibri import ColibriFit
from ntkpdf.fits.n3fit import N3Fit


def load_fit(fit):
    """The reader of ``fit``. An n3fit fit has n3fit's ``nnfit/`` folder (and may also
    carry a ``pdf_model.pkl``); a colibri fit pickles its model in ``pdf_model.pkl``."""
    if (fit.path / "nnfit").is_dir():
        return N3Fit(fit)
    if (fit.path / "pdf_model.pkl").is_file():
        return ColibriFit(fit)
    raise ConfigError(f"Fit {fit.name} is neither an n3fit fit (no nnfit/) nor a colibri fit (no pdf_model.pkl).")
