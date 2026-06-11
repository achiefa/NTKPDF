"""
ntkpdf.api.py

This module contains the `reportengine` programmatic API, initialized with the
NTK providers and Validphys config and environment classes.
"""

import logging

from reportengine import api
from colibri.config import Environment

from ntkpdf.app import ntk_providers
from ntkpdf.config import ntkConfig
from ntkpdf.log import setup_logger

log = logging.getLogger(__name__)

# The CLI App configures logging via reportengine's init_logging; the API path does
# not, so log records would be silently dropped in a notebook. Install the
# notebook-aware colour handler here (idempotent) so API/notebook logs appear --
# HTML in Jupyter, the usual reportengine ANSI colours in a terminal/script.
setup_logger()

# API needed its own module, so that it can be used with any Matplotlib backend
# without breaking validphys.app
API = api.API(ntk_providers, ntkConfig, Environment)
