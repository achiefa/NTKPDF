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

log = logging.getLogger(__name__)

# API needed its own module, so that it can be used with any Matplotlib backend
# without breaking validphys.app
API = api.API(ntk_providers, ntkConfig, Environment)
