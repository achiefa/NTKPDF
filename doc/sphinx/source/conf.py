"""Sphinx setup following colibri's source/conf.py, without optional extensions."""

from datetime import datetime
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "doc" / "sphinx" / "build"
sys.path.insert(0, str(ROOT))
# Autodoc imports the real package. Keep import-time caches inside this tree.
sys.dont_write_bytecode = True
os.environ["MPLCONFIGDIR"] = str(BUILD / "matplotlib")
os.environ["KERAS_HOME"] = str(BUILD / "keras")
os.environ["XDG_CACHE_HOME"] = str(BUILD / "cache")
os.environ["MPLBACKEND"] = "Agg"

project = "ntkpdf"
author = "Amedeo Chiefa"
copyright = f"{datetime.now().year}, {author}"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]
source_suffix = [".rst"]
master_doc = "index"
language = "en"
exclude_patterns = []
html_theme = "sphinx_rtd_theme"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = True
# Dataclass fields are also autodoc members. Use fields for NumPy Attributes
# sections rather than creating a second set of attribute index entries.
napoleon_use_ivar = True
