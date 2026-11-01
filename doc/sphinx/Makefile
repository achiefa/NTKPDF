# Build the HTML documentation: `make html` here, or `make -C doc/sphinx html` from the
# repository root. PYTHON is the environment's interpreter (override: PYTHON=...).
PYTHON ?= python
SPHINXOPTS ?=
SOURCEDIR = source
BUILDDIR = build

.PHONY: html help
html:
	$(PYTHON) -m sphinx -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html

help:
	$(PYTHON) -m sphinx -M help $(SOURCEDIR) $(BUILDDIR) $(SPHINXOPTS)
