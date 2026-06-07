"""
conftest.py

Pytest fixtures for the ntkpdf test suite. Mirrors the layout of the upstream
validphys/n3fit suites: the test fit is referenced *by name* (the validphys
loader downloads it on demand, so nothing is committed to the repo) plus the
platform-marker plumbing copied from those suites.
"""

import sys

import pytest

# A small NTK fit (3 replicas, epoch snapshots 100..1000) hosted on the NNPDF
# server; ``FallbackLoader`` downloads it on demand. Used by the end-to-end
# eigenvalue tests. The fit-free config tests do not need it.
FIT = "260607-ac-serial"


@pytest.fixture(scope="session")
def ntk_fit():
    """The test fit, located -- and downloaded if absent -- via the loader."""
    from validphys.loader import FallbackLoader

    return FallbackLoader().check_fit(FIT)


@pytest.fixture
def clean_eigenvalue_cache(ntk_fit):
    """Delete any ``ntk_eigenvalues_*`` cache files created during a test.

    The eigenvalue providers serialise into ``<fit>/fit_replicas/replica_*/``
    (see colibri's ``produce_replicas_path``), i.e. *inside* the fit directory.
    This fixture snapshots the cache files present before the test and removes
    any that appear during it, so the (possibly shared, downloaded) fit is left
    untouched.
    """
    replicas_dir = ntk_fit.path / "fit_replicas"
    pattern = "replica_*/ntk_eigenvalues_*.npz"
    before = set(replicas_dir.glob(pattern))
    yield
    for created in set(replicas_dir.glob(pattern)) - before:
        created.unlink()


def pytest_runtest_setup(item):
    ALL = {"darwin", "linux"}
    supported_platforms = ALL.intersection(mark.name for mark in item.iter_markers())
    plat = sys.platform
    if supported_platforms and plat not in supported_platforms:
        pytest.skip(f"cannot run on platform {plat}")
