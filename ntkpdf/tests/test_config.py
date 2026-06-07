"""
test_config.py

Fit-free unit tests for ``ntkpdf.config``: the model-selector -> kwargs mapping
that decides which model modifier the NTK / eigenvalue computation uses, and the
name-keyed serialisation contract.

These exercise the *real* reportengine resolution path through the API
(``API.kwargs`` -> ``ntkConfig.produce_kwargs``) without building any model or
loading a fit -- ``produce_kwargs`` depends only on ``name``. This is the heart
of the ``Eigenvaluesconfigs`` wiring: each entry there carries only a ``name``
(``model``/``nn``), and that name alone must select the right modifier.
"""

import pytest

from reportengine.configparser import ConfigError

from ntkpdf.api import API
from ntkpdf.config import FULL_MODEL, NN, NO_PREFACTORS, NO_SMR


@pytest.mark.parametrize(
    "name, expected",
    [
        ("model", FULL_MODEL),
        ("nn", NN),
        ("no_prefactors", NO_PREFACTORS),
        ("no_smr", NO_SMR),
    ],
)
def test_kwargs_resolves_to_modifier(name, expected):
    """Each selector ``name`` resolves, through ``produce_kwargs``, to the
    modifier kwargs that get passed down to the NTK computation."""
    assert API.kwargs(name=name) == expected


def test_full_model_imposes_no_modifier():
    """``model`` must impose *no* modifier -- the full, unmodified model."""
    assert dict(API.kwargs(name="model")) == {}


def test_model_and_nn_differ():
    """The two ``Eigenvaluesconfigs`` selectors must map to *different*
    modifiers; otherwise both report variants would compute the same NTK."""
    assert API.kwargs(name="model") != API.kwargs(name="nn")


def test_unknown_selector_raises():
    with pytest.raises(ConfigError):
        API.kwargs(name="not_a_selector")


def test_cache_filename_is_name_keyed():
    """Distinct selector names must serialise to distinct cache files, so one
    config's eigenvalues can never be loaded in place of another's. This is the
    other half of the safety property: both the modifier *and* the on-disk cache
    key derive from the same ``name``, so they cannot desync."""
    from colibri.ntk.ntkutils import generate_filename

    model_file = generate_filename(1, "model")
    nn_file = generate_filename(1, "nn")
    assert model_file != nn_file
    assert "model" in model_file
    assert "nn" in nn_file
