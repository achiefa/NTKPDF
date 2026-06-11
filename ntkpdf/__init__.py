import logging
import os

# Match the environment the analysis notebooks/scripts expect, *before* any
# tensorflow/keras-backed module (the `ntk` package) is imported.
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# NB: double precision is configured at model-build time, not here. colibri's
# ``Environment`` enables ``jax_enable_x64`` and ``produce_base_metamodels`` sets
# keras ``floatx`` to float64 just before building the model. Toggling JAX state
# at import (before jax initialises) proved fragile, so it is deliberately done
# lazily, once colibri has set up the backend.
from ntkpdf.plotting.style import setup

# Silence the chatty loggers
for _name in ("validphys", "n3fit", "keras"):
    logging.getLogger(_name).setLevel(logging.ERROR)

# Setup plotting style
setup()
