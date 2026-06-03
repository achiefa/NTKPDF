import logging
import os
from ntkpdf.plotting.style import setup

# Match the environment the analysis notebooks/scripts expect, *before* any
# tensorflow/keras-backed module (the `ntk` package) is imported.
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Silence the chatty loggers
for _name in ("validphys", "n3fit", "keras"):
    logging.getLogger(_name).setLevel(logging.ERROR)

# Setup plotting style
setup()