import logging
import os

# Configure the keras/jax backend and precision *before* importing anything
# (including the plotting style) that might pull in keras/jax -- once jax is
# imported the env var below is read, and once a model is built the floatx is
# baked into its variables. Keep this block at the very top.
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Run the analysis in *genuine* double precision. Both halves are required, and
# both must be in place before the first model build / jax computation, so that
# ntkpdf's own ``generate_pdf_model`` builds float64 *weights* -- not merely a
# float64-typed array holding a float32 computation:
#   - jax defaults to 32-bit and silently downcasts float64 -> float32 unless
#     x64 is enabled. ``config.update`` is reliable post-import/pre-computation
#     (an env var only works if set before jax is first imported).
#   - keras' default ``floatx`` is float32; ``set_floatx`` makes model variables
#     float64, matching what colibri-n3fit's ``N3FitPDFModel`` forces on the NTK
#     path so the two model-build paths agree.
import jax

jax.config.update("jax_enable_x64", True)
import keras

keras.backend.set_floatx("float64")

from ntkpdf.plotting.style import setup

# Silence the chatty loggers
for _name in ("validphys", "n3fit", "keras"):
    logging.getLogger(_name).setLevel(logging.ERROR)

# Setup plotting style
setup()