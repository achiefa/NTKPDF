Fit readers and model interfaces
====================================

Readers isolate storage, initialisation and loss conventions. Ensembles provide
batched PDF evaluation; the single-replica model interface provides pure
parameter-dependent functions for JAX differentiation.

Reader selection
----------------

.. automodule:: ntkpdf.fits
   :members:
   :undoc-members:

Common containers
-----------------

.. automodule:: ntkpdf.fits.common
   :members:
   :undoc-members:

n3fit reader and ensemble
-----------------------------

.. automodule:: ntkpdf.fits.n3fit
   :members:
   :undoc-members:

Colibri reader and ensemble
-------------------------------

.. automodule:: ntkpdf.fits.colibri
   :members:
   :undoc-members:

Differentiable model and loader patch
-----------------------------------------

.. automodule:: ntkpdf.model
   :members:
   :undoc-members:
   :show-inheritance:
