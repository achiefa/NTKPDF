Application and configuration
=============================

These modules wire the shared provider list into the CLI and programmatic API.
``ntkConfig`` is the boundary between declarative resource names and fit-reader
operations. Importing ``app`` also installs the model-loader patch.

Configuration
-------------

.. automodule:: ntkpdf.config
   :members:
   :undoc-members:
   :show-inheritance:

CLI application
---------------

.. automodule:: ntkpdf.app
   :members:
   :undoc-members:
   :show-inheritance:

Python API
----------

``API`` is a ``reportengine.api.API`` instance, not a class defined by ntkpdf.
Use ``API.<resource>(**inputs)`` to request a named resource.

.. automodule:: ntkpdf.api
   :members:

Analysis script
---------------

.. automodule:: ntkpdf.scripts.ntkanalysis
   :members:
   :undoc-members:
   :show-inheritance:
