Grid interoperability and support
=================================

These modules implement basis transformations, validphys wrappers, grid
conversion and logging. They are not registered as ntkpdf provider modules.
Their functions can be called directly from Python.

Package initialisation
----------------------

Importing ntkpdf sets backend defaults and applies its plotting style.

.. automodule:: ntkpdf
   :members:

validphys interoperability
--------------------------

.. automodule:: ntkpdf.vpinterface
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. automodule:: ntkpdf.utils
   :members:
   :undoc-members:

Bases
-----

``EVOL_IDX_REORDER`` maps n3fit output order to validphys's evolution order.
``R_FROM_FLAV_TO_EVOL`` and ``R_FROM_EVOL_TO_FLAV`` use the latter order;
see :doc:`../design/analysis` before applying them to a raw model array.

.. automodule:: ntkpdf.bases
   :members:
   :undoc-members:

Logging
-------

.. automodule:: ntkpdf.log
   :members:
   :undoc-members:
   :show-inheritance:

Template resources
------------------

``template_path`` points to the analysis-driver YAML shipped in the package.

.. automodule:: ntkpdf.analysistemplate
   :members:

Scripts namespace
-----------------

.. automodule:: ntkpdf.scripts
   :members:
