Architecture and design
=======================

ntkpdf separates storage conventions from analysis. Readers reconstruct a fit's
model and loss ingredients. Shared providers then work with PDF grids,
``TrainValData`` and colibri's ``NTKStats`` rather than fit-specific files.

.. code-block:: text

   saved fit -> load_fit -> reader
                             |-- ensemble -> PDF grids and losses
                             |-- pdf_model -> colibri NTK eigensystem
                             `-- training_data -> FK, Cinv, y -> M
                                                        |
                             eigensystem + M -> features / evolution
                             pdf_model + FK, Cinv, y -> loss Hessian

Reportengine resolves these quantities by name. Its namespaces are part of the
memory model, not just a way to organise report headings.

.. toctree::
   :maxdepth: 1

   reportengine
   fits-models
   analysis
   reports
