Getting started
===============

Installation
------------

Use an environment in which NNPDF's ``n3fit`` and ``validphys``, and colibri,
are already importable. From the ntkpdf checkout, install with Poetry:

.. code-block:: console

   poetry install

``pyproject.toml`` pins colibri's ``ntk`` branch. NNPDF is supplied through
colibri rather than a second ntkpdf pin. Use compatible development checkouts;
this package uses their model and provider interfaces directly. A pickled
colibri model also needs the package defining its class to be importable
(for example, ``colibri_msht``).

Import ntkpdf before using its model stack. ``ntkpdf.__init__`` defaults
``KERAS_BACKEND`` to ``jax`` before the Keras-backed modules are loaded. It also
applies the plotting style. That style enables LaTeX text rendering; rendering
figures therefore needs a working LaTeX installation unless you override it.

Run a report
------------

For a prepared reportengine runcard:

.. code-block:: console

   ntkpdf runcard.yaml
   ntkpdf runcard.yaml -o report-output

The default output directory is the runcard's stem. Actions in ``actions_``
request providers; see :doc:`api/resources` for ntkpdf's resource names.

To fill the bundled analysis template for one fit:

.. code-block:: console

   python -m ntkpdf.scripts.ntkanalysis FIT_ID \
       --title "NTK analysis" --author "Your name" --keywords ntk pdf \
       --epochs 500 1000 --replicas 1 --formats pdf -o report-output

Replace ``FIT_ID`` with a fit known to validphys's loader. Requested epochs
must have saved parameter files. The module entry point above calls
``CompareFitApp.main`` through the script's ``main`` function.

The following arguments are defined in
``ntkpdf.scripts.ntkanalysis.CompareFitApp.add_positional_arguments``:

.. list-table:: Analysis-script arguments
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Meaning
   * - ``current_fit``
     - Fit identifier. Required unless supplied interactively.
   * - ``--title``, ``--author``
     - Report metadata. Required in non-interactive mode.
   * - ``--keywords K1 K2 ...``
     - One or more metadata keywords. Required in non-interactive mode.
   * - ``--current_fit_label LABEL``
     - Display label; default ``Current Fit``.
   * - ``--thcovmat_if_present``
     - Set ``use_thcovmat_if_present`` for statistical estimators. Off by default.
   * - ``--epochs E1 E2 ...``
     - Integer snapshots for ``Epochspecs`` and the PDF-at-epoch pages.
       Default: no selected snapshots.
   * - ``--replicas R1 R2 ...``
     - 1-based replica selections for ``Replicaspecs`` on single-replica
       feature/eigenvalue pages. These pages are skipped when omitted.
   * - ``--show-fakepdf``
     - Overlay the closure-test underlying law on PDF-at-epoch plots.
       Off by default; the fit must define an underlying law if enabled.
   * - ``-i``, ``--interactive``
     - Prompt for missing metadata and the fit, and prompt for the fit label.

Pass ``--title`` even with ``-i``: the current ``interactive_title`` default
refers to ``reference_fit``, which this parser does not define.

The script also inherits reportengine/validphys options. ``-o`` selects the
output directory. ``--formats pdf`` avoids the default PNG plus PDF output.
The bundled ``analysis.yaml`` is not a complete standalone runcard: the script
supplies the missing fit and metadata through ``complete_mapping``.

.. warning::

   Reports and NTK API requests can write eigenvalue caches inside the fit's
   ``fit_replicas/replica_<r>/`` directories. They are not read-only operations
   on a shared fit installation. Eigenvalue cache filenames use the selector
   ``name``; do not reuse a cache name for different model modifiers.

Use the Python API
------------------

``API`` resolves the same resource names as the CLI. It is not necessary to
construct ``NTKApp``. These examples follow the notebooks in
``ntkpdf/examples/``; they require the named fits and snapshots to be available.

From ``plot_pdfs.ipynb``, request a PDF plotting grid at a saved epoch:

.. code-block:: python

   from ntkpdf.api import API

   grid = API.pdf_grid_at_epoch(fit="260611-ac-01-ntk-sgd", epoch=5000)

From ``h_val_grid.ipynb``, build a feature-eigenvalue trajectory:

.. code-block:: python

   from ntkpdf.api import API
   from ntkpdf.utils import EVOL_LIST

   common_dict = dict(
       dataset_inputs={"from_": "fit"},
       use_cuts="fromfit",
       theory={"from_": "fit"},
       theoryid={"from_": "theory"},
   )
   h_grid = API.h_val_grid(
       fit="260527-ac-01-ntk-sgd",
       **common_dict,
       training=True,
       name="model",
       evol_fl_names=EVOL_LIST,
       custom_epochs=[500, 1000, 2000, 5000],
   )

The notebook's ``from_`` context is convenient for isolated API requests.
Do not copy it unchanged into deeply nested report loops; see
:doc:`design/reportengine`. The current FK and training-data productions ask
the reader for its own data context, so they no longer need that global context
to reconstruct the metric.

``loss_hessian.ipynb`` uses the same API pattern:

.. code-block:: python

   H = API.loss_hessian_at_epoch(
       fit="260527-ac-01-ntk-sgd", epoch=500, training=True, **common_dict
   )
   eigenvalues = API.loss_hessian_eigenvalues_at_epoch(
       fit="260527-ac-01-ntk-sgd", epoch=500, training=True, **common_dict
   )

``H.data`` has shape ``(nreplicas, P, P)``, where ``P`` is the parameter count.
``analytical_solution.ipynb`` and ``evolution_operator.ipynb`` show the
``EvolutionOperator`` interface. Its time is an explicit argument, not a saved
epoch identifier; see :doc:`design/analysis` before converting between them.

Running tests
-------------

From the checkout, with pytest and the test dependencies available:

.. code-block:: console

   pytest -m "not fit"
   pytest

The first command excludes tests marked ``fit``. These include the external-fit
integration tests; the remaining tests include small synthetic n3fit models,
so “not fit” does not mean “no model construction”.

Plain ``pytest`` includes both kinds of integration test:

* ``ntk_fit`` in ``ntkpdf.tests.conftest`` asks ``FallbackLoader`` for
  ``260607-ac-sequential``, downloading it if absent.
* ``colibri_fit`` reuses ``ntkpdf_test_colibri_msht`` if present. Otherwise it
  builds two replicas for ten epochs with ``python -m colibri_msht.app``, using
  ``colibri_msht/runcards/fppdf_perturbative_charm_nosr_mc.yaml`` with
  ``record_every: 1``. It moves the result into the NNPDF results directory.

Thus the ``fit`` marker denotes integration tests needing an external fit,
not only tests that download data. The colibri tests require ``colibri_msht``
with ``FixedParamModel`` importable. These tests can train a fixture and write
outside the checkout. The analysis package itself does not perform that training.

Building these docs
-------------------

The Sphinx layout follows colibri: ``source/`` contains the reStructuredText
pages and ``conf.py``; ``build/`` holds the generated output. From the
repository root, with the environment's Python:

.. code-block:: console

   make -C doc/sphinx html

Add ``PYTHON=/path/to/python`` to use another interpreter. The pages are written
to ``doc/sphinx/build/html``. Autodoc imports the package from this checkout
(``conf.py`` puts it first on the path), so the environment must provide
ntkpdf's dependencies. The build executes no notebook or report, and keeps
import-time caches under ``build/``. Existing docstring problems are tracked in
``doc/sphinx/DOCSTRING_ISSUES.md`` rather than corrected in the Python source.
