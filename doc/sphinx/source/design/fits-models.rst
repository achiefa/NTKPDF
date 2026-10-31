Reading fits and evaluating models
======================================

Choose the reader at the boundary
---------------------------------

``ntkpdf.fits.load_fit`` accepts a validphys ``FitSpec``. It checks for an
``nnfit/`` directory first and returns ``N3Fit``. Otherwise, a ``pdf_model.pkl``
file selects ``ColibriFit``. Neither present means ``ConfigError``. An n3fit fit
may contain a pickle too; the directory test takes precedence.

The storage convention is selected once per reader construction. Shared model,
metric and plotting consumers do not repeat that test. The ``fit_kind``
production exposes the reader to config resolution; productions for the
ensemble, saved epochs, learning rate, underlying law and training data delegate
to its methods. Colibri workers may load a fresh reader through
``get_pdf_model``; “once” here means once at the loading boundary, not a global
singleton.

.. list-table:: Storage and interpretation
   :header-rows: 1
   :widths: 18 41 41

   * - Item
     - n3fit: ``N3Fit``
     - colibri: ``ColibriFit``
   * - Runcard
     - ``filter.yml`` via ``FitSpec.as_input()``.
     - ``filter.yml`` via ``FitSpec.as_input()``; the reader drops ``actions_``.
   * - Model
     - Rebuilt from runcard architecture and ``nnfit/replica_<r>/<fit>.json`` preprocessing. No pickle is needed.
     - Unpickled with dill from ``pdf_model.pkl``. Its class must be importable and the object must be a colibri ``PDFModel``.
   * - Replica discovery
     - Reader metadata from numerically sorted ``nnfit/replica_<r>/``.
     - Numerically sorted ``fit_replicas/replica_<r>/``.
   * - Parameter snapshots
     - Both readers use ``fit_replicas/replica_<r>/parameters/params_<e>.npz`` with a flat ``params`` array.
     - Same path and array key.
   * - Meaning of ``e``
     - Weights after ``e`` steps; ``params_0`` is the saved initialisation.
     - Parameters after the update of epoch ``e``: after ``e+1`` updates in full-batch training. The initial point is not saved.
   * - ``stored_epochs``
     - Reads replica 1, assumes common snapshots, and removes replica 1's best epoch (which n3fit also saves).
     - Intersects saved epochs across all replicas, including early-stopped replicas.
   * - Training ingredients
     - Saved replica pseudodata and the fitting-covariance eigensystem in ``tables/``.
     - Regenerated from the runcard using colibri's ``central_covmat_index`` and ``mc_pseudodata``.
   * - Data coordinates
     - Covariance-diagonal coordinates; ``diagonal_basis: true`` is required.
     - Original data points, with dense inverse covariance blocks.
   * - NTK selectors
     - ``model``, ``no_prefactors``, ``nn``, ``no_smr``.
     - ``model`` only.
   * - Report settings
     - ``current`` includes fit, LHAPDF PDF and theory. Supplies direct global data/theory inputs; retains validphys summary tables.
     - ``current`` contains the fit and label only. Disables validphys tables and replaces selector settings with full-model settings.

The snapshot distinction follows n3fit's weight callback and colibri's
``run_gradient_descent``/``write_mc_replica``. Colibri records parameters after
the epoch's update loop. With multiple batches, one epoch contains multiple
optimizer steps, so do not read ``e+1`` as an unconditional batch-step count.
Its ``mc_loss.csv`` records the training loss evaluated during updates, not a
fresh loss evaluated at the saved post-update parameters. For full-batch
training the row at recording index ``k`` is the loss before the update of epoch
``k * record_every``. This explains why recomputed snapshot losses need not
match that row.

Reader and ensemble contracts
-----------------------------

There is no abstract reader base class. The interface is the methods used by
``ntkConfig``, the model loader and the report script:

.. list-table:: Shared reader interface
   :header-rows: 1
   :widths: 33 67

   * - Member
     - Purpose
   * - ``fit``, ``selectors``
     - Original fit specification and applicable NTK selector names.
   * - ``replicas()``, ``stored_epochs()``
     - Ordered replica IDs and supported common snapshot epochs.
   * - ``pdf_model(replica=None)``
     - Single-replica colibri ``PDFModel`` for differentiation.
   * - ``ensemble(nreplicas, replica_ids=None)``
     - Batched PDF evaluation with ``at(weights, selector)`` and ``at_init(selector)``.
   * - ``data()``, ``training_data(fk)``
     - Fit-specific data context and a common ``TrainValData`` result.
   * - ``learning_rate()``, ``summary()``
     - Constant-rate evolution input and fit-summary fields. The colibri reader rejects a learning-rate schedule for ``learning_rate()``.
   * - ``fakepdf()``, ``report_settings(current_fit)``
     - Closure-law identifier and analysis-runcard overrides.

The legacy ``produce_replica_paths`` and ``produce_replicas`` still enumerate
``nnfit/``; they are n3fit-specific resources, not the portable reader interface.
Shared weight loading and replica counting use ``fit_replicas/``.

To add another fit kind, implement the shared reader methods, return the same
grid ordering and ``TrainValData`` contract, then add one detection case in
``load_fit``. Keep storage decisions in that reader. A new NTK selector also
needs an explicit mapping in ``config.SELECTOR_KWARGS`` and reader validation;
PDF-ensemble selector mappings are a separate interface. Extend report settings
where upstream tables do not apply. Test file discovery, snapshot semantics,
loss reconstruction and agreement between ensemble and differentiation paths.
Do not add a fit-kind branch to each analysis provider.

Two n3fit paths, for two jobs
---------------------------------

.. code-block:: text

   N3Fit.ensemble                 N3Fit.pdf_model(replica)
          |                              |
   N3FitEnsemble                  NTKPDFN3Fit
   multi-replica MetaModel         single-replica PDFModel
   predict PDF grids              grid_values_func(XGRID, **kwargs)
          |                              |
   detached NumPy NTKStats         pure function: flat parameters -> PDFs
   plotting and loss trajectories  JAX Jacobian / NTK / loss Hessian

``N3FitEnsemble`` uses n3fit's ``generate_pdf_model`` to evaluate multiple
replicas together. Building it is expensive, so ``produce_pdf_ensemble`` shares
it across requests. Its Keras model is mutable. ``at`` loads the flat network
weights and fitted preprocessing before prediction; ``at_init`` restores a copy
of the freshly built weights. Returned grids are NumPy-backed, not live model
views. A later epoch load therefore cannot overwrite a previous result.

``at_init`` is a reconstructed initial ensemble built with fresh seeds
``0 .. nreplicas-1``. Do not confuse it with reading ``params_0`` from the fit.
The saved epoch-zero snapshot and this generated ensemble serve different
purposes.

``NTKPDFN3Fit`` instead implements colibri's ``PDFModel`` interface. Its
``grid_values_func`` unpacks a flat parameter vector and calls Keras
``stateless_call``. Differentiation sees parameters as JAX arguments, without
``set_weights`` side effects. The submodel for each ``exclude_layers`` choice
is cached on the instance so it is not rebuilt inside each trace. Both n3fit
build paths set Keras float precision to float64; the NTK model also enables
JAX x64 explicitly.

The NTK selector ``name`` resolves to ``kwargs``:

.. list-table:: n3fit differentiation modifiers
   :header-rows: 1

   * - Name
     - Change
   * - ``model``
     - No modifier.
   * - ``no_prefactors``
     - Set preprocessing exponents to the neutral values, alpha = 1 and beta = 0.
   * - ``no_smr``
     - Exclude the ``impose_msr`` sum-rule layer.
   * - ``nn``
     - Both of the preceding modifiers.

``layer_kwargs`` adds ``grad_layers``: 1-based Dense-layer indices whose
parameters remain differentiable. Other parameters pass through
``stop_gradient``; the forward PDF is unchanged. With ``cumulative=True`` it
includes all layers up to the largest selected index.

Do not equate these NTK modifiers with the ``model_selectors`` presentation
list. The PDF ensemble currently interprets its ``no_prefactors`` flag;
``analysis.yaml`` gives those presentation selectors the names
``with_prefactors`` and ``without_prefactors``. They are not NTK selector names.

The colibri path
----------------

``ColibriFit.pdf_model`` unpickles the model and checks its type.
``ColibriEnsemble`` wraps ``pdf_model.grid_values_func(XGRID)`` in
``jax.jit(jax.vmap(...))``. The same forward map supports batched grids and
single-replica differentiation, so no separate Keras reconstruction is needed.
``ColibriFit.ensemble`` regenerates initial parameters with colibri's
``pdf_initial_parameters`` and the replica ID as seed, using the saved
initialiser settings. It can add fresh replica IDs for initial-ensemble studies.
Neither reader trains the model.

Only load trusted fit pickles. Unpickling a model is code execution, not a
portable numerical-data interchange format.

Why the model-loader monkeypatch exists
-------------------------------------------

Colibri's NTK providers load models themselves inside worker threads. They
pass a fit name to ``get_pdf_model`` rather than accept a model dependency.
The upstream loader expects a pickle and imports ``colibri_n3fit`` for a saved
n3fit initialisation dictionary. ntkpdf needs to rebuild n3fit models from the
runcard without importing that fitting adapter, while still accepting native
colibri model pickles.

``ntkpdf.model.install`` replaces ``get_pdf_model`` in ``colibri.utils``,
``colibri.ntk.ntkutils``, ``colibri.ntk.ntk`` and ``colibri.checks``. Rebinding
only ``colibri.utils`` is insufficient: ``from ... import get_pdf_model`` gives
the consumers their own references. The replacement resolves a validphys
``FitSpec`` and delegates to ``load_fit(...).pdf_model(...)``.

``ntkpdf.app`` installs it at module import. Both the CLI and ``ntkpdf.api``
import that module; putting the patch only in ``NTKApp.__init__`` would miss
the API. A notebook using colibri providers directly must call ``install()``
itself, as ``examples/layer_ntk.ipynb`` does. After upgrading colibri, inspect
new import sites: ``install`` detects a missing symbol in a known target, but
cannot detect a new consumer retaining the upstream loader.
