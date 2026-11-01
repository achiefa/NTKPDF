Runcard resource catalogue
==============================

This catalogue lists the public functions defined in the ntkpdf modules of
``ntkpdf.app.ntk_providers``, their named collections, and all productions
defined on ``ntkConfig``. These names can be requested as actions or
dependencies, or through ``API.<name>(**inputs)``. Many are intermediate
objects rather than renderable report elements. Figure and table providers
are the usual template targets.

The registry also includes validphys and colibri resources, which are not
enumerated here. Reportengine looks up module attributes, so imported helpers
can be visible too; they are not additional ntkpdf-defined report providers.
Classes and private helpers below the module level are described in the
autodoc pages, not presented as runcard actions.

Imported ntkpdf helpers
------------------------------

The following public functions are also module attributes of registered
providers, so reportengine can technically resolve their names. They are
implementation helpers rather than the intended report-facing interface:

* ``ntkpdf.data_theory`` imports ``flav_to_evol_matrix`` and ``peak_rss_gb``.
* ``ntkpdf.ntkdecomposition`` imports ``plotting_grid_from_ntkstat`` and
  ``peak_rss_gb``.
* ``ntkpdf.plotting.pdfplots_providers`` imports ``plot_grids`` and
  ``peak_rss_gb``.
* ``ntkpdf.plotting.loss_providers`` imports ``make_figure``, ``draw_bounds``
  and ``draw_line``.

Their signatures and behaviour are documented in :doc:`support` and
:doc:`plotting`. The registry also exposes imported classes, constants and
third-party helpers as module attributes; this catalogue does not recommend
requesting those as report resources.

Data, grids and summaries
-----------------------------

.. list-table:: Defined providers and collections
   :header-rows: 1
   :widths: 25 42 33

   * - Module
     - Resource
     - Result / collection scope
   * - ``ntkpdf.data_theory``
     - ``get_train_val_data``
     - Training/validation pseudodata pair.
   * - Same
     - ``fk_train_val``
     - Training/validation FK pair.
   * - Same
     - ``cinv_train_val``
     - Training/validation inverse covariance pair.
   * - Same
     - ``m_matrix_train_val``
     - Training/validation metric pair.
   * - Same
     - ``loss_function_grid``
     - Training grid and optional validation grid.
   * - Same
     - ``loss_function_grid_by_fit``
     - Collect ``loss_function_grid`` over ``fits``.
   * - ``ntkpdf.pdfgrids``
     - ``collected_init_grids``
     - Collect ``init_grids_by_name`` over ``fits``.
   * - Same
     - ``pdf_grid_all_epochs``
     - Collect ``pdf_grid_at_epoch`` over ``Epochspecs``.
   * - Same
     - ``all_epochs``
     - Collect ``epoch`` over ``Epochspecs``.
   * - ``ntkpdf.summary``
     - ``ntk_fit_summary``
     - One fit's summary table.
   * - Same
     - ``collected_ntk_fit_summaries``
     - Collect ``ntk_fit_summary`` over ``fits``.
   * - Same
     - ``summarise_ntk_fits``
     - Combined fit-summary table.

Decomposition and curvature
---------------------------

All decomposition resources in this table belong to
``ntkpdf.ntkdecomposition``; the last two belong to ``ntkpdf.hessian``.

.. list-table:: Defined providers and collections
   :header-rows: 1
   :widths: 52 48

   * - Resource
     - Result / collection scope
   * - ``compute_ntk_decomposition_by_replica_at_epoch``
     - One-replica decomposition; normally called by the ensemble helper with explicit arrays.
   * - ``compute_ntk_decomposition_ensemble``
     - ``EvolutionOperator`` at one epoch.
   * - ``evolution_operator``
     - Alias provider for that ensemble result.
   * - ``feature_columns_at_epoch``
     - ``FeatureColumns`` containing only q and cuts.
   * - ``feature_grids_at_epoch``
     - Per-rank feature plotting grids.
   * - ``h_val_grid``
     - Streaming feature-eigenvalue trajectory.
   * - ``h_val_grids_by_fit``
     - Collect ``h_val_grid`` over ``fits``.
   * - ``frobenius_norm_grid``
     - Kernel-norm trajectory.
   * - ``frobenius_norm_grids_by_fit``
     - Collect ``frobenius_norm_grid`` over ``fits``.
   * - ``loss_hessian_at_epoch``
     - Per-replica parameter-space Hessians.
   * - ``loss_hessian_eigenvalues_at_epoch``
     - Descending Hessian spectra.

Figure providers
----------------

These are defined in ``ntkpdf.plotting.pdfplots_providers``:

* ``plot_pdfs_at_initialisation``
* ``compare_fits_at_init``
* ``plot_features_at_epoch``
* ``plot_features_at_epoch_single_replica``
* ``plot_pdfs_at_epoch``
* ``plot_pdfs_all_epochs``

These are defined in ``ntkpdf.plotting.loss_providers``:

* ``plot_losses_all_replicas``
* ``plot_losses_ensemble``

These are defined in ``ntkpdf.plotting.feature_plots``:

* ``plot_feature_eigvals_by_rank``
* ``plot_feature_eigvals_by_fit``
* ``plot_feature_eigvals_replica_by_fit``
* ``plot_eigvals_replica_by_fit``
* ``plot_feature_eigvals_rank``
* ``plot_feature_eigvals_replica_rank``
* ``plot_eigvals_rank``
* ``plot_eigvals_replica_rank``
* ``plot_frobenius_norm``
* ``plot_frobenius_norm_replica``

The ``*_rank`` wrappers consume a single fit-anchored grid. The ``*_by_fit``
and ``*_by_rank`` comparison wrappers consume collections over fits. Prefer
single-grid wrappers inside the bundled report's presentation loops to avoid
request-site collections of large trajectories.

Config productions
------------------

Request the left-hand name, not the Python method's ``produce_`` prefix.
The dependencies shown are the method's argument names; defaults are in
:doc:`application`.

.. list-table:: Productions defined by ``ntkpdf.config.ntkConfig``
   :header-rows: 1
   :widths: 28 42 30

   * - Resource
     - Dependencies
     - Purpose
   * - ``fit_kind``
     - ``fit``
     - Select the reader.
   * - ``kwargs``
     - ``fit_kind``, ``name``
     - Validate the NTK selector and return its modifiers.
   * - ``wreplica_paths``
     - ``fit``
     - Numerically ordered weight-replica paths.
   * - ``replica_paths``
     - ``fit``
     - n3fit ``nnfit/`` paths.
   * - ``replicas``
     - ``fit``
     - n3fit replica namespace list.
   * - ``nreplicas``
     - ``fit``, ``replica_ids``, ``custom_nreplicas``
     - Replica count, with overrides.
   * - ``fit_weights``
     - ``fit``, ``epoch``, ``replica_ids``
     - Load flat parameters; scalar epoch or one epoch per selected replica.
   * - ``stored_epochs``
     - ``fit_kind``
     - Reader's saved-epoch list.
   * - ``epochs``
     - ``stored_epochs``, ``custom_epochs``
     - Optional trajectory-epoch override.
   * - ``learning_rate``
     - ``fit_kind``
     - Reader's constant learning rate.
   * - ``pdf_ensemble``
     - ``fit_kind``, ``nreplicas``, ``replica_ids``
     - Shared batched evaluator.
   * - ``init_grids_by_name``
     - ``pdf_ensemble``, ``model_selectors``
     - Initial plotting-grid mapping.
   * - ``pdf_at_init``
     - ``pdf_ensemble``, ``model_selector``
     - Reconstructed initial ensemble PDFs.
   * - ``pdf_at_epoch``
     - ``pdf_ensemble``, ``fit_weights``, ``model_selector``
     - Snapshot ensemble PDFs.
   * - ``pdf_grid_at_epoch``
     - ``pdf_at_epoch``
     - Snapshot ``XPlottingGrid``.
   * - ``fakepdf``
     - ``fit_kind``
     - Parsed underlying-law PDF.
   * - ``fakepdf_grid``
     - ``fit_kind``, ``show_fakepdf``
     - Optional underlying-law grid; ``None`` when disabled.
   * - ``ntk_fast_kernel_arrays``
     - ``fit_kind``, ``padding``, ``flavour_basis``
     - FK DataFrame on the common grid.
   * - ``training_data``
     - ``fit_kind``, ``ntk_fast_kernel_arrays``
     - Reader's ``TrainValData``.

The class also defines ``parse_model_selectors`` for a supplied global list.
Inherited parsers/productions (for example ``fit`` and ``replicas_path``) come
from validphys and colibri, not this catalogue. ``replicas_path`` is the
colibri singular directory resource; it is distinct from ntkpdf's n3fit-only
``replica_paths`` list.
