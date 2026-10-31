Data, metrics and NTK analysis
==================================

One grid and one ordering
-------------------------

``colibri.constants.XGRID`` is the common x-grid. The default FK builder,
``ntkpdf.data_theory._build_ntk_fast_kernel_arrays``, pads the DIS flavour axis
to 14 components, maps FK x-points onto ``XGRID`` with ``closest_indices``, and
flattens flavour first, then x. Rows are ``(dataset, id)``. Columns are
``ntkpdf.utils.EVOL_INDEX``. The builder rejects multiple FK specifications per
dataset; it uses n3fit's ``DIS`` layer helpers, not a general hadronic bilinear
convolution interface.

For a flattened PDF grid of length ``D = 14 * len(XGRID)``, the prediction is
``FK @ f``. Here ``D`` is a PDF-grid dimension, **not** the number ``P`` of
trainable model parameters.

Training ingredients belong to the reader
---------------------------------------------

``ntkConfig.produce_ntk_fast_kernel_arrays`` builds the FK from
``fit_kind.data()``. ``produce_training_data`` asks the reader to transform and
split it consistently with the fitted loss. This prevents report presentation
namespaces from choosing a different theory or split for the metric.

``fits.common.TrainValData`` is a frozen dataclass with three fields. Each is a
``(training, validation)`` pair of ``NTKStats``:

.. list-table:: Per-replica loss ingredients
   :header-rows: 1

   * - Field
     - Exposing provider
     - Shape of ``.data`` for one subset
   * - ``y``
     - ``get_train_val_data``
     - ``(R, N, 1)``
   * - ``fk``
     - ``fk_train_val``
     - ``(R, N, D)``
   * - ``cinv``
     - ``cinv_train_val``
     - ``(R, N, N)``

``R`` is the replica count; ``N`` is the subset's data-coordinate count.
``data_idx`` labels start at 1 and are shared by these blocks. This matters
because ``NTKStats`` checks matching inner labels before matrix multiplication;
equal shapes alone are insufficient. Replica order must also match.

For n3fit, ``N3Fit.training_data`` reads saved pseudodata with the single-replica
validphys reader and reads the covariance table. It orders eigenmodes
numerically and applies the replica's saved training/validation split.
The table supplies a transformation ``R`` satisfying
:math:`R C R^T = \operatorname{diag}(\Lambda)`. More precisely, the saved
:math:`\Lambda` are correlation-matrix eigenvalues and the transformation
includes inverse standard deviations, so this is not merely an orthogonal
rotation of the raw covariance. FK becomes ``R @ FK`` and the inverse covariance
becomes ``diag(1 / Lambda)`` before restriction to the subset. Fits with
``diagonal_basis: false`` are rejected.

For colibri, ``ColibriFit.training_data`` regenerates the level-1 data and
configured covariance with ``central_covmat_index``. Its
``general_covariance_matrix`` dependency selects experimental or t0 covariance
according to ``use_t0_covmat``. It then uses ``mc_pseudodata`` for each
replica's pseudodata and split. It selects the original-data rows of FK and
inverts each covariance submatrix: ``inv(C[subset, subset])``. This is not the
same as selecting a block of ``inv(C)``. The reader assumes colibri's default
``--trval_index 0``, because that index was not recorded in the fit. This
assumption matters when there is a validation split.

The metric and reconstructed loss
---------------------------------

``m_matrix_train_val`` forms the PDF-grid-space metric for each subset:

.. math::

   M = FK^T C^{-1} FK,
   \qquad
   L(\theta) = \frac{1}{N}
       (y - FK f(\theta))^T C^{-1}(y - FK f(\theta)).

The metric is not divided by ``N``. The reconstructed scalar loss is.
Changing the data coordinates consistently in ``y``, ``FK`` and ``C`` leaves
this quadratic form unchanged. Changing which coordinates belong to the
training subset is a different operation; n3fit's eigenmode split must not be
replaced by a colibri-style data-point split.

``loss_function_grid`` evaluates this loss at every ``stored_epochs`` snapshot.
It streams parameters and PDF grids instead of collecting one large grid per
epoch. The result is a training ``LossGrid`` and either a validation ``LossGrid``
or ``None`` when there are no validation data. The loss plotting providers
handle that ``None`` explicitly.

.. important::

   In the current source, ``custom_epochs`` changes the ``epochs`` resource used
   by ``h_val_grid`` and ``frobenius_norm_grid``. It does **not** shorten
   ``loss_function_grid``, whose dependency is ``stored_epochs``. Also, it does
   not necessarily limit the upstream eigenvalue-cache computation, which
   enumerates saved parameter files separately.

Kernel eigensystems, features and evolution
-----------------------------------------------

Colibri supplies the kernel and its eigensystems. In
``colibri.ntk.ntkutils.compute_ntk``, ``jax.jacfwd`` differentiates the PDF-grid
function. With :math:`J = \partial f/\partial\theta`, its flattened contraction
is :math:`K = J J^T` in the same flavour-major ordering as FK. Colibri's
``compute_eigendecomposition`` sorts eigenvalues descending and returns
eigenvectors as columns. ``eigenvalue_grid`` supplies trajectories;
``eigenvalues_at_epoch`` selects a snapshot. ``eigenvectors_at_epoch`` supplies
labelled vectors for a snapshot. Eigenvalue caches are written per replica;
their filenames are keyed by ``name``, not the full modifier dictionary.

``ntkpdf.ntkdecomposition`` combines that eigensystem with the loss metric.
It retains modes satisfying :math:`\lambda_i/\lambda_0 > \mathrm{tol}` and
forms

.. math::

   H_\perp = \sqrt{\Lambda_K}\,Z_\perp^T M Z_\perp\,\sqrt{\Lambda_K}.

Here :math:`\Lambda_K` contains retained **kernel** eigenvalues, not the
covariance-table eigenvalues above. The eigenvalues of :math:`H_\perp` are the
feature values ``h``. With its eigenvectors ``W``, the feature columns are
:math:`Q = Z_\perp\sqrt{\Lambda_K}W`.

``compute_ntk_decomposition_by_replica_at_epoch`` returns the projectors,
feature matrices and spectra for one replica. ``compute_ntk_decomposition_ensemble``
stacks them into ``EvolutionOperator``, padding to accommodate replica-dependent
cuts. The default tolerance is ``1e-12`` for the full decomposition and
``1e-7`` for ``feature_columns_at_epoch`` and ``h_val_grid``; choose it
explicitly when comparing these paths.

For plotting, ``feature_columns_at_epoch`` retains only ``q`` and the cuts,
not the full operator's four large matrices. ``feature_grids_at_epoch`` turns
1-based rank selections into PDF-like grids. It excludes replicas without a
requested mode from ensemble bands; a rank missing from every replica is drawn
as zero with a warning. ``h_val_grid`` keeps only the small spectra while
streaming eigenvectors. ``hValGrid`` inherits colibri's ``EigenvalueGrid`` so
the existing trajectory plotting machinery can consume it.

``EvolutionOperator`` implements
:math:`f(t) = u(t) f_0 + v(t)y` for the fixed kernel/metric decomposition.
Its methods use exponential factors with ``2 * training_time`` and contain no
implicit ``1/N``. Do not treat this as a new fit or a general exact solution of
nonlinear training. Time normalisation must be chosen consistently with the
loss and update convention. The evolution notebooks pass ``epoch * learning_rate``;
that is an explicit notebook choice, not a conversion performed by the operator.
In particular, colibri's epoch-zero snapshot is already post-update.

``frobenius_norm_grid`` computes :math:`\|K\|_F = \sqrt{\sum_i\lambda_i^2}`
from cached eigenvalues, without constructing eigenvectors.

The full loss Hessian
---------------------

``ntkpdf.hessian.loss_hessian_at_epoch`` uses the same FK, inverse covariance,
data and model selector as the loss reconstruction. It applies ``jax.hessian``
to the normalised scalar loss, one replica at a time. This is a
parameter-space ``(P, P)`` object, not a PDF-grid-space kernel. The eigenvalue
provider sorts its spectrum descending and does not clip negative eigenvalues.

Its Gauss--Newton part is :math:`(2/N)J^T M J`. Its nonzero spectrum corresponds
to :math:`(2/N)h` before numerical mode truncation. The full Hessian also contains
the residual-weighted second derivatives of the model and can show negative
curvature. It is not interchangeable with ``h_val_grid``.

The current Hessian provider indexes the full training-data ensembles with
``replica_id - 1``. Use consecutive 1-based replica IDs for this path; a sparse
directory numbering is not mapped through a separate ID-to-frame table.

Bases and orderings
-------------------

There are three distinct conventions, even though two use the word “evolution”.

.. list-table:: Flavour axes
   :header-rows: 1
   :widths: 25 75

   * - Convention
     - Ordering
   * - n3fit output / FK: ``EVOL_LIST``
     - ``photon, sigma, gluon, V, V3, V8, V15, V24, V35, T3, T8, T15, T24, T35``
   * - validphys evolution basis
     - ``sigma, V, T3, V3, T8, V8, T15, V15, T24, V24, T35, V35, g, photon``
   * - Flavour basis: ``EXPORT_LABELS``
     - ``TBAR, BBAR, CBAR, SBAR, UBAR, DBAR, GLUON, D, U, S, C, B, T, PHT``

``ntkpdf.bases.EVOL_IDX_REORDER`` reorders model output into validphys basis
order. ``R_FROM_FLAV_TO_EVOL`` and its inverse use that validphys order.
``utils.flav_to_evol_matrix`` instead constructs a transformation for an explicit
requested evolution-flavour order; it translates ``sigma`` and ``gluon`` aliases.
``EVOL_INDEX`` and ``FLAVOUR_INDEX`` label flattened grids on ``XGRID``. Their
``x`` level is a 1-based grid-position index, not the floating-point x value.

Colibri eigenvectors default to FK names such as ``singlet`` and ``g`` where
ntkpdf uses ``sigma`` and ``gluon``. Their component order is the same, but their
labels are not. The reduced feature calculations deliberately use NumPy
positional products. For label-aware evolution products, pass
``evol_fl_names=list(EVOL_LIST)`` as the notebooks and analysis runcard do.
Do not silently bypass label checks unless you have established that positional
order really matches.

``ntkpdf.vpinterface.NTKPDF`` and ``NTKLHAPDF`` let validphys consume in-memory
``NTKStats`` grids. They synthesise PDF metadata rather than require an LHAPDF
info file. ``grid_values`` rotates to flavour basis for validphys requests.
These are wrappers around an existing grid, not interpolation or scale-evolution
engines: the implementation does not use requested x-values to recompute PDFs
and repeats the stored grid for requested Q entries.
