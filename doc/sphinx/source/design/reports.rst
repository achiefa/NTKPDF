Plotting and extending reports
==============================

Shared drawing primitives
-------------------------

``ntkpdf.plotting.style`` defines ``NTKPDF_STYLE``, ``make_figure``,
``draw_bounds``, ``draw_line`` and ``select_draw``. The draw functions accept an
axis, a Stats-compatible object with replicas on axis zero, and an x-grid.
They return a ``HandlerSpec`` for the caller's legend. This separates the
statistical presentation from the quantity being drawn: a loss trajectory and
a PDF curve can use the same uncertainty convention.

``draw_bounds`` supports mean or median, standard-deviation shading and 68%
interval edges. ``draw_line`` draws individual members. ``make_figure`` creates
a bare Matplotlib ``Figure``, avoiding pyplot's global figure registry when
reportengine holds many figures. ``setup`` applies the shared style on package
import; ``NTKApp.default_style`` supplies it on the CLI path too.

``pdfplots_utils.plot_grids`` orchestrates one plot per requested flavour and
can instead draw into supplied axes. Labels, plotting arguments and y-label
lengths are checked. Ratio normalisation uses a **1-based** ``normalise_to``
grid index and divides whole grids, so all grids must share the flavour axis.
For ``plot_pdfs_at_epoch``, the fitted grid is first and an enabled underlying
law is second: a ratio to that law requires ``normalise_to: 2``.

``feature_plots`` reuses colibri's generic ``ntk_plot_provider`` for kernel and
feature-eigenvalue trajectories. These wrappers use colibri's drawing interface,
not the ``style.draw_*`` signature. Signed feature columns in
``plot_features_at_epoch`` are drawn with a linear y-axis even inside a
``Rankspecs`` scope carrying ``yscale: log`` for positive spectra.

How the bundled report is assembled
-----------------------------------

``ntkpdf/analysistemplate/analysis.yaml`` is the driver. Its ``actions_``
requests ``report(main=true)`` and its top-level template is ``report.md``.
The template links sub-reports, each defined by a mapping with ``meta: Null``
and a Markdown ``template`` filename.

.. code-block:: text

   analysis.yaml + CompareFitApp.complete_mapping()
       |
       `-- report.md
             |-- pdf_init.md          (Customreplicas, Selectors, scales)
             |-- loss_all_replicas.md / loss_ensemble.md
             |-- pdf_at_epochs.md / pdf_all_epochs.md
             |-- ntk.md              (Eigenvaluesconfigs, ranks, replicas)
             `-- h_vals.md

``CompareFitApp.get_config`` reads the driver and updates it with
``complete_mapping``. That method adds metadata and a direct singular ``fit``,
then merges ``fit_kind.report_settings(currentmap)``. Finally it adds
``Epochspecs``, ``Replicaspecs``, ``show_fakepdf`` and
``use_thcovmat_if_present`` from CLI arguments. This keeps n3fit-only report
assumptions out of the generic script. Colibri disables ``Validphyssummaries``
and overrides the model selector lists in its reader.

``--epochs`` drives snapshot pages through ``Epochspecs``. It does not select
the h-value trajectory's epochs. ``custom_epochs`` overrides ``epochs`` for
providers depending on that resource; the loss grid instead uses
``stored_epochs``. Choose the namespace by the dependency actually consumed,
not by similar names in comments or notebook descriptions.

``produce_fakepdf_grid`` gates an optional dependency. With ``show_fakepdf``
false it returns ``None`` without looking up an underlying law. Declaring an
unconditional ``fakepdf`` parameter in the plotting provider would instead
force closure-test resolution even when the overlay is not wanted.

Worked recipe: add a compact loss page
--------------------------------------

The following is a development recipe, not a change made by this documentation.
It adds a page using an existing provider, so no new computation or Python
module is needed.

1. Add a sub-report mapping beside ``loss_ensemble_report`` in ``analysis.yaml``:

   .. code-block:: yaml

      compact_loss_report:
        meta: Null
        template: compact_loss.md

2. Create ``ntkpdf/analysistemplate/compact_loss.md`` with:

   .. code-block:: text

      %Ensemble loss

      {@plot_losses_ensemble@}

   Its dependencies resolve through ``loss_function_grid``. The driver already
   supplies the default mean, standard-deviation and interval settings.

3. Link it from ``report.md``:

   .. code-block:: text

      [Compact loss summary]({@compact_loss_report report@})

4. If the page needs a new observable, add a plain provider to an existing
   registered module, or add its module to ``ntk_providers``. Let it depend on
   ``training_data``/``pdf_ensemble`` rather than reopen storage-specific files.
   Use ``@figure`` for one figure or ``@figuregen`` for several; distinct
   ``(figure, suffix)`` yields follow ``plot_losses_all_replicas``. Reuse
   ``make_figure`` and ``draw_*`` for new ntkpdf-style plots.

5. For a new user selection, add a parser argument to
   ``CompareFitApp.add_positional_arguments`` and map it in ``complete_mapping``.
   A list of mappings creates a loop namespace; a Boolean gates optional work.
   Follow the ``Epochspecs`` or ``show_fakepdf`` pattern, respectively.

Before nesting the page under ranks, scales or replicas, trace where every
large dependency is placed. Reuse a shallow fit-level result and select small
views in the page. Do not introduce a collect over heavy per-epoch arrays just
to produce a small trajectory; stream those arrays in one provider instead.
