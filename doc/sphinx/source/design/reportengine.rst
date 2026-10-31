Reportengine: resolution and memory
=======================================

Providers and productions
-------------------------

``ntkpdf.app.ntk_providers`` combines validphys providers, colibri's NTK
providers and eight ntkpdf provider modules. A provider's parameter names
declare the resources it needs. For example,
``m_matrix_train_val(fk_train_val, cinv_train_val)`` asks reportengine to resolve
the FK and inverse-covariance pairs before it runs. Renaming a parameter changes
the graph, not only the local Python interface.

``ntkConfig`` inherits ``colibriConfig``. It supplies input parsers and
productions in addition to the provider modules:

.. list-table:: Three resolution paths
   :header-rows: 1
   :widths: 24 38 38

   * - Definition
     - When used
     - Example
   * - Module provider
     - Scheduled as a graph node after its dependencies are resolved.
     - ``ntkpdf.data_theory.m_matrix_train_val``
   * - ``parse_<key>``
     - Parses a supplied input value for ``key``.
     - ``ntkConfig.parse_model_selectors``
   * - ``produce_<key>``
     - Computes an absent input from config-resolvable dependencies during graph construction.
     - ``ntkConfig.produce_training_data``

Productions are eager. They cannot generally depend on a deferred module
provider. Keep that boundary in mind when moving work from one path to another.
``produce_training_data`` can consume ``ntk_fast_kernel_arrays`` because that
resource is itself a production.

If a key has a production but no parser, supplying it as a raw runcard/API
input normally raises “conflicts with a production rule and no parser is
defined”. Config resolution precedes ordinary provider lookup. Avoid a new
parameter name that accidentally names an existing production. For instance,
the loss plotting option is ``replica_indices``, not ``replicas`` (an n3fit
namespace resource) or ``replica_ids`` (used by weight loading and replica count).

Namespaces and template loops
-----------------------------

A namespace is an input/result scope layered over its parent. A report loop
expands requests into child scopes; each child sees its local values and the
values inherited from its parents. This runcard fragment supplies two scopes:

.. code-block:: yaml

   Epochspecs:
     - epoch: 500
     - epoch: 1000

The corresponding reportengine template fragment is:

.. code-block:: text

   {@with Epochspecs@}
   ## Epoch {@epoch@}
   {@plot_pdfs_at_epoch@}
   {@endwith@}

This requests a figure provider for each selected epoch. Nested ``with`` blocks
multiply the request sites. Templates are reportengine Markdown, distinct from
the reStructuredText used for this documentation.

Where sharing succeeds or fails
-------------------------------

The graph deduplicates a full ``CallSpec``: function, argument names, result
name and resolved namespace. It does not cache solely by function or by the
numerical values of its inputs. ``ResourceBuilder._make_callspec`` places a
provider according to the deepest dependency scope and reuses an existing
identical call specification.

.. code-block:: text

   fit scope:       pdf_ensemble, training_data, M
                        |               |
   presentation:    selector / rank / scale / replica
                        `--- small views or figures

This is why ntkpdf builds ``pdf_ensemble`` independently of presentation
selectors, and why ``init_grids_by_name`` computes all initialisation selectors
in a Python loop. The presentation layer indexes the mapping. Sharing depends
on keeping the dependencies in a shallow scope; declaring a production alone
does not guarantee one evaluation for the entire report.

``from_`` is not a neutral alias for caching. ``Config.parse_from_`` sets
``_tainted`` and can write the resolved value into the current scope; its
mapping branch returns scope index zero. ``Config._resolve_key`` also resets
the taint flag immediately before invoking a production and, if the invocation
sets it, forces that production's placement to index zero. Dependencies that
were placed locally can keep their consumers local too. In a nested report,
this can turn a per-fit quantity into one cached quantity per leaf. It is not
accurate to treat every ``from_`` anywhere in a runcard as tainting every
production: the effect follows the actual resolution path and scope writes.

``CompareFitApp.complete_mapping`` therefore injects the singular ``fit``
directly. ``N3Fit.report_settings`` also supplies direct top-level
``dataset_inputs``, ``theory`` and ``theoryid``. The current FK production goes
further: it asks ``fit_kind.data()`` for the fit's own data, outside the report's
floating data namespaces. The training metric should vary with the fit, not
with a page's scale or rank group.

Collections need the same care. ``ResourceBuilder._process_requirement``
special-cases ``collect`` before looking for an existing config result.
``_make_collect`` rebuilds its request-site collector and returns scope index
zero. It does this at every request site. Its element providers may still share
an identical shallow ``CallSpec``; it is too strong to say that every collect
unconditionally recomputes every descendant. What is unsafe is assuming the
collector itself will be hoisted out of presentation loops. The leaf wrappers
in ``ntkpdf.plotting.feature_plots`` take ``h_val_grid`` or ``eigenvalue_grid``
directly rather than a per-fit collect for precisely this reason.

Retention and epoch streaming
-----------------------------

``ResourceExecutor.set_result`` stores each result in its namespace. The
sequential executor has no last-consumer eviction step. Config productions are
also written into namespaces during graph construction. Results remain
referenced for the run; making one large node per epoch retains one large result
per epoch.

Use a plain Python loop when only a small reduction of each epoch is needed:

* ``loss_function_grid`` loads parameters, predicts one PDF grid and retains
  only the scalar training/validation losses for each replica and epoch.
* ``h_val_grid`` calls colibri's ``eigenvectors_ensemble_at_epoch`` directly,
  reduces it to ``h`` values, and deletes the local eigenvector array. Upstream's
  bounded one-entry eigenvector cache may retain a recent result, but it does
  not retain all epochs.

This bounds transient large arrays to a small number of epochs. It does not
remove the cost of the shared FK, metric or requested snapshot results.
Likewise, ``@figuregen`` is not a promise of bounded report memory:
reportengine's figure wrapper materialises the generator before saving.

Verification points for maintainers
---------------------------------------

The rules above follow the installed reportengine implementation:

.. list-table:: Source locations to revisit after upgrades
   :header-rows: 1
   :widths: 40 60

   * - Source
     - Behaviour
   * - ``configparser.Config._resolve_key`` and ``parse_from_``
     - Parser/production precedence, eager evaluation, placement and taint.
   * - ``resourcebuilder.ResourceBuilder._process_requirement``, ``_make_callspec``, ``_make_collect``
     - Provider lookup, dependency placement, node reuse and request-site collectors.
   * - ``dag.DAG.add_or_update_node``
     - Deduplication by node value.
   * - ``resourcebuilder.ResourceExecutor.execute_sequential`` and ``set_result``
     - Execution and retained namespace results.
   * - ``report._process_template_text`` and ``ResourceBuilder._make_collect_targets``
     - Nested template targets expanded into namespace requests.

Provider checks run in ``_make_callspec``. Production methods are called by the
config parser instead, so validate them inline and raise ``ConfigError``, as
``produce_kwargs`` does.
