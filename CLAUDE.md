# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`ntkpdf` is a small analysis app for studying **already-trained** NNPDF/n3fit PDF fits
through the lens of the **Neural Tangent Kernel (NTK)**. It does not run fits; it loads
the weights a fit saved to disk (at multiple training epochs), rebuilds the n3fit PDF
model in memory, and produces eigenvalue spectra, PDF grids, and comparison reports.

It is a thin **reportengine** layer stacked on top of two upstream packages:

- **validphys / n3fit** (NNPDF) — data/theory loading, the PDF model (`generate_pdf_model`,
  `N3PDF`), plotting grids, and the base `App`/`Config` classes.
- **colibri** — provides `colibriConfig`, the `Environment`, the `NTKStats` container, and
  the `colibri.ntk.*` providers (eigenvalues, eigenvectors, NTK computation, plots).

When working here you will constantly need the upstream conventions. Read these companion
files; this repo assumes their patterns:

- `<nnpdf>/CLAUDE.md` — validphys/n3fit + reportengine.
- `<colibri>/CLAUDE.md` — colibri's reportengine setup, JAX, NTK code.

Both are installed as dependencies in the active environment, so **discover their checkout
locations from the installed packages** rather than assuming a path (the checkout dir differs
per machine):

```bash
# <colibri> repo root: the CLAUDE.md is one level above the package
python -c "import colibri, os; print(os.path.dirname(os.path.dirname(colibri.__file__)))"
# <nnpdf> repo root: the package lives under src/, so the root is three levels up
python -c "import validphys, os; print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(validphys.__file__)))))"
```

Run these with the project's environment Python. grep those checkouts directly when a
provider name or symbol isn't defined in this repo. If a package isn't importable, ask the
user to pin down its location.

## Reportengine: the one thing you must internalise

This app is **declarative**. You almost never call functions directly. Instead:

- Every public function in the modules listed in `ntk_providers` (`ntkpdf/app.py`) is a
  **provider**. Its **parameter names are its dependencies** — reportengine resolves each
  parameter either to another provider's output or to a runcard key.
- `ntkConfig.produce_*` / `parse_*` methods (`ntkpdf/config.py`) are also providers; e.g.
  `produce_metamodels` is what gets called when something depends on `metamodels`.
- **Renaming a parameter or function silently rewires the dependency graph.** To understand
  what runs, trace parameter names backwards from the action in the runcard.
- To add a feature: add a function to an existing provider module (or a new module added to
  `ntk_providers`). It then becomes available app-wide by name.

Two equivalent entry points:

- **CLI**: `ntkpdf runcard.yaml` runs a report from a YAML runcard.
- **API**: `from ntkpdf.api import API; API.<provider_name>(**leaf_kwargs)` calls any
  provider directly (best for notebooks/debugging). `api.py` is deliberately a separate
  module so it can pick its own Matplotlib backend without touching `ntkpdf.app`.

### Reportengine resolution rules you must respect

These are non-obvious and each one has bitten this codebase. Verify in the reportengine
source (`reportengine/src/reportengine/{resourcebuilder,configparser,dag}.py`) before
relying on caching/memory behaviour.

- **`parse_` vs `produce_`.** `parse_<key>` runs when the *user supplies* `<key>` (runcard/API
  kwarg); `produce_<key>` *computes* it from dependencies. A key with only a `produce_` rule
  **cannot be supplied** as input — doing so raises *"conflicts with a production rule and no
  parser is defined."* **A provider/parameter name that collides with an existing `produce_`
  resource resolves to that production, not your arg** (e.g. `replicas` → `produce_replicas`
  returns an `NSList` and fails a `list` type-check). Also avoid `replica_ids` — it feeds
  `produce_nreplicas`/`produce_fit_weights`. Pick a fresh name (e.g. `replica_indices`).
- **The unit of caching is the DAG node `(provider, namespace)`.** Reportengine *does*
  memoize — but per node, keyed by the namespace it resolves in (`dag.py` dedups nodes;
  `resourcebuilder.set_result` stores each once). So a quantity that resolves at **one
  shallow namespace** is a single node → computed once → shared by every leaf below. The
  *same* function resolved at a **different namespace per leaf** is a *different* node → its
  own cache slot → computed and **retained N times**. "Recomputed per leaf" therefore never
  means caching failed; it means the quantity became N nodes instead of 1. Two things turn
  one per-fit constant into N nodes: (a) `from_:` taint, (b) `collect` re-making (below). The
  fix for both: **collapse the heavy work to one shallow node and do the rank/replica/epoch
  fan-out in plain Python** inside the provider (invisible to namespace resolution) — e.g.
  `_grouped_trajectory_plots` (builds the grid once per fit/name, loops ranks/scales/replicas
  in Python) and `produce_init_grids_by_name`.
- **`from_:` taints resolution.** Resolving any `from_:` sets a taint flag that pins the
  dependent production to the *deepest* namespace in scope — so inside a `{@with …@}` loop it
  becomes a **distinct node per leaf** (recomputed and separately retained per iteration; the
  per-node cache can't help because they aren't the same node). The value is unchanged; only
  the node multiplies. Keep `from_` off hot paths: inject the value **directly** instead. The
  singular `fit` is injected directly in `ntkanalysis.complete_mapping`
  (`autosettings['fit'] = currentmap`, *not* `fit: {from_: current}`); the data/theory chain
  (`dataset_inputs`, `theory`, `theoryid`) is injected the same way (read once from
  `fit.as_input()`), so `data → ntk_fast_kernel_arrays → fitting_covmat_eigensystem` resolve
  once instead of reloading all commondata/FK per leaf (an OOM — see case study below).
- **`collect` is re-made at every request site.** A `collect` nested inside presentation
  loops recomputes its whole subtree per iteration. Build the heavy thing **once** (a
  `produce_` anchored at a shallow namespace, depending only on the things it truly varies
  with) and have the presentation index into it. See `produce_init_grids_by_name` (built once
  per fit/nreplicas, indexed by selector name) vs the old per-selector collect.
- **Nothing is freed.** `produce_` results are evaluated eagerly at *graph-build* time and
  cached; the executor never frees node results. So collecting an expensive/large quantity
  over a big dimension (e.g. 1000 epochs) keeps every copy alive → OOM. To bound memory,
  **stream in a single provider loop** instead of collecting: `loss_function_grid`
  (`data_theory.py`) loads/predicts/discards one epoch at a time, and `h_val_grid`
  (`ntkdecomposition.py`) loops epochs calling colibri's `eigenvectors_ensemble_at_epoch`
  directly and `del`-ing each epoch's `(nreplicas, n, n)` eigenvectors before the next. The
  give-away that you need this: a per-epoch object is large *and* you're collecting it.
- **`@check`/`make_argcheck` fire only on providers, not `produce_` methods.** Validate
  `produce_` inputs inline and raise `reportengine.configparser.ConfigError`.

**Case study — the `ntkanalysis` data/metric-chain OOM.** The full report (deeply nested
`{@with@}` loops × `Customreplicas`/`Selectors`/`Epochspecs`/`Rankspecs`/`PDFscalespecs`/
`Replicaspecs`) OOM-killed at >15 GB. Cause was the rules above compounding: the *per-fit
constant* data→FK→covmat→metric chain became a node per presentation leaf (~80+), each heavy
object recomputed **and retained** (nothing freed). Localise with a per-node RSS print in
`resourcebuilder.execute_sequential`, or a watchdog thread dumping `faulthandler` all-thread
stacks + `gc` type counts at RSS thresholds — these named each culprit precisely (commondata
reload → millions of ruamel YAML objects; `fitting_covmat_eigensystem` ~0.6 GB×13; the
`(Ntr×Ntr)` `cinv_diagonal_basis_train_val`). **Do not guess from a mental model — the leak
was *not* where intuition pointed (eigenvectors / the keras MetaModel rebuild); it was the
validphys data chain.** Fixes (each removed GBs, all verified bit-identical): (1)
`fitting_covmat_eigensystem` made a `produce_` (loads the eigensystem CSV once; the FK-index
relabel — which needs a provider — moved to its one consumer `fk_diagonal_basis_train_val`);
(2) `dataset_inputs`/`theory`/`theoryid` injected directly (taint break, above). The residual
`m_matrix` chain (`fk/cinv_diagonal_basis_train_val`, tainted via
`read_replica_pseudodata → context_index`) needs the same build-once-and-index treatment.

## Commands

```bash
# Install (Poetry; nnpdf + colibri must already be importable in the env)
poetry install

# Run a report from a runcard (output dir defaults to the runcard's stem)
ntkpdf path/to/runcard.yaml
ntkpdf path/to/runcard.yaml -o output_dir

# Generate a full NTK comparison report for one fit via the bundled template.
# Fills ntkpdf/analysistemplate/analysis.yaml with the chosen fit, then runs it.
ntkanalysis <fit_id> --title "..." --author "..." --keywords k1 k2
ntkanalysis -i        # interactive: prompts for fit/title/author/keywords

# Useful ntkanalysis flags:
#   --epochs 100 500 1000   PDF-at-epoch snapshot pages (-> Epochspecs namespace)
#   --show-fakepdf          overlay the closure-test underlying law on those plots
#   --thcovmat_if_present   use the theory covmat for the statistical estimators
#   --formats pdf           figure formats (default png+pdf; this avoids the PNGs)
```

There is no test suite in the repo yet (the `test` extra wires up `pytest`/`hypothesis`).
Do not invent test commands.

`ntkpdf/scripts/260526-ac-02-ntk-sgd/` is a captured example run (input runcard, figures,
tables, `meta.yaml`) — a good reference for what a completed report looks like.

`ntkpdf/examples/*.ipynb` are the best **API usage** references (they call providers directly,
the fastest way to prototype/debug a new one): `h_val_grid.ipynb` (build + plot an `hValGrid`)
and `plot_pdfs.ipynb` (PDF grids, fakepdf, `plot_grids`). The recurring `common_dict` there
(`dataset_inputs`/`use_cuts`/`theory`/`theoryid` all `from_: fit`) is the minimal data/theory
context the FK chain needs.

## Critical environment detail

`ntkpdf/__init__.py` sets `KERAS_BACKEND=jax` (and silences TF/validphys/n3fit logs)
**before** any keras/tensorflow-backed module is imported. Anything that pulls in the n3fit
model or `colibri.ntk` must go through package import first — do not import those modules in
a way that bypasses `ntkpdf/__init__.py`, or the backend will default to TensorFlow.

## Architecture

### Loading a fit from disk (`ntkpdf/config.py`)

The `produce_*` methods know the on-disk layout of an n3fit/colibri fit and turn it into
in-memory objects. Key files inside a fit directory:

- `fit_replicas/replica_<r>/parameters/params_<epoch>.npz` — raw weight vectors per epoch
  (`produce_fit_weights`). This is the NTK-specific artifact; "epochs" means these snapshots.
- `nnfit/replica_<r>/<fitname>.json` — preprocessing exponents, `best_epoch`
  (`produce_fit_preprocessing`, `produce_best_epochs`).
- `nnfit/replica_<r>/chi2exps.log` — per-epoch train/val losses (`log_losses_replica`).

The model pipeline: `produce_model_info` (reads the fit's runcard architecture) →
`produce_base_metamodels` (builds a fresh `generate_pdf_model` with random init **once** per
fit/nreplicas, and snapshots its preprocessing into `_ntk_base_preprocessing`) →
`produce_metamodels` (thin wrapper that applies a `model_selector` via the module-level
`_apply_model_selector`, which only overwrites the cheap prefactor weights — `no_prefactors`
sets alpha=1/beta=0 — so selectors share one expensive build) → `produce_model_at_epoch`
(loads a saved epoch's weights via `set_epoch_weights`/`_unpack_params`) →
`produce_pdf_at_epoch` (predicts on the x-grid and wraps the result in `NTKStats`, via
`predict_pdf_grid`). `produce_init_grids_by_name` builds the per-selector PDF grids at
initialisation in one shared production.

The base model is mutated in place and reused; this is safe under reportengine's sequential
execution because each grid is materialised to NumPy before the next mutation.

### The NTK model interface (`ntkpdf/model.py`)

**There are two model-build paths and they are not the same object.** The one above
(`produce_base_metamodels`) builds a *multi-replica* keras model for PDF **grids/plots**.
The NTK path needs something different: a *single-replica* model exposed as a **pure JAX
function of the flat weight vector**, so `jax.jacfwd` can differentiate it. That is
`NTKPDFN3Fit.grid_values_func(xgrid, exclude_layers, remove_prefactors, grad_layers)`,
consumed by `colibri.ntk.ntkutils.compute_ntk` and by `ntkpdf/hessian.py`. The kwargs come
from the selector frozensets in `config.py` (`FULL_MODEL`/`NO_PREFACTORS`/`NN`/`NO_SMR`,
and `layer_kwargs(...)` for the layer-restricted kernel). Both paths force keras
`floatx=float64` so they agree bit-for-bit.

This module replaces **colibri-n3fit**, which ntkpdf used to depend on. That package exists
to drive *colibri's fitting machinery* with an n3fit network; ntkpdf never runs a fit, so
only the model class was needed. Do not reintroduce the dependency.

**The monkeypatch — the non-obvious part.** colibri's NTK providers do not *receive* a
model; they *load* one themselves, by fit **name**, from inside `ThreadPoolExecutor` workers
(`colibri/ntk/ntkutils.py`, `colibri/ntk/ntk.py`), via `colibri.utils.get_pdf_model`. Only a
string crosses into the worker, so there is no argument through which ntkpdf can inject its
model. `ntkpdf.model.install()` therefore rebinds `get_pdf_model` over colibri's. Two traps:

- Those modules bind the name with `from colibri.utils import get_pdf_model`, so each holds
  **its own reference** — patching `colibri.utils` alone reaches none of them. Every module
  in `_PATCH_TARGETS` must be rebound. (`ntkpdf/hessian.py` imports it *inside* the function,
  so it resolves at call time and needs no special handling.)
- `install()` is called at **module level in `ntkpdf/app.py`** — not in `ntkpdf/__init__.py`
  (which would drag jax/keras into every bare `import ntkpdf`) and not in `NTKApp.__init__`
  (which the API path never calls). Both entry points import `app.py`: the CLI via `NTKApp`,
  the API because `ntkpdf/api.py` imports `ntk_providers` from it. A notebook that calls
  colibri's NTK providers **directly**, without going through `ntkpdf.api`, must call
  `install()` itself — see `examples/layer_ntk.ipynb`.

Unlike colibri's loader, ours takes no `pdf_model.pkl`: it builds from the fit's own
`filter.yml` (`_model_init_args`, the same runcard keys `produce_model_info` reads) and loads
the replica's fitted preprocessing exponents from `nnfit/replica_<r>/<fit>.json`. The pkl
that n3fit's `model_trainer.py` writes for colibri is now unused by ntkpdf.

The patch is invisible from colibri's side: if colibri grows a *new* `from colibri.utils
import get_pdf_model` site, that site silently keeps the old loader and will try to import
`colibri_n3fit`. `install()` raises if a target loses the symbol, but cannot see a new one —
so after bumping colibri, re-run a report with `colibri_n3fit` **uninstalled**. That is the
only test that catches it.

### NTKStats and the diagonal-covmat convention

`NTKStats` (from `colibri.ntk.ntkutils`) is the central container: a stack of per-replica
pandas DataFrames presented as a `(n_replicas, …, …)` array, with a validphys `Stats`
interface and an index-aware `@` (matmul aligns inner indices, so FK/Cinv chains must share
labels). `STAT_TYPES["NKT"] = NTKStats` is registered in `vpinterface.py`.

`ntkpdf/data_theory.py` builds the FK tables and covmat machinery used to reconstruct the
fit's training metric. Two fit conventions are handled throughout (gated on
`fit.as_input()["diagonal_basis"]`, default `True`):

- **diagonal_basis=True** (modern): pseudodata and FK are rotated into the diagonalised
  covariance eigenbasis; rows are labelled `"eigenmode k"` and sorted ascending by `k`.
  `fitting_covmat_eigensystem` reads the eigensystem CSV written by `vp-setupfit`.
- **diagonal_basis=False** (legacy): everything stays in the `(group, dataset, id)`
  MultiIndex and is reindexed against `groups_index`.

The train/val pipeline (`train_val_masks` → `fk_diagonal_basis_train_val` /
`cinv_diagonal_basis_train_val` → `m_matrix_train_val`) reproduces the per-replica training
metric `M = (V·FK)ᵀ diag(1/Λ) (V·FK)` exactly as used during the fit.

### Bases and rotations (`ntkpdf/bases.py`, `ntkpdf/utils.py`)

Two orderings coexist and are a frequent source of bugs:

- **n3fit model output order** = `EVOL_LIST` (`photon, sigma, g, V, T3, …`).
- **validphys "evolution" basis order** (`sigma, V, T3, …, g, photon`).
- **flavour basis** = `EXPORT_LABELS` (`tbar … gluon … t, photon`).

`bases.py` builds `R_FROM_FLAV_TO_EVOL` / `R_FROM_EVOL_TO_FLAV` and `EVOL_IDX_REORDER`;
`utils.py` has `flav_to_evol_matrix` and the `EVOL_INDEX` / `FLAVOUR_INDEX` MultiIndices on
the shared `XGRID`. When you index or einsum across flavours, confirm which ordering the
array is in.

### Exposing the model to validphys (`ntkpdf/vpinterface.py`)

`NTKPDF` / `NTKLHAPDF` wrap an `NTKStats` so it quacks like a validphys `PDF` / `LHAPDFSet`
(no LHAPDF info file — a fake `_info` is synthesised). This lets standard validphys actions
(`xplotting_grid`, chi², etc.) operate on the in-memory model. Output is always in the
evolution basis; `grid_values` rotates to flavour basis on request.

### NTK decomposition & feature (h) values (`ntkpdf/ntkdecomposition.py`)

Turns the per-replica NTK eigensystem (from colibri) plus the data metric `M` (from
`m_matrix_train_val`) into the evolution operator and the **feature ("h") values**:

- `compute_ntk_decomposition_by_replica_at_epoch(eigenvalues, eigenvectors, M, tol)` — one
  replica: keep eigenmodes above `tol`, form `H_perp = √Λ (Zᵀ M Z) √Λ`, eigendecompose it →
  an `NTKDecomposition` (`P_parallel`/`P_perp`/`Q`/`Qinv`/`h`/…).
- `compute_ntk_decomposition_ensemble(...)` — stacks per-replica results into an
  `EvolutionOperator` (NTKStats `q`/`qinv`/`h`/…) at one epoch; `EvolutionOperator` can evolve
  a PDF (`u`, `v`, `__call__(t, data, f0)`, `plotting_grid(...)`).
- `hValGrid` subclasses colibri's `EigenvalueGrid` (only the legend label changes `λ → h`), so
  it satisfies the `NTKGrid` interface and plugs straight into colibri's NTK plot providers.
  `h_val_grid` builds it over epochs; `feature_plots.py` wraps colibri's `ntk_plot_provider`
  (`plot_feature_eigvals_by_fit` / `_by_rank`).

Two gotchas specific to this module:

- **Flavour-label mismatch (Z vs M).** colibri eigenvectors are labelled with `FK_FLAVOURS`
  names (`photon, singlet, g, …`); `M`/FK use n3fit `EVOL_LIST` (`photon, sigma, gluon, …`).
  The orderings are **positionally identical** — only the label strings differ — so a pandas
  label-aligned `@` raises *"matrices are not aligned"*. The h path therefore uses **numpy
  positional** matmul (`np.asarray(...)`), which is also faster.
- **Streaming, not collecting** (see the "Nothing is freed" rule): `h_val_grid` only needs the
  tiny `h` vector but the eigenvectors are `(nreplicas, n, n)`, so it streams them per epoch.

### Plotting (`ntkpdf/plotting/`)

There is **one shared, styled drawing layer**; do not re-implement `ax.plot`/styling in a
provider.

- `style.py` — the matplotlib style (`setup()` / `NTKPDF_STYLE`, also wired as
  `NTKApp.default_style` for the CLI), `make_figure` (a bare `Figure`, never pyplot-managed,
  to avoid the "More than 20 figures" warning under reportengine), the legend handler
  (`HandlerSpec` + `ComposedHandler`), and the **container-agnostic draw primitives**
  `draw_bounds` / `draw_line` / `select_draw`. Each draw fn takes `(ax, stats, xgrid, …)`
  where `stats` is an `NTKStats`/`Stats` with replicas on axis 0, and **returns** a
  `HandlerSpec` (the caller collects it for the legend).
- `pdfplots_utils.plot_grids` orchestrates per-flavour figures from a list of
  `XPlottingGrid`s using those primitives; `_resolve_flavours` + `DEFAULT_FLAVOURS` /
  `DEFAULT_YLABELS` (in `pdfplots_providers.py`) keep flavour/ylabel defaults paired so they
  can't desync. `plot_grids` normalises by whole-grid division, so grids plotted together
  must share the same flavour axis (build them all on the same `flavours` list).
- Style note: `colibri.ntk.plotntk` sets a global font via module-level `rc()` on import,
  applied *after* `init_style` on the CLI — so don't reuse colibri's draw layer for ntkpdf
  plots if you want the NTKPDF serif style.
- Figure formats default to **png + pdf** (reportengine `--formats` default in its `app.py`);
  pass `--formats pdf` to avoid PNGs.
- **Reusing colibri's NTK plot providers.** Any `NTKGrid` (`EigenvalueGrid`, `EigenvectorGrid`,
  `hValGrid`) works with colibri's generic `ntk_plot_provider` and the `plot_eigvals_*`
  wrappers; pass the grids-by-fit collect (e.g. `h_val_grids_by_fit`). Legend labels come from
  the grid's `get_plotting_label`.
- **`frame_center` caveat.** colibri's `ntk_plot_provider` auto-scales y via validphys
  `frame_center`, which percentiles values in the central 10% of the x-range. With a
  sparse/uneven x-axis (a few `custom_epochs` whose midpoint gap has no sample) that window is
  empty → `IndexError`. Dense axes (the ~1000-epoch eigenvalue grid) never hit it. This is
  guarded in `colibri/ntk/plotntk.py` (falls back to `ax.autoscale_view()`). Note some fixes
  like this live in **colibri**, which is also yours — patch it there, not in ntkpdf.

### Reports and templates

`ntkpdf/analysistemplate/` holds the driver runcard (`analysis.yaml`) and the markdown
templates consumed by `ntkanalysis`. The top-level `report.md` links **sub-reports**
(`initialisation_report`, `loss_report`, `pdf_epochs_report`, `ntk_report`), each declared in
`analysis.yaml` as `name_report: {meta: Null, template: foo.md}` and referenced as
`{@name_report report@}`. Inside a template, `{@ … @}` blocks call providers and `{@with
Namespace@}…{@endwith@}` iterates a runcard namespace list (`Customreplicas`, `Selectors`,
`PDFscalespecs`, `Epochspecs`, …) — each element becomes a sub-namespace its providers see.

Two epoch knobs that are easy to confuse:

- **`--epochs E1 E2 …`** (CLI) → injected as the `Epochspecs` namespace; drives the
  **PDFs-at-epochs** snapshot page (one figure per chosen epoch, `plot_pdfs_at_epoch`).
- **`custom_epochs: [...]`** (runcard) overrides `produce_epochs`, limiting which epochs the
  **trajectory** grids (`loss_function_grid`, `h_val_grid`) iterate — useful for speed, but too
  few/uneven values trip `frame_center` (see Plotting), and a trajectory over 3–4 points is
  barely a curve.

**Optional closure overlay:** `--show-fakepdf` overlays the closure-test underlying-law PDF
(`closuretest: fakepdf`) on the PDF-at-epoch plots. It is gated through
`produce_fakepdf_grid(fit, show_fakepdf=False)`, which returns `None` when off — so the page
still works for fits that are not closure tests (a plain `fakepdf_grid` dependency would have
*forced* the lookup and errored). This "gate an optional/expensive dependency behind a flag
that returns None" is the general pattern for optional overlays.

### Adding a report page / feature (worked recipe)

Use the "PDFs at epochs" page as the template (provider `plot_pdfs_at_epoch`):

1. **Data/grid provider** — a plain provider (e.g. `pdf_grid_at_epoch` in `pdfgrids.py`) or a
   `produce_` in `config.py`. Gate optional/closure-only or expensive dependencies behind a
   flag that returns `None` when off (see `produce_fakepdf_grid(fit, show_fakepdf=False)`) so
   a mere parameter dependency doesn't *force* the build/error.
2. **Plot provider** — `@figuregen` (yields one figure per iteration; lazy, good for many
   replicas/epochs) or `@figure`, in `plotting/`. Reuse `plot_grids` / the `draw_*`
   primitives; yield `(fig, suffix)` for distinct filenames. Add the module to `ntk_providers`
   (`app.py`) if it is new.
3. **Sub-report** — add `name_report: {meta: Null, template: foo.md}` to `analysis.yaml`,
   create `analysistemplate/foo.md` (`%title` + `{@with Xspecs@} … {@your_provider@} …`), and
   link it from `report.md`.
4. **Let the user choose on the CLI** — add an argument in
   `ntkanalysis.CompareFitApp.add_positional_arguments`, then inject it in `complete_mapping`
   as either a **namespace list** (one figure per element, e.g.
   `autosettings['Epochspecs'] = [{'epoch': e} for e in …]`) or a **scalar flag** (e.g.
   `autosettings['show_fakepdf'] = …`).
5. **Respect the resolution rules above**: don't collide parameter names with `produce_`
   resources; if the page builds `data`/FK tables, expose top-level `theoryid`
   (`theory: {from_: fit}` + `theoryid: {from_: theory}`); prefer faceting via namespaces +
   indexing a shared production over collecting heavy work inside loops.

## Conventions

- Module docstrings start with the dotted module name; provider modules mark "meant to be
  used as providers" vs internal helpers with `_` prefixes and section comment banners.
- Validphys/colibri symbols are imported and reused rather than reimplemented — prefer
  finding the upstream provider/util over writing a new one.
- `XGRID` (the x-grid) comes from `colibri.constants`; keep all grids on it so indices align.
