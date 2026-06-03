# ntkpdf

A tool for the **Neural Tangent Kernel (NTK)** analysis of trained
[NNPDF](https://github.com/NNPDF/nnpdf)/`n3fit` Parton Distribution Function fits.

`ntkpdf` does not run fits. It takes a fit that has already been trained, loads
the weights it saved to disk (at multiple training epochs), rebuilds the `n3fit`
PDF model in memory, and produces a report with eigenvalue spectra, PDF grids at
initialisation, training/validation loss curves, and related NTK diagnostics.

It is a thin [`reportengine`](https://github.com/NNPDF/reportengine) application
layered on top of two upstream packages:

- **validphys / n3fit** (NNPDF) — data/theory loading, the PDF model, plotting
  grids, and the base `App`/`Config` classes.
- **colibri** — `colibriConfig`, the execution `Environment`, the `NTKStats`
  container, and the `colibri.ntk.*` providers (eigenvalues, eigenvectors, NTK).

## Requirements

`ntkpdf` runs inside an environment that already provides the NNPDF stack and
colibri:

- `nnpdf` (validphys, n3fit) and `colibri`, importable in the same environment
- `reportengine`
- `jax` (the PDF model is evaluated with the **JAX** Keras backend — `ntkpdf`
  sets `KERAS_BACKEND=jax` on import)
- `h5py`, `prompt_toolkit`, plus the usual scientific Python stack
  (numpy/pandas/matplotlib)

A working LaTeX installation is needed for the figures (`text.usetex` is on).

## Installation

In an environment where `nnpdf` and `colibri` are already installed (e.g. a
conda env), install `ntkpdf` from the repository root:

```bash
pip install -e .
```

## Usage

### Generate an NTK analysis report for a fit

The driver fills a template runcard (`ntkpdf/analysistemplate/`) for the chosen
fit and produces the report:

```bash
python ntkpdf/scripts/ntkanalysis.py <FIT_ID> \
    --title "My NTK analysis" \
    --author "Your Name" \
    --keywords NTK \
    --output <output_dir>
```

Add `-i` to be prompted interactively for the missing fields, and
`--thcovmat_if_present` to use the theory covariance matrix for the statistical
estimators when available.

### Run a custom runcard

Any runcard built around the `ntkpdf` providers can be executed directly:

```bash
ntkpdf path/to/runcard.yaml -o <output_dir>
```

### Programmatic API

Every provider can be called directly, which is convenient for notebooks and
scripting (no runcard needed):

```python
from ntkpdf.api import API

common = dict(
    dataset_inputs={"from_": "fit"},
    use_cuts="fromfit",
    theory={"from_": "fit"},
    theoryid={"from_": "theory"},
)

figs = API.plot_losses_all_replicas(fit="<FIT_ID>", **common)
```

## What the report contains

- **Summary / statistical summary** — fit metadata and estimators.
- **t0 losses**, **theory covariance summary**, **dataset properties**.
- **Initialisation** — PDFs at initialisation, comparing model selectors
  (e.g. with / without preprocessing prefactors) at chosen replica counts.
- **Loss functions all replicas** — training and validation loss vs epoch, one
  figure per replica.
- **NTK plots** — eigenvalue spectra of the NTK (full model and NN-only).

## Architecture (brief)

`ntkpdf` is declarative: behaviour is built from **providers** (plain functions)
and **config productions** (`produce_*` / `parse_*` methods on `ntkConfig`),
which `reportengine` resolves into a dependency graph from the runcard. Key
modules:

- `ntkpdf/config.py` — `ntkConfig`: loads a fit's weights/preprocessing/epochs
  from disk and rebuilds the `n3fit` model.
- `ntkpdf/data_theory.py` — FK tables, fitting-covmat eigensystem, and the
  (streaming) training/validation loss curves.
- `ntkpdf/vpinterface.py` — wraps an `NTKStats` so it behaves like a validphys
  `PDF`.
- `ntkpdf/plotting/` — the shared, styled drawing layer and the plot providers.
- `ntkpdf/analysistemplate/` — the report template and sub-report pages.

For a fuller architecture description aimed at contributors, see `CLAUDE.md`.

## Notes

- Loss/FK-table analyses assume the fit was run with `diagonal_basis=True`.
- The `n3fit` model is evaluated with the JAX Keras backend; importing `ntkpdf`
  sets `KERAS_BACKEND=jax` before any TensorFlow/Keras-backed module is loaded.
