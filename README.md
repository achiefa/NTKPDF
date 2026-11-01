[![DOI](https://zenodo.org/badge/1260769576.svg)](https://doi.org/10.5281/zenodo.23495650)

# ntkpdf

A tool for the **Neural Tangent Kernel (NTK)** analysis of trained Parton Distribution
Function fits. It reads two kinds of fits:

- **n3fit** fits: the neural networks of [NNPDF](https://github.com/NNPDF/nnpdf);
- **colibri** Monte Carlo fits of any [colibri](https://github.com/HEP-PBSP/colibri) PDF
  model, e.g. the FPPDF/MSHT fixed parametrisation of
  [colibri_msht](https://github.com/HEP-PBSP/colibri_msht).

ntkpdf does not run fits. It loads the parameters a fit saved at several training epochs
and produces a report: the PDFs at initialisation and at chosen epochs, the training and
validation losses, the NTK eigenvalue spectra and the feature (h) values.

This repository was used to produce the results of
[arXiv:2610.36034](https://arxiv.org/abs/2610.36034).

This README takes a new machine from nothing to a report. Developers, and coding
assistants, should also read [`CLAUDE.md`](CLAUDE.md). The full documentation is in
`doc/sphinx` (see [Tests and documentation](#tests-and-documentation)).

## 1. Set up the environment

You need conda, git and LaTeX: the figures use LaTeX text, and PNG figures also need
`dvipng`.

```bash
git clone https://github.com/achiefa/NTKPDF.git
cd NTKPDF
conda env create -f environment.yml    # creates the environment "ntkpdf-dev"
conda activate ntkpdf-dev
```

`environment.yml` takes LHAPDF, MPI and ultranest from conda-forge, then installs ntkpdf
with `pip install -e .`, which also installs:

- colibri, branch [`ntk`](https://github.com/HEP-PBSP/colibri/tree/ntk);
- nnpdf (validphys and n3fit), branch
  [`ntk_colibri`](https://github.com/NNPDF/nnpdf/tree/ntk_colibri), pinned by colibri.

To produce or analyse colibri_msht fits, add colibri_msht, branch `msht`, to the same
environment:

```bash
git clone -b msht https://github.com/HEP-PBSP/colibri_msht.git
pip install --no-deps -e colibri_msht
```

`--no-deps` matters: colibri_msht's own requirements would replace colibri's `ntk` branch
with colibri's default branch.

Check the installation, from the NTKPDF folder:

```bash
python -c "import ntkpdf, colibri, validphys; print(ntkpdf.__file__)"
pytest -m "not fit"
```

**Data.** Theories, PDF sets and fits are downloaded from NNPDF's servers the first time a
command needs them. `vp-get` downloads them beforehand, e.g. `vp-get theoryID 40000000`,
`vp-get pdf NNPDF40_nnlo_as_01180` or `vp-get fit <fit>`. ntkpdf finds fits in the NNPDF
results folder:

```bash
python -c "from validphys.loader import Loader; print(Loader().resultspath)"
```

## 2. Produce a fit ntkpdf can read

ntkpdf needs each replica's parameters at several epochs, in
`fit_replicas/replica_<r>/parameters/params_<e>.npz`: `params_0` holds the initial
parameters and `params_<e>` those after e epochs. The fit folder must be in the NNPDF
results folder.

### colibri_msht fits

The runcards in `colibri_msht/runcards/` set up FPPDF's fitted-charm and
perturbative-charm fits, each with and without sum rules. They already save what ntkpdf
needs:

```yaml
max_epochs: 20000
record_parameters: true   # save the parameters...
record_every: 50          # ...every 50 epochs, the initial ones included
```

Each run fits one Monte Carlo replica. Run one per replica (three here), then move the fit
into the results folder:

```bash
RUNCARDS=$(python -c "import colibri_msht, os; print(os.path.dirname(colibri_msht.__file__))")/runcards
cp $RUNCARDS/fppdf_perturbative_charm_mc.yaml my_msht_fit.yaml
for r in 1 2 3; do colibri_msht my_msht_fit.yaml -rep $r; done
mv my_msht_fit "$(python -c 'from validphys.loader import Loader; print(Loader().resultspath)')/"
```

The fit is written to a folder named after the runcard, in the current directory. On a
cluster, submit one job per replica. ntkpdf needs neither postfit nor evolution.
[colibri_msht's README](https://github.com/HEP-PBSP/colibri_msht/tree/msht) describes the
runcards.

### n3fit fits

Use n3fit from nnpdf's `ntk_colibri` branch (installed above), with these runcard keys:

```yaml
diagonal_basis: true     # ntkpdf works in the covariance eigenbasis
fitting:
  savepseudodata: true   # ntkpdf rebuilds each replica's training data from it
save_checkpoints: true   # save the parameters...
checkpoint_freq: 50      # ...every 50 epochs, the initial ones included
```

Then run the fit as usual: `vp-setupfit`, `n3fit` for each replica, `evolven3fit` and
`postfit` (NNPDF's tutorial
[How to run a PDF fit](https://docs.nnpdf.science/tutorials/run-fit.html)). The NTK pages
need only the training output. The report's validphys tables also need the fit's LHAPDF
set, which evolution and postfit produce.

## 3. Run the analysis

```bash
ntkanalysis <fit> --title "NTK analysis" --author "Your name" --keywords ntk \
    --epochs 0 1000 -o my_report
```

Open `my_report/index.html`. `python -m ntkpdf.scripts.ntkanalysis` runs the same command.
The main options:

- `--epochs E1 E2 ...`: add pages with the PDFs at these epochs, which must be saved;
- `--replicas R1 R2 ...`: add single-replica eigenvalue and feature pages;
- `--show-fakepdf`: overlay a closure test's underlying law on the PDFs at those epochs;
- `--formats pdf`: save the figures as PDF only (the default is PNG and PDF);
- `-i`: prompt for missing fields; pass `--title` anyway, as its prompt is broken.

The docs (`getting-started`) list every option. The report writes the NTK eigenvalues it
computes into the fit's replica folders (`ntk_eigenvalues_*.npz`).

Two other ways to run ntkpdf:

- a custom runcard: `ntkpdf runcard.yaml -o output_dir`;
- the Python API, which calls any provider by name:

  ```python
  from ntkpdf.api import API

  epochs = API.stored_epochs(fit="<fit>")
  pdfs = API.pdf_at_epoch(fit="<fit>", epoch=1000)   # NTKStats, replicas on the first axis
  ```

  The notebooks in `ntkpdf/examples/` show more.

## Tests and documentation

```bash
pytest -m "not fit"   # fast tests
pytest                # also the end-to-end tests on real fits
```

The end-to-end tests download an n3fit test fit and run a small colibri_msht fit (two
replicas, ten epochs), so they need colibri_msht installed. Both fits stay in the NNPDF
results folder for the next run.

The documentation is built with Sphinx:

```bash
pip install sphinx sphinx_rtd_theme sphinx-copybutton
make -C doc/sphinx html     # then open doc/sphinx/build/html/index.html
```

## Notes

- ntkpdf sets `KERAS_BACKEND=jax` when imported, before Keras is loaded.
- If ntkpdf, colibri or colibri_msht is an editable install of another checkout,
  `python -c "import ntkpdf; print(ntkpdf.__file__)"` shows which code runs. Put a
  checkout first on `PYTHONPATH` to run that one instead.
- Fits made with colibri before commit 9b8aa8ea, or with nnpdf before 8e0b89e3b, don't
  save the initial parameters. The older colibri fits also save each snapshot one epoch
  later than its name says.
