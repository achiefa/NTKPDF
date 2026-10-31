"""
ntkpdf.fits.colibri

A fit made with colibri's Monte Carlo gradient descent, for any colibri ``PDFModel``
(e.g. colibri_msht's MSHT parametrisation). On disk:

- ``filter.yml``: the runcard;
- ``pdf_model.pkl``: the model, pickled with dill;
- ``fit_replicas/replica_<r>/parameters/params_<e>.npz``: the parameters *after* the
  update of epoch e, i.e. after e + 1 steps (``colibri/gradient_descent.py``); the
  initial parameters are not saved;
- ``fit_replicas/replica_<r>/mc_loss.csv``: row k is the training loss *before* the
  update of epoch k * record_every.

colibri does not save the pseudodata, the covmat or the initial parameters: they are
regenerated here with colibri's own functions, from the runcard.
"""

import dill
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from colibri.config import colibriConfig
from colibri.mc_utils import mc_pseudodata
from colibri.param_initialisation import pdf_initial_parameters
from colibri.pdf_model import PDFModel
from colibri.training_validation import trval_seed
from reportengine.configparser import ConfigError

from ntkpdf.fits.common import TrainValData, data_idx, labelled, pdf_stats, sort_replicas
from ntkpdf.utils import XGRID

# Runcard keys of colibri's ``mc_pseudodata``; when absent, its defaults apply, as in the fit.
PSEUDODATA_OPTIONS = ("mcseed", "shuffle_indices", "positive_pseudodata", "mc_validation_fraction")


class ColibriFit:
    """How a colibri fit stores its model, epochs and training data."""

    selectors = ("model",)

    def __init__(self, fit):
        self.fit = fit
        self.runcard = {k: v for k, v in fit.as_input().items() if k != "actions_"}

    def replicas(self) -> list[int]:
        return [int(d.name.split("_")[-1]) for d in self._replica_dirs()]

    def stored_epochs(self) -> list[int]:
        """The epochs saved by every replica (a replica that stops early saves fewer)."""
        saved = [
            {int(p.stem.split("_")[-1]) for p in (d / "parameters").glob("params_*.npz")}
            for d in self._replica_dirs()
        ]
        return sorted(set.intersection(*saved))

    def learning_rate(self) -> float:
        optimizer = self._optimizer_settings()
        if optimizer["scheduler"] is not None:
            raise ConfigError(f"Fit {self.fit.name} uses a learning-rate schedule, not a constant learning rate.")
        return optimizer["optimizer_hyperparams"]["learning_rate"]

    def summary(self) -> dict:
        pdf_model = self.pdf_model()
        optimizer = self._optimizer_settings()
        return {
            r"Model": type(pdf_model).__name__,
            r"Parameters": f"{pdf_model.n_parameters}",
            r"Optimizer": f"{optimizer['optimizer']}",
            r"Learning rate": f"{optimizer['scheduler'] or optimizer['optimizer_hyperparams']['learning_rate']}",
            r"Epochs": f"{self.runcard['max_epochs']}",
            r"Initialiser": f"{self.runcard['param_initialiser_settings']['type']}",
        }

    def pdf_model(self, replica: int | None = None) -> PDFModel:
        """The model colibri pickled; the same for every replica."""
        with open(self.fit.path / "pdf_model.pkl", "rb") as f:
            pdf_model = dill.load(f)
        if not isinstance(pdf_model, PDFModel):
            raise ConfigError(
                f"pdf_model.pkl of fit {self.fit.name} is not a colibri PDFModel "
                "(a colibri-n3fit fit?), which ntkpdf does not read."
            )
        return pdf_model

    def ensemble(self, nreplicas: int, replica_ids=None) -> "ColibriEnsemble":
        """The model for ``nreplicas`` replicas, with their initial parameters drawn as
        colibri draws them (the seed is the replica index): those of ``replica_ids``,
        or by default the fit's own replicas, followed by fresh ones if ``nreplicas``
        asks for more."""
        pdf_model = self.pdf_model()
        replicas = replica_ids or self._replicas_and_fresh_ones(nreplicas)
        # colibri reads only the forward map's ``param_names``, which are the model's
        # (for a fit without extra, non-PDF parameters).
        settings = self.runcard["param_initialiser_settings"]
        initial = np.stack([pdf_initial_parameters(pdf_model, settings, r) for r in replicas])
        return ColibriEnsemble(pdf_model, initial)

    def data(self):
        """The fit's data, with its own theory and cuts, as colibri builds them."""
        from colibri.api import API

        return API.data(**self.runcard)

    def training_data(self, fk: pd.DataFrame) -> TrainValData:
        """The pseudodata, FK and inverse covmat of each replica's losses, in the basis
        of the data points, as in the fit: the level-1 data and the t0 covmat ``C``
        (colibri's single covmat, for the noise and the chi2), then each replica's
        pseudodata and split from colibri's ``mc_pseudodata``; ``cinv = C[tr, tr]^-1``.

        The split uses colibri's default ``--trval_index 0``, which the fit does not
        record; it matters only when ``mc_validation_fraction`` is not 0.
        """
        from colibri.api import API

        central = API.central_covmat_index(**self.runcard)
        covmat = np.asarray(central.covmat)
        options = {k: self.runcard[k] for k in PSEUDODATA_OPTIONS if k in self.runcard}
        y, fks, cinvs = ([], []), ([], []), ([], [])
        for r in self.replicas():
            replica = mc_pseudodata(central, r, trval_seed(0), **options)
            for i, idx in enumerate((replica.training_indices, replica.validation_indices)):
                idx = np.asarray(idx, dtype=int)
                y[i].append(np.asarray(replica.pseudodata)[idx, None])
                fks[i].append(fk.to_numpy()[idx])
                cinvs[i].append(np.linalg.inv(covmat[np.ix_(idx, idx)]))
        return TrainValData(
            fk=tuple(labelled(blocks, fk.columns) for blocks in fks),
            cinv=tuple(labelled(blocks, data_idx(len(blocks[0]))) for blocks in cinvs),
            y=tuple(labelled(blocks) for blocks in y),
        )

    def fakepdf(self) -> str:
        try:
            return self.runcard["closure_test_pdf"]
        except KeyError as e:
            raise ConfigError(
                f"fit {self.fit.name} has no 'closure_test_pdf' (not a closure test); "
                "set show_fakepdf: False (the default) to skip the underlying-law overlay."
            ) from e

    def report_settings(self, current_fit: dict) -> dict:
        """The report's settings for this fit, ``current_fit`` being its ``{id, label}``,
        where they differ from ``analysis.yaml``, which is written for n3fit fits: no
        LHAPDF set or ``theory`` in the ``current`` namespace, no validphys fit tables
        (they need both, and n3fit's replica files), and only the full-model selector
        (a colibri model has no prefactors or network layers)."""
        return {
            "current": {"fit": current_fit, "speclabel": current_fit["label"]},
            "description": self.runcard.get("description", self.fit.name),
            "Validphyssummaries": [],
            "Eigenvaluesconfigs": [{"name": "model", "title_eigenvalues": "full model"}],
            "model_selectors": [{"name": "model", "no_prefactors": False}],
            "Selectors": [{"selectors": ["model"], "labels": ["Model"], "selector_name": "Full model"}],
        }

    def _replica_dirs(self) -> list:
        return sort_replicas((self.fit.path / "fit_replicas").glob("replica_*"))

    def _replicas_and_fresh_ones(self, nreplicas: int) -> list[int]:
        """The first ``nreplicas`` of: the fit's replicas, then new indices after them."""
        fitted = self.replicas()
        return (fitted + list(range(max(fitted) + 1, max(fitted) + 1 + nreplicas)))[:nreplicas]

    def _optimizer_settings(self) -> dict:
        """The runcard's optimizer settings, with colibri's defaults."""
        return colibriConfig({}).parse_optimizer_settings(self.runcard["optimizer_settings"])


class ColibriEnsemble:
    """The model evaluated for several replicas at once."""

    def __init__(self, pdf_model: PDFModel, initial_parameters: np.ndarray):
        self._pdfs = jax.jit(jax.vmap(pdf_model.grid_values_func(XGRID)))
        self.initial_parameters = initial_parameters
        self.nreplicas = len(initial_parameters)

    def at(self, weights, selector=None):
        """PDFs of the ensemble's replicas with one epoch's parameters ``weights``, one
        vector per replica (extra vectors are ignored, as by the n3fit ensemble)."""
        if selector is not None and selector.get("no_prefactors"):
            raise ConfigError("A colibri model has no preprocessing prefactors to remove.")
        return pdf_stats(self._pdfs(jnp.asarray(np.stack(weights[: self.nreplicas]))))

    def at_init(self, selector=None):
        """PDFs of every replica with its initial parameters."""
        return self.at(self.initial_parameters, selector)
