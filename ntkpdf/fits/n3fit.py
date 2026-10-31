"""
ntkpdf.fits.n3fit

A fit made with n3fit, NNPDF's neural network. On disk:

- ``filter.yml``: the runcard, from which the network is rebuilt;
- ``nnfit/replica_<r>/<fit>.json``: each replica's fitted preprocessing exponents and
  best epoch;
- ``nnfit/replica_<r>/*pseudodata*.csv`` and ``tables/*fitting_covmat_table.csv``: the
  pseudodata and the covmat, in the covmat's eigenbasis (``diagonal_basis``);
- ``fit_replicas/replica_<r>/parameters/params_<e>.npz``: the network weights after e
  steps (``params_0`` is the initialisation); the best epoch is saved too.
"""

import json
import logging
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from colibri.ntk.ntkutils import NTKStats
from n3fit.backends import MetaModel
from n3fit.backends.keras_backend.MetaModel import PREPROCESSING_LAYER_ALL_REPLICAS
from n3fit.model_gen import ReplicaSettings, generate_pdf_model
from reportengine.configparser import ConfigError
from validphys.pseudodata import read_replica_pseudodata

from ntkpdf.fits.common import TrainValData, data_idx, pdf_stats, sort_replicas
from ntkpdf.model import NTKPDFN3Fit
from ntkpdf.utils import INPUT_GRID

log = logging.getLogger(__name__)


class N3Fit:
    """How an n3fit fit stores its model, epochs and training data."""

    selectors = ("model", "no_prefactors", "nn", "no_smr")

    def __init__(self, fit):
        self.fit = fit
        self.runcard = fit.as_input()

    def replicas(self) -> list[int]:
        return [int(d.name.split("_")[-1]) for d in self._replica_dirs()]

    def stored_epochs(self) -> list[int]:
        """The saved epochs (replica 1's; all replicas share them), without the best
        epoch, which n3fit saves as well."""
        best_epoch = self.best_epochs()[0]
        params = self.fit.path / "fit_replicas/replica_1/parameters"
        epochs = (int(p.stem.split("_")[-1]) for p in params.glob("params_*.npz"))
        return sorted(e for e in epochs if e != best_epoch)

    def best_epochs(self) -> list[int]:
        return [self._replica_json(d)["best_epoch"] for d in self._replica_dirs()]

    def preprocessing(self, replica_ids: Sequence[int] | None = None) -> list[list[float]]:
        """Each replica's fitted preprocessing exponents, ``smallx, largex`` per flavour
        (the order of the preprocessing layer's weights)."""
        if replica_ids is None:
            dirs = self._replica_dirs()
        else:
            dirs = [self.fit.path / f"nnfit/replica_{r}" for r in replica_ids]
        return [
            [x for fl in self._replica_json(d)["preprocessing"] for x in (fl["smallx"], fl["largex"])]
            for d in dirs
        ]

    def model_info(self) -> dict:
        parameters = self.runcard["parameters"]
        return {
            "basis_info": self.runcard["fitting"]["basis"],
            "fitbasis": self.runcard["fitting"]["fitbasis"],
            "nodes": parameters["nodes_per_layer"],
            "activations": parameters["activation_per_layer"],
            "initializer": parameters["initializer"],
            "initializer_scale": parameters.get("initializer_scale", 1.0),
            # The init-width exponent the fit used (default 1.0 = standard glorot), so
            # that the network at initialisation matches the fit's.
            "initializer_gamma": parameters.get("initializer_gamma", 1.0),
            "layer_type": parameters["layer_type"],
            "dropout": parameters["dropout"],
            "impose_sumrule": self.runcard.get("sum_rules", "All"),
            "optimizer": parameters["optimizer"]["optimizer_name"],
            "learning_rate": parameters["optimizer"]["learning_rate"],
            "clipnorm": parameters["optimizer"]["clipnorm"],
        }

    def learning_rate(self) -> float:
        return self.model_info()["learning_rate"]

    def summary(self) -> dict:
        info = self.model_info()
        return {
            r"Nodes/layer": f"{info['nodes']}",
            r"Activations/layer": f"{info['activations']}",
            r"Layer type": f"{info['layer_type']}",
            r"Initializer": f"{info['initializer']}",
            r"Initializer scale": f"{info['initializer_scale']}",
            r"Initializer gamma": f"{info['initializer_gamma']}",
            r"Dropout": f"{info['dropout']}",
            r"Optimizer": f"{info['optimizer']}",
            r"Learning rate": f"{info['learning_rate']}",
            r"Clipnorm": f"{info['clipnorm']}",
            r"Impose sum rule": f"{info['impose_sumrule']}",
        }

    def pdf_model(self, replica: int | None = None) -> NTKPDFN3Fit:
        """The single-replica network of the NTK, built from the runcard, with
        ``replica``'s fitted preprocessing exponents (or the ones it was built with).
        These matter whenever the prefactors are kept (the ``model``/``no_smr``
        selectors)."""
        pdf_model = NTKPDFN3Fit(
            replica_index=1,
            replica_range_settings={"min_replica": 1, "max_replica": 1},
            **_model_init_args(self.runcard),
        )
        if replica is not None:
            exponents = self.preprocessing([replica])[0]
            pdf_model.n3fit_model.get_layer(PREPROCESSING_LAYER_ALL_REPLICAS).set_weights(
                [np.array([[x]]) for x in exponents]
            )
        return pdf_model

    def ensemble(self, nreplicas: int, replica_ids: Sequence[int] | None = None) -> "N3FitEnsemble":
        return N3FitEnsemble(self.model_info(), self.preprocessing(replica_ids), nreplicas)

    def data(self):
        """The fit's data, with its own theory and cuts."""
        from validphys.api import API

        return API.data(
            fit=self.fit.name,
            use_cuts="fromfit",
            theoryid=self.runcard["theory"]["theoryid"],
            dataset_inputs=self.runcard["dataset_inputs"],
        )

    def training_data(self, fk: pd.DataFrame) -> TrainValData:
        """The pseudodata, FK and inverse covmat of each replica's losses, in the
        covmat's eigenbasis, as the fit used them: ``R C Rᵀ = diag(Λ)`` (saved in
        ``tables/``), so ``fk = R FK`` and ``cinv = diag(1/Λ)``, restricted to each
        replica's training or validation eigenmodes."""
        if not self.runcard.get("diagonal_basis", True):
            raise ConfigError(f"Fit {self.fit.name} was not run with diagonal_basis=True, which ntkpdf needs.")
        eigvals, rotation = self._covmat_eigensystem()
        pseudodata = [read_replica_pseudodata(self.fit, None, r) for r in self.replicas()]
        masks = _train_val_masks(pseudodata)
        return TrainValData(
            fk=_fk_train_val(rotation, fk, masks),
            cinv=_cinv_train_val(eigvals, masks),
            y=_data_train_val(pseudodata),
        )

    def fakepdf(self) -> str:
        try:
            return self.runcard["closuretest"]["fakepdf"]
        except KeyError as e:
            raise ConfigError(
                f"fit {self.fit.name} has no 'closuretest: fakepdf' (not a closure test); "
                "set show_fakepdf: False (the default) to skip the underlying-law overlay."
            ) from e

    def report_settings(self, current_fit: dict) -> dict:
        """The report's settings for this fit, ``current_fit`` being its ``{id, label}``:
        the ``current`` namespace (the fit, its LHAPDF set and theory) and the
        data/theory context, injected as direct values (see
        ``ntkanalysis.complete_mapping``)."""
        return {
            "current": {
                "fit": current_fit,
                "pdf": current_fit,
                "theory": {"from_": "fit"},
                "theoryid": {"from_": "theory"},
                "speclabel": current_fit["label"],
            },
            "dataset_inputs": self.runcard["dataset_inputs"],
            "theory": self.runcard["theory"],
            "theoryid": self.runcard["theory"]["theoryid"],
        }

    def _replica_dirs(self) -> list:
        return sort_replicas(p for p in (self.fit.path / "nnfit").glob("replica_*") if p.is_dir())

    def _replica_json(self, replica_dir) -> dict:
        with open(replica_dir / f"{self.fit.name}.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def _covmat_eigensystem(self):
        """``Λ`` and ``R``, as written by vp-setupfit (``Λ`` are the eigenvalues of the
        correlation matrix and ``R = Vᵀ diag(1/σ)``)."""
        from validphys.api import API

        table = pd.read_csv(
            API.fitting_covmat_name(fit=self.fit.name),
            index_col=[0], header=[0], sep="\t|,", engine="python",
        )
        return table.iloc[:, 0].values, table.iloc[:, 1:]


class N3FitEnsemble:
    """n3fit's multi-replica network (a keras ``MetaModel``), built once with the fit's
    architecture and fresh seeds 0..n-1, then loaded with each epoch's weights.

    The model is shared and mutated in place: every method sets the weights it needs
    and returns a NumPy-backed ``NTKStats``, so no result depends on the order of
    the calls."""

    def __init__(self, model_info: dict, preprocessing: list, nreplicas: int):
        self.model = _build_metamodel(model_info, nreplicas)
        self.preprocessing = preprocessing
        self.nreplicas = nreplicas
        self.initial_weights = [_copy_weights(self.model.get_replica_weights(i)) for i in range(nreplicas)]

    def at(self, weights: Sequence[np.ndarray], selector: Mapping | None = None):
        """PDFs of every replica with one epoch's flat network ``weights`` and the fit's
        preprocessing (neutralised by a ``no_prefactors`` selector)."""
        set_epoch_weights(self.model, weights, self.preprocessing, self.nreplicas, selector)
        return self._predict()

    def at_init(self, selector: Mapping | None = None):
        """PDFs of the network as initialised, with the exponents it was built with."""
        for i, initial in enumerate(self.initial_weights):
            weights = _copy_weights(initial)
            if _has_no_prefactors(selector):
                weights["preprocessing_factor"] = _neutral_preprocessing(weights["preprocessing_factor"])
            self.model.set_replica_weights(weights, i)
        return self._predict()

    def _predict(self):
        grids = self.model.predict({"pdf_input": INPUT_GRID.T}, verbose=False)[0]  # (nrep, nx, 14)
        return pdf_stats(grids.transpose((0, 2, 1)))


# ----------------------------------------
# The network
#########################################
def _model_init_args(fit_input: dict) -> dict:
    """:class:`~ntkpdf.model.NTKPDFN3Fit` constructor arguments read from a fit's
    ``filter.yml``: the fields n3fit bakes into ``pdf_model.pkl`` (``model_trainer.py``'s
    ``_init_args``)."""
    parameters = fit_input["parameters"]
    return {
        "nnseed": fit_input["nnseed"],
        "flav_info": fit_input["fitting"]["basis"],
        "fitbasis": fit_input["fitting"]["fitbasis"],
        "nodes": parameters["nodes_per_layer"],
        "activations": parameters["activation_per_layer"],
        "initializer_name": parameters["initializer"],
        "layer_type": parameters["layer_type"],
        # n3fit's ``performfit`` defaults ``sum_rules`` to True; the trainer passes it
        # straight through to the model as ``impose_sumrule``.
        "impose_sumrule": fit_input.get("sum_rules", True),
    }


def _build_metamodel(model_info: dict, nreplicas: int) -> MetaModel:
    """The multi-replica network, randomly initialised with seeds 0..n-1, in double
    precision so that its *weights* are float64, as on the NTK path
    (:class:`~ntkpdf.model.NTKPDFN3Fit`)."""
    log.info(f"Building base PDF model for {nreplicas} replicas.")
    import keras

    keras.backend.set_floatx("float64")
    replica_settings = [
        ReplicaSettings(
            seed=i,
            nodes=model_info["nodes"],
            activations=model_info["activations"],
            architecture=model_info["layer_type"],
            initializer=model_info["initializer"],
            initializer_scale=model_info["initializer_scale"],
            initializer_gamma=model_info["initializer_gamma"],
            dropout_rate=model_info["dropout"],
            regularizer=None,
            regularizer_args={},
        )
        for i in range(nreplicas)
    ]
    return generate_pdf_model(
        replicas_settings=replica_settings,
        flav_info=model_info["basis_info"],
        fitbasis=model_info["fitbasis"],
        out=14,
        impose_sumrule=model_info["impose_sumrule"],
        scaler=None,
    )


def _copy_weights(weights: dict) -> dict:
    return {layer: [np.array(w, copy=True) for w in ws] for layer, ws in weights.items()}


def _unpack_params(shapes, weights, prepr) -> dict:
    """A flat weight vector in the layout of ``MetaModel.set_replica_weights``."""
    weight_list = []
    pos = 0
    for shape in shapes:
        size = np.prod(shape)
        weight_list.append(weights[pos:pos + size].reshape(shape))
        pos += size
    return dict(all_NNs=weight_list, preprocessing_factor=prepr)


def _neutral_preprocessing(prepr):
    """Preprocessing weights are ordered [alpha, beta] per flavour, and the prefactor
    x^(1-alpha) (1-x)^beta is neutralised by alpha=1, beta=0."""
    return [np.full_like(w, 1.0 if j % 2 == 0 else 0.0) for j, w in enumerate(prepr)]


def _has_no_prefactors(model_selector: Mapping | None) -> bool:
    return model_selector is not None and model_selector.get("no_prefactors")


def set_epoch_weights(metamodels, weights, fit_preprocessing, nreplicas, model_selector=None):
    """Load one epoch's per-replica flat ``weights`` and ``fit_preprocessing`` into
    ``metamodels`` in place, and return it. Loading rewrites the preprocessing layer,
    so a ``no_prefactors`` selector is applied here."""
    if _has_no_prefactors(model_selector):
        fit_preprocessing = [_neutral_preprocessing(p) for p in fit_preprocessing]
    shapes = [list(v.shape) for v in metamodels.get_replica_weights(0)["all_NNs"]]
    for idx in range(nreplicas):
        metamodels.set_replica_weights(_unpack_params(shapes, weights[idx], fit_preprocessing[idx]), idx)
    return metamodels


# ----------------------------------------
# Training data, in the covmat's eigenbasis
#########################################
def _sorted_eigenmode_index(idx: pd.Index) -> pd.Index:
    """Sort a flat ``"eigenmode k"`` index ascending by ``k``."""
    return pd.Index(sorted(idx, key=lambda s: int(s.split()[-1])))


def _train_val_masks(pseudodata):
    """Per-replica boolean training and validation masks over the eigenmodes, sorted
    by ``k`` (the order of the covmat table)."""
    tr_frames, val_frames = [], []
    for pdr in pseudodata:
        # The training and validation modes are saved one after the other, so the
        # saved index is not ordered by eigenmode.
        data_index = _sorted_eigenmode_index(pdr.pseudodata.index)
        tr_mask = data_index.isin(pdr.tr_idx)
        tr_frames.append(pd.DataFrame(tr_mask, index=data_index))
        val_frames.append(pd.DataFrame(~tr_mask, index=data_index))
    return NTKStats(tr_frames), NTKStats(val_frames)


def _data_train_val(pseudodata):
    """Per-replica training and validation pseudodata, rows labelled ``data_idx``."""
    tr_data_list, val_data_list = [], []
    for pdr in pseudodata:
        data = pdr.pseudodata.reindex(_sorted_eigenmode_index(pdr.pseudodata.index))
        data.columns = data.columns.str.replace(r"\s+\d+$", "", regex=True)
        tr_data_list.append(data.loc[pdr.tr_idx])
        val_data_list.append(data.loc[~data.index.isin(pdr.tr_idx)])
    tr_data = NTKStats(tr_data_list)
    val_data = NTKStats(val_data_list)
    tr_data.set_index(data_idx(tr_data.shape[0]), tr_data._df_columns)
    val_data.set_index(data_idx(val_data.shape[0]), val_data._df_columns)
    return tr_data, val_data


def _fk_train_val(rotation, fk, masks):
    """Per-replica ``R FK``, split by each replica's training and validation modes,
    rows labelled ``data_idx`` (the eigenmode labels would not match Cinv's)."""
    rotation.columns = fk.index  # so that the product aligns the data points
    fk_rotated = rotation @ fk
    tr_masks, val_masks = masks
    fk_tr = NTKStats([fk_rotated[m.iloc[:, 0]] for m in tr_masks.frames])
    fk_vl = NTKStats([fk_rotated[m.iloc[:, 0]] for m in val_masks.frames])
    fk_tr.set_index(data_idx(fk_tr.shape[0]), fk_tr._df_columns)
    fk_vl.set_index(data_idx(fk_vl.shape[0]), fk_vl._df_columns)
    return fk_tr, fk_vl


def _cinv_train_val(eigvals, masks):
    """Per-replica ``diag(1/Λ)`` on each replica's training and validation modes."""
    tr_masks, val_masks = masks
    cinv_tr = NTKStats([pd.DataFrame(np.diag(1.0 / eigvals[m.iloc[:, 0].values])) for m in tr_masks.frames])
    cinv_vl = NTKStats([pd.DataFrame(np.diag(1.0 / eigvals[m.iloc[:, 0].values])) for m in val_masks.frames])
    tr_idx = data_idx(cinv_tr.shape[0])
    vl_idx = data_idx(cinv_vl.shape[0])
    cinv_tr.set_index(tr_idx, tr_idx)
    cinv_vl.set_index(vl_idx, vl_idx)
    return cinv_tr, cinv_vl
