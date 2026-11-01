"""
ntkpdf.fits.common

What every kind of fit hands to the rest of ntkpdf.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from colibri.ntk.ntkutils import NTKStats

from ntkpdf.utils import EVOL_INDEX


@dataclass(frozen=True)
class TrainValData:
    """The per-replica ingredients of the training and validation losses
    ``(y - fk f)^T cinv (y - fk f) / n``. Each field is a ``(training, validation)``
    pair of ``NTKStats`` whose data axis is labelled ``data_idx``."""

    fk: tuple
    cinv: tuple
    y: tuple


def sort_replicas(replicas: Iterable[Path]) -> list[Path]:
    """``replica_<r>`` paths in numerical order of ``r``."""
    return sorted(replicas, key=lambda r: int(r.stem.split("_")[-1]))


def data_idx(n: int) -> pd.MultiIndex:
    """``data_idx``-named MultiIndex 1..n, shared by FK, Cinv and data so that their
    ``NTKStats`` products align."""
    return pd.MultiIndex.from_arrays([[i + 1 for i in range(n)]], names=["data_idx"])


def labelled(blocks, columns=None) -> NTKStats:
    """Per-replica blocks of equal shape ``(n, m)`` as one ``NTKStats`` whose rows are
    labelled ``data_idx``."""
    stats = NTKStats(np.stack(blocks))
    stats.set_index(data_idx(stats.shape[0]), columns)
    return stats


def pdf_stats(grids) -> NTKStats:
    """PDF grids of shape ``(nrep, 14, nx)``, flavours in the FK order, as an ``NTKStats``
    of shape ``(nrep, 14 nx, 1)`` on ``EVOL_INDEX`` (flavour-major)."""
    grids = np.asarray(grids)
    stats = NTKStats(grids.reshape(grids.shape[0], -1, 1))
    stats.set_index(EVOL_INDEX, None)
    return stats
