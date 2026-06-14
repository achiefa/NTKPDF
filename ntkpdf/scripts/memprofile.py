#!/usr/bin/env python
"""
memprofile.py -- expose memory growth of the per-epoch NTK eigenvector computation.

This mirrors exactly what ``ntkpdf.ntkdecomposition.h_val_grid`` does in its loop --
compute the NTK eigenvectors for an epoch and discard them -- with NO reportengine and
NO API in the loop, so anything that grows is the *computation*, not DAG retention.

It reports, per epoch:
  - current and peak process RSS (resident memory),
  - the number and total size of *live JAX arrays* (``jax.live_arrays()``) -- if these
    grow, the eigenvectors/intermediates are being retained on the JAX side,
  - the Python object *types* that grew most since the last report -- if e.g.
    ``ndarray`` or a keras type keeps climbing, that's the reference being held.

Usage (from ntkpdf/scripts/):
    python memprofile.py 260526-ac-02-ntk-sgd
    python memprofile.py 260526-ac-02-ntk-sgd --epochs 60 --replicas 1 2 3
    python memprofile.py 260526-ac-02-ntk-sgd --no-gc      # see the un-collected growth
    python memprofile.py 260526-ac-02-ntk-sgd --kwargs nn  # the NN-only (remove_prefactors) config

Read it as: if `curRSS` keeps climbing while `jaxArrs`/`jaxGB` stay flat, the leak is
not retained Python/JAX objects -- it's JAX/keras per-evaluation C-level memory.
"""
import argparse
import gc
import os
import resource
import subprocess
import sys
from collections import Counter

import ntkpdf  # MUST be imported first: sets KERAS_BACKEND=jax before keras is loaded
import jax
from validphys.loader import FallbackLoader
from colibri.ntk.eigenvector import eigenvectors_ensemble_at_epoch

# Model-config presets matching the report's Eigenvaluesconfigs.
KWARGS_PRESETS = {
    "model": frozenset(),
    "nn": frozenset({("exclude_layers", ("impose_msr",)), ("remove_prefactors", True)}),
}


def current_rss_gb():
    """Current resident memory in GB (not the peak)."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        pass
    try:  # `ps` RSS is in KB on macOS and Linux
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])
        return int(out.strip()) / 1e6
    except Exception:
        m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return m / 1e9 if sys.platform == "darwin" else m / 1e6


def peak_rss_gb():
    m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return m / 1e9 if sys.platform == "darwin" else m / 1e6


def jax_live():
    arrs = jax.live_arrays()
    return len(arrs), sum(a.nbytes for a in arrs) / 1e9


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fit", help="fit name (downloaded via the validphys loader if needed)")
    ap.add_argument("--epochs", type=int, default=40, help="how many stored epochs to sweep (default 40)")
    ap.add_argument("--replicas", type=int, nargs="+", default=[1, 2, 3], help="replica ids (default 1 2 3)")
    ap.add_argument("--kwargs", choices=list(KWARGS_PRESETS), default="model", help="model config (default model)")
    ap.add_argument("--no-gc", action="store_true", help="disable per-epoch gc.collect() (shows raw growth)")
    ap.add_argument("--every", type=int, default=5, help="report every N epochs (default 5)")
    args = ap.parse_args()

    fit = FallbackLoader().check_fit(args.fit)
    replicas_path = fit.path / "fit_replicas"
    kwargs = KWARGS_PRESETS[args.kwargs]

    rep_dir = replicas_path / f"replica_{args.replicas[0]}" / "parameters"
    all_epochs = sorted(int(f.stem.split("_")[-1]) for f in rep_dir.glob("params_*.npz"))
    epochs = all_epochs[: args.epochs]

    print(
        f"fit={args.fit} | stored epochs={len(all_epochs)}, sweeping {len(epochs)} "
        f"| replicas={args.replicas} | config={args.kwargs} | gc={'OFF' if args.no_gc else 'on'}"
    )
    print(f"{'epoch':>7} {'curRSS':>9} {'peakRSS':>9} {'jaxArrs':>8} {'jaxGB':>7}  top-object-growth")

    prev = None
    for k, ep in enumerate(epochs):
        # Exactly the heavy call h_val_grid makes; then drop it like h_val_grid does.
        eigenvectors = eigenvectors_ensemble_at_epoch(
            fit, replicas_path, ep, replica_index_list=tuple(args.replicas), kwargs=kwargs
        )["eigenvectors_data"]
        del eigenvectors
        if not args.no_gc:
            gc.collect()

        if k % args.every == 0 or k == len(epochs) - 1:
            n_arr, gb = jax_live()
            objs = Counter(type(o).__name__ for o in gc.get_objects())
            growth = ""
            if prev is not None:
                top = sorted(((objs[t] - prev.get(t, 0), t) for t in objs), reverse=True)[:5]
                growth = ", ".join(f"{t}+{d}" for d, t in top if d > 20)
            print(
                f"{ep:>7} {current_rss_gb():>8.2f}G {peak_rss_gb():>8.2f}G "
                f"{n_arr:>8} {gb:>6.2f}G  {growth}",
                flush=True,
            )
            prev = objs

    print(f"\nfinal: curRSS={current_rss_gb():.2f} GB  peakRSS={peak_rss_gb():.2f} GB")


if __name__ == "__main__":
    main()
