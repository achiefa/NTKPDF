%Comparing NTK eigenvalues of {@ current fit @}

# NTK eigenvalues

The eigenvalue grid is built **once** per (fit, model config); the rank groups and
replicas are iterated inside the providers (so the grid is not rebuilt per figure).
Each figure is titled by its rank group (and replica). The single-replica plots are
empty unless `--replicas` was given.

## Ensemble

{@plot_eigvals_grouped@}

## Single replicas

{@plot_eigvals_replicas_grouped@}
