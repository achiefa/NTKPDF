%Comparing feature eigenvalues of {@ current fit @}

# Feature eigenvalues

The h grid is built **once** per (fit, model config); the x-scales, rank groups and
replicas are iterated inside the providers (so the grid is not rebuilt per figure).
Each figure is titled by its x-scale, rank group (and replica). The single-replica
plots are empty unless `--replicas` was given.

## Ensemble

{@plot_feature_eigvals_grouped@}

## Single replicas

{@plot_feature_eigvals_replicas_grouped@}
