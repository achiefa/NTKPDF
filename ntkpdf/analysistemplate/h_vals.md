%Comparing feature eigenvalues of {@ current fit @}

# Feature eigenvalues

The `h_val_grid` is built **once** per (fit, model config) and *sliced* by the rank
groups / x-scales below -- the template with/endwith loops drive only the section
structure and the rank selection, they do not rebuild the (full-trajectory) grid. The
single-replica plots are empty unless `--replicas` was given.

## Ensemble

{@with PDFscalespecs@}
### {@Xscaletitle@} x-scale
{@with Rankspecs@}
#### {@rank_title@}
{@plot_feature_eigvals_rank@}
{@endwith@}
{@endwith@}

## Single replicas

{@with Replicaspecs@}
### Replica {@replica_index@}
{@with PDFscalespecs@}
#### {@Xscaletitle@} x-scale
{@with Rankspecs@}
##### {@rank_title@}
{@plot_feature_eigvals_replica_rank@}
{@endwith@}
{@endwith@}
{@endwith@}
