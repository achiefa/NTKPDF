%Comparing feature eigenvalues of {@ current fit @}

# Feature eigenvalues

## Ensemble

{@with PDFscalespecs@}
### {@Xscaletitle@}
{@with Rankspecs@}
#### {@rank_title@}
{@plot_feature_eigvals_by_fit@}
{@endwith@}
{@endwith@}

# Single replicas

One figure per selected replica (`--replicas`); empty if none were selected.

{@with Replicaspecs@}
## Replica {@replica_index@}
{@with PDFscalespecs@}
### {@Xscaletitle@}
{@with Rankspecs@}
#### {@rank_title@}
{@plot_feature_eigvals_replica_by_fit@}
{@endwith@}
{@endwith@}
{@endwith@}
