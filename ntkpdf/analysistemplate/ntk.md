%Comparing NTK eigenvalues of {@ current fit @}

# NTK eigenvalues

## Ensemble

{@with Rankspecs@}
### {@rank_title@}
{@plot_eigvals_by_fit@}
{@endwith@}

# Single replicas

One figure per selected replica (`--replicas`); empty if none were selected.

{@with Replicaspecs@}
## Replica {@replica_index@}
{@with Rankspecs@}
### {@rank_title@}
{@plot_eigvals_replica_by_fit@}
{@endwith@}
{@endwith@}
