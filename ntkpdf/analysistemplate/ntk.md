%Comparing NTK eigenvalues of {@ current fit @}

# NTK eigenvalues

The `eigenvalue_grid` is built **once** per (fit, model config) and *sliced* by the
rank groups below -- the template with/endwith loops drive only the section structure
and the rank selection, they do not rebuild the (full-trajectory) grid. The
single-replica plots are empty unless `--replicas` was given.

## Ensemble

{@with Rankspecs@}
### {@rank_title@}
{@plot_eigvals_rank@}
{@endwith@}

## Single replicas

{@with Replicaspecs@}
### Replica {@replica_index@}
{@with Rankspecs@}
#### {@rank_title@}
{@plot_eigvals_replica_rank@}
{@endwith@}
{@endwith@}
