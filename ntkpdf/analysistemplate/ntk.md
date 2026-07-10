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

# NTK Frobenius norm

The Frobenius norm $\|K\|_F = \sqrt{\sum_i \lambda_i^2}$ of the NTK as a function of
epoch, computed from the same `eigenvalue_grid`. The single-replica plots are empty
unless `--replicas` was given.

## Ensemble

{@plot_frobenius_norm@}

## Single replicas

{@with Replicaspecs@}
### Replica {@replica_index@}
{@plot_frobenius_norm_replica@}
{@endwith@}
