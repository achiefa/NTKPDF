%Features $q$ of {@ current fit @}

# Features $q$

Each feature $q^{(k)}$ is a vector over $(\text{flavour}, x)$, so it is plotted like
a PDF: value versus $x$, one figure per flavour. The ranks of each group are
overlaid on the same axes, at the epochs selected on the command line (`--epochs`).

**Note on padding / removed replicas.** Each replica retains only the $q$ columns
up to its own NTK cut; columns beyond that cut are zero-padded, meaning the feature
is *absent* for that replica at that rank. Such replicas are **excluded** from the
ensemble band for that rank, so different ranks may be averaged over different
numbers of replicas. (When this happens the API/CLI logs how many replicas were
dropped for each rank.)

<!-- # Ensemble

{@with Epochspecs@}
### Epoch {@epoch@}
{@with Rankspecs@}
#### {@rank_title@}
{@plot_features_at_epoch@}
{@endwith@}
{@endwith@} -->

# Single replicas

One figure per selected replica (`--replicas`); empty if none were selected. Ranks
are drawn as individual lines (no band).

{@with Replicaspecs@}
## Replica {@replica_index@}
{@with Epochspecs@}
### Epoch {@epoch@}
{@with Rankspecs@}
#### {@rank_title@}
{@with PDFscalespecs@}
##### {@Xscaletitle@}
{@plot_features_at_epoch_single_replica@}
{@endwith@}
{@endwith@}
{@endwith@}
{@endwith@}
