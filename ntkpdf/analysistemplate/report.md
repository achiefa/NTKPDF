%NTK analysis of {@ current fit_id @}

Summary
-------

We are analysing `{@ current fit_id @}`: {@ current description @}

{@ summarise_ntk_fits @}

Statistical summary
-------------------

{@ summarise_fits @}

t0 losses
---------
{@ dataspecs::t0_info t0_chi2_info_table @}

Theory covariance summary
-------------------------
{@summarise_theory_covmat_fits@}

Dataset properties
------------------
{@current fit_datasets_properties_table@}

Initialisation
--------------
{@with Customreplicas@}
[PDFs at initialisation with {@nreplicas_title@}]({@initialisation_report report@})
{@endwith@}

Loss functions
--------------
[Loss functions all replicas]({@loss_report report@})

PDFs at epochs
--------------
{@with Epochspecs@}
[PDFs at epoch {@epoch@}]({@pdf_epochs_report report@})
{@endwith@}

NTK plots
---------
{@with Eigenvaluesconfigs@}
[Plots NTK ({@title_eigenvalues@})]({@ntk_report report@})
{@endwith@}

Feature eigenmodes
------------------
{@with Eigenvaluesconfigs@}
[Plots H ({@title_eigenvalues@})]({@h_val_report report@})
{@endwith@}