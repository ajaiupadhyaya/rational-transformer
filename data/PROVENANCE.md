# Provenance of data/

## aklmt_figure2.csv

Derived by `ewh.data.aklmt` (repository `expectations-without-hindsight`, commit 57e9084) from
the replication package of Afrouzi, Kwon, Landier, Ma and Thesmar (2023, Quarterly Journal of
Economics), Experiment 1. It reproduces their Figure II: implied persistence (OLS of the
one-step forecast on last period's realisation) and the error-on-revision coefficient, by rho,
with participant-clustered standard errors.

Only these derived statistics (six rows, one per rho) are in this repository. The replication
package itself is not. Columns: `rho, implied_rho, implied_rho_se, b_eor, b_eor_se,
b_eor_se_cr1, n_eor, participants`.

`analysis.human.aklmt_design_check` reads the raw package from the local
`expectations-without-hindsight` checkout when it is present (read only, never copied) to compare
the experiment's series with the Family B prior.
