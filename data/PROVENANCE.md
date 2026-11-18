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

## spf_panel.parquet

Written by `scripts/export_spf_panel.py`, run once inside the environment of the read-only
companion repository `expectations-without-hindsight` (commit b3b77cf; its `ewh/data` code is
identical to tag `pap-v1.2`), raw-data snapshot 2026-11-11:

    PYTHONDONTWRITEBYTECODE=1 uv run --project ~/Documents/expectations-without-hindsight \
        --no-sync python scripts/export_spf_panel.py --out data/spf_panel.parquet

sha256 `644df8e9a0c6bc5db738911903e567e6d4c94dc02a7245747cb696be02e0e285`, 623,051 bytes,
1,351 rows (7 variables x 193 survey quarters, 1968Q4-2016Q4).

Sources: only public raw files, through EWH's own raw-file pipeline (the "ours" path of its
SPF replication: `ewh.data.spf.spf_pairs(var, "bgms")`, `spf.sample`, `spf.trim(spf.load(var))`,
`ewh.data.realtime`):
- Philadelphia Fed Survey of Professional Forecasters individual microdata
  (`SPFmicrodata.xlsx`): the forecasts, errors and revisions;
- Philadelphia Fed Real-Time Data Set for Macroeconomists (RTDSM: `NOUTPUTQvQd`,
  `ROUTPUTQvQd`, `PQvQd`, `cpiQvMd`, `rucQvMd`): real-time histories and outcomes;
- FRED `TB3MS` and `GS10` (Board of Governors H.15 rates, not revised): histories and outcomes
  of TB3M and TN10Y, as in EWH.

Nothing in this file comes from the BGMS (`118422-V1.zip`) or Coibion-Gorodnichenko
(`112862-V1.zip`) replication packages; the export refuses to open either archive. EWH's
replication check reports that this pipeline reproduces BGMS Table 3; the consensus b computed
from this file equals EWH's `cons_ours` for all seven variables.

Variables: EWH's headline set (NGDP, RGDP, PGDP, CPI, UNEMP, TB3M, TN10Y). Target: the
forecast made at t of quarter t+3 and its revision from the t-1 survey (BGMS's f3q). Columns
are documented in the export script's docstring: real-time history of vintage t as a list
(`hist`, from `hist_first` to t-1, with its source vintage in `hist_source`), the outcome, the
consensus error and revision, the cross-sectional mean SPF forecast for quarters t-1..t+4, and
the individual errors and revisions (`ind_id`, `ind_err`, `ind_rev`).

Real-time conventions inherited from EWH: vintage t ends at t-1; when x[t-1] was released late
(1996Q1 for NGDP, RGDP and PGDP) it is taken from the next vintage that has it
(`last_filled`); real-time CPI vintages begin in 1994Q3, so earlier CPI histories are the
1994Q3 vintage cut to observations before t.
