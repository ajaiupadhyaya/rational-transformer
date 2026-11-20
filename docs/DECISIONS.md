# Decisions

Dates in this log are UTC; commit and tag timestamps in git are local time (UTC−5).

A running log of choices fixed during the project.

## 2026-11-16: Family B prior constants

Family B draws sigma^2 ~ InvGamma(a=3, b=2) (shape a, scale b; mean b/(a-1) = 1)
and mu | sigma^2 ~ N(0, 25 * sigma^2), i.e. `B_PRIOR = dict(a=3.0, b=2.0, k=25.0)`
in `rt/sim.py`. The series starts at x_0 = mu.

## 2026-11-16: Evaluation conventions (Task 6)

- Bayes benchmarks use a 2001-point rho grid for both families (`bayes_b`'s default of
  401 is off by up to 1e-2 sigma near rho = 1).
- Family B diagnostics, including every MSE column and the convergence error, are computed
  on standardised deviations `(x - mu) / sigma`, `(F - mu) / sigma` with each sequence's
  true mu and sigma, for the model and every benchmark alike. Family A is used raw.
- Evaluation sets: set i (RHOS order, then prior = 6) is drawn from
  `SeedSequence([10_000 + i, family tag])`. Benchmark caches under `results/eval/` are keyed
  by family, set, size, seed, window, grid and a digest of the data and of
  `rt/bayes.py` + `rt/baselines.py`.
- Convergence compares the last two checkpoints of the schedule the run was trained under
  (10,000 and 30,000 for the main config).

## 2026-11-16: Local learning check (Task 6 Step 5)

Cell `A-s-wfull-s0`, trained with `--max-steps 3000` (checkpoints 100, 300, 1000, 3000)
into `results/gate` on mps under `nice`, then evaluated on the full default evaluation sets
(6 x 20,000 + 100,000 sequences). Training was shortened from 30,000 steps by ruling: a
full run takes ~1.7 h locally and 2,000 steps already reached Bayes-level loss.

**Pass.** `excess_mse_pct` against `bayes`, h = 1, `prior` set, step 3000: **2.290** (< 5).
Trajectory on the same row: 123.06 (step 100), 33.03 (300), 5.10 (1000), 2.29 (3000).
`converged` = False (step 3000 vs step 1000 differ by more than 0.1%, as expected this early).

Step 3000, h = 1, bench `bayes` (identical to `window_bayes` for the full window):

| rho   | gap     | excess_mse_pct |
|-------|---------|----------------|
| 0.0   | +0.1381 | 5.320          |
| 0.2   | +0.0771 | 0.827          |
| 0.4   | -0.0106 | -0.529         |
| 0.6   | -0.0630 | 1.055          |
| 0.8   | -0.0248 | 2.812          |
| 1.0   | -0.0780 | 148.174        |
| prior | -0.0054 | 2.290          |

At rho = 1 the 3000-step model is far from Bayes (implied persistence 0.907 vs 0.985):
random-walk paths reach magnitudes the model rarely trains on. It is the end points of the
grid, not the prior average, that will need the full 30,000 steps. Bayes' own `eor_b` on the
prior set is -0.0007 (a martingale check: should be ~0).

Wall times (M2 Pro, mps, niced): training 533 s for 3000 steps; evaluation of all four
checkpoints 105 s including the first-time Bayes cache build, peak RSS 2.0 GB. A windowed
cell (A-s-w4, one checkpoint) evaluates in 79 s cold (window-Bayes build) and 40 s warm,
peak RSS 1.8 GB.

## 2026-11-16: Architecture probe (Task 5b)

Question: is the gap from Bayes at the ends of the rho grid in the local check above an
artifact of how scalars enter and leave the network (raw value in through `Linear(1, d)`,
pre-LayerNorm blocks, output `head(ln_f(h))` bounded by the head weights)?

Two config options were added, both defaulting to the original network (`configs/grid.json`:
`"head": "linear"`, `"inputs": "raw"`; a config or checkpoint without the keys means the
defaults). They are saved in every checkpoint, are part of the resume check, and evaluation
rebuilds the model from the checkpoint's saved config. `rt.train` takes `--head` and
`--inputs`.

- `head = "affine"`: the same `head(ln_f(h))` path gives 12 numbers per position,
  (a_h, b_h), and the forecast is `a_h * x_t + b_h` with x_t the raw value at that position
  (the last position of a padded window is its last real value). The a_h rows start at zero.
- `inputs = "raw+asinh"`: the value is embedded from `[x, asinh(x)]` by one linear layer.

Both see only the window's observations and their time indices (tested: causality, pad
invariance and the window rule hold for the affine head).

Protocol: cell `A-s-wfull-s0`, 3000 steps (checkpoints 100, 300, 1000, 3000), mps, under
`nice`, otherwise identical config; each variant into `results/probe/<head>-<inputs>/`,
evaluated on the full default sets. The baseline (linear, raw) is the existing
`results/gate` run. One seed per variant: differences of a few tenths of a percent
between variants are within what another seed could move.

Step 3000, bench `bayes`, h = 1 (`excess_mse_pct` / `gap`):

| rho   | linear, raw (baseline) | affine, raw       | linear, raw+asinh  | affine, raw+asinh |
|-------|------------------------|-------------------|--------------------|-------------------|
| 0.0   | 5.320 / +0.1381        | 2.897 / +0.0831   | 5.024 / +0.1303    | 2.519 / +0.0747   |
| 0.2   | 0.827 / +0.0771        | 0.291 / +0.0374   | 0.750 / +0.0684    | 0.295 / +0.0337   |
| 0.4   | -0.529 / -0.0106       | 0.059 / -0.0202   | -0.264 / -0.0138   | 0.097 / -0.0162   |
| 0.6   | 1.055 / -0.0630        | 1.171 / -0.0348   | 1.303 / -0.0520    | 0.960 / -0.0285   |
| 0.8   | 2.812 / -0.0248        | 1.190 / -0.0081   | 2.244 / -0.0190    | 1.015 / -0.0064   |
| 1.0   | 148.174 / -0.0780      | 4.181 / -0.0085   | 150.514 / -0.0806  | 1.410 / -0.0029   |
| prior | 2.290 / -0.0054        | 0.838 / -0.0019   | 2.162 / -0.0066    | 0.719 / -0.0006   |

Implied persistence at rho = 1, h = 1 (Bayes 0.985): 0.907, 0.976, 0.904, 0.982.

Step 3000, bench `bayes`, h = 6:

| rho   | linear, raw (baseline) | affine, raw       | linear, raw+asinh  | affine, raw+asinh |
|-------|------------------------|-------------------|--------------------|-------------------|
| 0.0   | 0.197 / -0.0301        | 0.033 / +0.0015   | 0.067 / -0.0057    | 0.030 / +0.0030   |
| 0.6   | 0.597 / +0.0281        | 0.296 / +0.0105   | 0.493 / +0.0183    | 0.284 / +0.0107   |
| 1.0   | 18.295 / -0.0577       | 1.046 / +0.0010   | 14.967 / -0.0501   | 0.765 / -0.0014   |
| prior | 0.662 / -0.0075        | 0.268 / +0.0010   | 0.569 / -0.0056    | 0.237 / +0.0018   |

`prior` h = 1 excess at step 1000: 5.101, 1.766, 5.338, 1.415 (same column order); at
step 300: 33.03, 6.13, 34.91, 5.21.

Throughput: the same for every variant, ~5.75 steps/s solo on mps (a 200-step timing of
the real training step for each; the baseline's own 3000-step run logged 5.68). The probe
runs' wall times (953-962 s for the two run side by side, 644 s for the third, which
overlapped evaluations) reflect contention, not the architecture.

Windowed check, `A-s-w4-s0`, 3000 steps, bench `window_bayes`, step 3000:

| rho   | h=1 linear, raw   | h=1 affine, raw+asinh | h=6 linear, raw   | h=6 affine, raw+asinh |
|-------|-------------------|-----------------------|-------------------|-----------------------|
| 0.0   | 2.011 / +0.0354   | 1.903 / +0.0333       | 0.088 / -0.0087   | 0.102 / -0.0009       |
| 0.2   | 0.385 / +0.0187   | 0.452 / +0.0183       |                   |                       |
| 0.4   | -0.484 / +0.0037  | -0.265 / +0.0059      |                   |                       |
| 0.6   | -0.471 / -0.0091  | -0.294 / -0.0037      | 0.622 / +0.0070   | 0.314 / +0.0049       |
| 0.8   | 0.888 / -0.0098   | 0.595 / -0.0039       |                   |                       |
| 1.0   | 206.143 / -0.1048 | 1.866 / +0.0024       | 25.656 / -0.0862  | -0.063 / +0.0029      |
| prior | 1.621 / -0.0087   | 0.359 / +0.0041       | 0.631 / -0.0078   | 0.132 / +0.0035       |

(`prior` h = 1 excess at step 1000: 4.660 vs 0.918.) Runs under `results/probe/linear-raw/`
and `results/probe/affine-raw+asinh/`.

Findings:

- **The rho = 1 failure is an output-scale artifact, and the affine head removes it.**
  Excess error at rho = 1, h = 1 goes from 148% to 4.2% (affine, raw) and 1.4% (affine,
  raw+asinh); the persistence gap from -0.078 to -0.009 / -0.003. The windowed path shows
  the same thing (206% to 1.9%). Prior-average excess also falls (2.29% to 0.72%), and the
  affine variants learn much faster (step 300: 6% vs 33%).
- **asinh inputs alone do nothing** for the high-rho end (150.5% at rho = 1, essentially the
  baseline). The input side was not the bottleneck; the bounded output was. On top of the
  affine head asinh is slightly better everywhere, but by margins one seed cannot separate.
- **Closest to Bayes at both grid ends: (affine, raw+asinh)** — rho = 1 gap -0.0029, rho = 0
  gap +0.0747, the smallest of the four at both ends and on `prior` (-0.0006).
- **Not fixed: the low end.** At rho = 0 every variant still over-states persistence at
  h = 1 (+0.075 to +0.138; implied 0.195 vs Bayes 0.120 for the best variant). The affine
  head halves the gap but does not close it, and in the windowed check the two variants are
  identical there (+0.033 vs +0.035). So this part of the gap is not a scalar-scale artifact;
  it may close with training time or be a real feature of what the network learns, and the
  study can interpret it.
- **Worse somewhere:** at rho = 0.4 (and for affine, raw at 0.6) the baseline has the lower
  excess error (-0.529 vs +0.059 / +0.097). A negative excess means beating Bayes at one
  fixed rho by leaning toward it, which costs elsewhere under the prior; it is not a loss of
  ability, but it is a cell where the new heads look worse on the table.

The default is unchanged (`linear`, `raw`); whether to switch is the controller's call.

## 2026-11-17: Default architecture is the affine head with raw+asinh inputs

The grid config (and the fallback defaults in `rt/train.py` / `rt/model.py`, so a config
without the keys means the same thing) now use `head: "affine"`, `inputs: "raw+asinh"`,
by controller ruling after the architecture probe (see the probe section above): the linear
head over a LayerNorm output cannot scale with x_t, which is what left the 3000-step model
at 148% excess at rho = 1. The old linear/raw learning-check outputs are kept at
`results/probe/linear-raw-gate`.

Gate check 3 is now bound to the code it certifies: `python -m rt.gate --learn --device mps`
runs the learning cell as subprocesses and writes `results/gate/learning.json` with the code
hash; the gate passes check 3 only when that hash equals the current one. The gate also
fails if the code hash changes while it runs.

Learning check, A-s-wfull-s0, affine + raw+asinh, 3000 steps, mps (training 516 s, eval 60 s).
Prior set, h = 1, vs bayes: **excess_mse_pct 0.719** (gap -0.0006, kappa +0.0048);
trajectory 78.86 (100), 5.21 (300), 1.42 (1000), 0.72 (3000).

Step 3000, h = 1, bench bayes:

| rho   | gap     | excess_mse_pct |
|-------|---------|----------------|
| 0.0   | +0.0747 | 2.519          |
| 0.2   | +0.0337 | 0.295          |
| 0.4   | -0.0162 | 0.097          |
| 0.6   | -0.0285 | 0.960          |
| 0.8   | -0.0064 | 1.015          |
| 1.0   | -0.0029 | 1.410          |
| prior | -0.0006 | 0.719          |

## 2026-11-17: owner sign-off of the pre-registration

The owner signed off `docs/PREREG.md` with these decisions (PREREG, "Owner decisions"):

- The plateau rule (evaluation error at step 30,000 within 0.1% of step 10,000) no longer
  overrides verdicts. It cannot separate under-training from bias under a learning rate that
  decays to zero. It stays as a printed column. Every primary "different" verdict (H1, H2) and
  every adjusted-positive H3 combination is re-trained at 60,000 steps and counts only if it
  survives; if the budget does not cover the re-training it is reported as "not tested for
  training length".
- H3 uses a Bonferroni adjustment over the eight primary cell-rho combinations.
- The mismatched-prior and mid-sequence-break arms are dropped.
- Analysis code lives in `analysis/`, outside the gated `rt/` and `configs/`.
- Margins two-sided as coded; budget rule never cuts size `l`.

Gate re-run on mains power, 2026-11-17: GATE PASS at code hash `ecb0d185…`. Learning check
reproduced exactly (prior h=1 excess 0.719% at step 3,000), so no new exposure. Then tagged
`prereg-v1`.

## 2026-11-17: cloud pilot (timing only; no diagnostic was looked at)

One RTX A6000 (Thunder Compute, prototyping mode, 8 vCPU), 17:36:46 to 19:15:49 UTC (99 min),
instance 3lnydbl8, deleted. Tests 104 passed, 1 skipped; gate PASS at code hash `ecb0d185…` on
CUDA with the same benchmark numbers and learning check (0.719%) as the Mac. Logs:
`results/cloud/pilot/`.

Training steps/second (A-<size>-<window>-s0, fp32 unless noted; "agg" = combined over jobs):

| cell | 1 job | 4 jobs agg | 8 jobs agg | TF32, 1 job |
|---|---|---|---|---|
| xs full | 44.6 | 58.9 | 60.0 (135.7 with 1 CPU thread per job) | |
| s full | 39.5 | 46.1 | 47.4 (46.6 with 1 thread) | |
| m full | 15.6 | 15.5 | 15.5 | 19.4 |
| l full | 3.53 | 3.32 | out of GPU memory (~10 GB per job) | 5.16 |
| l w16 | 14.7 | 14.1 | | 19.3 |
| l w2 | 22.7 | 40.5 | | |
| m w16 | 32.1 | 60.9 | | |

Evaluation of one checkpoint (size l): w4 286 s cold / 198 s warm (167 s with TF32); full window
65 s; Family B w4 737 s cold (benchmark cache built on the instance CPU) / 197 s warm. Profile
(A-l-w4 warm): ~170 s GPU forward, ~45 s diagnostics, ~30 s input building. Windowed evaluation
cost scales with W (every origin is a separate forward over 220,000 sequences). The benchmark
cache for both families and all windows (70 files, 8 GB) was then built on the Mac in ~10 min.

Projection for the configured grid on this GPU, even with TF32 and xs/s moved to the Mac: about
57 GPU-hours (size l ~49 h, size m ~8 h), against ~28-34 h that $12 buys. Over the cap: per
PREREG section 10 the sweep does not start; the owner decides.

## 2026-11-17: deviation prereg-v1.1, Plan A (budget), decided before any confirmatory run

The pilot projected the configured grid at ~57 GPU-hours, over the $12 sweep cap. Per PREREG
section 10 the owner decided. Owner chose "Plan A" on 2026-11-17. No sweep result existed.

1. **Checkpoints `[3000, 10000, 30000]`** instead of `[100, 300, 1000, 3000, 10000, 30000]`
   (`configs/grid.json`). Saving a checkpoint does not change training (per-step RNG seeds), so
   the step-30,000 models are the same; only the exploratory training-time traces lose their
   three earliest points. The convergence column still compares steps 10,000 and 30,000.
   Changes the code hash: gate re-run on the Mac 2026-11-17, PASS at `a2682c12…` (learning check 0.719%, identical); tagged `prereg-v1.1`.
2. **Cloud runs use TF32 matrix multiplies and one CPU thread per process**, set by
   environment (`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, `OMP_NUM_THREADS=1`), not code. The
   instance gate (learning check) runs under the same settings. Mac runs (mps) are unaffected.
3. **Sizes `xs` and `s` train on the Mac** (PREREG section 10, first step).
4. **Size `m`: seeds 0, 1, 2** (PREREG section 10, second step); verdicts use t with 2 df.
5. **Family B (descriptive only): windows full and 4, seeds 0, 1, 2.** The overlay shows these
   two windows; the other Family B cells are not run and are reported as not run.
6. **Size `l`, Family A (every hypothesis) is not cut** and runs first.

Cloud order: A-l (25 cells), then A-m seeds 0-2 (15), then B-l full and w4 seeds 0-2 (6). Budget
guard $11 for the sweep, leaving room under the $15 cap (pilot ~$0.71) for the training-length
re-checks of section 6.

## 2026-11-18: analysis rule for incomplete seed groups (before any real verdict was computed)

PREREG section 6 makes the seed the unit and computes every verdict from "the five seeds" (three
for size `m` under Plan A), but does not say what happens if a run fails or never finishes.
Rule adopted for `analysis/` (implemented in `analysis/hypotheses.py`): a (family, size, window,
step, rho, h, bench) group with fewer seeds than Plan A schedules is **incomplete**; its numbers
are printed and marked "incomplete", and it never enters a decision. A hypothesis that needs an
incomplete or missing group is "not established", unless a complete group already rejects it.
Reason: a verdict from a subset of seeds would be a different test from the one registered, and
which seeds fail is not random with respect to training difficulty. Cost: four seeds that all
show a difference read "not established" instead of "rejected"; the four seed values are still
printed. Decided on 2026-11-18 by the controller, with the independent reviewer's concurrence,
before `analysis.report` had been run on any real metrics.

## 2026-11-17: Survey comparison conventions (Task 11; exploratory, PREREG section 8)

Fixed before any Family B checkpoint exists or is loaded. Code: `analysis/spf.py`; data:
`data/spf_panel.parquet` (provenance in `data/PROVENANCE.md`).

- Variables and sample: EWH's headline SPF replication set and window (NGDP, RGDP, PGDP, CPI,
  UNEMP, TB3M, TN10Y; 1968Q4-2016Q4), consensus = mean of individual errors and revisions.
- Series fed to the models: first differences of the real-time levels of vintage t -- the
  annualised quarterly percent growth rate for NGDP, RGDP, PGDP and CPI, the change for UNEMP,
  TB3M and TN10Y -- standardised by their mean and standard deviation over every observation
  dated before t. Forecasts are mapped back by loc + scale * F.
- Horizons: the survey target is quarter t+3 (growth from x[t-1] for growth variables). The
  model's last input is d[t-1], so the forecast at t uses horizons 1-4 and the t-1 forecast
  (made from vintage t-1) horizons 2-5 (growth, compounded) or 1-5 (levels, summed onto
  x[t-2]). All within the models' 1-6: horizons match.
- Context: the full-window model gets the last 80 scaled values (its trained context) with time
  indices 1..n; window 4 gets the last 4 with indices 77..80.
- Drop rule range: the central 99% of standardised Family B training values (x - mu) / sigma,
  t = 1..80, rho ~ U[0, 1] (100,000 simulated paths, seed (30000, 0x5BF)): about [-5.72, 5.79].
  A variable-quarter is out of range if any fed value is outside it (or it cannot be scaled);
  the fraction is pooled over variables, per Family B window, over variable-quarters with a
  survey error and revision and a strictly real-time history (see the 2026-11-18 entries for
  the real-time rule, the per-cell rule and the current figures).

## 2026-11-18: Survey comparison, strict real time (review C1, C2; controller ruling)

Decided before any trained Family B model was run on SPF data. Supersedes the 2026-11-17
conventions where they differ.

- A history fed to a model must be vintage t itself (`hist_vintage` == t), ending at t-1, never
  completed from a later vintage. The export no longer fills a late x[t-1] from the next vintage
  (that figure was published after the survey deadline), and no longer uses the first available
  later vintage when vintage t does not exist. `analysis/spf.py` refuses any row with
  `last_filled` True, a vintage other than t, or a missing x[t-1].
- Consequences: NGDP, RGDP and PGDP lose 1996Q1 (no x[1995Q4] in the 1996Q1 vintage) and 1996Q2
  (its revision needs the 1996Q1 forecast); CPI comparison quarters start in 1994Q4 (the first
  real-time CPI vintage is 1994Q3, and the revision needs the t-1 vintage too). The survey's b
  is computed on the same reduced quarters.

## 2026-11-18: Survey comparison, per-cell 20% rule (review I1; controller ruling)

Decided before any trained Family B model was run on SPF data. Reason: the closest reading of
PREREG section 8's "matched cleanly"; under the pooled rule alone, TB3M's full-window cell
(about 55% of its quarters out of range) would have been printed.

- Any variable-window cell whose own out-of-range fraction exceeds 20% is omitted from the
  comparison table (its b is not printed) and listed with the reason. The pooled 20% rule that
  drops the whole arm stays as well.
- Current figures on the real panel (scaled histories only, no model): full window 92.8% in
  range pooled (77 of 1,075 real-time variable-quarters out, all TB3M 1981Q4-2000Q4: the 1980Q2
  and 1980Q4 bill-rate changes reach 6.9 sd and stay in the last 80 values); window 4 99.9%
  (1 out: TB3M 1981Q4). The arm is kept; the TB3M full-window cell (54.6%) is omitted.

## 2026-11-20: sweep spend over the $12 sweep cap (after all runs finished)

All 96 Plan A runs finished (cloud sweep done 2026-11-20 03:06 UTC; last Mac run 03:54 UTC).
Cloud spend, estimated from instance time at $0.35/h: about $14.35 over four instances
(~$2.11, ~$4.29, ~$4.60, ~$3.35), ~41 instance-hours, pilot included. That is over the $12
sweep cap and the $15 total of the Plan A entry is nearly used. Cause: on 2026-11-18 the
watcher's file copy (rsync, no timeout) hung from 08:07 UTC, about 6 GPU-hours of work after
the last checkpoint was lost and the instance was not deleted on time; later instances
resumed from checkpoints. Fixed in commit 71b69df (rsync/ssh timeouts, process-group time
limit). The owner had set an overall hard limit of $20, which held. No cell was added, dropped
or re-run selectively because of the overrun; the grid is exactly Plan A. Nothing was flagged,
so the 60,000-step training-length re-train was not needed and was not run.

## 2026-11-20: venue verdict (plan Task 12 step 2), after H1–H3 were read

**Verdict: preprint (arXiv and SSRN) plus the esnupi report; a workshop submission later.**
Owner asked for SSRN in addition on 2026-11-20.

Reasons, from what H2 and H3 showed:
- H3 is a clean, pre-registered null with tight intervals (no 95% upper end above +0.0008
  against a +0.02 threshold), and H2 holds at every window against an information-matched
  benchmark. The novelty search (docs/novelty_check.md) found no paper that measures the
  expectations-literature slope diagnostics against an exact window-Bayes forecaster across
  context lengths. That is worth a citable, timestamped preprint.
- H1 is a near-replication (Balim et al. 2023; Akram and Vikalo 2024), the evidence is one
  architecture on simulated AR(1) data, and the result is what the squared-error optimum
  predicts once training succeeds. That is not a main-conference contribution on its own.
- arXiv reaches the ML readers (cs.LG primary; econ.GN or q-fin cross-list); SSRN reaches the
  expectations and behavioural-finance readers the paper is in conversation with (AKLMT, BGMS,
  Coibion–Gorodnichenko). Both accept a PDF and allow the same paper on both.
- A workshop on in-context learning or ML for economics is the right next step if the owner
  wants peer review; it would want the evaluation-set bootstrap the review suggested (paper
  section 05 notes that this error is not quantified).

Preconditions before posting: the repository is public with the pre-registration tags (the
paper cites it), the cover's report and issue numbers are set, and the cloud spend is
confirmed in the Thunder console. arXiv may ask a first-time submitter for an endorsement in
cs.LG. Posting is the owner's action; nothing is submitted on the owner's behalf without an
explicit go.

## 2026-11-20: timing of deviations v1.2 to v1.4, resumed runs (final review)

Timing, from the ledger, file times and annotated tag dates (UTC). `prereg-v1.1` was tagged
2026-11-17 19:44; the first confirmatory run started 19:47 and finished 19:51 (A-xs-w2-s0, s1).
Primary cells A-l-w2-s0 to s2 had metrics files by 2026-11-18 00:18. The `analysis/` code was
first committed at 00:33, `prereg-v1.2` (incomplete-seed rule) was tagged 00:45, `v1.3` (SPF
arm) 01:19 and `v1.4` (SPF report section) 01:27. So v1.2 to v1.4 were fixed **after
confirmatory runs had started and their metrics files existed, but before the analysis code had
been run on any real metrics and before any trained model was run on SPF data.** `rt.evaluate`
prints a run's prior-set excess error, gap and kappa to its log, so per-run diagnostics existed
in readable form; no verdict or H1–H3 summary had been computed. The paper states this timing.

Resumed and crashed runs (PREREG section 11). Twelve runs were resumed from a checkpoint after
an interruption (instance deletion, Mac sleep or unplugging): A-l-w4-s1, s2, s3 (from step
10,000); A-m-w16-s0, s1, s2 (3,000); A-s-w4-s1, s2 (3,000); A-s-wfull-s2, s3 (10,000);
A-xs-w8-s2, s3 (3,000). A resume restores model, optimiser and scheduler state, and batches are
seeded per step, so a resumed run follows the same data stream as an uninterrupted one. One
attempt of A-s-wfull-s2 stopped on an MPS internal error (`results/logs/A-s-wfull-s2.log`) and
was resumed; no run failed in the end (all 96 ledger rows have return code 0).

Checkpoints of A-l-w8-s0 to s4 and A-l-w4-s4, and steps 3,000 and 10,000 of A-l-w16-s0, were
lost with a cloud instance; their evaluation CSVs had been copied and are complete.
Cloud evaluation used TF32: re-evaluating two cloud cells in fp32 moves excess error by up to
0.04 points and the gap by up to 2e-4, far inside the margins; Mac cells reproduce to 1e-6.

## 2026-11-20: repository made public

Owner approved. Created https://github.com/ajaiupadhyaya/rational-transformer (public) at 04:54 UTC and pushed `main` (8c30667) and tags
`prereg-v1` to `prereg-v1.4` unchanged. The tags existed only locally until then, so they are the
author's record, not an independent timestamp; the paper says so.

## 2026-11-20: cost accounting corrected (pilot), at the paper's fact-check

The 2026-11-20 spend entry above says "about $14.35 … pilot included". That is not
established. The four sweep instances sum to about $14.35 (~41 instance-hours at $0.35/h); the
first figure (~$2.11) was read from the Thunder console and it is not recorded whether it
included the separate pilot instance 3lnydbl8 (~99 min, "~$0.71" in the pilot entry). If the
pilot was billed separately, total cloud spend is about $15.06. So spend is $14.35 to about
$15.06, at or just over the $15 pre-registered cloud cap (not "within" it), and within the
owner's $20 hard limit either way. The figures are estimates from instance time, not yet
reconciled against the provider's bill. The paper and the report edition state the range.

## 2026-11-20: paper posted (Zenodo published, SSRN submitted)

The academic paper (docs/paper/_output/paper.pdf, commit 2a78e5c, md5
e9e7c3bc65e3b0be3b6ad4bcb9ee7ef6) was published on Zenodo as record 24387529, DOI
10.5281/zenodo.24387529 (concept DOI 10.5281/zenodo.24387528), CC BY 4.0, with the owner's
approval, and submitted to SSRN as abstract 7733526 (preprint, CC BY), now in SSRN's review.
arXiv submission 8401422 is open and waits for an endorsement in cs.LG. The posted PDF says
the DOI is "reserved"; that was true when it was built and the file cannot change on Zenodo.

## 2026-11-20: Zenodo version 2 (title page corrected)

At the owner's request the title page was changed to the standard academic form (no
"Working paper" label, no "This version" line) and the paper now cites the Zenodo concept
DOI 10.5281/zenodo.24387528, which resolves to the latest version. No result, number or text
of the analysis changed. Published as Zenodo version 2, record 24400796, DOI
10.5281/zenodo.24400796 (paper.pdf md5 f02726235997899521dc5ced6ebe6946, commit 032b9d3);
version 1 (24387529) stays as history. SSRN 7733526 still holds the first file; it is to be
replaced through Revise once SSRN approves the submission.
