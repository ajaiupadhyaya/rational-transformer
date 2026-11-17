# Decisions

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
