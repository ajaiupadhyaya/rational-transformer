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
