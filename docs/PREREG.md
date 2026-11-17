# The Rational Transformer: pre-registration

Status: **signed off by the owner on 2026-11-17.** It binds when the commit that holds it is
tagged `prereg-v1`. Authority: the design
(`docs/superpowers/specs/2026-11-16-rational-transformer-design.md`) for the question,
hypotheses, margins and benchmarks; the code under `rt/` and `configs/grid.json` for everything
that is computed. Where the two differ, the difference is listed in the first section, not
resolved silently.

## Owner decisions (2026-11-17)

The draft listed seven open points. The owner approved the following on 2026-11-17, before any
confirmatory run existed. The sections below are written to these decisions.

1. **Gate.** The gate must be green at the tagged code. `results/gate.json` and
   `results/gate/learning.json` certified code hash `5b028b39…`; commit `2c41231` changed
   `rt/gate.py`, `rt/sweep.py` and `rt/train.py`. The learning check and the gate are re-run
   before the tag, and the re-run's numbers are added to section 9 and `docs/DECISIONS.md` as
   one more pre-freeze exposure.
2. **Convergence: a training-length check replaces the plateau override.** The design asks that
   an under-trained cell never be reported as bias. The plateau rule in the code (evaluation
   error at step 30,000 within 0.1% of step 10,000) cannot do that job under a learning rate
   that decays to zero at step 30,000: error can still be falling by more than 0.1% while the
   model is already inside the 1% equivalence margin, so the rule would mark most cells "not
   converged". The `converged` column is still computed and printed, per run and per cell (a
   cell is converged only if all five runs are), but it no longer overrides a verdict. Instead,
   every primary finding of a difference is re-tested at twice the training length (section 6).
3. **Margins are two-sided,** as coded: a model that beats the benchmark at a fixed rho by more
   than the margin is not "equivalent".
4. **Budget rule** as in section 10: small sizes move to the laptop first, then size `m` drops
   to three seeds; size `l` is never cut; if the projection still exceeds the cap the sweep
   does not start and the owner decides.
5. **H3 multiplicity:** Bonferroni across the eight primary cell-rho combinations (section 7).
6. **Scope.** The mismatched-prior and mid-sequence-break arms are dropped from this study; they
   are not built and are not run. All analysis code (verdicts, selection, the H3 rule, the
   human overlay, the survey comparison) lives outside `rt/` and `configs/`, in `analysis/`, so
   that writing it does not change the gated code hash. It must implement sections 6–7 exactly
   as written here. The full-information rational and least-squares forecasts are reported from
   `analysis/` as descriptive rows.
7. **Family B priors and AKLMT.** The Family B priors are fixed (section 2). Whether they cover
   AKLMT's mean and noise ranges is checked when the overlay is built and reported with it. It
   affects only the descriptive overlay.

## 1. Question and logic

Does a transformer trained only to minimise squared forecast error on autoregressive data end
up forecasting like a rational Bayesian, or does it show the over- and underreaction humans
show, and under what constraints?

A rational forecaster who must learn the persistence rho is unbiased only on average over the
prior. At a fixed rho its posterior is pulled toward the prior: its implied persistence sits
above rho when rho is low and below rho when rho is high. Part of the human pattern "more
overreaction when persistence is low" is therefore what a Bayesian does anyway. A forecaster
who sees only the last W observations is pulled further, because it has less data. So a
deviation from the full-information forecast, or even from full Bayes, is not evidence of a
bias: it may be the information the forecaster lacks. **Nothing in this study is called a bias
unless it exceeds the Bayes forecaster with the same prior and the same information, at the same
rho.** For a window-limited network that benchmark is window-Bayes. The sharp test is whether a
trained network ever overreacts where the information-matched Bayesian underreacts (rho = 0.6
and 0.8), which rational learning cannot produce.

## 2. Data-generating processes

Simulation code: `rt/sim.py`. Each path is simulated to t = 86 so that six-step-ahead targets
exist for every forecast origin t = 1..80. Arrays are float64.

**Family A (confirmatory).** x_0 = 0; x_t = rho · x_{t−1} + eps_t, eps_t ~ N(0, 1) i.i.d.;
rho ~ Uniform[0, 1] drawn once per sequence (or fixed, for evaluation sets). Mean zero and unit
noise scale are known to every forecaster. x_0 is never a model input.

**Family B (descriptive only; no hypothesis reads it).** sigma² ~ InvGamma(shape a = 3,
scale b = 2) (mean 1); mu | sigma² ~ N(0, 25 · sigma²); x_0 = mu;
x_t = mu + rho · (x_{t−1} − mu) + sigma · eps_t; rho ~ Uniform[0, 1]. Constants:
`B_PRIOR = dict(a=3.0, b=2.0, k=25.0)` (`docs/DECISIONS.md`, 2026-11-16). x_0 (= mu) is never a
model input.

## 3. Models and training

Everything below is `configs/grid.json`, `rt/model.py` and `rt/train.py` at the tagged commit.

**Architecture.** A pre-LayerNorm causal transformer over real-valued inputs. Each position
embeds its value from the two features [x_s, asinh(x_s)] by one linear layer (`inputs:
"raw+asinh"`) and adds a learned embedding of its true time index s (indices 1..86; index 0 is
padding and is never attended to). Blocks: multi-head self-attention and a 4·d GELU MLP, each
with a residual connection and a LayerNorm before it; a final LayerNorm. Output head
(`head: "affine"`): a linear layer maps the final state at each position to 12 numbers
(a_h, b_h), h = 1..6, and the forecast of x_{t+h} is a_h · x_t + b_h, where x_t is the raw value
at that position. The a_h rows start at zero. No input normalisation.

| Size | Layers | d_model | Heads |
|---|---|---|---|
| xs | 1 | 32 | 2 |
| s | 2 | 64 | 4 |
| m | 4 | 128 | 4 |
| l | 8 | 256 | 8 |

**Windows as total information limits.** W ∈ {2, 4, 8, 16, full}. With window W, the forecast
at origin t is computed from the inputs x[max(1, t − W + 1) .. t] and their true time indices
only, left-padded with (value 0, time 0) to length W; padding is masked from attention, so the
forecast equals the unpadded window's. This is input truncation, not a per-layer attention mask,
so a deep model cannot see further back than W. With the full window the model sees x[1..t] in
one causal pass over x[1..80]. The benchmark is given exactly the same observations and time
indices.

**Training.** Every step draws a fresh batch of 1,024 sequences from the family's prior with the
generator `SeedSequence([seed, 0xA11CE, step])`. Full window: the loss is the squared error
summed over the six horizons, averaged over all 80 positions and the batch. Window W: each
sequence gets one origin t ~ Uniform{1, ..., 80} from the same generator, and the loss is taken
at that origin only (the last window position). Because the generator depends only on the seed
and the step, **all cells of a family at a given seed see the same training paths** (and, for
windowed cells, the same origins): cells are not independent of each other; only seeds are.
Parameter initialisation uses `torch.manual_seed(seed)`.

| Setting | Value |
|---|---|
| Optimiser | AdamW, lr 1e-3, betas (0.9, 0.999) |
| Weight decay | 0.01, applied to **all** parameters (weights, biases, LayerNorm gains and biases, embeddings): `AdamW(model.parameters(), …)` with no parameter groups |
| Schedule | linear warm-up over 500 steps, then cosine to zero at step 30,000 |
| Steps | 30,000 |
| Batch | 1,024 sequences |
| Checkpoints | steps 100, 300, 1,000, 3,000, 10,000, 30,000 |
| Seeds | 0, 1, 2, 3, 4 |
| Horizons | 1–6 (a revision at horizon h needs the horizon-(h+1) forecast from the period before) |

**Grid** (`rt.sweep.grid`). Family A: 4 sizes × 5 windows × 5 seeds = 100 runs. Family B: size
`l` × 5 windows × 5 seeds = 25 runs. 125 runs in all; each run trains and then evaluates every
checkpoint.

## 4. Benchmarks

All are computed on the same evaluation paths as the model (`rt/bayes.py`, `rt/baselines.py`).

- **Full-information rational (FIRE):** rho^h · x_t (Family A); mu + rho^h (x_t − mu)
  (Family B), with the true rho and mu. Reported descriptively; not a row of the metrics CSV and
  not used by H1–H3.
- **Bayes (Family A):** the exact posterior over rho under the training prior U[0, 1] from
  x[1..t], with x_0 = 0 known; forecast E[rho^h | data] · x_t. Integration on 2,001 equally
  spaced rho points on [0, 1] by the trapezoid rule (weight ½ at the two end points, 1
  elsewhere, in numerator and denominator alike).
- **Window-Bayes (Family A):** the same posterior from x[s..t], s = max(1, t − W + 1), and t.
  The first observation in the window enters with its exact marginal likelihood given rho and
  its time index: x_s ~ N(0, V_s(rho)), V_s(rho) = sum_{k<s} rho^{2k}. It is the best forecast
  available to anyone who has exactly the network's information. For the full window it is
  Bayes.
- **Bayes and window-Bayes (Family B):** mu and sigma² integrated analytically under the
  Normal-Inverse-Gamma prior of section 2; rho on the same 2,001-point trapezoid grid; forecast
  E[rho^h x_t + (1 − rho^h) E(mu | rho, data) | data].
- **Least-squares learner:** OLS of x_s on x_{s−1} with an intercept over x[1..t], iterated h
  steps (copied from EWH `ewh/sim/benchmarks.py` @ 57e9084); undefined for t < 3. Reported
  descriptively for continuity with EWH; not used by H1–H3.

The gate (`rt/gate.py`) has checked that Bayes' own error-on-revision coefficient is zero on
prior-drawn data, that a point prior reproduces FIRE, and that on h = 1 squared error
Bayes ≤ window-Bayes for every window and window-Bayes ≤ windowed least squares (W = 8, 16), in
both families (numbers in section 9).

## 5. Evaluation sets and diagnostics

**Sets** (`rt/evaluate.py`). Per family: six fixed-rho sets, rho ∈ {0, 0.2, 0.4, 0.6, 0.8, 1},
of 20,000 sequences each, and one prior set of 100,000 sequences with rho ~ U[0, 1]. Set i (in
that order, prior = 6) is drawn from `SeedSequence([10000 + i, tag])`, tag 0xE0A (Family A) or
0xE0B (Family B), a stream separate from every training stream. The same sets are used for
every run.

**Positions and horizons.** All diagnostics pool (sequence, t) pairs over forecast origins
t = 41..80 and drop pairs with a missing value. Horizons h = 1..6 (error-on-revision h = 1..5).

**Notation.** F[t, h] is a forecaster's forecast of x_{t+h} made at t. slope(y on z) is the OLS
slope of y on z with an intercept, pooled as above. For each run, checkpoint, rho set, horizon
and benchmark B ∈ {bayes, window_bayes}, one row of `results/metrics/<cell>.csv` holds:

| Column | Definition (`rt/diagnostics.py`) |
|---|---|
| `implied_persistence` | IP_h(model) = slope(F_model[t, h] on x_t) |
| `bench_implied_persistence` | IP_h(B) |
| `gap` | IP_h(model) − IP_h(B). Positive: the model reacts more to the current value than the benchmark with the same information (overreaction beyond it) |
| `excess_mse_pct` | 100 · (mean (x_{t+h} − F_model[t, h])² / mean (x_{t+h} − F_B[t, h])² − 1), over pairs where both forecasts exist |
| `eor_b`, `bench_eor_b` | slope((x_{t+h} − F[t, h]) on (F[t, h] − F[t−1, h+1])), h = 1..5, for the model and for B (BGMS individual-level form). Missing at h = 6 |
| `kappa` | slope((F_model[t, 1] − F_bayes[t, 1]) on (x_t − F_bayes[t−1, 1])); only in rows with B = bayes, h = 1 |
| `converged` | the run-level rule of section 6 |

plus `cell_id, family, size, window ("full" or W), seed, step, rho (0.0 … 1.0 or "prior"), h,
bench`. For the full window the `window_bayes` rows equal the `bayes` rows.

**Family B standardisation.** Every Family B diagnostic, including every squared-error column
and the convergence error, is computed on standardised deviations x̃ = (x − mu)/sigma and
F̃ = (F − mu)/sigma using each sequence's true mu and sigma, identically for the model and every
benchmark. Reason: Family B sequences differ in level and scale, so a regression pooled over
sequences on raw values would mostly measure between-sequence differences in the mean, not how
a forecast responds to news within a sequence. Family A has mu = 0 and sigma = 1 and is used
raw.

## 6. Unit of inference and decision rule

**Unit.** The seed. For a given family, size, window, checkpoint, rho, horizon, benchmark and
column, the five seeds give five numbers. The interval is the 95% Student-t interval of their
mean: mean ± t_{0.975, 4} · sd / √5, t_{0.975, 4} = 2.776. Seeds are independent of each other;
cells at the same seed are not (same training paths), so no statement treats two cells as
independent replications. All seeds are scored on the same evaluation sets, so sampling error of
the evaluation sets is common to every seed and is not in the interval; the sets (20,000 × 40
pairs per rho, 100,000 × 40 for the prior) are sized to make it small relative to the margins,
and the gap and excess-error columns compare model and benchmark on the same pairs.

**Margins** (fixed in the design). `excess_mse_pct`: 1 (percentage point). `gap`: 0.02.

**Verdict** (`rt.diagnostics.verdict`, with lo and hi the interval's ends and m the margin):
- **equivalent** if −m < lo and hi < m;
- **different** if lo > m or hi < −m (the sign of the mean is reported with it);
- **inconclusive** otherwise.

Both margins are two-sided (owner decision 3).

**Convergence column (descriptive).** A run is converged if the model's squared error at h = 1
on the prior evaluation set (standardised for Family B) at step 30,000 is within 0.1% (relative)
of its value at step 10,000 (`rt.evaluate.evaluate_all`; `converged` is False if either
checkpoint is missing). A cell (family, size, window) is converged only if all five of its runs
are. The column is printed for every run and cell. It does not change a verdict (owner
decision 2).

**Training-length check (binding).** Under-training is separated from bias by training longer,
not by a plateau test. A *flagged finding* is, at size `l`: any "different" verdict in H1 or
H2, or any H3 combination that is positive after the adjustment. For every cell with a flagged
finding, all five seeds are re-trained from scratch under a configuration identical to
`configs/grid.json` except `max_steps` = 60,000 (warm-up 500, cosine to zero at 60,000) and a
checkpoint added at 60,000, written to a separate results directory, and evaluated on the same
evaluation sets. The finding **survives** if the same verdict holds at step 60,000 in the same
column, at the same rho, with the same sign (for H3: positive after the same adjustment). Then:

- survives: the finding stands and is reported with both training lengths;
- does not survive: reported as **"not robust to training length"**, never as a bias. For H1
  and H2 the hypothesis is then "not established" at that cell, unless the 60,000-step verdicts
  are all "equivalent", which is reported as such next to the 30,000-step result but does not
  turn the pre-registered 30,000-step decision into "supported";
- not run, because the budget cap does not cover it: reported as **"different at 30,000 steps;
  not tested for training length"**, and no claim of bias is made from it.

"Equivalent" and "inconclusive" verdicts are not re-tested. The doubled configuration is a new
effective config, so it needs its own green gate; it is logged as a deviation tag
(`prereg-v1.x`, section 10) with nothing else changed. If more cells are flagged than the
budget covers, they are run in this order: H3 positives, then H1, then H2 from the smallest
window up.

**Checkpoint.** Every confirmatory verdict uses step 30,000, the last checkpoint.

## 7. Hypotheses

Each hypothesis reads only the rows named. "Every grid rho" means rho ∈ {0.0, 0.2, 0.4, 0.6, 0.8,
1.0}; the `prior` row is printed alongside and is not part of any decision.

### H1 (near-replication): the largest model with the full window is Bayes

**Rows:** `family = A, size = l, window = full, bench = bayes, h = 1, step = 30000`, every grid
rho; columns `excess_mse_pct` and `gap`. That is 6 rho × 2 columns = 12 verdicts.

**Decision:** H1 is **supported** if all 12 are "equivalent". It is **rejected** if any is
"different"; the rho, column and sign are the finding. Otherwise (any "inconclusive") H1 is **not established**. A "different" verdict is subject to the
training-length check of section 6 before it is reported as a finding. Requiring all 12 is an intersection test, so no
multiplicity adjustment is needed.

### H2 (information or learning): window-limited models against window-Bayes

**Rows:** for each W ∈ {2, 4, 8, 16} separately: `family = A, window = W, bench = window_bayes,
h = 1, step = 30000`, every grid rho; columns `excess_mse_pct` and `gap`; 12 verdicts per
window. **Size `l` is primary.** Sizes `xs`, `s` and `m` get the same verdicts, reported in the
same table, and do not enter the decision. Two-sided.

**Decision and meaning, per window:**
- All 12 "equivalent": **rational given its information.** Everything the window-W network gets
  "wrong" relative to full Bayes is the information limit.
- Any "different": **a departure from rational use of the window.** The finding is the column,
  sign and the rho where it appears. A positive gap is overreaction beyond window-Bayes, a
  negative gap underreaction beyond it. A positive excess error with an equivalent gap is a
  forecast that responds to x_t correctly on average but is noisier than window-Bayes; a negative
  excess error beyond −1 is beating window-Bayes at that rho by leaning toward it, which must
  cost elsewhere under the prior.
- Otherwise: **not established** at that window ("inconclusive").

Every "different" verdict at size `l` is subject to the training-length check of section 6.

The four windows are reported separately; no pooled statement across windows is made.

### H3 (the human signature): overreaction where the matched Bayesian underreacts

**Rows:** `family = A, window ∈ {2, 4, 8, 16}, bench = window_bayes, h = 1, step = 30000,
rho ∈ {0.6, 0.8}`, column `gap`. **Size `l` is primary:** 4 windows ×
2 rho = **8 cell-rho combinations.** A combination is **positive** if the lower end of the 95%
interval of the mean gap is above +0.02.

**The rule (owner decision 5).** H3 is **supported** only if at least one of the 8 primary
combinations is positive after a Bonferroni adjustment for 8: the lower end of the
1 − 0.05/8 (99.375%) Student-t interval, mean − 5.261 · sd / √5, is above +0.02. The
denominator is always 8. A combination that is positive after the adjustment is a flagged
finding and counts as support only if it survives the training-length check of section 6. A combination that is
positive at 95% but not after the adjustment is reported as "k of 8 positive at 95%, none after
adjustment", not as support. The other sizes (xs, s, m: 24 more combinations) are reported with
their count of positives at 95% out of 24, descriptively. A single positive anywhere is reported
with the number of combinations it was found among, never as a headline on its own.

The full window is covered by H1: a "different" verdict with a positive gap at rho = 0.6 or 0.8
there would be reported under H1.

## 8. Exploratory analyses (labelled as such wherever reported)

- **Training-time traces** of `gap` and `excess_mse_pct` across the six checkpoints: does a
  network pass through an overreacting phase before converging?
- **Model size**: the size profile of all diagnostics, beyond the H2/H3 reporting.
- **Horizons above 1**: `gap`, `excess_mse_pct` and `eor_b` against `bench_eor_b` for h = 2..6
  (h = 2..5 for error-on-revision), and `kappa` (D4).
- **The rho = 0 behaviour seen before the freeze** (section 9): at h = 1 the small model
  overstated persistence at rho = 0 against Bayes (+0.075 to +0.138, depending on architecture).
  Whether it closes with training and size is described, not tested.
- **Mismatched prior** and **mid-sequence break** in rho, listed as exploratory in the design:
  **dropped** (owner decision 6). Not built, not run; the write-up says so.
- **Family B human overlay (descriptive):** implied persistence against rho for Family B size
  `l` at each window, with the Family B Bayes curve and the AKLMT human-subject curve copied
  from EWH with provenance.
- **Survey comparison (illustrative, droppable):** histories of the SPF variables already
  replicated in EWH are fed to the Family B models, scaled using only data before each forecast
  date; b is computed at the horizons that match the survey and printed beside the SPF's b for
  the same variable-quarters. **Drop rule:** if horizons or scaling cannot be matched cleanly,
  the arm is dropped and the write-up says so. Real series are not AR(1) and the models never
  saw them; nothing here is a test.

## 9. Everything seen before the freeze

All of the following were seen before the tag. Every model result is **one seed (0) of size
`s`, trained 3,000 steps** under a schedule whose cosine ends at step 3,000, so none of these runs
is part of the confirmatory grid as configured (30,000 steps). The confirmatory size-`s`, seed-0
cells will see the same training paths (same seed), with a different learning-rate schedule
after step 500; size `s` is not primary for any hypothesis. Source: `docs/DECISIONS.md`.

**Gate, benchmark and planted checks** (`results/gate.json`, code hash `5b028b39…`).
Bayes' error-on-revision b at h = 1 on 100,000 prior-drawn sequences: +0.00096 (Family A),
+0.00048 (Family B); tolerance ±0.01. Squared error at h = 1 (20,000 sequences, Family B
standardised), Bayes / window-Bayes for W = 2, 4, 8, 16: Family A 1.0143 / 1.0850, 1.0737,
1.0584, 1.0398; Family B 1.0326 / 1.2677, 1.2092, 1.1575, 1.1038. Windowed least squares:
Family A 1.1926 (W = 8), 1.0746 (W = 16); Family B 1.4763, 1.1770. Planted kappa values −0.3, 0,
0.2, 0.5 and a planted gap of 0.15 were recovered exactly. Bayes' own `eor_b` on the Family A
prior evaluation set: −0.0007.

**First learning check: linear head, raw inputs** (cell A-s-wfull-s0). Prior set, h = 1,
excess over Bayes: 123.06% (step 100), 33.03% (300), 5.10% (1,000), 2.29% (3,000). Passed the
gate's < 5% bar; `converged` = False. Step 3,000, h = 1, against Bayes:

| rho | gap | excess_mse_pct |
|---|---|---|
| 0.0 | +0.1381 | 5.320 |
| 0.2 | +0.0771 | 0.827 |
| 0.4 | −0.0106 | −0.529 |
| 0.6 | −0.0630 | 1.055 |
| 0.8 | −0.0248 | 2.812 |
| 1.0 | −0.0780 | 148.174 |
| prior | −0.0054 | 2.290 |

At rho = 1 the model failed: implied persistence 0.907 against Bayes 0.985, 148% excess error.

**Architecture probe** (A-s-wfull-s0, 3,000 steps, one run per variant), step 3,000, against
Bayes, `excess_mse_pct / gap`:

h = 1:

| rho | linear, raw | affine, raw | linear, raw+asinh | affine, raw+asinh |
|---|---|---|---|---|
| 0.0 | 5.320 / +0.1381 | 2.897 / +0.0831 | 5.024 / +0.1303 | 2.519 / +0.0747 |
| 0.2 | 0.827 / +0.0771 | 0.291 / +0.0374 | 0.750 / +0.0684 | 0.295 / +0.0337 |
| 0.4 | −0.529 / −0.0106 | 0.059 / −0.0202 | −0.264 / −0.0138 | 0.097 / −0.0162 |
| 0.6 | 1.055 / −0.0630 | 1.171 / −0.0348 | 1.303 / −0.0520 | 0.960 / −0.0285 |
| 0.8 | 2.812 / −0.0248 | 1.190 / −0.0081 | 2.244 / −0.0190 | 1.015 / −0.0064 |
| 1.0 | 148.174 / −0.0780 | 4.181 / −0.0085 | 150.514 / −0.0806 | 1.410 / −0.0029 |
| prior | 2.290 / −0.0054 | 0.838 / −0.0019 | 2.162 / −0.0066 | 0.719 / −0.0006 |

Implied persistence at rho = 1, h = 1 (Bayes 0.985): 0.907, 0.976, 0.904, 0.982.

h = 6:

| rho | linear, raw | affine, raw | linear, raw+asinh | affine, raw+asinh |
|---|---|---|---|---|
| 0.0 | 0.197 / −0.0301 | 0.033 / +0.0015 | 0.067 / −0.0057 | 0.030 / +0.0030 |
| 0.6 | 0.597 / +0.0281 | 0.296 / +0.0105 | 0.493 / +0.0183 | 0.284 / +0.0107 |
| 1.0 | 18.295 / −0.0577 | 1.046 / +0.0010 | 14.967 / −0.0501 | 0.765 / −0.0014 |
| prior | 0.662 / −0.0075 | 0.268 / +0.0010 | 0.569 / −0.0056 | 0.237 / +0.0018 |

Prior-set h = 1 excess at step 1,000: 5.101, 1.766, 5.338, 1.415; at step 300: 33.03, 6.13,
34.91, 5.21 (same column order).

Windowed probe, A-s-w4-s0, 3,000 steps, step 3,000, against window-Bayes,
`excess_mse_pct / gap`:

| rho | h=1 linear, raw | h=1 affine, raw+asinh | h=6 linear, raw | h=6 affine, raw+asinh |
|---|---|---|---|---|
| 0.0 | 2.011 / +0.0354 | 1.903 / +0.0333 | 0.088 / −0.0087 | 0.102 / −0.0009 |
| 0.2 | 0.385 / +0.0187 | 0.452 / +0.0183 | | |
| 0.4 | −0.484 / +0.0037 | −0.265 / +0.0059 | | |
| 0.6 | −0.471 / −0.0091 | −0.294 / −0.0037 | 0.622 / +0.0070 | 0.314 / +0.0049 |
| 0.8 | 0.888 / −0.0098 | 0.595 / −0.0039 | | |
| 1.0 | 206.143 / −0.1048 | 1.866 / +0.0024 | 25.656 / −0.0862 | −0.063 / +0.0029 |
| prior | 1.621 / −0.0087 | 0.359 / +0.0041 | 0.631 / −0.0078 | 0.132 / +0.0035 |

Prior-set h = 1 excess at step 1,000: 4.660 (linear, raw) and 0.918 (affine, raw+asinh).

**The architecture choice.** From the probe the controller switched the default to the affine
head with [x, asinh x] inputs (DECISIONS, 2026-11-17). The reason was an output-scale artifact: a
linear head reading a LayerNorm output cannot scale its forecast with x_t, which is what left
the linear-head model at 148% excess error at rho = 1 (206% in the W = 4 run); the affine head
removed it (1.4% and 1.9%). asinh inputs alone did nothing at rho = 1. The choice was made on
these tables, which include numbers at rho = 0.6 and 0.8 and in a windowed cell, the regions H2
and H3 read. Two things limit what that exposure can do: the choice was made on a scale failure
at rho = 1, not on the sign of the gap at 0.6–0.8 (where every variant, full window and W = 4,
showed a small negative gap at h = 1), and the confirmatory claims are about **this architecture**. They do not say that
transformers in general, or this network with a linear head, behave this way.

**Second learning check: affine head, raw+asinh inputs** (A-s-wfull-s0, 3,000 steps; the gate's
learning evidence). Prior set, h = 1, against Bayes: excess 0.719% (gap −0.0006, kappa +0.0048);
trajectory 78.86% (step 100), 5.21% (300), 1.42% (1,000), 0.72% (3,000). Step 3,000, h = 1,
against Bayes: rho 0.0: gap +0.0747, excess 2.519; 0.2: +0.0337, 0.295; 0.4: −0.0162, 0.097;
0.6: −0.0285, 0.960; 0.8: −0.0064, 1.015; 1.0: −0.0029, 1.410. (These equal the probe's
"affine, raw+asinh" column.)

Nothing from sizes `xs`, `m` or `l`, from any seed other than 0, from Family B models, from
windows other than full and 4, or from 30,000-step training has been seen. The gate was re-run
before the tag (owner decision 1):

**Gate re-run, 2026-11-17** (`results/gate.json`, code hash `ecb0d185…`, the tagged code; mps).
GATE PASS. Benchmark and planted checks gave the same values as listed above. The learning check
(A-s-wfull-s0, affine head, raw+asinh, 3,000 steps) reproduced the second learning check exactly:
prior set, h = 1, excess over Bayes 78.862% (step 100), 5.208% (300), 1.415% (1,000), 0.719%
(3,000); gap −0.0006, kappa +0.0048 at step 3,000. Training is deterministic for a given seed and
device, so this re-run exposed nothing new.

## 10. Deviations and budget

**Deviations.** After the tag, any change to code under `rt/`, to `configs/`, or to the analysis
defined in sections 5–8 is logged in `docs/DECISIONS.md` with its reason and the date, committed,
and tagged `prereg-v1.1`, `prereg-v1.2`, …. The write-up lists every deviation and says whether
it was made before or after any confirmatory diagnostic was seen. A changed code hash re-opens
the gate.

**Budget.** Cloud spend is capped at $15 of the $20 Thunder Compute credit. A pilot on the cloud
GPU measures time per run and how many runs share the GPU, and projects the total. If the
projected cost exceeds the cap: first, sizes `xs` and `s` are moved to the laptop (Family A, all
windows and seeds); then, if still over, size `m` drops from 5 seeds to 3 (seeds 0, 1, 2), and
its verdicts use the t interval with 2 degrees of freedom (t_{0.975, 2} = 4.303). Size `l` is not
cut. If the projection still exceeds the cap, the sweep does not start and the owner decides.
These choices are made from the pilot's timing, before any confirmatory diagnostic is seen, and
are logged as deviations. The sweep's budget guard stops launching runs at the cap; runs it did
not launch are reported as not run.

## 11. Reported regardless of outcome

- Every verdict of H1, H2 and H3 as defined, including "inconclusive", with the seed mean, the
  interval and the five seed values; for every flagged finding, the outcome of the
  training-length check or the statement that it was not run.
- The full metrics CSVs for every run and checkpoint, and the convergence status of every run
  and cell.
- The H2 tables for all four sizes and the H3 counts for every size.
- Runs that failed or were not launched, the cost ledger and the hardware each run used.
- Every deviation from this document.
- The pre-freeze exposure of section 9.
- The Family B overlay and the survey comparison, or the statement that an arm was dropped and
  why.
