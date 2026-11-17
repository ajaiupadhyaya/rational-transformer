# The Rational Transformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train small transformers on AR(1) forecasting and measure, against exact Bayes benchmarks, whether they are rational or show human-like expectation biases.

**Architecture:** A Python package `rt` with a simulator, exact grid-posterior benchmarks, a small transformer, a set of diagnostics, and a gate-bound resumable sweep launcher. Every run writes one row-per-(cell, seed, checkpoint, rho, horizon) CSV; the report is built only from those CSVs. The gate and the diagnostics are built and proven locally before any cloud spend.

**Tech Stack:** Python 3.12, uv, PyTorch, NumPy, pandas, matplotlib, pytest. Thunder Compute (one A100) for the sweep.

**Spec:** `docs/superpowers/specs/2026-11-16-rational-transformer-design.md` (copied into the repo in Task 1).

## Global Constraints

- Repo: `~/Documents/rational-transformer`. `git init`, then set `user.email` to `ajaiupadhyaya@users.noreply.github.com` **before the first commit**. Work on `main`.
- `uv` only (`uv init`, `uv add`, `uv run`). Never pip.
- Thunder Compute: **hard cap $15 spend, $5 reserve.** Price read from the live pricing page and quoted to AJ before an instance is created. AJ creates the account and token. Instance deleted once results are copied off and verified.
- Local jobs: at most 4 processes, `nice`, battery guard (stop under 30% battery), resumable.
- Never edit `~/Documents/expectations-without-hindsight` (EWH). Copy from pinned commit `57e9084` with a provenance header naming file and commit.
- The public repo must not contain the AKLMT replication package; only the derived table `aklmt_figure2.csv`.
- Equivalence margins, fixed: excess squared error under 1%; |implied-persistence gap| under 0.02.
- Evaluation positions: t = 41..80 for both families. Horizons 1..6. AKLMT grid `RHOS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)`.
- No hypothesis or analysis change after tag `prereg-v1`. Any bound-code change after the tag is a logged deviation and a new tag.
- **Stop points (AJ must say go):** end of Task 8 (pre-registration sign-off), Task 9 step 3 (price and projection before the sweep), Task 12 step 3 (making the repo public).
- End each working session with done/total/left and the next task.

## Review Focus

1. **rho = 1.** The first-observation variance `(1 - rho^(2s)) / (1 - rho^2)` is 0/0 at rho = 1; it must equal `s`. Tested in Task 2.
2. **Stale gate.** Any edit to `rt/` or `configs/` after the gate passed must make the sweep refuse to start. Tested in Task 7.
3. **Interrupted sweep.** A run killed mid-training (instance stopped, laptop closed) must resume or restart cleanly, and a half-written results file must never be read as complete. Tested in Task 7.
4. **Budget overrun.** The sweep must stop launching runs once elapsed time × hourly price reaches the budget, even if runs remain. Tested in Task 7.
5. **History shorter than the window.** At t < W the model and window-Bayes must both use the same shorter prefix. Tested in Tasks 2 and 6.

---

## File map

| File | Responsibility |
|---|---|
| `rt/sim.py` | Draw Family A and Family B sequences |
| `rt/bayes.py` | Exact Bayes and window-Bayes forecasts, both families |
| `rt/baselines.py` | Full-information rational and least-squares learner (copied from EWH) |
| `rt/diagnostics.py` | Slopes, error-on-revision, kappa, excess error, equivalence verdict |
| `rt/model.py` | The transformer |
| `rt/train.py` | Train one cell, checkpoint, convergence flag |
| `rt/evaluate.py` | Forecast arrays for a model and its benchmarks → metrics rows |
| `rt/gate.py` | The three gate checks, `results/gate.json` |
| `rt/sweep.py` | Grid, gate binding, resume, budget guard |
| `rt/human.py` | AKLMT human curve |
| `rt/spf.py` | Real-data comparison |
| `rt/report.py` | Figures and `RESULTS.md` |
| `configs/grid.json` | Sizes, windows, seeds, training settings |
| `cloud/` | Thunder Compute setup, push, pull scripts and notes |
| `docs/PREREG.md`, `docs/DECISIONS.md`, `docs/novelty_check.md` | Frozen hypotheses, decision log, literature check |

Shared conventions used by every task:

- A batch of sequences is `x: np.ndarray` of shape `(n, T+1)` with `x[:, 0]` the starting value and `T = 86` (80 forecast positions plus 6 steps of targets).
- A forecast array is `F: np.ndarray` of shape `(n, 81, 6)`: `F[i, t, h-1]` is the forecast of `x[i, t+h]` made at time `t`. Positions not forecast are NaN.
- `window=None` means full history.

---

### Task 1: Repo and simulator

**Files:**
- Create: `pyproject.toml`, `rt/__init__.py`, `rt/sim.py`, `tests/test_sim.py`, `README.md`, `docs/DECISIONS.md`
- Copy in: the spec and this plan under `docs/superpowers/`

**Interfaces:**
- Produces:
  - `@dataclass Batch: x: np.ndarray (n, T+1); rho: np.ndarray (n,); mu: np.ndarray (n,); sigma: np.ndarray (n,)`
  - `sample_family_a(n: int, rng: np.random.Generator, rho: float | None = None, T: int = 86) -> Batch` — `x_0 = 0`, `mu = 0`, `sigma = 1`, `rho ~ U[0,1]` unless given.
  - `sample_family_b(n, rng, rho=None, T=86) -> Batch` — `sigma² ~ InvGamma(a=3, b=2)`, `mu | sigma² ~ N(0, 25·sigma²)`, `x_0 = mu`, `x_t = mu + rho·(x_{t-1} - mu) + sigma·eps`.
  - Constants `RHOS`, `T_EVAL = range(41, 81)`, `HORIZONS = (1, 2, 3, 4, 5, 6)`, `B_PRIOR = dict(a=3.0, b=2.0, k=25.0)`.

- [ ] **Step 1:** Create the repo per Global Constraints; `uv init`, `uv add numpy pandas torch matplotlib`, `uv add --dev pytest`.
- [ ] **Step 2: Write failing tests**

```python
def test_family_a_moments():
    b = sample_family_a(200_000, np.random.default_rng(0), rho=0.6)
    assert b.x.shape == (200_000, 87) and np.all(b.x[:, 0] == 0)
    # stationary variance 1/(1-0.36) = 1.5625, reached by t=80
    assert abs(b.x[:, 80].var() - 1.5625) < 0.03
    assert abs(np.corrcoef(b.x[:, 79], b.x[:, 80])[0, 1] - 0.6) < 0.01

def test_family_a_random_walk_variance_grows():
    b = sample_family_a(200_000, np.random.default_rng(1), rho=1.0)
    assert abs(b.x[:, 50].var() - 50) < 1.0

def test_family_a_prior_draw():
    b = sample_family_a(100_000, np.random.default_rng(2))
    assert 0 <= b.rho.min() and b.rho.max() <= 1 and abs(b.rho.mean() - 0.5) < 0.01

def test_family_b_centres_on_mu():
    b = sample_family_b(200_000, np.random.default_rng(3), rho=0.0)
    assert abs(np.mean(b.x[:, 40] - b.mu)) < 0.02
    assert abs(np.mean(b.sigma**2) - 1.0) < 0.05   # InvGamma(3,2) mean = 1
```

- [ ] **Step 3:** Run `uv run pytest tests/test_sim.py -v`; expect import failure.
- [ ] **Step 4:** Implement `rt/sim.py`.
- [ ] **Step 5:** Run the tests; expect 4 passed.
- [ ] **Step 6:** Commit: `feat: repo scaffold and AR(1) simulator`.

---

### Task 2: Bayes and window-Bayes, Family A

**Files:** Create `rt/bayes.py`, `rt/baselines.py`, `tests/test_bayes_a.py`

**Interfaces:**
- Consumes: `Batch`, constants from Task 1.
- Produces:
  - `first_obs_var(rho: np.ndarray, s: int) -> np.ndarray` — `sum_{k<s} rho^(2k)`, equal to `s` at rho = 1.
  - `bayes_a(x: np.ndarray, window: int | None, grid: int = 2001, prior: tuple[float, float] = (0.0, 1.0)) -> np.ndarray` — returns `F` of shape `(n, 81, 6)` for t = 1..80. Posterior over rho on `grid` points uniform on `prior`. With `window=W` at time t, uses observations `x[max(1, t-W+1) .. t]`: the likelihood of the first of them given `(rho, s)` is `N(0, first_obs_var(rho, s))` where `s` is its time index, then one-step transitions. Full window is the case `s = 1`. Forecast `E[rho^h | data] · x_t`. float64, log-space.
  - `fire(x, rho, mu=None) -> F` — `rho^h · x_t` (with mean: `mu + rho^h (x_t - mu)`).
  - `ls_learner(x, intercept: bool) -> F` — copied from EWH `ewh/sim/benchmarks.py` at `57e9084`, adapted to the `F` shape.

- [ ] **Step 1: Write failing tests**

```python
def test_first_obs_var_at_one():
    assert first_obs_var(np.array([1.0]), 7)[0] == 7.0
    assert abs(first_obs_var(np.array([0.5]), 50)[0] - 1 / 0.75) < 1e-9

def test_point_prior_equals_fire():
    b = sample_family_a(500, np.random.default_rng(0), rho=0.8)
    F = bayes_a(b.x, None, grid=3, prior=(0.8, 0.8))
    assert np.nanmax(np.abs(F - fire(b.x, 0.8))) < 1e-9

def test_bayes_error_orthogonal_to_revision_under_prior():
    b = sample_family_a(100_000, np.random.default_rng(1))       # rho from prior
    F = bayes_a(b.x, None)
    assert abs(eor_b(F, b.x, h=1)) < 0.01      # eor_b lands in Task 4; inline OLS here

def test_bayes_beats_ls_and_window_is_no_better_than_full():
    b = sample_family_a(50_000, np.random.default_rng(2))
    mse = lambda F: np.nanmean((b.x[:, 42:82] - F[:, 41:81, 0]) ** 2)
    full, w4, ls = bayes_a(b.x, None), bayes_a(b.x, 4), ls_learner(b.x, intercept=False)
    assert mse(full) < mse(w4) and mse(full) < mse(ls)

def test_window_longer_than_history_equals_full():
    b = sample_family_a(200, np.random.default_rng(3))
    assert np.allclose(bayes_a(b.x, 16)[:, 1:16], bayes_a(b.x, None)[:, 1:16])

def test_boundaries_do_not_nan():
    for rho in (0.0, 1.0):
        b = sample_family_a(200, np.random.default_rng(4), rho=rho)
        assert np.isfinite(bayes_a(b.x, 2)[:, 1:81]).all()
```

- [ ] **Step 2:** Run; expect failures.
- [ ] **Step 3:** Implement. Vectorise over sequences and grid; process time steps cumulatively for the full window and by slice for windows.
- [ ] **Step 4:** Run `uv run pytest tests/test_bayes_a.py -v`; expect 6 passed.
- [ ] **Step 5:** Commit: `feat: exact Bayes and window-Bayes for Family A`.

---

### Task 3: Bayes and window-Bayes, Family B

**Files:** Modify `rt/bayes.py`; create `tests/test_bayes_b.py`

**Interfaces:**
- Produces: `bayes_b(x, window: int | None, grid: int = 401, prior=B_PRIOR) -> F`. Grid over rho only. Given rho, the model is a Gaussian linear regression in `mu` with unknown variance and the Normal–Inverse-Gamma prior from Task 1, so the marginal likelihood and `E[mu | rho, data]` are closed-form. Rows: each transition `x_j - rho·x_{j-1} = mu·(1 - rho) + sigma·eps`; the first observation in the window at time index `s` is `x_s = mu + sigma·sqrt(first_obs_var(rho, s))·eps` (for the full window the first row is the transition from `x_0 = mu`, i.e. `x_1 = mu + sigma·eps`). Forecast: posterior mean over rho of `rho^h·x_t + (1 - rho^h)·E[mu | rho, data]`.

- [ ] **Step 1: Write failing tests**

```python
def test_b_error_orthogonal_to_revision_under_prior():
    b = sample_family_b(100_000, np.random.default_rng(0))
    assert abs(eor_slope(bayes_b(b.x, None), b.x, 1)) < 0.01

def test_b_beats_ls_with_intercept_and_window():
    b = sample_family_b(50_000, np.random.default_rng(1))
    mse = lambda F: np.nanmean((b.x[:, 42:82] - F[:, 41:81, 0]) ** 2)
    full = bayes_b(b.x, None)
    assert mse(full) < mse(bayes_b(b.x, 8)) and mse(full) < mse(ls_learner(b.x, intercept=True))

def test_b_matches_brute_force_on_tiny_case():
    # 3-D grid posterior over (rho, mu, sigma) on 5 sequences, t = 6, horizon 1
    b = sample_family_b(5, np.random.default_rng(2))
    assert np.allclose(bayes_b(b.x, None)[:, 6, 0], brute_force_b(b.x, t=6), rtol=2e-3)
```

`brute_force_b` lives in the test file: an explicit 3-D grid, no conjugacy.

- [ ] **Step 2:** Run; expect failures.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run; expect 3 passed.
- [ ] **Step 5:** Commit: `feat: exact Bayes for Family B`.

---

### Task 4: Diagnostics and verdicts

**Files:** Create `rt/diagnostics.py`, `tests/test_diagnostics.py`

**Interfaces:**
- Produces (all pool over `T_EVAL`, OLS with intercept, ignore NaN rows):
  - `implied_persistence(F, x, h: int = 1) -> float` — slope of `F[:, t, h-1]` on `x[:, t]`.
  - `eor_b(F, x, h: int) -> float` — slope of `x[t+h] - F[t, h-1]` on `F[t, h-1] - F[t-1, h]`. Valid for h = 1..5.
  - `kappa(F_model, F_bayes, x) -> float` — slope of `F_model[t, 0] - F_bayes[t, 0]` on `x[t] - F_bayes[t-1, 0]`.
  - `excess_mse_pct(F_model, F_bench, x, h: int) -> float` — `100·(mse_model / mse_bench - 1)`.
  - `gap(F_model, F_bench, x, h=1) -> float` — difference of implied persistences.
  - `verdict(values: Sequence[float], margin: float) -> Literal["equivalent", "different", "inconclusive"]` — 95% t-interval of the mean across seeds: wholly inside `(-margin, margin)` → equivalent; wholly outside `[-margin, margin]` → different; otherwise inconclusive.

- [ ] **Step 1: Write failing tests**

```python
def test_planted_kappa_recovered():
    b = sample_family_a(100_000, np.random.default_rng(0))
    Fb = bayes_a(b.x, None)
    for theta in (-0.3, 0.0, 0.2, 0.5):
        Fm = Fb.copy()
        Fm[:, 2:, 0] += theta * (b.x[:, 2:81] - Fb[:, 1:80, 0])
        assert abs(kappa(Fm, Fb, b.x) - theta) < 0.01

def test_planted_persistence_offset_recovered():
    b = sample_family_a(100_000, np.random.default_rng(1), rho=0.4)
    Fm = fire(b.x, 0.55)
    assert abs(gap(Fm, fire(b.x, 0.4), b.x) - 0.15) < 0.005
    assert abs(implied_persistence(fire(b.x, 0.4), b.x) - 0.4) < 1e-9

def test_fire_eor_is_zero_and_overreaction_is_negative():
    b = sample_family_a(100_000, np.random.default_rng(2), rho=0.6)
    assert abs(eor_b(fire(b.x, 0.6), b.x, 1)) < 0.01
    assert eor_b(fire(b.x, 0.9), b.x, 1) < -0.05      # too-persistent forecaster overreacts

def test_verdict():
    assert verdict([0.001, -0.002, 0.003, 0.0, 0.001], 0.02) == "equivalent"
    assert verdict([0.08, 0.09, 0.07, 0.085, 0.075], 0.02) == "different"
    assert verdict([0.01, 0.03, 0.02, 0.025, 0.015], 0.02) == "inconclusive"
```

- [ ] **Step 2:** Run; expect failures.
- [ ] **Step 3:** Implement. Replace the inline OLS in Task 2 and 3 tests with `eor_b`.
- [ ] **Step 4:** Run `uv run pytest -v`; all pass.
- [ ] **Step 5:** Commit: `feat: diagnostics and equivalence verdict`.

---

### Task 5: Model and training

**Files:** Create `rt/model.py`, `rt/train.py`, `configs/grid.json`, `tests/test_train.py`

**Interfaces:**
- Produces:
  - `SIZES = {"xs": (1, 32, 2), "s": (2, 64, 4), "m": (4, 128, 4), "l": (8, 256, 8)}` — (layers, d_model, heads).
  - `class Forecaster(nn.Module)`: `forward(values: Tensor (B, L), times: Tensor (B, L) int) -> Tensor (B, L, 6)`. Causal self-attention. Input embedding = linear(value) + learned embedding of the absolute time index (87 entries). Pre-norm blocks, MLP ratio 4, no dropout.
  - `@dataclass Cell: family: Literal["A", "B"]; size: str; window: int | None; seed: int`, with `cell.id -> str` like `A-l-w4-s3` (`wfull` for None).
  - `make_batch(cell, rng, batch_size) -> (values, times, targets, mask)`. Full window: the whole sequence t = 1..80, loss at every position. Window W: draw t uniformly in 1..80, input is `x[max(1, t-W+1)..t]` with its true time indices, left-padded, loss at the last position only. Targets are `x[t+1..t+6]`.
  - `train_cell(cell, cfg, out_dir, device) -> Path` — AdamW, lr `1e-3`, 500 warmup steps then cosine, batch 1024, fresh simulated data every step, `max_steps` 30,000, squared-error loss summed over horizons. Saves checkpoints at steps `[100, 300, 1000, 3000, 10000, 30000]` to `out_dir/<cell.id>/step<k>.pt` (write to a temp name, then rename) and `train_log.csv`. Resumes from the latest checkpoint if one exists. Records device, wall time, torch version.
  - `configs/grid.json` holds all of the above numbers plus `windows: [2, 4, 8, 16, null]`, `seeds: [0, 1, 2, 3, 4]`, `eval_n_per_rho: 20000`, `eval_n_prior: 100000`, and the convergence rule: a cell is converged if evaluation squared error at step 30,000 is within 0.1% of its value at step 10,000.

- [ ] **Step 1: Write failing tests**

```python
def test_forward_shape_and_causality():
    m = Forecaster(*SIZES["xs"])
    v, t = torch.randn(4, 80), torch.arange(1, 81).repeat(4, 1)
    out = m(v, t); assert out.shape == (4, 80, 6)
    v2 = v.clone(); v2[:, 50:] += 1.0
    assert torch.allclose(m(v2, t)[:, :50], out[:, :50], atol=1e-6)

def test_window_batch_never_shows_more_than_w():
    cell = Cell("A", "xs", 4, 0)
    values, times, targets, mask = make_batch(cell, np.random.default_rng(0), 256)
    assert values.shape[1] == 4 and (mask.sum(1) <= 4).all()

def test_tiny_training_reduces_loss_and_resumes(tmp_path):
    cfg = load_cfg(max_steps=300, checkpoints=[100, 300], batch_size=256)
    p = train_cell(Cell("A", "xs", None, 0), cfg, tmp_path, "cpu")
    log = pd.read_csv(p / "train_log.csv")
    assert log.loss.iloc[-1] < 0.7 * log.loss.iloc[0]
    before = (p / "step300.pt").stat().st_mtime
    train_cell(Cell("A", "xs", None, 0), cfg, tmp_path, "cpu")     # already done
    assert (p / "step300.pt").stat().st_mtime == before
```

- [ ] **Step 2:** Run; expect failures.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest tests/test_train.py -v`; expect 3 passed.
- [ ] **Step 5:** Commit: `feat: transformer and training loop`.

---

### Task 6: Evaluation pipeline

**Files:** Create `rt/evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `model_forecasts(model, x: np.ndarray, window: int | None, device) -> F` — at t < W uses the shorter prefix, exactly as `make_batch` does.
  - `eval_sets(family, cfg) -> dict[str, Batch]` — keys `"rho=0.0"`…`"rho=1.0"` and `"prior"`, from fixed evaluation seeds (`10_000 + index`) that training never uses; cached to `results/eval/` with the benchmark forecasts (`bayes`, `window_bayes`, `fire`, `ls`) so they are computed once per (family, window).
  - `evaluate_cell(cell, step, cfg, device) -> pd.DataFrame` with columns `cell_id, family, size, window, seed, step, converged, rho, h, bench, implied_persistence, bench_implied_persistence, gap, eor_b, bench_eor_b, excess_mse_pct, kappa`. `bench` is `bayes` and `window_bayes` (identical for the full window); `rho` is a grid value or `"prior"`; `kappa` is filled only for `bench == "bayes"`, `h == 1`. Written atomically to `results/metrics/<cell.id>.csv`.

- [ ] **Step 1: Write failing tests**

```python
def test_model_forecasts_respect_window(tmp_path):
    m = Forecaster(*SIZES["xs"])
    b = sample_family_a(8, np.random.default_rng(0))
    x2 = b.x.copy(); x2[:, :40] += 5.0                 # change only old history
    F1, F2 = model_forecasts(m, b.x, 4, "cpu"), model_forecasts(m, x2, 4, "cpu")
    assert np.allclose(F1[:, 44:], F2[:, 44:], atol=1e-5)

def test_short_history_matches_full_window_input():
    m = Forecaster(*SIZES["xs"]); b = sample_family_a(8, np.random.default_rng(1))
    assert np.allclose(model_forecasts(m, b.x, 16, "cpu")[:, 1:16],
                       model_forecasts(m, b.x, None, "cpu")[:, 1:16], atol=1e-5)

def test_evaluate_cell_schema(tmp_path):
    ...  # train xs for 100 steps with eval_n shrunk to 500, call evaluate_cell,
         # assert the columns above, 7 rho values x 6 horizons x 2 benches rows per step,
         # and that a row with bench == "bayes" for an untrained model has excess_mse_pct > 0
```

- [ ] **Step 2:** Run; expect failures.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run; expect 3 passed.
- [ ] **Step 5: Local learning check.** Run `uv run python -m rt.train --cell A-s-wfull-s0 --device mps` then `uv run python -m rt.evaluate --cell A-s-wfull-s0` (background, `nice`). Expected: `excess_mse_pct` against Bayes at h = 1 on the `prior` set under 5 at the last checkpoint. If it is not, stop and debug before going on; record the number in `docs/DECISIONS.md`.
- [ ] **Step 6:** Commit: `feat: evaluation pipeline`.

---

### Task 7: Gate and sweep launcher

**Files:** Create `rt/gate.py`, `rt/sweep.py`, `tests/test_gate_sweep.py`

**Interfaces:**
- Produces:
  - `code_hash() -> str` — sha256 over the sorted contents of `rt/*.py` and `configs/*.json`.
  - `run_gate(cfg) -> dict` and `python -m rt.gate` — three checks, written to `results/gate.json` as `{"passed": bool, "code_hash": str, "checks": {...}}`:
    1. benchmark: Bayes `eor_b` on the prior set within ±0.01 for both families; point-prior Bayes equals `fire`; `bayes` squared error ≤ `window_bayes` ≤ a W-observation least-squares fit, for every window.
    2. planted: the Task 4 planted `kappa` and `gap` values recovered within 0.01.
    3. learning: the Task 6 Step 5 cell exists with `excess_mse_pct < 5`.
  - `grid(cfg) -> list[Cell]` — Family A: 4 sizes × 5 windows × 5 seeds = 100; Family B: size `l` × 5 windows × 5 seeds = 25. Ordered smallest size first.
  - `python -m rt.sweep --device cuda --budget-usd 12 --price-per-hour P --jobs J [--only A-l-*]` — refuses to start unless `gate.json` has `passed` and a `code_hash` equal to the current one; skips cells whose metrics CSV is complete (all six checkpoints present); runs up to `J` cells at once; before launching each cell checks `elapsed_hours × P ≥ budget` and, if so, launches nothing more and exits with a message listing the cells not run; appends to `results/ledger.csv` (cell, start, end, seconds, device).
  - `python -m rt.sweep --time-only --cell A-l-wfull-s0 --steps 500` — prints steps per second and nothing else (no diagnostics), for the cloud pilot.

- [ ] **Step 1: Write failing tests**

```python
def test_sweep_refuses_without_green_gate(tmp_path): ...
    # no gate.json -> SystemExit; gate.json with a different code_hash -> SystemExit

def test_code_hash_changes_when_config_changes(tmp_path): ...

def test_resume_skips_complete_and_redoes_partial(tmp_path): ...
    # complete CSV for cell 1, CSV with 3 of 6 checkpoints for cell 2 (and a stray .tmp file):
    # planned work == [cell 2, cell 3, ...]

def test_budget_guard_stops_launching(tmp_path): ...
    # fake clock: 2 h elapsed, price 6, budget 12 -> no further launches, exit message names remaining cells

def test_grid_counts():
    cells = grid(load_cfg()); assert len(cells) == 125
    assert sum(c.family == "B" for c in cells) == 25
```

- [ ] **Step 2:** Run; expect failures.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest -v`; all pass. Then `uv run python -m rt.gate`; expected output ends `GATE PASS`.
- [ ] **Step 5:** Commit: `feat: gate and gate-bound resumable sweep`.

---

### Task 8: Novelty check and pre-registration — STOP POINT

**Files:** Create `docs/novelty_check.md`, `docs/PREREG.md`, `scripts/novelty_openalex.py` (copied from EWH with provenance)

- [ ] **Step 1:** Run the OpenAlex search for: transformers / in-context learning × (overreaction, expectations, Coibion–Gorodnichenko, diagnostic expectations, working memory, AR(1) forecasting bias). Record queries, hit counts and the closest five papers with one line each on how they differ. Known neighbours to place: arXiv 2308.08536, 2410.16546, 2303.16158, AKLMT, "Overreaction and Working Memory". **If a paper already runs these diagnostics on trained-from-scratch transformers against a Bayes benchmark, stop and report to AJ before anything else.**
- [ ] **Step 2:** Write `docs/PREREG.md`: H1–H3 exactly as in the spec; the margins; the convergence rule; the unit of inference (seed) and interval; the evaluation sets and seeds; which rows of the metrics CSV each hypothesis reads (H1: `size=l, window=full, bench=bayes, h=1`, every grid rho, both `excess_mse_pct` and `gap`; H2: every window below full, `bench=window_bayes`, same two columns, largest size primary and other sizes reported; H3: `rho ∈ {0.6, 0.8}`, `bench=window_bayes`, windows below full, converged cells, positive means the interval's lower bound is above +0.02); the exploratory list; the SPF arm as illustrative and droppable; what was seen before the tag (gate numbers and the one local learning-check cell only).
- [ ] **Step 3:** Commit. **Stop. Give AJ the novelty summary and PREREG for sign-off.** On go: `git tag prereg-v1`.

---

### Task 9: Cloud pilot and sweep — STOP POINT at step 3

**Files:** Create `cloud/README.md`, `cloud/setup.sh`, `cloud/push.sh`, `cloud/pull.sh`

- [ ] **Step 1:** Read the current Thunder Compute docs and pricing page. Write into `cloud/README.md`: the CLI commands to create, connect to, stop and delete an instance; the A100 hourly price today; whether a stopped instance bills; storage cost. AJ signs up and logs the CLI in himself.
- [ ] **Step 2:** Write the scripts. `setup.sh`: install uv, `uv sync`, run `uv run pytest -q` and `uv run python -m rt.gate` on the instance (gate must pass there too, same code hash). `push.sh`: rsync the repo excluding `results/`. `pull.sh`: rsync `results/metrics`, `results/ledger.csv`, logs and final checkpoints back, never deleting anything local.
- [ ] **Step 3: Pilot. Quote the price to AJ and wait for go.** Create one A100 instance, run setup, then `--time-only` for one cell of each size at `--jobs` 1, 4 and 8. From steps per second compute projected hours and dollars for the 125 cells plus evaluation. **Report the projection to AJ and wait for go.** If it exceeds $12: first move sizes `xs` and `s` to the Mac (≤4 processes, `nice`, battery guard), then cut seeds 5 → 3 for size `m`, and record whichever is used as a deviation in `docs/DECISIONS.md` before any diagnostic is looked at.
- [ ] **Step 4:** Run `python -m rt.sweep --device cuda --budget-usd 12 --price-per-hour <P> --jobs <J>` in tmux. Run `cloud/pull.sh` every 30 minutes and at the end.
- [ ] **Step 5:** Verify locally: every expected metrics CSV present with all six checkpoints; ledger row count matches; `results/` backed up. Then delete the instance and confirm from the CLI that nothing is running. Record total spend in `docs/DECISIONS.md`.
- [ ] **Step 6:** Commit metrics, ledger and logs (not checkpoints; keep those on disk, git-ignored): `data: sweep results`.

---

### Task 10: Human overlay and results pack

**Files:** Create `rt/human.py`, `rt/report.py`, `data/aklmt_figure2.csv` (copied from EWH `data/processed/` with a provenance note), `tests/test_report.py`, `RESULTS.md`, `figures/`

**Interfaces:**
- Produces:
  - `human_curve() -> pd.DataFrame` — columns `rho, implied_rho, implied_rho_se, b_eor, b_eor_se`.
  - `summarise(metrics_dir) -> pd.DataFrame` — per (family, size, window, step, rho, h, bench): seed mean, 95% interval, `verdict` for `gap` (margin 0.02) and `excess_mse_pct` (margin 1), with non-converged cells labelled `not converged`.
  - `hypotheses(summary) -> dict` — the H1, H2, H3 answers computed only from the rows named in PREREG.
  - `python -m rt.report` — writes `figures/fig1_gap_map.png` (gap against window-Bayes by window and rho, size `l`), `fig2_human_overlay.png` (Family B implied persistence against rho: transformer by window, Family B Bayes, window-Bayes, human points with error bars), `fig3_training_traces.png` (gap through training), `fig4_size.png`, and `RESULTS.md`.

- [ ] **Step 1: Write failing tests**

```python
def test_every_number_in_results_traces_to_summary(tmp_path): ...
    # build a synthetic metrics dir, run the report, parse the numeric table cells in RESULTS.md,
    # assert each equals the corresponding summary value to the printed precision

def test_not_converged_never_called_biased(): ...
    # a cell with converged == False and gap == 0.2 gets verdict "not converged"

def test_hypotheses_read_only_prereg_rows(): ...
    # changing rows outside the PREREG selection leaves hypotheses() unchanged
```

- [ ] **Step 2:** Run; expect failures. Implement. Run; all pass.
- [ ] **Step 3:** Run `uv run python -m rt.report` on the real results. `RESULTS.md` states H1–H3 verdicts with numbers and intervals, lists deviations, and reports numbers without interpretation beyond the pre-registered readings.
- [ ] **Step 4:** Open each figure and check axes, units and that the human points match `aklmt_figure2.csv`.
- [ ] **Step 5:** Commit: `feat: results pack`.

---

### Task 11: Real-data comparison (droppable)

**Files:** Create `scripts/export_spf_panel.py`, `rt/spf.py`, `tests/test_spf.py`

- [ ] **Step 1:** `scripts/export_spf_panel.py` is run inside EWH's environment, read-only: `uv run --project ~/Documents/expectations-without-hindsight python scripts/export_spf_panel.py --out data/spf_panel.parquet`. It calls EWH's `bgms_panel(var)` for the variables in its SPF replication and writes, per variable and quarter, the real-time history available at the forecast date, the survey forecasts at each horizon, and the realised values.
- [ ] **Step 2:** `spf_comparison(models, panel) -> pd.DataFrame`: for each variable-quarter, scale the history by the mean and standard deviation of first differences computed from data before that date only, feed the last W (or all) values to the Family B model, unscale, and compute `eor_b` at the horizons that match the survey; print beside the survey's own `eor_b` on the same variable-quarters.
- [ ] **Step 3:** Test with a synthetic panel that scaling uses no observation dated on or after the forecast quarter, and that a pure AR(1) panel returns the model's Family B coefficients within 0.05.
- [ ] **Step 4:** **Drop rule:** if survey horizons cannot be matched to model horizons, or the scaled histories fall outside the range Family B was trained on for more than 20% of variable-quarters, do not report the arm; write the reason in `RESULTS.md` and `docs/DECISIONS.md`.
- [ ] **Step 5:** Add the table to `RESULTS.md` under "Illustrative comparison". Commit.

---

### Task 12: Paper and publication — STOP POINT at step 3

- [ ] **Step 1:** Build a short paper PDF under `paper/`, reusing the build and house style in `~/Documents/lookahead-probe/paper/` (copy, do not link). Sections: question, setup and benchmarks, pre-registered results, exploratory results, illustrative comparison, limits. Every number comes from `RESULTS.md`. Verify the PDF builds and read it page by page.
- [ ] **Step 2:** Write the venue verdict in `docs/DECISIONS.md`: blog-only, workshop, or arXiv/SSRN, with the reason, based on what H2 and H3 showed.
- [ ] **Step 3: Ask AJ before making anything public.** On go: create the GitHub repo, push `main` and tags, confirm the AKLMT package and checkpoints are absent from history.
- [ ] **Step 4:** Hand AJ the figures and numbers for the BASIS piece. Update the project memory note. State done/total/left.
