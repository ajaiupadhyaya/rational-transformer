"""Turn a trained checkpoint into the table of numbers the study reports.

For one cell this produces the model's forecasts and the benchmarks' forecasts on the
same fixed evaluation sets and runs the diagnostics on both.

Evaluation sets (per family, never used in training): "rho=0.0" .. "rho=1.0", each
`eval_n_per_rho` sequences at that fixed rho, and "prior", `eval_n_prior` sequences with
rho drawn from the prior. Set i (RHOS order, then prior = 6) is drawn from
SeedSequence([10_000 + i, family tag]).

Benchmarks: `bayes` (full history), `window_bayes` (the cell's window; the same array as
`bayes` for the full window), `fire`, `ls`. Bayes uses a 2001-point rho grid for both
families. Bayes and least-squares forecasts are cached under results/eval/ as .npy files
whose name carries the family, set, size, seed, window and grid plus a digest of the
evaluation data and of rt/bayes.py and rt/baselines.py, so a changed config or changed
benchmark code can never read a stale file. `fire` is closed form and is not cached.

Family B diagnostics are computed on standardised deviations, using each sequence's
true mu and sigma: x~ = (x - mu) / sigma and F~ = (F - mu) / sigma, identically for the
model and every benchmark (a pooled regression on raw values would mostly measure
between-sequence differences in the mean). Family A has mu = 0, sigma = 1 and is used raw.

Usage: uv run python -m rt.evaluate --cell A-s-wfull-s0 --device mps
         [--ckpt-dir results/checkpoints] [--out results/metrics]
         [--eval-n-per-rho N] [--eval-n-prior N]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from rt.baselines import fire, ls_learner
from rt.bayes import bayes_a, bayes_b
from rt.diagnostics import eor_b, excess_mse_pct, implied_persistence, kappa
from rt.model import Forecaster
from rt.sim import HORIZONS, RHOS, T_EVAL, Batch
from rt.train import (SAMPLERS, T_MAX, Cell, _checkpoints, batch_from_x, build_model,
                      load_cfg)

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "results" / "eval"
CKPT_DIR = ROOT / "results" / "checkpoints"
METRICS_DIR = ROOT / "results" / "metrics"

N_T, N_H = T_MAX + 1, len(HORIZONS)
GRID = {"A": 2001, "B": 2001}                 # rho grid points for bayes_a / bayes_b
EVAL_SEED_BASE = 10_000
EVAL_FAMILY_TAG = {"A": 0xE0A, "B": 0xE0B}    # a separate evaluation stream per family
EVAL_SPECS = [(f"rho={r}", float(r)) for r in RHOS] + [("prior", None)]
EVAL_KEYS = [k for k, _ in EVAL_SPECS]
BENCHES = ("bayes", "window_bayes")           # the benchmarks the metrics rows compare against
TOKENS_PER_BATCH = 1 << 17                    # model inputs per forward pass (memory bound)
COLUMNS = ["cell_id", "family", "size", "window", "seed", "step", "converged", "rho", "h",
           "bench", "implied_persistence", "bench_implied_persistence", "gap", "eor_b",
           "bench_eor_b", "excess_mse_pct", "kappa"]
_T = np.array(T_EVAL)


# ---- model forecasts --------------------------------------------------------

@torch.no_grad()
def model_forecasts(model: torch.nn.Module, x: np.ndarray, window: int | None,
                    device) -> np.ndarray:
    """F (n, 81, 6) float64: F[i, t, h-1] is the model's forecast of x[i, t+h], NaN at t = 0.

    Inputs are built by `rt.train.batch_from_x`, exactly as in training: the full window
    is one causal pass over x[1..80]; window W gives, for each origin t, the inputs
    x[max(1, t-W+1)..t] (left-padded) and reads the forecast at the last position.
    The model is moved to `device` (and left there); its train/eval mode is restored.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    F = np.full((n, N_T, N_H), np.nan, dtype=np.float64)
    dev = torch.device(device)
    was_training = model.training
    model.to(dev).eval()
    try:
        if window is None:
            rows = max(1, TOKENS_PER_BATCH // T_MAX)
            for i0 in range(0, n, rows):
                values, times, _, _ = batch_from_x(x[i0:i0 + rows], None)
                F[i0:i0 + rows, 1:] = model(values.to(dev), times.to(dev)).cpu().numpy()
        else:
            rows = max(1, TOKENS_PER_BATCH // window)
            for i0 in range(0, n, rows):
                xc = x[i0:i0 + rows]
                c = xc.shape[0]
                for t in range(1, T_MAX + 1):
                    values, times, _, _ = batch_from_x(xc, window, np.full(c, t))
                    out = model(values.to(dev), times.to(dev))[:, -1]
                    F[i0:i0 + c, t] = out.cpu().numpy()
    finally:
        model.train(was_training)
    return F


# ---- evaluation sets and benchmarks ----------------------------------------

def eval_seed(key: str) -> int:
    return EVAL_SEED_BASE + EVAL_KEYS.index(key)


def eval_sets(family: str, cfg: dict) -> dict[str, Batch]:
    """The fixed evaluation sets for `family`, keyed "rho=0.0" .. "rho=1.0", "prior"."""
    out = {}
    for key, rho in EVAL_SPECS:
        n = int(cfg["eval_n_prior"] if rho is None else cfg["eval_n_per_rho"])
        rng = np.random.default_rng(np.random.SeedSequence([eval_seed(key),
                                                            EVAL_FAMILY_TAG[family]]))
        out[key] = SAMPLERS[family](n, rng, rho=rho)
    return out


def _code_digest() -> bytes:
    h = hashlib.sha256()
    for name in ("bayes.py", "baselines.py"):
        h.update((ROOT / "rt" / name).read_bytes())
    return h.digest()


def _save_atomic_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)


def bench_forecasts(family: str, window: int | None, key: str, batch: Batch, *,
                    grid: int | None = None, eval_dir=None,
                    names=BENCHES) -> dict[str, np.ndarray]:
    """Benchmark forecasts for one evaluation set, computed once and cached.

    `names` picks from "bayes", "window_bayes", "fire", "ls". Cached arrays come back
    memory-mapped read-only. For window None, "window_bayes" is the "bayes" array.
    """
    grid = GRID[family] if grid is None else int(grid)
    cache = Path(EVAL_DIR if eval_dir is None else eval_dir)
    cache.mkdir(parents=True, exist_ok=True)
    x = np.ascontiguousarray(batch.x, dtype=np.float64)
    digest = hashlib.sha256(x.tobytes() + _code_digest()).hexdigest()[:16]
    stem = f"{family}_{key}_n{x.shape[0]}_seed{eval_seed(key)}_grid{grid}_{digest}"
    bayes_fn = bayes_a if family == "A" else bayes_b

    def cached(tag: str, compute: Callable[[], np.ndarray]) -> np.ndarray:
        path = cache / f"{stem}_{tag}.npy"
        if not path.exists():
            _save_atomic_npy(path, compute())
        return np.load(path, mmap_mode="r")

    loaded: dict = {}

    def bayes_w(w):          # one array object per window, so window None aliases bayes
        if w not in loaded:
            tag = "bayes-wfull" if w is None else f"bayes-w{w}"
            loaded[w] = cached(tag, lambda: bayes_fn(x, w, grid=grid))
        return loaded[w]

    out = {}
    for name in names:
        if name == "bayes":
            out[name] = bayes_w(None)
        elif name == "window_bayes":
            out[name] = bayes_w(window)
        elif name == "fire":
            out[name] = fire(x, batch.rho, batch.mu)
        elif name == "ls":
            out[name] = cached("ls", lambda: ls_learner(x))
        else:
            raise ValueError(f"unknown benchmark {name!r}")
    return out


# ---- diagnostics on one evaluation set --------------------------------------

def _standardise(batch: Batch, x: np.ndarray, *Fs: np.ndarray) -> tuple[np.ndarray, ...]:
    """(x - mu) / sigma and (F - mu) / sigma per sequence; identity when mu = 0, sigma = 1."""
    mu, sigma = np.asarray(batch.mu, np.float64), np.asarray(batch.sigma, np.float64)
    if np.all(mu == 0.0) and np.all(sigma == 1.0):          # Family A: exact no-op
        return (x, *Fs)
    out, done = [(x - mu[:, None]) / sigma[:, None]], {}
    for F in Fs:
        if id(F) not in done:                                # window_bayes is bayes: once
            done[id(F)] = (F - mu[:, None, None]) / sigma[:, None, None]
        out.append(done[id(F)])
    return tuple(out)


def set_metrics(batch: Batch, F_model: np.ndarray,
                benches: dict[str, np.ndarray]) -> tuple[list[dict], float]:
    """Diagnostics of the model against each benchmark in `BENCHES`, on one set.

    Returns one dict per (bench, h) and the model's mean squared error at h = 1 over
    T_EVAL (on standardised values, like every MSE column), used by the convergence rule.
    """
    x, Fm, *Fb = _standardise(batch, batch.x, F_model, *(benches[b] for b in BENCHES))
    Fb = dict(zip(BENCHES, Fb))
    rows = []
    for name in BENCHES:
        for h in HORIZONS:
            ip_m = implied_persistence(Fm, x, h)
            ip_b = implied_persistence(Fb[name], x, h)
            rows.append({
                "h": h, "bench": name,
                "implied_persistence": ip_m, "bench_implied_persistence": ip_b,
                "gap": ip_m - ip_b,
                "eor_b": eor_b(Fm, x, h) if h < N_H else np.nan,
                "bench_eor_b": eor_b(Fb[name], x, h) if h < N_H else np.nan,
                "excess_mse_pct": excess_mse_pct(Fm, Fb[name], x, h),
                "kappa": kappa(Fm, Fb["bayes"], x) if name == "bayes" and h == 1 else np.nan,
            })
    mse_h1 = float(np.mean((x[:, _T + 1] - Fm[:, _T, 0]) ** 2))
    return rows, mse_h1


# ---- one checkpoint, every checkpoint ---------------------------------------

def _load_model(cell: Cell, path: Path) -> tuple[Forecaster, dict]:
    """The checkpoint's model, rebuilt from the config it was trained under (architecture
    options included; never from the evaluation config), and that config."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ck["config"], cell.size)
    model.load_state_dict(ck["model"])
    return model, ck["config"]


def _evaluate_step(cell: Cell, step: int, model, sets: dict[str, Batch], device,
                   eval_dir) -> tuple[pd.DataFrame, float]:
    rows, prior_mse = [], None
    for key, rho in EVAL_SPECS:
        b = sets[key]
        F = model_forecasts(model, b.x, cell.window, device)
        r, mse = set_metrics(b, F, bench_forecasts(cell.family, cell.window, key, b,
                                                   eval_dir=eval_dir))
        del F
        for row in r:
            row["rho"] = "prior" if rho is None else rho
        rows += r
        if rho is None:
            prior_mse = mse
    df = pd.DataFrame(rows)
    df["cell_id"], df["family"], df["size"] = cell.id, cell.family, cell.size
    df["window"] = "full" if cell.window is None else cell.window
    df["seed"], df["step"] = cell.seed, step
    df["converged"] = pd.array([pd.NA] * len(df), dtype="boolean")
    return df[COLUMNS], prior_mse


def evaluate_cell(cell: Cell, step: int, cfg: dict, device, *, ckpt_dir=CKPT_DIR,
                  eval_dir=None) -> pd.DataFrame:
    """Metrics rows for checkpoint `step` of `cell`.

    `converged` is a property of the whole run, so it is left NA here; `evaluate_all`
    fills it.
    """
    path = Path(ckpt_dir) / cell.id / f"step{step}.pt"
    model, _ = _load_model(cell, path)
    df, _ = _evaluate_step(cell, step, model, eval_sets(cell.family, cfg), device, eval_dir)
    return df


def _ckpt_schedule(train_cfg: dict) -> list[int]:
    """The checkpoint steps `rt.train.train_cell` writes under `train_cfg`."""
    max_steps = int(train_cfg["max_steps"])
    return sorted({int(k) for k in train_cfg["checkpoints"] if 0 < k <= max_steps} | {max_steps})


def evaluate_all(cell: Cell, cfg: dict, device, *, ckpt_dir=CKPT_DIR, out_dir=METRICS_DIR,
                 eval_dir=None) -> pd.DataFrame:
    """Evaluate every checkpoint of `cell`, fill `converged`, write <out_dir>/<cell.id>.csv.

    Convergence: the model's squared error on the prior set at h = 1 at the last
    scheduled checkpoint is within `cfg["convergence"]["rel_tol"]` (relative) of its value
    at the second-to-last one; False if either checkpoint is missing. The schedule is the
    one the run was trained under (the config saved in its checkpoints).
    """
    ckpts = _checkpoints(Path(ckpt_dir) / cell.id)
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints for {cell.id} under {ckpt_dir}")
    sets = eval_sets(cell.family, cfg)
    frames, mse, train_cfg = [], {}, None
    for step in sorted(ckpts):
        model, train_cfg = _load_model(cell, ckpts[step])
        df, mse[step] = _evaluate_step(cell, step, model, sets, device, eval_dir)
        frames.append(df)
    schedule = _ckpt_schedule(train_cfg)
    rel_tol = float(cfg["convergence"]["rel_tol"])
    converged = False
    if len(schedule) >= 2 and schedule[-2] in mse and schedule[-1] in mse:
        prev, last = mse[schedule[-2]], mse[schedule[-1]]
        converged = bool(abs(last - prev) <= rel_tol * prev)
    out = pd.concat(frames, ignore_index=True)
    out["converged"] = converged
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cell.id}.csv"
    tmp = path.with_name(path.name + ".tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate every checkpoint of one cell.")
    ap.add_argument("--cell", required=True, help="e.g. A-s-wfull-s0 or B-l-w4-s3")
    ap.add_argument("--device", default="cpu", help="cpu, mps or cuda")
    ap.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    ap.add_argument("--out", default=str(METRICS_DIR), help="directory for <cell>.csv")
    ap.add_argument("--eval-n-per-rho", type=int, default=None)
    ap.add_argument("--eval-n-prior", type=int, default=None)
    args = ap.parse_args(argv)
    cell = Cell.parse(args.cell)
    over = {k: v for k, v in (("eval_n_per_rho", args.eval_n_per_rho),
                              ("eval_n_prior", args.eval_n_prior)) if v is not None}
    cfg = load_cfg(**over)
    t0 = time.perf_counter()
    df = evaluate_all(cell, cfg, args.device, ckpt_dir=args.ckpt_dir, out_dir=args.out)
    sel = df[(df.rho == "prior") & (df.h == 1) & (df.bench == "bayes")]
    for _, r in sel.iterrows():
        print(f"{cell.id} step {int(r.step):>6}: prior h=1 vs bayes  excess_mse_pct "
              f"{r.excess_mse_pct:8.3f}  gap {r.gap:+.4f}  kappa {r.kappa:+.4f}")
    print(f"converged={bool(df.converged.iloc[0])}  {time.perf_counter() - t0:.0f}s on "
          f"{args.device}  -> {Path(args.out) / (cell.id + '.csv')}")


if __name__ == "__main__":
    main()
