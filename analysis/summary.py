"""Per-run metrics CSVs -> one summary table (seed means, t intervals, verdicts).

Reads only the directory it is given. Convergence is reported (per run and per cell) and
never changes a verdict (PREREG owner decision 2).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from rt.diagnostics import verdict as _verdict
from rt.sweep import grid
from rt.train import Cell, load_cfg

MARGINS = {"gap": 0.02, "excess_mse_pct": 1.0}
MEAN_ONLY = ["implied_persistence", "bench_implied_persistence", "eor_b", "bench_eor_b", "kappa"]
KEY = ["family", "size", "window", "step", "rho", "h", "bench"]
INSUFFICIENT = "insufficient seeds"
STEPS_REQUIRED = 30000

# The grid actually run (Plan A, deviation prereg-v1.1).
WINDOWS = ("2", "4", "8", "16", "full")


def _plan_a() -> dict[tuple[str, str, str], tuple[int, ...]]:
    """{(family, size, window): seeds actually planned}."""
    out: dict[tuple[str, str, str], tuple[int, ...]] = {}
    for size in ("xs", "s", "l"):
        for w in WINDOWS:
            out[("A", size, w)] = (0, 1, 2, 3, 4)
    for w in WINDOWS:
        out[("A", "m", w)] = (0, 1, 2)
    for w in ("full", "4"):
        out[("B", "l", w)] = (0, 1, 2)
    return out


PLAN_A = _plan_a()


def expected_seeds(family: str, size: str, window: str) -> int:
    return len(PLAN_A.get((family, size, str(window)), (0, 1, 2, 3, 4)))


def norm_rho(r) -> str:
    return "prior" if str(r) == "prior" else f"{float(r):.1f}"


def t_interval(values: list[float]) -> tuple[float, float, float, float]:
    """mean, sd, lo, hi of the 95% Student-t interval (n - 1 df); NaN sd/interval if n < 2."""
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    mean = float(v.mean())
    if n < 2:
        return mean, float("nan"), float("nan"), float("nan")
    sd = float(v.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1)) * sd / np.sqrt(n)
    return mean, sd, mean - half, mean + half


def load_metrics(metrics_dir) -> pd.DataFrame:
    """Concatenate every readable <cell>.csv in `metrics_dir` (unreadable files are skipped)."""
    frames = []
    for path in sorted(Path(metrics_dir).glob("*.csv")):
        try:
            f = pd.read_csv(path, dtype={"rho": str, "window": str, "cell_id": str})
        except Exception:
            continue
        if len(f):
            frames.append(f)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["rho"] = df["rho"].map(norm_rho)
    df["window"] = df["window"].astype(str)
    df["step"] = df["step"].astype(int)
    df["h"] = df["h"].astype(int)
    df["seed"] = df["seed"].astype(int)
    df["converged"] = df["converged"].map(lambda b: b is True or str(b) == "True")
    return df


def _as_df(source) -> pd.DataFrame:
    return source if isinstance(source, pd.DataFrame) else load_metrics(source)


def summarise(source) -> pd.DataFrame:
    """One row per (family, size, window, step, rho, h, bench, column) for gap and excess_mse_pct.

    `source` is a metrics directory or the frame `load_metrics` returns. Values are ordered
    by seed; `seeds` and `values` are JSON lists stored as text.
    """
    df = _as_df(source)
    cols = (KEY + ["column", "n_seeds", "seeds", "values", "mean", "sd", "lo", "hi", "verdict"]
            + MEAN_ONLY + ["all_converged"])
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df.sort_values("seed")
    rows = []
    for key, g in df.groupby(KEY, sort=True):
        means = {c: (float(g[c].mean()) if g[c].notna().any() else float("nan")) for c in MEAN_ONLY}
        for column, margin in MARGINS.items():
            gg = g[np.isfinite(g[column].to_numpy(dtype=float))]
            vals = [float(v) for v in gg[column]]
            if not vals:
                continue
            mean, sd, lo, hi = t_interval(vals)
            ver = _verdict(vals, margin) if len(vals) >= 2 else INSUFFICIENT
            rows.append({**dict(zip(KEY, key)), "column": column, "n_seeds": len(vals),
                         "seeds": json.dumps([int(s) for s in gg["seed"]]),
                         "values": json.dumps(vals), "mean": mean, "sd": sd, "lo": lo,
                         "hi": hi, "verdict": ver, **means,
                         "all_converged": bool(g["converged"].all())})
    return pd.DataFrame(rows, columns=cols)


def run_table(source) -> pd.DataFrame:
    """One row per run: cell id, identifiers, the steps present, and the run's converged flag."""
    df = _as_df(source)
    if df.empty:
        return pd.DataFrame(columns=["cell_id", "family", "size", "window", "seed", "steps",
                                     "converged"])
    rows = []
    for cell_id, g in df.groupby("cell_id", sort=True):
        r = g.iloc[0]
        rows.append({"cell_id": cell_id, "family": r["family"], "size": r["size"],
                     "window": r["window"], "seed": int(r["seed"]),
                     "steps": sorted(int(s) for s in g["step"].unique()),
                     "converged": bool(g["converged"].all())})
    return pd.DataFrame(rows)


def convergence_cells(runs: pd.DataFrame) -> pd.DataFrame:
    """Per (family, size, window): runs present, runs converged, and the cell flag.

    A cell is converged only if every one of its expected runs is present and converged.
    """
    cols = ["family", "size", "window", "n_runs", "n_converged", "expected_runs", "converged"]
    if runs.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for (fam, size, w), g in runs.groupby(["family", "size", "window"], sort=True):
        exp = expected_seeds(fam, size, w)
        rows.append({"family": fam, "size": size, "window": w, "n_runs": len(g),
                     "n_converged": int(g["converged"].sum()), "expected_runs": exp,
                     "converged": bool(g["converged"].all() and len(g) >= exp)})
    return pd.DataFrame(rows, columns=cols)


def coverage(runs: pd.DataFrame, cfg: dict | None = None) -> dict:
    """Cells of the configured grid by status: run, incomplete, missing, or not run by design."""
    cfg = load_cfg() if cfg is None else cfg
    have = {r.cell_id: set(r.steps) for r in runs.itertuples()} if len(runs) else {}
    planned, by_design, missing, incomplete, done = [], [], [], [], []
    for c in grid(cfg):
        w = "full" if c.window is None else str(c.window)
        if c.seed not in PLAN_A.get((c.family, c.size, w), ()):
            by_design.append(c.id)
            continue
        planned.append(c.id)
        if c.id not in have:
            missing.append(c.id)
        elif STEPS_REQUIRED not in have[c.id]:
            incomplete.append(c.id)
        else:
            done.append(c.id)
    extra = sorted(set(have) - {c.id for c in grid(cfg)})
    return {"planned": len(planned), "complete": len(done), "not_run_by_design": by_design,
            "missing": missing, "incomplete": incomplete, "unexpected_files": extra}


def cell_ids(family: str, size: str, window: str, seeds=range(5)) -> list[str]:
    w = None if str(window) == "full" else int(window)
    return [Cell(family, size, w, s).id for s in seeds]
