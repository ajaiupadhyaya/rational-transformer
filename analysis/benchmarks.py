"""Descriptive benchmark-only rows (PREREG owner decision 6): Family A, h = 1.

Implied persistence of fire, ls, bayes and window_bayes (W = 2, 4, 8, 16) at every grid rho
and the prior set. Not a model result; read from the cached benchmark forecasts.
"""
from __future__ import annotations

import pandas as pd

from rt.diagnostics import implied_persistence
from rt.evaluate import bench_forecasts, eval_sets
from rt.sim import Batch
from rt.train import load_cfg

WINDOWS = (2, 4, 8, 16)
COLUMNS = ["rho", "bench", "window", "implied_persistence"]


def benchmark_rows(eval_dir=None, cfg: dict | None = None, grid: int | None = None) -> pd.DataFrame:
    """One row per (rho set, benchmark, window); window is '-' where it does not apply."""
    cfg = load_cfg() if cfg is None else cfg
    rows = []
    for key, batch in eval_sets("A", cfg).items():
        rho = "prior" if key == "prior" else key.split("=", 1)[1]
        x = batch.x

        def ip(F):
            return float(implied_persistence(F, x, 1))

        fb = bench_forecasts("A", None, key, batch, grid=grid, eval_dir=eval_dir,
                             names=("fire", "ls", "bayes"))
        for name in ("fire", "ls", "bayes"):
            rows.append({"rho": rho, "bench": name, "window": "-", "implied_persistence": ip(fb[name])})
        for w in WINDOWS:
            wb = bench_forecasts("A", w, key, batch, grid=grid, eval_dir=eval_dir,
                                 names=("window_bayes",))["window_bayes"]
            rows.append({"rho": rho, "bench": "window_bayes", "window": str(w),
                         "implied_persistence": ip(wb)})
    return pd.DataFrame(rows, columns=COLUMNS)
