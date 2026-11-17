"""Diagnostics and verdicts: turn forecast arrays into the numbers the study reports.

Conventions: `x` is (n, 87); a forecast array `F` is (n, 81, 6) with
F[i, t, h-1] the forecast of x[i, t+h] made at time t (NaN where not forecast).
Everything pools (sequence, t) pairs over t in T_EVAL, drops pairs with a NaN
regressand or regressor, and uses OLS with an intercept.
"""
from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from scipy import stats

from rt.sim import T_EVAL

_T = np.array(T_EVAL)


def _slope(y: np.ndarray, z: np.ndarray) -> float:
    """OLS slope of y on z (intercept included), NaN pairs dropped."""
    y, z = np.ravel(y), np.ravel(z)
    ok = np.isfinite(y) & np.isfinite(z)
    y, z = y[ok], z[ok]
    zc = z - z.mean()
    return float((zc * (y - y.mean())).sum() / (zc * zc).sum())


def implied_persistence(F: np.ndarray, x: np.ndarray, h: int = 1) -> float:
    """Slope of F[:, t, h-1] on x[:, t]."""
    return _slope(F[:, _T, h - 1], x[:, _T])


def eor_b(F: np.ndarray, x: np.ndarray, h: int) -> float:
    """Error-on-revision slope: x[t+h] - F[t, h-1] on F[t, h-1] - F[t-1, h]."""
    if h not in range(1, 6):
        raise ValueError(f"h must be in 1..5; got {h}")
    err = x[:, _T + h] - F[:, _T, h - 1]
    rev = F[:, _T, h - 1] - F[:, _T - 1, h]
    return _slope(err, rev)


def kappa(F_model: np.ndarray, F_bayes: np.ndarray, x: np.ndarray) -> float:
    """Slope of F_model[t,0] - F_bayes[t,0] on x[t] - F_bayes[t-1,0] (horizon 1)."""
    dev = F_model[:, _T, 0] - F_bayes[:, _T, 0]
    surprise = x[:, _T] - F_bayes[:, _T - 1, 0]
    return _slope(dev, surprise)


def _sq_errors(F: np.ndarray, x: np.ndarray, h: int) -> np.ndarray:
    return (x[:, _T + h] - F[:, _T, h - 1]) ** 2


def excess_mse_pct(F_model: np.ndarray, F_bench: np.ndarray, x: np.ndarray, h: int) -> float:
    """100 * (mse_model / mse_bench - 1) over pairs where both forecasts exist."""
    se_m, se_b = _sq_errors(F_model, x, h), _sq_errors(F_bench, x, h)
    ok = np.isfinite(se_m) & np.isfinite(se_b)
    return float(100.0 * (se_m[ok].mean() / se_b[ok].mean() - 1.0))


def gap(F_model: np.ndarray, F_bench: np.ndarray, x: np.ndarray, h: int = 1) -> float:
    """Implied persistence of the model minus that of the benchmark."""
    return implied_persistence(F_model, x, h) - implied_persistence(F_bench, x, h)


def verdict(values: Sequence[float], margin: float) -> Literal["equivalent", "different", "inconclusive"]:
    """95% Student-t interval of the mean across seeds against (-margin, margin)."""
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    if n < 2:
        raise ValueError("verdict needs at least 2 values")
    half = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    lo, hi = v.mean() - half, v.mean() + half
    if -margin < lo and hi < margin:
        return "equivalent"
    if lo > margin or hi < -margin:
        return "different"
    return "inconclusive"
