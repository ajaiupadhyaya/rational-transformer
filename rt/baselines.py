"""Non-Bayesian benchmark forecasters, in the shared `F` convention.

`F` has shape (n, 81, 6): F[i, t, h-1] forecasts x[i, t+h] from x[i, 1..t]; F[:, 0] is NaN.

Provenance: `ls_coefficients` and the iteration in `ls_learner` are copied from
expectations-without-hindsight, file `ewh/sim/benchmarks.py` at commit 57e9084
(functions `ls_coefficients`, `ls_learner`). The original maps one shown history
(U, T) to one h-step forecast; here `ls_learner` applies it at every t = 1..80 with
the shown history x[1..t] (x_0 is not part of the shown history, matching
`window=None` in `rt.bayes`) and returns all six horizons. Origin times with too
few pairs to fit (t = 1, or t <= 2 with an intercept) are NaN.
"""
from __future__ import annotations

import numpy as np

N_T = 81    # time index 0..80; t = 0 carries no forecast
N_H = 6


def _empty_f(n: int) -> np.ndarray:
    return np.full((n, N_T, N_H), np.nan, dtype=np.float64)


def _check_x(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < N_T:
        raise ValueError(f"x must be (n, >= {N_T}); got {x.shape}")
    return x


def fire(x: np.ndarray, rho, mu=None) -> np.ndarray:
    """Full-information rational forecast: rho^h x_t, or mu + rho^h (x_t - mu).

    `rho` and `mu` are scalars or per-sequence arrays of shape (n,).
    """
    x = _check_x(x)
    n = x.shape[0]
    r = np.broadcast_to(np.asarray(rho, dtype=np.float64), (n,))[:, None, None]
    m = 0.0 if mu is None else np.broadcast_to(np.asarray(mu, dtype=np.float64), (n,))[:, None, None]
    h = np.arange(1, N_H + 1, dtype=np.float64)
    F = _empty_f(n)
    F[:, 1:] = m + r ** h * (x[:, 1:N_T, None] - m)
    return F


# --- copied verbatim from ewh/sim/benchmarks.py @ 57e9084 ----------------------
def ls_coefficients(hist: np.ndarray, intercept: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Per-row OLS of x_s on x_{s-1} over the shown history. Returns (a, b), each (U,)."""
    y, x = hist[:, 1:], hist[:, :-1]
    if not intercept:
        b = (x * y).sum(1) / (x * x).sum(1)
        return np.zeros(len(hist)), b
    xm, ym = x.mean(1, keepdims=True), y.mean(1, keepdims=True)
    b = ((x - xm) * (y - ym)).sum(1) / ((x - xm) ** 2).sum(1)
    a = ym[:, 0] - b * xm[:, 0]
    return a, b
# -------------------------------------------------------------------------------


def ls_learner(x: np.ndarray, intercept: bool = True) -> np.ndarray:
    """Least-squares AR(1) learner on the shown history x[1..t], iterated h steps.

    Adapted from `ls_learner` in ewh/sim/benchmarks.py @ 57e9084: same OLS fit and the
    same iteration f <- a + b f, evaluated at every origin t and every horizon.
    """
    x = _check_x(x)
    F = _empty_f(x.shape[0])
    t_min = 3 if intercept else 2       # need >= 2 pairs to fit a line, >= 1 for a slope
    for t in range(t_min, N_T):
        a, b = ls_coefficients(x[:, 1:t + 1], intercept)
        f = x[:, t].copy()
        for h in range(N_H):
            f = a + b * f
            F[:, t, h] = f
    return F
