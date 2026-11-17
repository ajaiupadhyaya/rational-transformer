"""Exact (grid) Bayes and window-Bayes forecasts for Family A.

Model: x_0 = 0, x_t = rho x_{t-1} + eps_t, eps_t ~ N(0, 1), rho ~ U[prior] continuous,
integrated by the trapezoid rule on `grid` equally spaced points (weights 1/2 at the two
endpoints, 1 elsewhere, in both numerator and denominator of every posterior moment).
At origin t with window W the forecaster sees x[s..t], s = max(1, t - W + 1) (s = 1 for
the full window). Unconditionally on the unseen past, x_s ~ N(0, V_s(rho)) with V_s = sum_{k<s} rho^(2k), so the log-likelihood
up to a rho-free constant is

    ll(rho) = -1/2 log V_s - x_s^2 / (2 V_s) + rho Sxy - rho^2 Sxx / 2,
    Sxy = sum_{j=s+1}^t x_j x_{j-1},   Sxx = sum_{j=s+1}^t x_{j-1}^2

(the rho-free sum of x_j^2 drops out). Sxy and Sxx are differences of cumulative sums,
so each origin costs one small matmul over the grid. The posterior is normalised in log
space (max-shift), and the forecast is E[rho^h | data] x_t.

The grid kernel runs on torch CPU tensors in float64 purely for speed (multithreaded
exp is ~4x numpy's); every value stays float64 and is cross-checked against a
numpy brute force in tests/test_bayes_a.py.

`F` convention: shape (n, 81, 6), F[i, t, h-1] forecasts x[i, t+h]; F[:, 0] is NaN.
"""
from __future__ import annotations

import numpy as np
import torch

N_T = 81
N_H = 6
_CHUNK_ELEMS = 1 << 24      # (sequences x grid) per working array: 128 MiB of float64


def first_obs_var(rho: np.ndarray, s: int) -> np.ndarray:
    """sum_{k<s} rho^(2k): variance of x_s given x_0 = 0 and unit shocks. Equals s at rho = 1."""
    rho = np.asarray(rho, dtype=np.float64)
    if s < 1:
        raise ValueError("s must be >= 1")
    # Direct sum rather than (1 - rho^2s) / (1 - rho^2): exact at rho = 1, no cancellation near it.
    return (rho[..., None] ** (2.0 * np.arange(s))).sum(-1)


def bayes_a(x: np.ndarray, window: int | None, grid: int = 2001,
            prior: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Posterior-mean forecasts E[rho^h | x[s..t]] x_t for t = 1..80, h = 1..6."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < N_T:
        raise ValueError(f"x must be (n, >= {N_T}); got {x.shape}")
    if window is not None and window < 1:
        raise ValueError("window must be None or >= 1")
    if grid < 1:
        raise ValueError("grid must be >= 1")
    if not 0.0 <= prior[0] <= prior[1] <= 1.0:
        raise ValueError(f"prior must satisfy 0 <= lo <= hi <= 1; got {prior}")
    n = x.shape[0]

    rho = np.linspace(prior[0], prior[1], grid, dtype=np.float64)              # (G,)
    pows = torch.from_numpy(rho[:, None] ** np.arange(N_H + 1, dtype=np.float64))  # (G, 7): 1, rho..rho^6
    # Per first-observation index s = 1..80: rows multiplying [x_s^2, Sxy, Sxx, 1]
    # (the last row is the rho-only part: first-observation log-variance + log prior weight).
    V = np.stack([first_obs_var(rho, s) for s in range(1, N_T)])                # (80, G); row s-1
    coef = np.zeros((N_T, 4, grid))
    coef[1:, 0] = -0.5 / V
    coef[1:, 1] = rho
    coef[1:, 2] = -0.5 * rho * rho
    # Trapezoid quadrature of the continuous uniform prior, carried as log-weights in the
    # constant column. With lo == hi all points coincide, so the weights cancel exactly.
    log_w = np.zeros(grid)
    if grid > 1:
        log_w[[0, -1]] = np.log(0.5)
    coef[1:, 3] = -0.5 * np.log(V) + log_w
    coef = torch.from_numpy(coef)

    F = np.full((n, N_T, N_H), np.nan, dtype=np.float64)
    chunk = max(1, _CHUNK_ELEMS // grid)
    for i0 in range(0, n, chunk):
        xc = x[i0:i0 + chunk, :N_T]
        c = xc.shape[0]
        # Cxy[:, t] = sum_{j=2}^t x_j x_{j-1};  Cxx[:, t] = sum_{j=2}^t x_{j-1}^2  (zero for t <= 1)
        cxy = np.zeros((c, N_T))
        cxx = np.zeros((c, N_T))
        np.cumsum(xc[:, 2:] * xc[:, 1:-1], axis=1, out=cxy[:, 2:])
        np.cumsum(xc[:, 1:-1] ** 2, axis=1, out=cxx[:, 2:])
        stats = np.ones((c, 4))
        for t in range(1, N_T):
            s = 1 if window is None else max(1, t - window + 1)
            stats[:, 0] = xc[:, s] ** 2
            stats[:, 1] = cxy[:, t] - cxy[:, s]
            stats[:, 2] = cxx[:, t] - cxx[:, s]
            ll = torch.from_numpy(stats) @ coef[s]                              # (c, G) log-lik
            ll -= ll.amax(dim=1, keepdim=True)
            ll.exp_()
            m = (ll @ pows).numpy()                                             # (c, 7)
            F[i0:i0 + c, t] = m[:, 1:] / m[:, :1] * xc[:, t:t + 1]
    return F
