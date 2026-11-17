"""Exact (grid) Bayes and window-Bayes forecasts for Families A and B.

Family A (`bayes_a`)
--------------------

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

Family B (`bayes_b`)
--------------------
Model: sigma^2 ~ InvGamma(a, b), mu | sigma^2 ~ N(0, k sigma^2), x_0 = mu,
x_t = mu + rho (x_{t-1} - mu) + sigma eps_t, rho ~ U[0, 1] (trapezoid grid, as above).
Given rho, the shown data x[s..t] are a Gaussian linear regression in mu with m = t - s + 1
rows y_i = z_i mu + sigma e_i:
    first row (x_s ~ N(mu, sigma^2 V), V = first_obs_var(rho, s); V = 1 when s = 1):
        y = x_s / sqrt(V),  z = 1 / sqrt(V)   (Jacobian of the rescaling: -1/2 log V)
    transitions j = s+1..t:  y = x_j - rho x_{j-1},  z = 1 - rho.
With the Normal-Inverse-Gamma prior (prior precision 1/k on mu / sigma^2, mean 0):
    L     = 1/k + sum z^2 = 1/k + 1/V + (t - s)(1 - rho)^2      (data-free)
    Szy   = sum z y = x_s / V + (1 - rho) A - rho (1 - rho) B,   A = sum x_j, B = sum x_{j-1}
    Q     = sum y^2 - Szy^2 / L                                  (>= 0)
    E[mu | rho, data] = Szy / L,   b_n = b + Q / 2,   a_n = a + m / 2
    log p(data | rho) = -1/2 log V + 1/2 log((1/k) / L) - a_n log b_n + (rho-free terms)
(sums over j = s+1..t). Q is linear in nine data statistics -- x_s^2, sum x_j^2,
sum x_j x_{j-1}, sum x_{j-1}^2, x_s A, x_s B, A^2, A B, B^2 -- with rho-only coefficients,
so each origin is one (c, 9) @ (9, G) matmul plus a (c, 3) @ (3, G) one for Szy.
Forecast: posterior mean over rho of rho^h x_t + (1 - rho^h) Szy / L.

`F` convention: shape (n, 81, 6), F[i, t, h-1] forecasts x[i, t+h]; F[:, 0] is NaN.
"""
from __future__ import annotations

import numpy as np
import torch

from rt.sim import B_PRIOR

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


def _trapezoid_log_weights(grid: int) -> np.ndarray:
    log_w = np.zeros(grid)
    if grid > 1:
        log_w[[0, -1]] = np.log(0.5)
    return log_w


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
    coef[1:, 3] = -0.5 * np.log(V) + _trapezoid_log_weights(grid)
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


def bayes_b(x: np.ndarray, window: int | None, grid: int = 401,
            prior: dict = B_PRIOR) -> np.ndarray:
    """Posterior-mean forecasts E[x_{t+h} | x[s..t]] for Family B, t = 1..80, h = 1..6.

    Grid over rho only; mu and sigma^2 are integrated analytically (module docstring).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < N_T:
        raise ValueError(f"x must be (n, >= {N_T}); got {x.shape}")
    if window is not None and window < 1:
        raise ValueError("window must be None or >= 1")
    if grid < 1:
        raise ValueError("grid must be >= 1")
    a, b, k = float(prior["a"]), float(prior["b"]), float(prior["k"])
    if not (a > 0 and b > 0 and k > 0):
        raise ValueError(f"prior a, b, k must be positive; got {prior}")
    n = x.shape[0]

    rho = np.linspace(0.0, 1.0, grid, dtype=np.float64)                         # (G,)
    omr = 1.0 - rho
    pows = rho[:, None] ** np.arange(N_H + 1, dtype=np.float64)                 # (G, 7)
    pows_t = torch.from_numpy(pows)
    log_w = _trapezoid_log_weights(grid)

    # Rho-only coefficient tables, one per (first index s, number of transitions d = t - s).
    def tables(s: int, d: int):
        V = first_obs_var(rho, s)
        L = 1.0 / k + 1.0 / V + d * omr * omr
        cz = np.stack([1.0 / V, omr, -rho * omr])                               # Szy over [x_s, A, B]
        cq = np.stack([
            (1.0 / k + d * omr * omr) / (V * L),     # x_s^2: 1/V - 1/(V^2 L), without cancellation
            np.ones(grid),                           # sum x_j^2
            -2.0 * rho,                              # sum x_j x_{j-1}
            rho * rho,                               # sum x_{j-1}^2
            -2.0 * omr / (V * L),                    # x_s A
            2.0 * rho * omr / (V * L),               # x_s B
            -omr * omr / L,                          # A^2
            2.0 * rho * omr * omr / L,               # A B
            -(rho * omr) ** 2 / L,                   # B^2
        ])
        const = -0.5 * np.log(V) - 0.5 * np.log(L) + log_w                       # (G,)
        mu_coef = (1.0 - pows[:, 1:]) / L[:, None]                               # (G, 6)
        return (torch.from_numpy(cz), torch.from_numpy(cq), torch.from_numpy(const),
                torch.from_numpy(mu_coef))

    cache: dict[tuple[int, int], tuple] = {}
    F = np.full((n, N_T, N_H), np.nan, dtype=np.float64)
    chunk = max(1, _CHUNK_ELEMS // grid)
    for i0 in range(0, n, chunk):
        xc = x[i0:i0 + chunk, :N_T]
        c = xc.shape[0]
        # Cumulative sums over transitions j = 2..t (zero for t <= 1), so that a sum over
        # j = s+1..t is C[:, t] - C[:, s].
        cA, cB, cyy, cxy, cxx = (np.zeros((c, N_T)) for _ in range(5))
        np.cumsum(xc[:, 2:], axis=1, out=cA[:, 2:])
        np.cumsum(xc[:, 1:-1], axis=1, out=cB[:, 2:])
        np.cumsum(xc[:, 2:] ** 2, axis=1, out=cyy[:, 2:])
        np.cumsum(xc[:, 2:] * xc[:, 1:-1], axis=1, out=cxy[:, 2:])
        np.cumsum(xc[:, 1:-1] ** 2, axis=1, out=cxx[:, 2:])
        sz = np.empty((c, 3))
        sq = np.empty((c, 9))
        for t in range(1, N_T):
            s = 1 if window is None else max(1, t - window + 1)
            d = t - s
            if (s, d) not in cache:
                cache[(s, d)] = tables(s, d)
            cz, cq, const, mu_coef = cache[(s, d)]
            xs = xc[:, s]
            A = cA[:, t] - cA[:, s]
            B = cB[:, t] - cB[:, s]
            sz[:, 0], sz[:, 1], sz[:, 2] = xs, A, B
            sq[:, 0] = xs * xs
            sq[:, 1] = cyy[:, t] - cyy[:, s]
            sq[:, 2] = cxy[:, t] - cxy[:, s]
            sq[:, 3] = cxx[:, t] - cxx[:, s]
            sq[:, 4], sq[:, 5] = xs * A, xs * B
            sq[:, 6], sq[:, 7], sq[:, 8] = A * A, A * B, B * B
            szy = torch.from_numpy(sz) @ cz                                     # (c, G)
            q = torch.from_numpy(sq) @ cq                                       # (c, G)
            # Q >= 0 exactly (Cauchy-Schwarz); clamp only guards rounding in the expansion.
            ll = q.clamp_(min=0.0).mul_(0.5).add_(b).log_().mul_(-(a + 0.5 * (d + 1))).add_(const)
            ll -= ll.amax(dim=1, keepdim=True)
            p = ll.exp_()
            m0 = (p @ pows_t).numpy()                                           # (c, 7)
            m1 = ((p * szy) @ mu_coef).numpy()                                  # (c, 6)
            F[i0:i0 + c, t] = (m0[:, 1:] * xc[:, t:t + 1] + m1) / m0[:, :1]
    return F
