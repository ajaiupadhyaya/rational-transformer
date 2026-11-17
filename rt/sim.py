"""AR(1) simulators for the two data families.

Family A: x_0 = 0, mu = 0, sigma = 1, rho ~ U[0, 1] (or fixed).
Family B: sigma^2 ~ InvGamma(a, b), mu | sigma^2 ~ N(0, k * sigma^2), x_0 = mu,
          x_t = mu + rho * (x_{t-1} - mu) + sigma * eps_t.

All arrays are float64. `x` has shape (n, T+1); column 0 is the starting value.
"""
from dataclasses import dataclass

import numpy as np

RHOS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
T_EVAL = range(41, 81)
HORIZONS = (1, 2, 3, 4, 5, 6)
B_PRIOR = dict(a=3.0, b=2.0, k=25.0)  # InvGamma(shape a, scale b); mu | s2 ~ N(0, k * s2)


@dataclass
class Batch:
    x: np.ndarray      # (n, T+1)
    rho: np.ndarray    # (n,)
    mu: np.ndarray     # (n,)
    sigma: np.ndarray  # (n,)


def _draw_rho(n: int, rng: np.random.Generator, rho: float | None) -> np.ndarray:
    if rho is None:
        return rng.uniform(0.0, 1.0, size=n)
    return np.full(n, float(rho), dtype=np.float64)


def _simulate(rho, mu, sigma, T, rng) -> np.ndarray:
    n = rho.shape[0]
    eps = rng.standard_normal((n, T))
    x = np.empty((n, T + 1), dtype=np.float64)
    x[:, 0] = mu
    for t in range(1, T + 1):
        x[:, t] = mu + rho * (x[:, t - 1] - mu) + sigma * eps[:, t - 1]
    return x


def sample_family_a(n: int, rng: np.random.Generator,
                    rho: float | None = None, T: int = 86) -> Batch:
    r = _draw_rho(n, rng, rho)
    mu = np.zeros(n)
    sigma = np.ones(n)
    return Batch(x=_simulate(r, mu, sigma, T, rng), rho=r, mu=mu, sigma=sigma)


def sample_family_b(n: int, rng: np.random.Generator,
                    rho: float | None = None, T: int = 86) -> Batch:
    a, b, k = B_PRIOR["a"], B_PRIOR["b"], B_PRIOR["k"]
    r = _draw_rho(n, rng, rho)
    sigma2 = b / rng.gamma(shape=a, scale=1.0, size=n)  # InvGamma(a, scale b)
    sigma = np.sqrt(sigma2)
    mu = rng.standard_normal(n) * np.sqrt(k * sigma2)
    return Batch(x=_simulate(r, mu, sigma, T, rng), rho=r, mu=mu, sigma=sigma)
