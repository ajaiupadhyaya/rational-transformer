import math

import numpy as np

from rt.baselines import ls_learner
from rt.bayes import bayes_b
from rt.sim import B_PRIOR, T_EVAL, sample_family_b


def _eor_slope(F, x, h):
    """OLS slope (with intercept), pooled over t in T_EVAL, of the forecast error
    x[t+h] - F[t, h-1] on the revision F[t, h-1] - F[t-1, h].

    Local stand-in for `eor_b`, which lands in Task 4 (copied from tests/test_bayes_a.py).
    """
    t = np.array(T_EVAL)
    err = (x[:, t + h] - F[:, t, h - 1]).ravel()
    rev = (F[:, t, h - 1] - F[:, t - 1, h]).ravel()
    rc = rev - rev.mean()
    return (rc * (err - err.mean())).sum() / (rc * rc).sum()


# --- brute-force reference: explicit 3-D grid over (rho, mu, sigma), no conjugacy -----

def _trapezoid_weights(g):
    w = np.ones(g)
    if g > 1:
        w[0] = w[-1] = 0.5
    return w


def _first_obs_var_by_recursion(rhos, s):
    """Var(x_s - mu) / sigma^2 given x_0 = mu: v_0 = 0, v_j = rho^2 v_{j-1} + 1.
    Deliberately not rt.bayes.first_obs_var (which is a direct power sum)."""
    v = np.zeros_like(rhos)
    for _ in range(s):
        v = rhos * rhos * v + 1.0
    return v


def brute_force_b(x, t, window=None, h=1, n_rho=161, n_mu=601, n_logsig=161):
    """E[x_{t+h} | x[s..t]] for each sequence, by explicit quadrature on a 3-D grid.

    The joint density at every grid point is the product of the prior densities
    InvGamma(a, b) on sigma^2, N(0, k sigma^2) on mu, U[0, 1] on rho, and the Gaussian
    density of every observed value given its AR(1) parent (or, for the first observed
    value x_s, its marginal N(mu, sigma^2 V_s(rho)) over the unseen x_1..x_{s-1}). The
    forecast integrand is the conditional mean mu + rho^h (x_t - mu). Nothing here uses
    Normal-Inverse-Gamma conjugacy.

    Grids: rho uniform on [0, 1] (trapezoid); log sigma uniform on a range set from the
    sample scale; mu uniform on a range covering the data, 0 (the prior mean) and a
    margin of 12 posterior-scale sigmas. All three are trapezoid rules; the change of
    variables to log sigma contributes the Jacobian sigma.
    """
    a, b, k = B_PRIOR["a"], B_PRIOR["b"], B_PRIOR["k"]
    rhos = np.linspace(0.0, 1.0, n_rho)
    w_rho = _trapezoid_weights(n_rho)
    s = 1 if window is None else max(1, t - window + 1)
    V = _first_obs_var_by_recursion(rhos, s)                       # (R,)
    out = np.empty(len(x))
    for i, xi in enumerate(np.asarray(x, dtype=np.float64)):
        obs = xi[s:t + 1]
        # sigma grid: wide in log space around a crude data scale (prior mode is ~0.7).
        scale = max(np.std(np.diff(obs)) if len(obs) > 1 else 0.0, abs(obs[0]) / 5, 0.5)
        logsig = np.linspace(np.log(scale) - 4.5, np.log(scale) + 4.0, n_logsig)
        sig = np.exp(logsig)
        # mu grid: covers min/max of the data and of 0, plus 12 x (an upper sigma scale).
        lo, hi = min(obs.min(), 0.0), max(obs.max(), 0.0)
        pad = 12.0 * scale * max(1.0, math.sqrt(s))
        mus = np.linspace(lo - pad, hi + pad, n_mu)
        w_mu, w_sig = _trapezoid_weights(n_mu), _trapezoid_weights(n_logsig)

        # Sum over observations of squared standardised residuals, on the (rho, mu) grid.
        R, M = rhos[:, None], mus[None, :]
        ss = (obs[0] - M) ** 2 / V[:, None]                         # first observation
        for j in range(1, len(obs)):
            ss = ss + (obs[j] - (M + R * (obs[j - 1] - M))) ** 2    # x_j | x_{j-1}
        n_obs = len(obs)
        S2 = sig[None, None, :] ** 2                                 # (1, 1, S)

        # log of: prod_j N(x_j; mean_j, sigma^2 v_j) x N(mu; 0, k sigma^2)
        #         x InvGamma(sigma^2; a, b) x |d sigma^2 / d log sigma| (= 2 sigma^2)
        loglik = (-0.5 * n_obs * np.log(2 * np.pi * S2) - 0.5 * np.log(V)[:, None, None]
                  - ss[:, :, None] / (2 * S2))
        logprior_mu = -0.5 * np.log(2 * np.pi * k * S2) - M[:, :, None] ** 2 / (2 * k * S2)
        logprior_s2 = (a * math.log(b) - math.lgamma(a) - (a + 1) * np.log(S2) - b / S2
                       + np.log(2 * S2))
        logp = loglik + logprior_mu + logprior_s2
        p = np.exp(logp - logp.max())
        p *= w_rho[:, None, None] * w_mu[None, :, None] * w_sig[None, None, :]
        cond_mean = M + R ** h * (xi[t] - M)                          # (R, M)
        out[i] = (p.sum(2) * cond_mean).sum() / p.sum()
    return out


# --- direct conjugate reference: explicit regression rows, no expanded sums -----------

def _direct_conjugate(xi, t, window, rhos, h_max=6):
    """Per-sequence, per-rho Normal-Inverse-Gamma posterior from explicit row vectors.

    Checks rt.bayes.bayes_b's expanded sufficient statistics (cumulative sums and a
    single matmul) against the textbook formulas evaluated on the residual vectors.
    """
    a, b, k = B_PRIOR["a"], B_PRIOR["b"], B_PRIOR["k"]
    s = 1 if window is None else max(1, t - window + 1)
    ll, mun = [], []
    for r in rhos:
        V = sum(r ** (2 * q) for q in range(s))
        y = np.concatenate([[xi[s] / math.sqrt(V)], xi[s + 1:t + 1] - r * xi[s:t]])
        z = np.concatenate([[1 / math.sqrt(V)], np.full(t - s, 1.0 - r)])
        lam = 1 / k + z @ z
        m = (z @ y) / lam
        bn = b + 0.5 * (y @ y - lam * m * m)
        an = a + 0.5 * len(y)
        ll.append(-0.5 * math.log(V) + 0.5 * math.log((1 / k) / lam) - an * math.log(bn))
        mun.append(m)
    ll, mun = np.array(ll), np.array(mun)
    w = np.exp(ll - ll.max()) * _trapezoid_weights(len(rhos))
    w /= w.sum()
    return np.array([(w * (rhos ** h * xi[t] + (1 - rhos ** h) * mun)).sum()
                     for h in range(1, h_max + 1)])


# --- brief tests ---------------------------------------------------------------

def test_b_error_orthogonal_to_revision_under_prior():
    b = sample_family_b(100_000, np.random.default_rng(0))
    assert abs(_eor_slope(bayes_b(b.x, None), b.x, 1)) < 0.01


def test_b_beats_ls_with_intercept_and_window():
    b = sample_family_b(50_000, np.random.default_rng(1))
    mse = lambda F: np.nanmean((b.x[:, 42:82] - F[:, 41:81, 0]) ** 2)
    full = bayes_b(b.x, None)
    assert mse(full) < mse(bayes_b(b.x, 8)) and mse(full) < mse(ls_learner(b.x, intercept=True))


def test_b_matches_brute_force_on_tiny_case():
    # 3-D grid posterior over (rho, mu, sigma) on 5 sequences, t = 6, horizon 1
    b = sample_family_b(5, np.random.default_rng(2))
    assert np.allclose(bayes_b(b.x, None)[:, 6, 0], brute_force_b(b.x, t=6), rtol=2e-3)


# --- additional checks (not in the brief) -----------------------------------------

def test_b_windowed_matches_brute_force():
    # s > 1: the first observed value's rho-dependent variance V_s and its Jacobian matter.
    b = sample_family_b(4, np.random.default_rng(3))
    b.x[0] = sample_family_b(1, np.random.default_rng(4), rho=1.0).x[0]
    for window, t, h in ((3, 7, 1), (4, 9, 3), (2, 30, 2)):
        ref = brute_force_b(b.x, t=t, window=window, h=h)
        assert np.allclose(bayes_b(b.x, window)[:, t, h - 1], ref, rtol=2e-3), (window, t)


def test_b_matches_direct_conjugate_formulas():
    paths = np.concatenate([sample_family_b(2, np.random.default_rng(20 + q), rho=rho).x
                            for q, rho in enumerate((0.0, 0.5, 0.97, 1.0))])
    paths[-1] *= 40.0                       # large-scale path: stresses the expanded sums
    rhos = np.linspace(0.0, 1.0, 51)
    for window in (None, 1, 2, 5, 16):
        F = bayes_b(paths, window, grid=51)
        assert F.shape == (len(paths), 81, 6) and np.isnan(F[:, 0]).all()
        for i in range(len(paths)):
            for t in (1, 2, 3, 6, 17, 40, 80):
                ref = _direct_conjugate(paths[i], t, window, rhos)
                assert np.allclose(F[i, t], ref, rtol=1e-9, atol=1e-9 * np.abs(paths[i]).max()), \
                    (window, i, t)


def test_b_default_grid_quadrature_error_is_small_and_second_order():
    # The default 401-point rho grid against an 8001-point one, in units of each path's
    # sigma. When the posterior piles up against rho = 1 the trapezoid rule is O(grid^-2)
    # (the integrand is cut off with non-zero slope), so the default grid is accurate to a
    # few thousandths of a shock sd in the worst cells, not to 1e-6. Measured on these
    # paths: max 3.4e-3 (full), 2.3e-3 (window 8); at grid 2001, 1.3e-4 and 8.5e-5.
    bs = [sample_family_b(3, np.random.default_rng(30 + q), rho=rho)
          for q, rho in enumerate((0.0, 0.6, 0.95, 1.0))]
    paths = np.concatenate([bb.x for bb in bs])
    sig = np.concatenate([bb.sigma for bb in bs])[:, None, None]
    for window in (None, 8):
        Ff = bayes_b(paths, window, grid=8001)[:, 1:]
        e401 = (np.abs(bayes_b(paths, window)[:, 1:] - Ff) / sig).max()
        e2001 = (np.abs(bayes_b(paths, window, grid=2001)[:, 1:] - Ff) / sig).max()
        assert e401 < 5e-3 and e2001 < e401 / 15, (window, e401, e2001)


def test_b_window_longer_than_history_equals_full():
    b = sample_family_b(200, np.random.default_rng(5))
    assert np.allclose(bayes_b(b.x, 16)[:, 1:16], bayes_b(b.x, None)[:, 1:16], rtol=1e-12)


def test_b_boundaries_do_not_nan():
    for rho in (0.0, 1.0):
        b = sample_family_b(300, np.random.default_rng(6), rho=rho)
        for window in (None, 1, 2):
            assert np.isfinite(bayes_b(b.x, window)[:, 1:81]).all()
