import numpy as np

from rt.baselines import fire, ls_learner
from rt.bayes import bayes_a, first_obs_var
from rt.sim import T_EVAL, sample_family_a


def _eor_slope(F, x, h):
    """OLS slope (with intercept), pooled over t in T_EVAL, of the forecast error
    x[t+h] - F[t, h-1] on the revision F[t, h-1] - F[t-1, h].

    Local stand-in for `eor_b`, which lands in Task 4.
    """
    t = np.array(T_EVAL)
    err = (x[:, t + h] - F[:, t, h - 1]).ravel()
    rev = (F[:, t, h - 1] - F[:, t - 1, h]).ravel()
    rc = rev - rev.mean()
    return (rc * (err - err.mean())).sum() / (rc * rc).sum()


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
    assert abs(_eor_slope(F, b.x, h=1)) < 0.01


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


# --- independent reference checks (not in the brief) ---------------------------

def _normal_logpdf(z, var):
    return -0.5 * np.log(2 * np.pi * var) - 0.5 * z * z / var


def _reference_bayes(xi, t, window, rhos):
    """Brute force for one sequence: explicit per-observation Gaussian log-densities."""
    s = 1 if window is None else max(1, t - window + 1)
    v = np.array([sum(r ** (2 * k) for k in range(s)) for r in rhos])
    ll = _normal_logpdf(xi[s], v)
    for j in range(s + 1, t + 1):
        ll = ll + _normal_logpdf(xi[j] - rhos * xi[j - 1], 1.0)
    w = np.exp(ll - ll.max())
    w /= w.sum()
    return np.array([(w * rhos ** h).sum() * xi[t] for h in range(1, 7)])


def test_bayes_matches_brute_force_reference():
    b = sample_family_a(6, np.random.default_rng(5))
    b.x[0] = sample_family_a(1, np.random.default_rng(6), rho=1.0).x[0]
    rhos = np.linspace(0.0, 1.0, 41)
    for window in (None, 1, 2, 4, 16):
        F = bayes_a(b.x, window, grid=41)
        assert F.shape == (6, 81, 6) and np.isnan(F[:, 0]).all()
        for i in range(6):
            for t in (1, 2, 3, 5, 17, 40, 80):
                ref = _reference_bayes(b.x[i], t, window, rhos)
                assert np.allclose(F[i, t], ref, rtol=1e-10, atol=1e-12), (window, i, t)


def test_ls_learner_matches_polyfit():
    b = sample_family_a(4, np.random.default_rng(7))
    for intercept in (False, True):
        F = ls_learner(b.x, intercept=intercept)
        assert F.shape == (4, 81, 6) and np.isnan(F[:, 0]).all()
        for i in range(4):
            for t in (3, 10, 80):
                y, z = b.x[i, 2:t + 1], b.x[i, 1:t]          # history x[1..t]
                if intercept:
                    bb, a = np.polyfit(z, y, 1)
                else:
                    a, bb = 0.0, np.linalg.lstsq(z[:, None], y, rcond=None)[0][0]
                f, ref = b.x[i, t], []
                for _ in range(6):
                    f = a + bb * f
                    ref.append(f)
                assert np.allclose(F[i, t], ref, rtol=1e-10, atol=1e-12)
