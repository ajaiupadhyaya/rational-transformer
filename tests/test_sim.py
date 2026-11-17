import numpy as np

from rt.sim import sample_family_a, sample_family_b


def test_family_a_moments():
    b = sample_family_a(200_000, np.random.default_rng(0), rho=0.6)
    assert b.x.shape == (200_000, 87) and np.all(b.x[:, 0] == 0)
    # stationary variance 1/(1-0.36) = 1.5625, reached by t=80
    assert abs(b.x[:, 80].var() - 1.5625) < 0.03
    assert abs(np.corrcoef(b.x[:, 79], b.x[:, 80])[0, 1] - 0.6) < 0.01


def test_family_a_random_walk_variance_grows():
    b = sample_family_a(200_000, np.random.default_rng(1), rho=1.0)
    assert abs(b.x[:, 50].var() - 50) < 1.0


def test_family_a_prior_draw():
    b = sample_family_a(100_000, np.random.default_rng(2))
    assert 0 <= b.rho.min() and b.rho.max() <= 1 and abs(b.rho.mean() - 0.5) < 0.01


def test_family_b_centres_on_mu():
    b = sample_family_b(200_000, np.random.default_rng(3), rho=0.0)
    assert abs(np.mean(b.x[:, 40] - b.mu)) < 0.02
    assert abs(np.mean(b.sigma**2) - 1.0) < 0.05   # InvGamma(3,2) mean = 1


def test_family_b_mu_prior_scale():
    # mu / sigma ~ N(0, k) exactly, with k = 25 (light-tailed, unlike mu itself, whose
    # variance 25 E[sigma^2] is a mixture over an inverse-gamma with shape 3).
    b = sample_family_b(400_000, np.random.default_rng(4), T=1)
    z = b.mu / b.sigma
    # sd of the sample variance is 25 * sqrt(2 / 400_000) ~= 0.056; tolerance ~5 sd.
    assert abs(z.var() - 25.0) < 0.3
    assert abs(z.mean()) < 0.05      # sd of the mean is 5 / sqrt(400_000) ~= 0.008
