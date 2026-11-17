import numpy as np
import pytest

from rt.baselines import fire
from rt.bayes import bayes_a
from rt.diagnostics import (eor_b, excess_mse_pct, gap, implied_persistence,
                            kappa, verdict)
from rt.sim import sample_family_a


def test_planted_kappa_recovered():
    b = sample_family_a(100_000, np.random.default_rng(0))
    Fb = bayes_a(b.x, None)
    for theta in (-0.3, 0.0, 0.2, 0.5):
        Fm = Fb.copy()
        Fm[:, 2:, 0] += theta * (b.x[:, 2:81] - Fb[:, 1:80, 0])
        assert abs(kappa(Fm, Fb, b.x) - theta) < 0.01


def test_planted_persistence_offset_recovered():
    b = sample_family_a(100_000, np.random.default_rng(1), rho=0.4)
    Fm = fire(b.x, 0.55)
    assert abs(gap(Fm, fire(b.x, 0.4), b.x) - 0.15) < 0.005
    assert abs(implied_persistence(fire(b.x, 0.4), b.x) - 0.4) < 1e-9


def test_fire_eor_is_zero_and_overreaction_is_negative():
    b = sample_family_a(100_000, np.random.default_rng(2), rho=0.6)
    assert abs(eor_b(fire(b.x, 0.6), b.x, 1)) < 0.01
    assert eor_b(fire(b.x, 0.9), b.x, 1) < -0.05      # too-persistent forecaster overreacts


def test_verdict():
    assert verdict([0.001, -0.002, 0.003, 0.0, 0.001], 0.02) == "equivalent"
    assert verdict([0.08, 0.09, 0.07, 0.085, 0.075], 0.02) == "different"
    assert verdict([0.01, 0.03, 0.02, 0.025, 0.015], 0.02) == "inconclusive"


def test_verdict_needs_two_values():
    with pytest.raises(ValueError):
        verdict([0.1], 0.02)


def test_eor_b_rejects_bad_horizon():
    b = sample_family_a(10, np.random.default_rng(3))
    for h in (0, 6):
        with pytest.raises(ValueError):
            eor_b(fire(b.x, 0.5), b.x, h)


def test_excess_mse_pct():
    b = sample_family_a(2_000, np.random.default_rng(4), rho=0.5)
    F = fire(b.x, 0.5)
    assert excess_mse_pct(F, F, b.x, 1) == 0.0
    G = F.copy()
    G[:, :, 0] += 1.0          # shifts the error; MSE must rise
    assert excess_mse_pct(G, F, b.x, 1) > 0
