import numpy as np
import pandas as pd
import pytest

import rt.evaluate as ev
from rt.bayes import bayes_a, bayes_b
from rt.evaluate import (COLUMNS, bench_forecasts, eval_sets, evaluate_all, evaluate_cell,
                         model_forecasts, set_metrics)
from rt.model import SIZES, Forecaster
from rt.sim import RHOS, Batch, sample_family_a, sample_family_b
from rt.train import Cell, load_cfg, train_cell


# ---- tests from the brief -------------------------------------------------

def test_model_forecasts_respect_window(tmp_path):
    m = Forecaster(*SIZES["xs"])
    b = sample_family_a(8, np.random.default_rng(0))
    x2 = b.x.copy(); x2[:, :40] += 5.0                 # change only old history
    F1, F2 = model_forecasts(m, b.x, 4, "cpu"), model_forecasts(m, x2, 4, "cpu")
    assert np.allclose(F1[:, 44:], F2[:, 44:], atol=1e-5)


def test_short_history_matches_full_window_input():
    m = Forecaster(*SIZES["xs"]); b = sample_family_a(8, np.random.default_rng(1))
    assert np.allclose(model_forecasts(m, b.x, 16, "cpu")[:, 1:16],
                       model_forecasts(m, b.x, None, "cpu")[:, 1:16], atol=1e-5)


def _tiny_cfg(**kw):
    base = dict(max_steps=100, checkpoints=[100], batch_size=256,
                eval_n_per_rho=500, eval_n_prior=500)
    return load_cfg(**{**base, **kw})


def test_evaluate_cell_schema(tmp_path):
    cfg = _tiny_cfg()
    cell = Cell("A", "xs", None, 0)
    train_cell(cell, cfg, tmp_path / "ck", "cpu")
    df = evaluate_cell(cell, 100, cfg, "cpu", ckpt_dir=tmp_path / "ck",
                       eval_dir=tmp_path / "eval")
    assert list(df.columns) == COLUMNS == [
        "cell_id", "family", "size", "window", "seed", "step", "converged", "rho", "h",
        "bench", "implied_persistence", "bench_implied_persistence", "gap", "eor_b",
        "bench_eor_b", "excess_mse_pct", "kappa"]
    assert len(df) == 7 * 6 * 2
    assert set(df.rho) == set(RHOS) | {"prior"}
    assert set(df.h) == set(range(1, 7)) and set(df.bench) == {"bayes", "window_bayes"}
    assert (df.cell_id == cell.id).all() and (df.window == "full").all() and (df.step == 100).all()
    bayes = df[df.bench == "bayes"]
    assert (bayes.excess_mse_pct > 0).all()            # a barely trained model is worse than Bayes
    # full window: window_bayes is bayes
    wb = df[df.bench == "window_bayes"]
    assert np.allclose(bayes.excess_mse_pct.values, wb.excess_mse_pct.values)
    # kappa only for bayes at h = 1; eor_b undefined at h = 6
    has_kappa = df.kappa.notna()
    assert (has_kappa == ((df.bench == "bayes") & (df.h == 1))).all()
    assert df[df.h == 6][["eor_b", "bench_eor_b"]].isna().all().all()
    assert df[df.h < 6][["eor_b", "bench_eor_b"]].notna().all().all()
    assert np.allclose(df.gap, df.implied_persistence - df.bench_implied_persistence)


# ---- evaluate_all: every checkpoint, convergence, atomic CSV ----------------

def test_evaluate_all_writes_metrics_and_convergence(tmp_path):
    cfg = _tiny_cfg(max_steps=60, checkpoints=[30, 60], eval_n_per_rho=200, eval_n_prior=300)
    cell = Cell("A", "xs", 4, 1)
    train_cell(cell, cfg, tmp_path / "ck", "cpu")
    df = evaluate_all(cell, cfg, "cpu", ckpt_dir=tmp_path / "ck", out_dir=tmp_path / "m",
                      eval_dir=tmp_path / "eval")
    path = tmp_path / "m" / f"{cell.id}.csv"
    assert path.exists() and not list((tmp_path / "m").glob("*.tmp"))
    back = pd.read_csv(path)
    assert list(back.columns) == COLUMNS and len(back) == 2 * 84
    assert sorted(set(back.step)) == [30, 60] and (back.window == 4).all()
    assert back.converged.dtype == bool and back.converged.nunique() == 1
    # windowed cell: window_bayes differs from bayes
    s = df[(df.step == 60) & (df.h == 1) & (df.rho == "prior")].set_index("bench")
    assert s.loc["bayes", "excess_mse_pct"] != s.loc["window_bayes", "excess_mse_pct"]

    # convergence = |mse(last) - mse(prev)| <= rel_tol * mse(prev), on prior h = 1
    def conv(rel_tol):
        c = load_cfg(**{**cfg, "convergence": {**cfg["convergence"], "rel_tol": rel_tol}})
        return bool(evaluate_all(cell, c, "cpu", ckpt_dir=tmp_path / "ck", out_dir=tmp_path / "m",
                                 eval_dir=tmp_path / "eval").converged.iloc[0])
    assert conv(1e9) and not conv(0.0)
    # a missing checkpoint means not converged, whatever the tolerance
    (tmp_path / "ck" / cell.id / "step30.pt").unlink()
    assert not conv(1e9)


# ---- Family B: diagnostics on standardised deviations ---------------------

def test_family_b_diagnostics_invariant_to_per_sequence_location_and_scale():
    rng = np.random.default_rng(3)
    b = sample_family_b(400, rng, rho=0.6)
    Fb = bayes_b(b.x, None, grid=101)
    Fw = bayes_b(b.x, 4, grid=101)
    # a stand-in "model": shrinks the Bayes forecast part-way toward the current value
    Fm = Fb.copy()
    Fm[:, 1:] = 0.7 * Fb[:, 1:] + 0.3 * b.x[:, 1:81, None]
    rows1, mse1 = set_metrics(b, Fm, {"bayes": Fb, "window_bayes": Fw})

    c = rng.normal(0.0, 20.0, size=400)                # a different shift and scale per sequence
    s = np.exp(rng.normal(0.0, 1.0, size=400))
    b2 = Batch(x=c[:, None] + s[:, None] * b.x, rho=b.rho, mu=c + s * b.mu, sigma=s * b.sigma)

    def tr(F):
        return c[:, None, None] + s[:, None, None] * F
    rows2, mse2 = set_metrics(b2, tr(Fm), {"bayes": tr(Fb), "window_bayes": tr(Fw)})
    d1, d2 = pd.DataFrame(rows1), pd.DataFrame(rows2)
    num = d1.select_dtypes("number").columns
    assert np.allclose(d1[num].values, d2[num].values, rtol=1e-8, atol=1e-10, equal_nan=True)
    assert np.isclose(mse1, mse2, rtol=1e-10)
    # and the pooled raw slope really would have moved (so the test has teeth)
    from rt.diagnostics import implied_persistence
    assert abs(implied_persistence(tr(Fm), b2.x) - d1.implied_persistence.iloc[0]) > 0.01


def test_family_a_metrics_use_raw_values():
    b = sample_family_a(300, np.random.default_rng(4), rho=0.4)
    Fb = bayes_a(b.x, None, grid=201)
    Fm = 0.9 * Fb
    rows, _ = set_metrics(b, Fm, {"bayes": Fb, "window_bayes": Fb})
    from rt.diagnostics import excess_mse_pct, implied_persistence
    r = pd.DataFrame(rows).query("h == 2 and bench == 'bayes'").iloc[0]
    assert r.implied_persistence == implied_persistence(Fm, b.x, 2)
    assert r.excess_mse_pct == excess_mse_pct(Fm, Fb, b.x, 2)


# ---- evaluation sets and the benchmark cache ------------------------------

def test_eval_sets_are_fixed_and_separate_per_family():
    cfg = load_cfg(eval_n_per_rho=30, eval_n_prior=40)
    a1, a2, bb = eval_sets("A", cfg), eval_sets("A", cfg), eval_sets("B", cfg)
    assert list(a1) == [f"rho={r}" for r in RHOS] + ["prior"]
    for r in RHOS:
        assert a1[f"rho={r}"].x.shape == (30, 87) and (a1[f"rho={r}"].rho == r).all()
    assert a1["prior"].x.shape == (40, 87) and a1["prior"].rho.std() > 0.1
    assert np.array_equal(a1["prior"].x, a2["prior"].x)
    assert not np.allclose(a1["rho=0.6"].x[:, 1:], bb["rho=0.6"].x[:, 1:])
    assert not np.allclose(a1["rho=0.6"].x, a1["rho=0.8"].x)


def test_benchmark_cache_is_keyed_by_size_seed_and_grid(tmp_path, monkeypatch):
    calls = []
    real = ev.bayes_a
    monkeypatch.setattr(ev, "bayes_a", lambda x, w, grid: calls.append((len(x), w, grid))
                        or real(x, w, grid=grid))
    sets = eval_sets("A", load_cfg(eval_n_per_rho=40, eval_n_prior=50))
    f1 = bench_forecasts("A", 4, "rho=0.6", sets["rho=0.6"], eval_dir=tmp_path)
    assert len(calls) == 2                               # full and window-4
    f1b = bench_forecasts("A", 4, "rho=0.6", sets["rho=0.6"], eval_dir=tmp_path)
    assert len(calls) == 2                               # cache hit
    assert np.array_equal(f1b["window_bayes"], f1["window_bayes"], equal_nan=True)
    assert np.array_equal(f1["bayes"], bayes_a(sets["rho=0.6"].x, None), equal_nan=True)

    # a changed evaluation size never reads the old file
    sets2 = eval_sets("A", load_cfg(eval_n_per_rho=60, eval_n_prior=50))
    f2 = bench_forecasts("A", 4, "rho=0.6", sets2["rho=0.6"], eval_dir=tmp_path)
    assert len(calls) == 4 and f2["bayes"].shape == (60, 81, 6)
    assert np.array_equal(f2["window_bayes"], bayes_a(sets2["rho=0.6"].x, 4), equal_nan=True)

    # a changed grid never reads the old file
    f3 = bench_forecasts("A", 4, "rho=0.6", sets["rho=0.6"], grid=101, eval_dir=tmp_path)
    assert len(calls) == 6
    assert np.array_equal(f3["bayes"], bayes_a(sets["rho=0.6"].x, None, grid=101), equal_nan=True)

    # the same size under another seed (another key) is another file too
    f4 = bench_forecasts("A", 4, "rho=0.8", sets["rho=0.8"], eval_dir=tmp_path)
    assert len(calls) == 8 and not np.allclose(f4["bayes"][:, 1:], f1["bayes"][:, 1:])
    assert not list(tmp_path.glob("*.tmp"))


def test_bench_forecasts_family_b_uses_grid_2001_and_full_window_alias(tmp_path):
    assert ev.GRID == {"A": 2001, "B": 2001}
    b = eval_sets("B", load_cfg(eval_n_per_rho=10, eval_n_prior=10))["rho=1.0"]
    f = bench_forecasts("B", None, "rho=1.0", b, eval_dir=tmp_path,
                        names=("bayes", "window_bayes", "fire", "ls"))
    assert np.array_equal(f["bayes"], bayes_b(b.x, None, grid=2001), equal_nan=True)
    assert f["window_bayes"] is f["bayes"]              # one array, standardised once
    assert np.allclose(f["fire"][:, 1:, 0], b.x[:, 1:81])       # rho = 1: random walk
    assert f["ls"].shape == (10, 81, 6)


# ---- architecture options survive the checkpoint ---------------------------

def test_affine_head_respects_window_in_model_forecasts():
    import torch
    torch.manual_seed(0)
    m = Forecaster(*SIZES["xs"], head="affine", inputs="raw+asinh")
    with torch.no_grad():
        m.head.weight[:6].normal_(0.0, 0.1)
    b = sample_family_a(8, np.random.default_rng(0))
    x2 = b.x.copy(); x2[:, :40] += 5.0                 # change only old history
    F1, F2 = model_forecasts(m, b.x, 4, "cpu"), model_forecasts(m, x2, 4, "cpu")
    assert np.allclose(F1[:, 44:], F2[:, 44:], atol=1e-5)
    assert not np.allclose(F1[:, 1:40], F2[:, 1:40], atol=1e-3)


@pytest.mark.parametrize("window", [None, 4])
def test_non_default_variant_checkpoint_evaluates_with_that_variant(tmp_path, window):
    import torch
    train = _tiny_cfg(max_steps=20, checkpoints=[20], head="affine", inputs="raw+asinh")
    cell = Cell("A", "xs", window, 0)
    train_cell(cell, train, tmp_path / "ck", "cpu")
    path = tmp_path / "ck" / cell.id / "step20.pt"
    model, saved = ev._load_model(cell, path)
    assert (model.head_kind, model.inputs_kind) == ("affine", "raw+asinh")
    assert saved["head"] == "affine" and saved["inputs"] == "raw+asinh"
    # the same forecasts as a model built by hand from the variant
    by_hand = Forecaster(*SIZES["xs"], head="affine", inputs="raw+asinh")
    by_hand.load_state_dict(torch.load(path, weights_only=True)["model"])
    b = sample_family_a(6, np.random.default_rng(2))
    assert np.allclose(model_forecasts(model, b.x, window, "cpu"),
                       model_forecasts(by_hand, b.x, window, "cpu"), equal_nan=True)
    # evaluation with the DEFAULT evaluation config still uses the checkpoint's variant
    ev_cfg = _tiny_cfg(eval_n_per_rho=50, eval_n_prior=50)
    assert ev_cfg["head"] == "linear"
    df = evaluate_cell(cell, 20, ev_cfg, "cpu", ckpt_dir=tmp_path / "ck",
                       eval_dir=tmp_path / "eval")
    assert len(df) == 84 and df.excess_mse_pct.notna().all()
    out = evaluate_all(cell, ev_cfg, "cpu", ckpt_dir=tmp_path / "ck", out_dir=tmp_path / "m",
                       eval_dir=tmp_path / "eval")
    assert np.allclose(out.excess_mse_pct.values, df.excess_mse_pct.values)
