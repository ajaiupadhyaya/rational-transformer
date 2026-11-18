"""Survey comparison (analysis/spf.py) and the SPF panel export (scripts/export_spf_panel.py).

Everything runs on synthetic panels and stand-in forecasters: no real checkpoint, no EWH."""
import importlib.util
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from analysis import spf
from rt.bayes import bayes_b
from rt.train import T_MAX, build_model, load_cfg

ROOT = Path(__file__).resolve().parent.parent
START = pd.Period("1950Q1", freq="Q")
RHO = 0.6


def _export_module():
    s = importlib.util.spec_from_file_location("export_spf_panel",
                                               ROOT / "scripts" / "export_spf_panel.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def _q(p):
    return f"{p.year}Q{p.quarter}"


def ar1_panel(kind, n_vars=2, n_q=90, burn=70, mu=None, sigma=None, seed=0, rho=RHO):
    """Pure AR(1) first differences d (mean mu, innovation sd sigma), levels built from them,
    no revisions: the history at t is x[START .. t-1], the outcome the realised target.
    Survey columns are placeholders (filled by the tests that need them)."""
    rng = np.random.default_rng(seed)
    mu = (3.0 if kind == "growth" else 0.3) if mu is None else mu
    sigma = (2.0 if kind == "growth" else 0.5) if sigma is None else sigma
    rows, raw = [], {}
    for v in range(n_vars):
        n = burn + n_q + 4
        d = np.empty(n)
        d[0] = mu
        for i in range(1, n):
            d[i] = mu + rho * (d[i - 1] - mu) + sigma * rng.standard_normal()
        x = np.empty(n + 1)
        x[0] = 100.0 if kind == "growth" else 5.0
        for i in range(n):
            x[i + 1] = x[i] * (1 + d[i] / 400) if kind == "growth" else x[i] + d[i]
        var = f"V{v}"
        raw[var] = x
        for k in range(burn, burn + n_q):          # survey quarter t = START + k
            t = START + k
            out = 100 * (x[k + 3] / x[k - 1] - 1) if kind == "growth" else x[k + 3]
            rows.append({"variable": var, "kind": kind, "target": "t+3", "t": _q(t),
                         "hist_first": _q(START), "hist": list(x[:k]), "hist_source": "synthetic",
                         "hist_vintage": _q(t), "last_filled": False, "outcome": out, "cons_err": 0.0, "cons_rev": 0.0,
                         "n_pairs": 0, **{f"spf_c{j}": np.nan for j in range(1, 7)},
                         "ind_id": [], "ind_err": [], "ind_rev": []})
    return pd.DataFrame(rows, columns=spf.PANEL_COLUMNS), raw


def plugin_ar1(rho=RHO):
    """True-rho AR(1) forecaster with the window's sample mean: exactly location-scale
    equivariant, so scaling and unscaling must not change its forecasts at all."""
    def f(z):
        z = np.asarray(z, float)
        m = z.mean()
        return m + rho ** np.arange(1, 7) * (z[-1] - m)
    return f


# ---- direct computations (independent of analysis.spf) ------------------------

def _diffs(x, kind):
    return 400 * (x[1:] / x[:-1] - 1) if kind == "growth" else np.diff(x)


def direct_err_rev(panel, raw, kind, window, forecast_raw):
    """err and rev per quarter computed on the unscaled series, forecasts from
    forecast_raw(list of 1-D raw windows) -> (n, 6)."""
    errs, revs = [], []
    for var, g in panel.groupby("variable", sort=False):
        x = raw[var]
        ks = [int((pd.Period(t, freq="Q") - START).n) for t in g.t]
        wins = []
        for k in ks:   # vintage t = START + k holds x[0..k-1]; differences d[0..k-2]
            d = _diffs(x[:k], kind)
            wins.append(d[-(T_MAX if window is None else window):])
        F = forecast_raw(wins)
        for i, k in enumerate(ks):
            if i == 0:
                continue
            Ft, Fp = F[i], F[i - 1]
            if kind == "growth":
                fc = 100 * (np.prod(1 + Ft[0:4] / 400) - 1)
                fp = 100 * (np.prod(1 + Fp[1:5] / 400) - 1)
                out = 100 * (x[k + 3] / x[k - 1] - 1)
            else:
                fc = x[k - 1] + Ft[0:4].sum()
                fp = x[k - 2] + Fp[0:5].sum()
                out = x[k + 3]
            errs.append(out - fc)
            revs.append(fc - fp)
    return np.array(errs), np.array(revs)


def slope(y, x):
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc * xc).sum())


def pooled_b(table):
    return table.set_index("variable")["b_model"]


# ---- (a) no look-ahead in the scaling ------------------------------------------

@pytest.mark.parametrize("kind", ["growth", "level"])
def test_scale_history_uses_only_the_past(kind):
    rng = np.random.default_rng(1)
    idx = pd.period_range("1960Q1", "2005Q4", freq="Q")
    x = pd.Series(100 + np.cumsum(rng.normal(0.5, 1.0, len(idx))), index=idx)
    as_of = pd.Period("1990Q1", freq="Q")
    z, (loc, scale) = spf.scale_history(x, as_of, kind)
    y = x.copy()
    y[y.index >= as_of] = rng.normal(1e3, 1e2, int((y.index >= as_of).sum()))
    z2, (loc2, scale2) = spf.scale_history(y, as_of, kind)
    pd.testing.assert_series_equal(z, z2)
    assert (loc, scale) == (loc2, scale2)
    assert z.index.max() == as_of - 1
    past = x[x.index < as_of].to_numpy()
    d = _diffs(past, kind)
    assert np.isclose(loc, d.mean()) and np.isclose(scale, d.std(ddof=1))
    np.testing.assert_allclose(z.to_numpy(), (d - d.mean()) / d.std(ddof=1))


def test_model_quarters_refuse_histories_not_ending_at_t_minus_1():
    """A row whose history runs into t or later (or stops before t-1) is never used."""
    panel, _ = ar1_panel("level", n_vars=1, n_q=12)
    f = plugin_ar1()
    assert spf.model_quarters(f, panel)["fc"].notna().all()
    for change in (lambda h: list(h) + [1e6, -1e6], lambda h: list(h)[:-1]):
        bad = panel.assign(hist=[change(h) for h in panel["hist"]])
        assert spf.model_quarters(f, bad)["fc"].isna().all()


# ---- (b) pure AR(1) panel: the pipeline reproduces the stand-in's own b ----------

@pytest.mark.parametrize("kind", ["growth", "level"])
@pytest.mark.parametrize("window", [None, 4])
def test_plugin_ar1_b_matches_direct(kind, window):
    panel, raw = ar1_panel(kind, n_vars=2, n_q=90, seed=2)
    f = plugin_ar1()
    q = spf.model_quarters(f, panel, window)
    err, rev = direct_err_rev(panel, raw, kind, window,
                              lambda wins: np.stack([f(w) for w in wins]))
    m = q.dropna(subset=["err", "rev"])
    np.testing.assert_allclose(m.err.to_numpy(), err, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(m.rev.to_numpy(), rev, rtol=1e-9, atol=1e-9)
    # with the stand-in's own errors as the "survey", model and survey b coincide
    panel = panel.set_index(["variable", "t"])
    panel.loc[[(v, t) for v, t in zip(m.variable, m.t)], "cons_err"] = err
    panel.loc[[(v, t) for v, t in zip(m.variable, m.t)], "cons_rev"] = rev
    panel = panel.reset_index()[spf.PANEL_COLUMNS]
    first = panel.groupby("variable").t.transform("min") == panel.t
    panel.loc[first, ["cons_err", "cons_rev"]] = np.nan
    tab = spf.spf_comparison(f, panel, window)
    assert (tab["T"] == 89).all()
    np.testing.assert_allclose(tab.b_model, tab.b_survey, atol=1e-9)
    for var, g in m.groupby("variable"):
        b_direct = slope(err[(m.variable == var).to_numpy()], rev[(m.variable == var).to_numpy()])
        assert abs(pooled_b(tab)[var] - b_direct) < 1e-9


@pytest.mark.parametrize("window", [None, 4])
def test_bayes_b_standin_b_matches_direct_within_005(window):
    """Family B Bayes is not exactly scale-equivariant (its priors are in absolute units), so
    the b of the scaled pipeline and of Bayes run directly on the raw series agree only
    approximately: within 0.05, per variable."""
    kind = "level"
    panel, raw = ar1_panel(kind, n_vars=2, n_q=70, seed=3)
    f = spf.bayes_forecaster(window, grid=201)
    tab = spf.spf_comparison(f, panel, window)

    def raw_bayes(wins):
        x = np.zeros((len(wins), T_MAX + 7))
        ts = []
        for i, w in enumerate(wins):
            n = len(w)
            t = n if window is None else T_MAX
            x[i, t - n + 1:t + 1] = w
            ts.append(t)
        F = bayes_b(x, window, grid=201)
        return F[np.arange(len(wins)), ts]

    err, rev = direct_err_rev(panel, raw, kind, window, raw_bayes)
    q = spf.model_quarters(f, panel, window).dropna(subset=["err", "rev"])
    for var in tab.variable:
        sel = (q.variable == var).to_numpy()
        b_direct = slope(err[sel], rev[sel])
        assert abs(pooled_b(tab)[var] - b_direct) < 0.05, (var, pooled_b(tab)[var], b_direct)


# ---- (c) drop rule ----------------------------------------------------------------

def test_drop_check_keeps_a_clean_ar1_panel():
    panel, _ = ar1_panel("level", n_vars=2, n_q=30)
    d = spf.drop_check(panel)
    assert not d.drop, d.reasons
    assert d.in_range["full"] == 1.0 and d.in_range["4"] == 1.0
    assert "kept" in d.text
    lo, hi = d.bounds
    assert -6.5 < lo < -5.0 and 5.0 < hi < 6.5     # central 99% of standardised Family B values


@pytest.mark.parametrize("change", ["target", "kind", "horizon"])
def test_drop_check_fires_on_unmatched_horizons(change):
    panel, _ = ar1_panel("growth", n_vars=2, n_q=10)
    kw = {}
    if change == "target":
        panel.loc[panel.variable == "V1", "target"] = "t+5"
    elif change == "kind":
        panel.loc[panel.variable == "V1", "kind"] = "ratio"
    else:
        kw["max_horizon"] = 4          # growth revisions need horizon 5
    d = spf.drop_check(panel, **kw)
    assert d.drop
    assert any("horizons cannot be matched" in r for r in d.reasons)
    assert "dropped" in d.text


def _with_outliers(panel, every):
    """A huge one-quarter jump in the 10th-last level of every `every`-th row's history: its
    scaled difference lands far outside the Family B range for both windows."""
    p = panel.copy()
    hist = []
    for i, h in enumerate(p["hist"]):
        h = list(h)
        if i % every == 0:
            h[-4:] = [v + 1e3 for v in h[-4:]]
        hist.append(h)
    p["hist"] = hist
    return p


def test_drop_check_fires_above_20pct_out_of_range():
    panel, _ = ar1_panel("level", n_vars=2, n_q=30)
    d = spf.drop_check(_with_outliers(panel, every=3))       # 20 of 60 rows: 33%
    assert d.drop
    assert any("outside the Family B range" in r for r in d.reasons)
    assert d.in_range["4"] == pytest.approx(40 / 60)
    d = spf.drop_check(_with_outliers(panel, every=6))       # 10 of 60 rows: 17%
    assert not d.drop, d.reasons
    assert d.in_range["4"] == pytest.approx(50 / 60)


def test_spf_arm_returns_empty_table_when_dropped():
    panel, _ = ar1_panel("level", n_vars=1, n_q=12)
    tab, d = spf.spf_arm({"m": (plugin_ar1(), None)}, _with_outliers(panel, every=1))
    assert d.drop and tab.empty and list(tab.columns) == spf.COMPARISON_COLUMNS
    tab, d = spf.spf_arm({"m": (plugin_ar1(), None), "m4": (plugin_ar1(), 4)}, panel)
    assert not d.drop and set(tab.model) == {"m", "m4"} and (tab["T"] == 11).all()


# ---- checkpoint wrapper (synthetic checkpoint, never a real one) --------------------

def _ckpt(root, cell_id, cfg, model):
    d = root / cell_id
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg, "model": model.state_dict()}, d / "step1.pt")
    return d / "step1.pt"


@pytest.mark.parametrize("window", [None, 4])
def test_forecaster_from_checkpoint_on_a_synthetic_checkpoint(tmp_path, window):
    cfg = load_cfg()
    torch.manual_seed(0)
    model = build_model(cfg, "xs")
    path = _ckpt(tmp_path, f"B-xs-w{'full' if window is None else window}-s0", cfg, model)
    f = spf.forecaster_from_checkpoint(path, window, size="xs")
    z = np.random.default_rng(0).standard_normal(120)
    out = f(z)
    assert out.shape == (6,) and out.dtype == np.float64 and np.all(np.isfinite(out))
    z2 = z.copy()
    z2[:-(T_MAX if window is None else window)] += 50.0     # outside the fed values
    np.testing.assert_array_equal(out, f(z2))
    z3 = z.copy()
    z3[-1] += 1.0
    assert not np.array_equal(out, f(z3))


# ---- (d) export script on a tiny fixture -------------------------------------------

class FixtureSource:
    def __init__(self):
        self.quarters = [pd.Period(q, freq="Q") for q in ("2000Q1", "2000Q2", "2000Q3")]

    def pairs(self, var):
        rows = []
        for t in self.quarters:
            for i in (1, 2, 3):
                err = np.nan if (i == 3 and t == self.quarters[0]) else 0.1 * i
                rows.append({"ID": i, "t": t, "fc": 1.0, "fc_prev": 0.5, "outcome": 2.0,
                             "err": err, "rev": 0.2 * i})
        return pd.DataFrame(rows)

    def forecasts(self, var):
        rows = [{"ID": i, "yq": t, **{f"c{k}": float(k + i) for k in range(1, 7)}}
                for t in self.quarters for i in (1, 2)]
        return pd.DataFrame(rows)

    def history(self, var, t):
        idx = pd.period_range("1999Q1", t + 1, freq="Q")         # includes t and t+1: dropped
        # UNEMP 2000Q2 claims a later vintage: the export must not use it
        vintage = t + 1 if (var == "UNEMP" and t.quarter == 2) else t
        return pd.Series(np.arange(len(idx), dtype=float) + 10, index=idx), "fixture", vintage


def test_export_schema_on_fixture(tmp_path):
    ex = _export_module()
    assert ex.PANEL_COLUMNS == spf.PANEL_COLUMNS
    panel = ex.build_panel(FixtureSource(), ("NGDP", "UNEMP"))
    out = tmp_path / "panel.parquet"
    ex.write_panel(panel, out)
    p = spf.load_panel(out)
    assert list(p.columns) == spf.PANEL_COLUMNS and len(p) == 6
    assert list(p.kind) == ["growth"] * 3 + ["level"] * 3
    assert set(p.target) == {"t+3"}
    assert list(p.t[:3]) == ["2000Q1", "2000Q2", "2000Q3"]
    for _, r in p.iterrows():
        lv = spf.row_levels(r)
        if r.variable == "UNEMP" and r.t == "2000Q2":
            assert len(lv) == 0 and r.hist_vintage == ""     # later vintage refused outright
            continue
        assert r.hist_vintage == r.t
        assert lv.index.min() == pd.Period("1999Q1", freq="Q")
        assert lv.index.max() == pd.Period(r.t, freq="Q") - 1       # nothing dated >= t
    assert not p.last_filled.any()
    r0 = p.iloc[0]
    assert r0.n_pairs == 2 and list(r0.ind_id) == [1, 2]
    assert r0.cons_err == pytest.approx(0.15) and r0.cons_rev == pytest.approx(0.4)
    assert r0.spf_c1 == pytest.approx(2.5) and r0.outcome == 2.0
    assert p.dtypes["outcome"] == np.float64 and p.dtypes["n_pairs"] == np.int64


def test_export_refuses_replication_packages(tmp_path, monkeypatch):
    ex = _export_module()
    monkeypatch.setattr(zipfile, "ZipFile", zipfile.ZipFile)    # restored after the test
    for sub in ("bgms", "cg", "spf"):
        (tmp_path / sub).mkdir()
        with zipfile.ZipFile(tmp_path / sub / "x.zip", "w") as z:
            z.writestr("a.txt", "a")
    ex._forbid_replication_packages(tmp_path)
    for sub in ("bgms", "cg"):
        with pytest.raises(PermissionError):
            zipfile.ZipFile(tmp_path / sub / "x.zip")
    zipfile.ZipFile(tmp_path / "spf" / "x.zip").close()


# ---- strict real time (review C1, C2) ------------------------------------------------

def _vintage_matrix():
    """Observations 1990Q1..1995Q4 x vintages 1994Q1..1996Q2. Vintage v holds obs < v, with
    values 1000 * (vintage index) + obs index so each vintage is identifiable; vintage 1996Q1
    lacks 1995Q4 (late release); vintages before 1994Q1 do not exist."""
    obs = pd.period_range("1990Q1", "1995Q4", freq="Q")
    vin = pd.period_range("1994Q1", "1996Q2", freq="Q")
    m = pd.DataFrame(np.nan, index=obs, columns=vin)
    for j, v in enumerate(vin):
        for i, o in enumerate(obs):
            if o < v:
                m.loc[o, v] = 1000.0 * (j + 1) + i
    m.loc[pd.Period("1995Q4", freq="Q"), pd.Period("1996Q1", freq="Q")] = np.nan
    return m


def test_export_history_never_comes_from_a_vintage_after_t():
    ex = _export_module()
    m = _vintage_matrix()
    for t in pd.period_range("1992Q1", "1996Q2", freq="Q"):
        lv, v = ex.vintage_history(m, t)
        if t < m.columns.min():
            assert v is None and len(lv) == 0           # no same-date vintage: nothing
            continue
        assert v == t
        assert (lv.index < t).all()
        np.testing.assert_array_equal(lv.to_numpy(), m[t].dropna().to_numpy())
        assert (lv // 1000 == list(m.columns).index(t) + 1).all()    # all from vintage t
    lv, _ = ex.vintage_history(m, pd.Period("1996Q1", freq="Q"))
    assert lv.index.max() == pd.Period("1995Q3", freq="Q")           # late x[t-1] not filled


def test_rows_not_strictly_real_time_drop_out_for_model_and_survey():
    """A filled row, a row whose x[t-1] is missing, or a row from another vintage: no model
    forecast at t, no revision at t+1, and both quarters leave the survey's b too."""
    f = plugin_ar1()
    panel, _ = ar1_panel("level", n_vars=1, n_q=20, seed=5)
    rng = np.random.default_rng(6)
    panel["cons_err"], panel["cons_rev"] = rng.standard_normal(20), rng.standard_normal(20)
    panel.loc[0, ["cons_err", "cons_rev"]] = np.nan
    base = spf.spf_comparison(f, panel)
    assert base["T"].iloc[0] == 19
    i = 10
    for change in ("filled", "missing", "vintage"):
        bad = panel.copy()
        if change == "filled":
            bad.loc[i, "last_filled"] = True
        elif change == "missing":
            h = list(bad.loc[i, "hist"])
            h[-1] = np.nan
            bad.at[i, "hist"] = h
        else:
            bad.loc[i, "hist_vintage"] = _q(pd.Period(bad.loc[i, "t"], freq="Q") + 1)
        assert spf.not_real_time(bad.loc[i]) is not None
        q = spf.model_quarters(f, bad).set_index("t")
        t, t1 = bad.loc[i, "t"], bad.loc[i + 1, "t"]
        assert np.isnan(q.loc[t, "fc"]) and np.isnan(q.loc[t1, "fc_prev"])
        assert q.drop(index=[t, t1])[["err", "rev"]].iloc[1:].notna().all().all()
        tab = spf.spf_comparison(f, bad)
        assert tab["T"].iloc[0] == 17
        keep = ~panel.t.isin([t, t1]) & panel.cons_err.notna()
        assert tab.b_survey.iloc[0] == pytest.approx(
            slope(panel.cons_err[keep].to_numpy(), panel.cons_rev[keep].to_numpy()))
        d = spf.drop_check(bad)
        assert d.table.not_real_time.iloc[0] == 1 and (d.table.n == 18).all()


# ---- per-cell 20% rule (review I1, controller ruling 2026-11-18) ----------------------

def test_spf_arm_omits_a_variable_window_cell_over_20pct():
    panel, _ = ar1_panel("level", n_vars=5, n_q=12)
    p = panel.copy()
    bad = p.variable == "V0"                          # 12 of 60 rows: pooled 20%, not > 20%
    p.loc[bad, "hist"] = pd.Series([list(h[:-4]) + [v + 1e3 for v in h[-4:]]
                                    for h in p.loc[bad, "hist"]], index=p.index[bad])
    d = spf.drop_check(p)
    assert not d.drop
    tbl = d.table.set_index(["variable", "window"])
    assert tbl.loc[("V0", "full"), "frac_out"] == 1.0 and tbl.loc[("V1", "4"), "frac_out"] == 0.0
    assert {(o["variable"], o["window"]) for o in d.omitted} == {("V0", "full"), ("V0", "4")}
    assert "Omitted from the table" in d.text and "V0 window full" in d.text
    tab, d2 = spf.spf_arm({"m": (plugin_ar1(), None), "m4": (plugin_ar1(), 4)}, p)
    assert "V0" not in set(tab.variable) and len(tab) == 8
    assert (tab.frac_out == 0.0).all() and list(tab.columns) == spf.COMPARISON_COLUMNS
    assert d2.omitted == d.omitted


def test_forecaster_from_checkpoint_validates_family_size_window(tmp_path):
    cfg = load_cfg()
    torch.manual_seed(0)
    model = build_model(cfg, "xs")
    a = _ckpt(tmp_path, "A-xs-w4-s0", cfg, model)
    b4 = _ckpt(tmp_path, "B-xs-w4-s0", cfg, model)
    bf = _ckpt(tmp_path, "B-xs-wfull-s0", cfg, model)
    with pytest.raises(ValueError, match="Family A"):
        spf.forecaster_from_checkpoint(a, 4, size="xs")
    with pytest.raises(ValueError, match="window is 4, expected full"):
        spf.forecaster_from_checkpoint(b4, None, size="xs")
    with pytest.raises(ValueError, match="window is full, expected 4"):
        spf.forecaster_from_checkpoint(bf, 4, size="xs")
    with pytest.raises(ValueError, match="size 'xs', expected 'l'"):
        spf.forecaster_from_checkpoint(b4, 4, size="l")
    odd = tmp_path / "somewhere"
    odd.mkdir()
    torch.save({"config": cfg, "model": model.state_dict()}, odd / "step1.pt")
    with pytest.raises(ValueError, match="not a cell id"):
        spf.forecaster_from_checkpoint(odd / "step1.pt", 4, size="xs")
    (b4.parent / "run.json").write_text('{"cell": "B-xs-w8-s0"}')
    with pytest.raises(ValueError, match="run.json"):
        spf.forecaster_from_checkpoint(b4, 4, size="xs")
    (b4.parent / "run.json").write_text('{"cell": "B-xs-w4-s0"}')
    assert spf.forecaster_from_checkpoint(b4, 4, size="xs")(np.zeros(10)).shape == (6,)


@pytest.mark.parametrize("window,n", [(4, 2), (8, 3), (None, 7)])
def test_short_history_is_padded_as_in_training(tmp_path, window, n):
    from rt.train import batch_from_x
    cfg = load_cfg()
    torch.manual_seed(1)
    model = build_model(cfg, "xs").eval()
    path = _ckpt(tmp_path, f"B-xs-w{'full' if window is None else window}-s0", cfg, model)
    f = spf.forecaster_from_checkpoint(path, window, size="xs")
    v = np.random.default_rng(3).standard_normal(n)
    x = np.zeros((1, 87))
    x[0, 1:n + 1] = v                                  # a series that has run for n periods
    values, times, _, _ = batch_from_x(x, window, t=None if window is None else np.array([n]))
    with torch.no_grad():
        out = model(values, times)
    ref = (out[0, n - 1] if window is None else out[0, -1]).numpy().astype(np.float64)
    np.testing.assert_allclose(f(v), ref, atol=1e-5)
