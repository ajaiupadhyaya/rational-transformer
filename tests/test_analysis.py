"""Tests for analysis/: synthetic metrics in temporary directories only (never results/)."""
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analysis import benchmarks, human, report
from analysis.hypotheses import (bonferroni_t, flagged_findings, hypotheses, prior_rows,
                                 resolve_decisions, training_length_check)
from analysis.summary import (convergence_cells, coverage, load_metrics, run_table, summarise)
from rt.sim import RHOS
from rt.sweep import grid
from rt.train import Cell, load_cfg

COLUMNS = ["cell_id", "family", "size", "window", "seed", "step", "converged", "rho", "h",
           "bench", "implied_persistence", "bench_implied_persistence", "gap", "eor_b",
           "bench_eor_b", "excess_mse_pct", "kappa"]
RHO_KEYS = [str(r) for r in RHOS] + ["prior"]
OFFS = [-0.002, -0.001, 0.0, 0.001, 0.002]          # seed offsets: sd 0.00158


def default_value(cell, step, rho, h, bench, column):
    base = 0.0 if column == "gap" else 0.1
    return base + (OFFS[cell.seed] if column == "gap" else OFFS[cell.seed] * 100)


def write_run(d, cell, value_fn=default_value, steps=(3000, 10000, 30000), converged=True):
    rows = []
    for step in steps:
        for rho in RHO_KEYS:
            for h in range(1, 7):
                for bench in ("bayes", "window_bayes"):
                    gap = value_fn(cell, step, rho, h, bench, "gap")
                    rows.append([cell.id, cell.family, cell.size,
                                 "full" if cell.window is None else cell.window, cell.seed, step,
                                 converged, rho, h, bench, 0.5 + gap, 0.5, gap, 0.1, 0.1,
                                 value_fn(cell, step, rho, h, bench, "excess_mse_pct"),
                                 0.01 if bench == "bayes" and h == 1 else np.nan])
    pd.DataFrame(rows, columns=COLUMNS).to_csv(Path(d) / f"{cell.id}.csv", index=False)


def cells_l(sizes=("l",), seeds=range(5), windows=(None, 2, 4, 8, 16), family="A"):
    return [Cell(family, s, w, sd) for s in sizes for w in windows for sd in seeds]


def make_dir(tmp_path, name, cells, **kw):
    d = tmp_path / name
    d.mkdir()
    for c in cells:
        write_run(d, c, **kw)
    return d


@pytest.fixture(scope="module")
def full_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("full")
    for c in cells_l():
        write_run(d, c)
    return d


def row(summary, **kw):
    m = pd.Series(True, index=summary.index)
    for k, v in kw.items():
        m &= summary[k].astype(str) == str(v)
    r = summary[m]
    assert len(r) == 1, (kw, len(r))
    return r.iloc[0]


# ---------------------------------------------------------------- summary

def test_generator_shape(tmp_path):
    d = make_dir(tmp_path, "g", [Cell("A", "l", 4, 0)])
    assert len(pd.read_csv(d / "A-l-w4-s0.csv")) == 252


def test_verdict_interval_five_seeds(full_dir):
    s = summarise(full_dir)
    r = row(s, family="A", size="l", window="full", step=30000, rho="0.6", h=1, bench="bayes",
            column="gap")
    assert r.n_seeds == 5 and json.loads(r["values"]) == OFFS
    assert r["mean"] == pytest.approx(0.0, abs=1e-12)
    sd = np.std(OFFS, ddof=1)
    half = 2.7764451051977934 * sd / np.sqrt(5)
    assert r["sd"] == pytest.approx(sd) and r["hi"] == pytest.approx(half)
    assert r["lo"] == pytest.approx(-half)
    assert r["verdict"] == "equivalent"
    e = row(s, family="A", size="l", window="full", step=30000, rho="0.6", h=1, bench="bayes",
            column="excess_mse_pct")
    assert e["verdict"] == "equivalent" and e["mean"] == pytest.approx(0.1)


def test_three_seed_m_uses_two_df(tmp_path):
    d = make_dir(tmp_path, "m", cells_l(sizes=("m",), seeds=range(3), windows=(4,)))
    r = row(summarise(d), size="m", step=30000, rho="0.6", h=1, bench="bayes", column="gap")
    assert r.n_seeds == 3
    sd = np.std(OFFS[:3], ddof=1)
    assert r["hi"] - r["mean"] == pytest.approx(4.302652729749462 * sd / np.sqrt(3))


def test_one_seed_insufficient(tmp_path):
    d = make_dir(tmp_path, "one", [Cell("A", "l", 4, 0)])
    r = row(summarise(d), step=30000, rho="0.6", h=1, bench="bayes", column="gap")
    assert r.n_seeds == 1 and r.verdict == "insufficient seeds"
    assert np.isnan(r["lo"]) and np.isnan(r["hi"])


def test_convergence_never_changes_verdict(tmp_path):
    d = make_dir(tmp_path, "nc", [Cell("A", "l", 4, s) for s in range(5)], converged=False,
                 value_fn=lambda c, st, rho, h, b, col: 0.2 + OFFS[c.seed] if col == "gap" else 0.1)
    df = load_metrics(d)
    s = summarise(df)
    r = row(s, step=30000, rho="0.6", h=1, bench="window_bayes", column="gap")
    assert r.verdict == "different" and not r.all_converged
    cc = convergence_cells(run_table(df))
    assert not cc.iloc[0].converged and cc.iloc[0].n_converged == 0


def test_cell_converged_needs_all_runs(tmp_path):
    d = make_dir(tmp_path, "cv", [Cell("A", "l", 4, s) for s in range(5)])
    write_run(d, Cell("A", "l", 4, 4), converged=False)
    cc = convergence_cells(run_table(load_metrics(d)))
    assert cc.iloc[0].n_converged == 4 and not cc.iloc[0].converged


# ---------------------------------------------------------------- hypotheses

def vfn(overrides=None, default=default_value):
    """value_fn with overrides {(size_window_key predicate)}: callable(cell, rho, column, bench) -> mean."""
    def f(cell, step, rho, h, bench, column):
        v = default(cell, step, rho, h, bench, column)
        if overrides and step == 30000 and h == 1 and cell.family == "A":
            shift = overrides(cell, rho, column, bench)
            if shift is not None:
                return v + shift
        return v
    return f


def hyp_from(tmp_path, name, overrides=None, cells=None):
    d = make_dir(tmp_path, name, cells or cells_l(), value_fn=vfn(overrides))
    return hypotheses(summarise(d)), d


def test_h1_supported(full_dir):
    h = hypotheses(summarise(full_dir))
    assert h["h1"]["decision"] == "supported" and len(h["h1"]["rows"]) == 12
    assert all(r["verdict"] == "equivalent" for r in h["h1"]["rows"])


def test_h1_rejected_and_flagged(tmp_path):
    h, _ = hyp_from(tmp_path, "h1r", lambda c, rho, col, b:
                    0.2 if (c.window is None and rho == "0.8" and col == "gap" and b == "bayes") else None)
    assert h["h1"]["decision"] == "rejected_pending_check"
    f = [x for x in h["flagged"] if x["hypothesis"] == "H1"]
    assert len(f) == 1 and f[0]["rho"] == "0.8" and f[0]["sign"] == "+"
    assert f[0]["cells"] == [f"A-l-wfull-s{i}" for i in range(5)]


def test_h1_not_established_inconclusive(tmp_path):
    h, _ = hyp_from(tmp_path, "h1n", lambda c, rho, col, b:
                    0.02 if (c.window is None and rho == "0.2" and col == "gap" and b == "bayes") else None)
    assert [r["verdict"] for r in h["h1"]["rows"] if r["rho"] == "0.2" and r["column"] == "gap"] == ["inconclusive"]
    assert h["h1"]["decision"] == "not_established" and h["flagged"] == []


def test_h2_per_window_independence(tmp_path):
    h, _ = hyp_from(tmp_path, "h2", lambda c, rho, col, b:
                    -3.0 if (c.window == 4 and rho == "0.4" and col == "excess_mse_pct"
                             and b == "window_bayes") else None)
    d = {w: h["h2"]["windows"][w]["decision"] for w in ("2", "4", "8", "16")}
    assert d == {"2": "rational_given_information", "4": "departure_pending_check",
                 "8": "rational_given_information", "16": "rational_given_information"}
    f = [x for x in h["flagged"] if x["hypothesis"] == "H2"]
    assert len(f) == 1 and f[0]["window"] == "4" and f[0]["sign"] == "-"
    assert f[0]["cells"] == [f"A-l-w4-s{i}" for i in range(5)]


def test_h2_other_sizes_do_not_enter_decision(tmp_path):
    cells = cells_l() + cells_l(sizes=("xs",))
    h, _ = hyp_from(tmp_path, "h2o", lambda c, rho, col, b: 0.5 if c.size == "xs" and col == "gap" else None,
                    cells=cells)
    assert h["h2"]["windows"]["4"]["decision"] == "rational_given_information"
    assert h["h2"]["other_sizes"]["xs"]["4"]["decision"] == "departure_pending_check"
    assert h["flagged"] == []
    assert h["h2"]["other_sizes"]["s"]["4"]["decision"] == "not_established"      # nothing run


def test_h3_bonferroni_multiplier(tmp_path):
    assert bonferroni_t(5) == pytest.approx(5.261, abs=1e-3)
    sd = np.std(OFFS, ddof=1)
    # mean m: lo95 = m - 2.776 sd/sqrt5 ; lo_adj = m - 5.261 sd/sqrt5.  Choose m between.
    lo95_mult, adj_mult = 2.7764451051977934, 5.261057575065803
    m = 0.02 + (lo95_mult + adj_mult) / 2 * sd / np.sqrt(5)
    h, _ = hyp_from(tmp_path, "h3a", lambda c, rho, col, b:
                    m if (c.window == 4 and rho == "0.6" and col == "gap" and b == "window_bayes") else None)
    c = [x for x in h["h3"]["combos"] if x["window"] == "4" and x["rho"] == "0.6"][0]
    assert c["positive_95"] and not c["positive_adjusted"]
    assert c["lo_adj"] == pytest.approx(m - adj_mult * sd / np.sqrt(5))
    assert h["h3"]["k_positive_95"] == 1 and h["h3"]["k_positive_adjusted"] == 0
    assert h["h3"]["decision"] == "not_supported"
    assert all(f["hypothesis"] != "H3" for f in h["flagged"])
    assert h["h3"]["statement"] == "1 of 8 positive at 95%, none after adjustment"
    # clearly adjusted-positive
    h2, _ = hyp_from(tmp_path, "h3b", lambda c, rho, col, b:
                     0.2 if (c.window == 8 and rho == "0.8" and col == "gap" and b == "window_bayes") else None)
    assert h2["h3"]["decision"] == "supported_pending_check"
    assert [f["hypothesis"] for f in h2["flagged"]].count("H3") == 1


def test_h3_denominator_eight_with_missing(tmp_path):
    cells = cells_l(windows=(2, 4))
    h, _ = hyp_from(tmp_path, "h3m", None, cells=cells)
    assert h["h3"]["denominator"] == 8 and len(h["h3"]["combos"]) == 8
    assert h["h3"]["n_missing"] == 4
    assert h["h3"]["decision"] == "not_established"
    assert "of 8 positive at 95%" in h["h3"]["statement"]


def test_missing_cells_not_established(tmp_path):
    cells = cells_l(windows=(None, 2, 4, 8))          # no window 16, no ... 
    h, _ = hyp_from(tmp_path, "miss", None, cells=cells)
    assert h["h2"]["windows"]["16"]["decision"] == "not_established"
    assert all(r["status"] == "not run" for r in h["h2"]["windows"]["16"]["rows"])
    assert h["h2"]["windows"]["8"]["decision"] == "rational_given_information"
    # a 'different' still rejects when other rows are missing
    h2, _ = hyp_from(tmp_path, "miss2", lambda c, rho, col, b:
                     0.3 if (c.window == 2 and rho == "0.0" and col == "gap") else None,
                     cells=cells_l(windows=(2,)))
    assert h2["h2"]["windows"]["2"]["decision"] == "departure_pending_check"
    assert h2["h1"]["decision"] == "not_established" and h2["h1"]["missing"]


def test_incomplete_seeds_not_used_in_decision(tmp_path):
    cells = cells_l(windows=(None,), seeds=range(4))
    h, _ = hyp_from(tmp_path, "inc", lambda c, rho, col, b: 0.3 if col == "gap" else None, cells=cells)
    assert h["h1"]["decision"] == "not_established"
    assert all(r["status"] == "incomplete" for r in h["h1"]["rows"])
    assert h["flagged"] == []


def test_hypotheses_read_only_prereg_rows(tmp_path):
    d = make_dir(tmp_path, "base", cells_l() + cells_l(family="B", seeds=range(3), windows=(None, 4)))
    base = summarise(d)
    h0 = hypotheses(base)

    def perturb(mask_fn):
        s = base.copy()
        m = mask_fn(s)
        assert m.any()
        s.loc[m, "values"] = json.dumps([0.9, 0.8, 0.95, 0.85, 0.9])
        s.loc[m, "mean"] = 0.88
        s.loc[m, "verdict"] = "different"
        return s

    for fn in (lambda s: s.h >= 2,
               lambda s: s.step != 30000,
               lambda s: s.rho == "prior",
               lambda s: s.family == "B",
               lambda s: (s.bench == "bayes") & (s.window != "full"),
               lambda s: (s.bench == "window_bayes") & (s.window == "full")):
        assert hypotheses(perturb(fn)) == h0
    # the prior rows are printed but separate
    assert prior_rows(perturb(lambda s: s.rho == "prior")) != prior_rows(base)


# ---------------------------------------------------------------- training-length check

def flagged_setup(tmp_path):
    ov = lambda c, rho, col, b: (0.3 if (c.window == 4 and rho == "0.6" and col == "gap" and b == "window_bayes") else
                                 0.2 if (c.window == 8 and rho == "0.8" and col == "gap" and b == "window_bayes") else None)
    h, _ = hyp_from(tmp_path, "fl", ov)
    return h


def long_dir(tmp_path, name, fn, windows):
    cells = cells_l(windows=windows)
    return make_dir(tmp_path, name, cells, steps=(60000,),
                    value_fn=lambda c, st, rho, hh, b, col: default_value(c, st, rho, hh, b, col)
                    + (fn(c, rho, col, b) or 0.0))


def test_flagged_and_training_length_outcomes(tmp_path):
    h = flagged_setup(tmp_path)
    fl = h["flagged"]
    assert {(f["hypothesis"], f["window"], f["rho"]) for f in fl} >= {("H2", "4", "0.6"), ("H2", "8", "0.8"),
                                                                       ("H3", "4", "0.6"), ("H3", "8", "0.8")}
    # survives: W=4 still different (+), W=8 gone; W=16 not re-trained
    d = long_dir(tmp_path, "l60", lambda c, rho, col, b: 0.3 if (c.window == 4 and rho == "0.6" and col == "gap"
                                                               and b == "window_bayes") else None, (4, 8))
    out = {(f["hypothesis"], f["window"], f["rho"], f["column"]): f
           for f in training_length_check(fl, summarise(d))}
    assert out[("H2", "4", "0.6", "gap")]["outcome"] == "survives"
    assert out[("H3", "4", "0.6", "gap")]["outcome"] == "survives"
    assert out[("H2", "8", "0.8", "gap")]["outcome"] == "not robust to training length"
    assert out[("H2", "8", "0.8", "gap")]["all_equivalent_at_60k"] is True
    assert out[("H3", "8", "0.8", "gap")]["outcome"] == "not robust to training length"
    # not run
    res = training_length_check(fl, None)
    assert all(f["outcome"] == "not run" for f in res)
    # sign flip does not survive
    d2 = long_dir(tmp_path, "l60b", lambda c, rho, col, b: -0.3 if (c.window == 4 and rho == "0.6"
                                                                    and col == "gap") else None, (4,))
    r = [f for f in training_length_check(fl, summarise(d2)) if f["hypothesis"] == "H2" and f["window"] == "4"][0]
    assert r["outcome"] == "not robust to training length"
    # resolved decisions
    dec = resolve_decisions(h, training_length_check(fl, summarise(d)))
    assert dec["h2_w4"] == "departure_from_rational_use" and dec["h2_w8"] == "not_established"
    assert dec["h3"] == "supported"
    dec_nr = resolve_decisions(h, training_length_check(fl, None))
    assert "not tested for training length" in dec_nr["h2_w4"]


def test_flagged_findings_function_matches(tmp_path):
    h = flagged_setup(tmp_path)
    assert flagged_findings(h) == h["flagged"]


# ---------------------------------------------------------------- coverage

def test_coverage_by_design_and_missing(tmp_path):
    cfg = load_cfg()
    d = make_dir(tmp_path, "cov", [Cell("A", "m", 4, 0), Cell("A", "m", 4, 1)])
    write_run(d, Cell("A", "m", 4, 2), steps=(3000, 10000))
    cov = coverage(run_table(load_metrics(d)), cfg)
    assert "A-m-w4-s3" in cov["not_run_by_design"] and "B-l-w2-s0" in cov["not_run_by_design"]
    assert "A-m-w4-s2" in cov["incomplete"] and "A-l-w4-s0" in cov["missing"]
    plan = len(grid(cfg)) - len(cov["not_run_by_design"])
    assert cov["planned"] == plan == 3 * 25 + 15 + 6


# ---------------------------------------------------------------- benchmarks, human

def test_human_curve_six_rows():
    h = human.human_curve()
    assert len(h) == 6 and {"rho", "implied_rho", "implied_rho_se", "b_eor", "b_eor_se"} <= set(h.columns)


def test_aklmt_design_check_absent_and_synthetic(tmp_path):
    assert human.aklmt_design_check(tmp_path / "nope.zip") is None
    rng = np.random.default_rng(0)
    recs = []
    for rho in RHOS:
        for a in range(3):
            x = 0.0
            for r in range(-5, 20):
                x = rho * x + rng.normal(0, 20)
                recs.append({"assignmentid": f"{rho}-{a}", "round": r, "actual": round(x), "rho": rho})
    p = tmp_path / "Replication.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(human.MEMBER, pd.DataFrame(recs).to_csv(index=False))
    t = human.aklmt_design_check(p)
    assert len(t) == 6 and (t["innov_sd"] > 10).all() and (t["innov_sd"] < 30).all()
    assert (t["prior_p_var_ge"] < 1e-6).all()


def test_benchmarks_tiny(tmp_path):
    cfg = load_cfg(eval_n_per_rho=40, eval_n_prior=40)
    b = benchmarks.benchmark_rows(eval_dir=tmp_path / "ev", cfg=cfg, grid=101)
    assert len(b) == 7 * 7 and np.isfinite(b["implied_persistence"]).all()
    fire = b[b.bench == "fire"]
    for r in RHOS:
        assert fire[fire.rho == str(r)].implied_persistence.iloc[0] == pytest.approx(r, abs=1e-9)
    assert set(b[b.bench == "window_bayes"].window) == {"2", "4", "8", "16"}


# ---------------------------------------------------------------- report

def test_every_number_in_results_traces(tmp_path):
    cells = cells_l() + cells_l(family="B", seeds=range(3), windows=(None, 4))
    d = make_dir(tmp_path, "metrics", cells,
                 value_fn=vfn(lambda c, rho, col, b: 0.2 if (c.window == 4 and rho == "0.6"
                                                             and col == "gap" and b == "window_bayes") else None))
    out = tmp_path / "out"
    cfg = load_cfg(eval_n_per_rho=40, eval_n_prior=40)
    H = report.run(d, out, eval_dir=tmp_path / "ev", cfg=cfg, aklmt_zip=tmp_path / "none.zip", grid=101)
    for p in ("RESULTS.md", "results/summary.csv", "results/hypotheses.json",
              "figures/fig1_gap_map.png", "figures/fig2_human_overlay.png",
              "figures/fig3_training_traces.png", "figures/fig4_size.png"):
        assert (out / p).stat().st_size > 0, p
    text = (out / "RESULTS.md").read_text()
    assert "could not be run" in text
    summary = pd.read_csv(out / "results" / "summary.csv", dtype={"window": str, "rho": str})
    num = re.compile(r"[+-]?\d+\.\d+")

    def close(printed, value):
        assert abs(float(printed) - value) <= 0.5e-4 + 1e-12, (printed, value)

    def line_for(prefix):
        ls = [l for l in text.splitlines() if l.startswith(prefix)]
        assert ls, prefix
        return ls

    checked = 0
    # H1 rows: "| rho | column | n | mean | sd | lo | hi | verdict | values |"
    for r in H["h1"]["rows"]:
        cells_ = [c.strip() for c in line_for(f"| {r['rho']} | {r['column']} | {r['n_seeds']} | ")[0].strip("| \n").split("|")]
        close(cells_[3], r["mean"]); close(cells_[4], r["sd"]); close(cells_[5], r["lo"]); close(cells_[6], r["hi"])
        assert cells_[7] == r["verdict"]
        pv = num.findall(cells_[8])
        assert len(pv) == 5
        for a, b in zip(pv, r["values"]):
            close(a, b)
        # and against summary.csv directly
        s = summary[(summary.family == "A") & (summary["size"] == "l") & (summary.window == "full") &
                    (summary.step == 30000) & (summary.h == 1) & (summary.bench == "bayes") &
                    (summary.rho == r["rho"]) & (summary.column == r["column"])].iloc[0]
        close(cells_[3], s["mean"]); close(cells_[5], s["lo"])
        checked += 1
    # H2 rows (size l): "| l | rho | column | n | mean | ..."
    for w, blk in H["h2"]["windows"].items():
        sect = text.split(f"### Window {w}")[1].split("###")[0]
        for r in blk["rows"]:
            ln = [l for l in sect.splitlines() if l.startswith(f"| l | {r['rho']} | {r['column']} | ")]
            assert len(ln) == 1
            c = [x.strip() for x in ln[0].strip("| ").split("|")]
            close(c[4], r["mean"]); close(c[6], r["lo"]); close(c[7], r["hi"])
            assert c[8] == r["verdict"]
            checked += 1
    # H3 rows
    for c3 in H["h3"]["combos"]:
        ln = line_for(f"| {c3['window']} | {c3['rho']} | {c3['n_seeds']} | ")
        c = [x.strip() for x in ln[0].strip("| ").split("|")]
        close(c[3], c3["mean"]); close(c[5], c3["lo"]); close(c[6], c3["lo_adj"])
        assert c[7] == ("yes" if c3["positive_95"] else "no")
        checked += 1
    assert checked == 12 + 48 + 8
    assert H["h3"]["statement"] in text
    assert H["flagged"] and any(f["cells"][0] in text for f in H["flagged"])
    assert "awaiting the training-length check" in text


# ---------------------------------------------------------------- report: 60k check and incomplete group

def test_results_trace_60k_and_incomplete(tmp_path):
    shift = lambda c, rho, col, b: (0.3 if col == "gap" and rho == "0.6" and b == "window_bayes"
                                    and c.window in (2, 4) else None)
    cells = [c for c in cells_l() if not (c.window == 2 and c.seed == 4)]
    d = make_dir(tmp_path, "metrics", cells, value_fn=vfn(shift))
    write_run(d, Cell("A", "l", 8, 0), converged=False)             # a non-converged run
    d60 = long_dir(tmp_path, "m60", lambda c, rho, col, b: 0.3 if (c.window == 4 and col == "gap"
                                                                  and rho == "0.6") else None, (4,))
    out = tmp_path / "out"
    H = report.run(d, out, metrics_60k_dir=d60, cfg=load_cfg(), aklmt_zip=tmp_path / "none.zip",
                   benchmarks=False)
    text = (out / "RESULTS.md").read_text()
    num = re.compile(r"[+-]?\d+\.\d+")

    def close(printed, value):
        assert abs(float(printed) - value) <= 0.5e-4 + 1e-12, (printed, value)

    # flagged table, with 60k columns
    chk = {(f["hypothesis"], f["window"], f["rho"], f["column"]): f for f in H["training_length_check"]}
    f = chk[("H2", "4", "0.6", "gap")]
    assert f["outcome"] == "survives"
    ln = [l for l in text.splitlines() if l.startswith("| H2 | 4 | 0.6 | gap | ")]
    assert len(ln) == 1
    c = [x.strip() for x in ln[0].strip("| ").split("|")]
    close(c[5], f["mean"]); close(c[6], f["lo"]); close(c[7], f["hi"])
    assert c[8] == "survives" and c[9] == str(f["n_60k"]) == "5"
    close(c[10], f["mean_60k"]); close(c[11], f["lo_60k"]); close(c[12], f["hi_60k"])
    assert c[13] == f["verdict_60k"] == "different" and c[14] == "-"
    assert c[15] == "no"
    for a, b in zip(num.findall(c[16]), f["values_60k"]):
        close(a, b)
    h3 = [l for l in text.splitlines() if l.startswith("| H3 | 4 | 0.6 | gap | ")][0]
    close([x.strip() for x in h3.strip("| ").split("|")][14], chk[("H3", "4", "0.6", "gap")]["lo_adj_60k"])
    # headlines carry the resolved decision and the 30k decision
    assert "after the training-length check: departure from rational use of the window (survived" in text
    assert "(Decision at 30,000 steps: departure from rational use of the window, pending" in text
    assert "h2_w4" not in text and "H2 window 4: departure" in text and "departure_from_rational_use" not in text
    assert "**Decision after the training-length check: supported." in text
    # incomplete H3 combination is marked and counted
    row = [l for l in text.splitlines() if l.startswith("| 2 | 0.6 | 4 | ")]
    assert len(row) == 1 and row[0].count("incomplete (4 of 5 seeds; not used)") == 2
    assert "not run or incomplete" in H["h3"]["statement"] and H["h3"]["statement"] in text
    assert H["h3"]["n_missing"] > 0
    # per-run convergence printed for every run
    assert len(H["runs"]) == len(cells)
    for r in H["runs"]:
        assert f"| {r['cell_id']} | 3000, 10000, 30000 | {'yes' if r['converged'] else 'no'} |" in text
    assert "| A-l-w8-s0 | 3000, 10000, 30000 | no |" in text
