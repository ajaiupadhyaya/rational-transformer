"""Metrics directory -> results/summary.csv, results/hypotheses.json, figures/, RESULTS.md.

    python -m analysis.report [--metrics DIR] [--out DIR] [--metrics-60k DIR]
                              [--eval-dir DIR] [--no-benchmarks]

Every number in RESULTS.md is read back from the two files this module just wrote and is
formatted by code. No sentence interprets a result beyond the pre-registered readings.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis import benchmarks as bm  # noqa: E402
from analysis import human as hm  # noqa: E402
from analysis.hypotheses import (GRID_RHOS, H2_WINDOWS, H3_RHOS, SURVIVES, hypotheses,  # noqa: E402
                                 prior_rows, resolve_decisions, training_length_check)
from analysis.summary import (convergence_cells, coverage, load_metrics, norm_rho,  # noqa: E402
                              run_table, summarise)
from rt.train import Cell  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SIZES = ("xs", "s", "m", "l")
WINDOWS = ("2", "4", "8", "16", "full")
STEPS = (3000, 10000, 30000)

DECISION_TEXT = {
    "supported": "supported",
    "rejected_pending_check": "rejected, pending the training-length check",
    "rejected": "rejected (survived the training-length check)",
    "not_established": "not established",
    "rational_given_information": "rational given its information",
    "departure_pending_check": "departure from rational use of the window, pending the "
                               "training-length check",
    "departure_from_rational_use": "departure from rational use of the window (survived the "
                                   "training-length check)",
    "supported_pending_check": "supported only if a flagged combination survives the "
                               "training-length check (pending)",
    "not_supported": "not supported",
}


# ---------------------------------------------------------------- helpers

def _clean(o):
    """JSON-safe copy: numpy scalars to Python, NaN/inf to None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _records(df: pd.DataFrame | None) -> list[dict]:
    return [] if df is None else df.to_dict("records")


def f4(v, sign=True) -> str:
    if v is None:
        return "-"
    return f"{v:+.4f}" if sign else f"{v:.4f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _vals(v) -> str:
    return ", ".join(f4(x) for x in v) if v else "-"


def _row_cells(r: dict) -> list[str]:
    """n, mean, sd, lo, hi, verdict, seed values."""
    if r["status"] == "not run":
        return ["-", "-", "-", "-", "-", "not run", "-"]
    note = r["verdict"] if r["status"] == "ok" else \
        f"incomplete ({r['n_seeds']} of {r['expected_seeds']} seeds; {r['verdict']}; not used)"
    return [str(r["n_seeds"]), f4(r["mean"]), f4(r["sd"], False), f4(r["lo"]), f4(r["hi"]),
            note, _vals(r["values"])]


def _compress(ids: list[str]) -> str:
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i in ids:
        c = Cell.parse(i)
        groups[(c.family, c.size, "full" if c.window is None else str(c.window))].append(c.seed)
    return "; ".join(f"A/B-{s}-w{w} (family {f}): seeds {','.join(map(str, sd))}"
                     for (f, s, w), sd in groups.items()) or "none"


# ---------------------------------------------------------------- figures

def _sel(summary: pd.DataFrame, **kw) -> pd.DataFrame:
    m = pd.Series(True, index=summary.index)
    for k, v in kw.items():
        m &= summary[k].astype(str) == str(v) if k in ("window", "rho") else summary[k] == v
    return summary[m]


def _errbar(ax, x, rows, color, label, dx=0.0):
    rows = rows.sort_values("_x")
    if rows.empty:
        return
    xs = rows["_x"].to_numpy(float) + dx
    ax.errorbar(xs, rows["mean"], yerr=[rows["mean"] - rows["lo"], rows["hi"] - rows["mean"]],
                fmt="o-", ms=4, lw=1.2, capsize=3, color=color, label=label)
    for xi, v in zip(xs, rows["values"]):
        ax.plot([xi] * len(json.loads(v)), json.loads(v), ".", color=color, alpha=0.35, ms=3)


def _palette(n):
    return plt.cm.viridis(np.linspace(0.05, 0.9, n))


def fig_gap_map(summary, path):
    d = _sel(summary, family="A", size="l", step=30000, h=1, bench="window_bayes", column="gap")
    d = d[d["rho"] != "prior"].copy()
    d["_x"] = d["rho"].astype(float)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4), sharey=True)
    for ax, w in zip(axes, WINDOWS):
        _errbar(ax, None, d[d["window"].astype(str) == w], "tab:blue", None)
        for y in (0.02, -0.02):
            ax.axhline(y, color="k", ls="--", lw=0.8)
        ax.axhline(0, color="gray", lw=0.6)
        ax.text(0.04, 0.95, f"window {w}", transform=ax.transAxes, va="top", fontsize=9)
        ax.set_xlabel("true persistence rho")
        ax.set_xlim(-0.05, 1.05)
    axes[0].set_ylabel("gap: model minus window-Bayes\nimplied persistence (slope)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_human_overlay(summary, path, human):
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls=":", label="45-degree line")
    base = dict(family="B", size="l", step=30000, h=1, column="gap")
    d = lambda w, b: _sel(summary, window=w, bench=b, **base).query("rho != 'prior'") \
        .assign(_x=lambda t: t["rho"].astype(float)).sort_values("_x")
    for w, b, col, lab in (("full", "bayes", "tab:blue", "model, full window"),
                           ("4", "window_bayes", "tab:orange", "model, W = 4")):
        r = d(w, b)
        if len(r):
            ax.plot(r["_x"], r["implied_persistence"], "o-", color=col, ms=4, label=lab)
    r = d("full", "bayes")
    if len(r):
        ax.plot(r["_x"], r["bench_implied_persistence"], "-", color="tab:blue", alpha=0.5,
                label="Bayes (Family B)")
    r = d("4", "window_bayes")
    if len(r):
        ax.plot(r["_x"], r["bench_implied_persistence"], "-", color="tab:orange", alpha=0.5,
                label="window-Bayes, W = 4 (Family B)")
    ax.errorbar(human["rho"], human["implied_rho"], yerr=1.96 * human["implied_rho_se"],
                fmt="s", color="tab:red", capsize=3, ms=4, label="human subjects (AKLMT), 95%")
    ax.set_xlabel("true persistence rho")
    ax.set_ylabel("implied persistence (slope of forecast on x_t)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _by_window_panels(summary, path, xkey, xs, xlabel, sizes_over=False, logx=False):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    cols = _palette(len(WINDOWS))
    for ax, rho in zip(axes, H3_RHOS):
        for w, col in zip(WINDOWS, cols):
            kw = dict(family="A", h=1, bench="window_bayes", column="gap", rho=rho, window=w)
            if sizes_over:
                d = _sel(summary, step=30000, **kw)
                d = d[d["size"].isin(xs)].assign(_x=lambda t: t["size"].map({s: i for i, s in enumerate(xs)}))
            else:
                d = _sel(summary, size="l", **kw)
                d = d[d["step"].isin(xs)].assign(_x=lambda t: t["step"].astype(float))
            _errbar(ax, None, d, col, f"window {w}", dx=0.0)
        for y in (0.02, -0.02):
            ax.axhline(y, color="k", ls="--", lw=0.8)
        ax.axhline(0, color="gray", lw=0.6)
        ax.text(0.04, 0.95, f"rho = {rho}", transform=ax.transAxes, va="top", fontsize=9)
        ax.set_xlabel(xlabel)
        if sizes_over:
            ax.set_xticks(range(len(xs)), xs)
        else:
            ax.set_xscale("log")
            ax.set_xticks(xs, [str(s) for s in xs])
            ax.minorticks_off()
    axes[0].set_ylabel("gap: model minus window-Bayes\nimplied persistence (slope)")
    axes[1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figures(summary, human, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_gap_map(summary, fig_dir / "fig1_gap_map.png")
    fig_human_overlay(summary, fig_dir / "fig2_human_overlay.png", human)
    _by_window_panels(summary, fig_dir / "fig3_training_traces.png", "step", list(STEPS),
                      "training step")
    _by_window_panels(summary, fig_dir / "fig4_size.png", "size", list(SIZES),
                      "model size", sizes_over=True)


# ---------------------------------------------------------------- RESULTS.md

def _h2_table(blocks: dict[str, dict]) -> str:
    rows = []
    for size, blk in blocks.items():
        for r in blk["rows"]:
            rows.append([size, r["rho"], r["column"], *_row_cells(r)])
    return _table(["size", "rho", "column", "n seeds", "mean", "sd", "lo (95%)", "hi (95%)",
                   "verdict", "seed values"], rows)


def _key_label(k: str) -> str:
    return f"H2 window {k[4:]}" if k.startswith("h2_w") else k.upper()


def _decision_line(label: str, decision: str, resolved: str | None) -> str:
    if resolved is None:
        return f"**{label}: {DECISION_TEXT[decision]}.**"
    return (f"**{label} after the training-length check: {DECISION_TEXT.get(resolved, resolved)}.** "
            f"(Decision at 30,000 steps: {DECISION_TEXT[decision]}.)")


def render(summary: pd.DataFrame, H: dict, metrics_label: str) -> str:
    h1, h2, h3 = H["h1"], H["h2"], H["h3"]
    prior = H["prior_rows"]
    L = []
    w = L.append
    w("# Results")
    w("")
    w(f"Generated by `python -m analysis.report` from `{metrics_label}`. Every number below is "
      "formatted by code from `results/summary.csv` and `results/hypotheses.json`. Decisions "
      "follow `docs/PREREG.md` sections 6 and 7 as written; nothing here goes beyond those "
      "readings. Gap is in slope units (margin 0.02); excess error is in percentage points "
      "(margin 1). Intervals are 95% Student-t intervals of the seed mean.")
    w("")
    cov = H["coverage"]
    w("## Run status")
    w("")
    w(f"Cells planned (Plan A, deviation prereg-v1.1): {cov['planned']}; complete through step "
      f"30000: {cov['complete']}; not run by design: {len(cov['not_run_by_design'])}; "
      f"missing: {len(cov['missing'])}; incomplete: {len(cov['incomplete'])}.")
    w("")
    w("## H1: the largest model with the full window is Bayes")
    w("")
    w(f"Rows: family A, size {h2['primary_size']}, full window, benchmark bayes, h = 1, step 30000.")
    w("")
    w(_decision_line("Decision", h1["decision"], H.get("resolved_decisions", {}).get("h1")))
    w("")
    w(_table(["rho", "column", "n seeds", "mean", "sd", "lo (95%)", "hi (95%)", "verdict",
              "seed values"], [[r["rho"], r["column"], *_row_cells(r)] for r in h1["rows"]]))
    w("")
    w("Prior set (printed, not part of the decision):")
    w("")
    w(_table(["rho", "column", "n seeds", "mean", "sd", "lo (95%)", "hi (95%)", "verdict",
              "seed values"], [[r["rho"], r["column"], *_row_cells(r)] for r in prior["h1"]]))
    w("")
    w("## H2: window-limited models against window-Bayes")
    w("")
    w("Rows: family A, benchmark window_bayes, h = 1, step 30000. Size l decides; other sizes "
      "are printed in the same tables and never enter a decision.")
    for win in H2_WINDOWS:
        blk = h2["windows"][win]
        w("")
        w(f"### Window {win}")
        w("")
        w(_decision_line(f"Decision (size {h2['primary_size']})", blk["decision"],
                         H.get("resolved_decisions", {}).get(f"h2_w{win}")))
        w("")
        blocks = {h2["primary_size"]: blk, **{s: h2["other_sizes"][s][win] for s in h2["other_sizes"]}}
        w(_h2_table(blocks))
        w("")
        w(f"Size {h2['primary_size']}, prior set (printed, not part of the decision):")
        w("")
        w(_table(["rho", "column", "n seeds", "mean", "sd", "lo (95%)", "hi (95%)", "verdict",
                  "seed values"], [[r["rho"], r["column"], *_row_cells(r)]
                                   for r in prior[f"h2_w{win}"]]))
    w("")
    w("## H3: overreaction where the matched Bayesian underreacts")
    w("")
    w(f"Rows: family A, size {h3['primary_size']}, benchmark window_bayes, h = 1, step 30000, "
      f"rho in {', '.join(H3_RHOS)}, column gap. A combination is positive at 95% if the lower end "
      f"of its 95% interval is above +0.02, and positive after adjustment if mean minus "
      f"{h3['bonferroni_t_n5']:.3f} sd / sqrt(n) (n = 5) is above +0.02. The denominator is "
      f"always {h3['denominator']}.")
    w("")
    w(_decision_line("Decision", h3["decision"], H.get("resolved_decisions", {}).get("h3"))
      + f" {h3['statement']}.")
    w("")
    rows = []
    for c in h3["combos"]:
        if c["status"] == "not run":
            rows.append([c["window"], c["rho"], "-", "-", "-", "-", "-", "not run", "-", "-"])
            continue
        if c["status"] == "ok":
            p95, padj = ("yes" if c["positive_95"] else "no"), ("yes" if c["positive_adjusted"] else "no")
        else:
            p95 = padj = f"incomplete ({c['n_seeds']} of {c['expected_seeds']} seeds; not used)"
        rows.append([c["window"], c["rho"], str(c["n_seeds"]), f4(c["mean"]), f4(c["sd"], False),
                     f4(c["lo"]), f4(c["lo_adj"]), p95, padj, _vals(c["values"])])
    w(_table(["window", "rho", "n seeds", "mean", "sd", "lo (95%)", "lo (adjusted)",
              "positive at 95%", "positive after adjustment", "seed values"], rows))
    w("")
    w("Other sizes (descriptive, positives at 95%):")
    w("")
    w(_table(["size", "combinations present", "not run or incomplete", "positive at 95%"],
             [[s, str(o["n_present"]), str(o["n_missing"]), str(o["k_positive_95"])]
              for s, o in h3["other_sizes"].items()]))
    w("")
    n_miss_other = sum(o["n_missing"] for o in h3["other_sizes"].values())
    w(f"Across xs, s and m: {h3['other_sizes_k_positive_95']} of {h3['other_sizes_denominator']} "
      f"positive at 95%; {n_miss_other} of {h3['other_sizes_denominator']} not run or incomplete.")
    w("")
    w("## Flagged findings and the training-length check")
    w("")
    fl = H["flagged"]
    if not fl:
        w("No size-l finding is flagged.")
    else:
        checked = H.get("training_length_check")
        w("A flagged finding is re-trained for twice as many steps (all five seeds) under "
          "PREREG section 6 before it is reported as a finding.")
        w("")
        rows = []
        for f in (checked or fl):
            outcome = f.get("outcome", "awaiting the training-length check")
            if outcome == "not run":
                outcome = "different at 30,000 steps; not tested for training length"
            base = [f["hypothesis"], f["window"], f["rho"], f["column"], f["sign"],
                    f4(f["mean"]), f4(f["lo"]), f4(f["hi"]), outcome]
            if checked:
                eq = "-" if f["hypothesis"] == "H3" else ("yes" if f["all_equivalent_at_60k"] else "no")
                base += [str(f["n_60k"]) if f.get("n_60k") is not None else "-", f4(f["mean_60k"]),
                         f4(f["lo_60k"]), f4(f["hi_60k"]), f.get("verdict_60k") or "-",
                         f4(f["lo_adj_60k"]) if f["hypothesis"] == "H3" else "-", eq,
                         _vals(f.get("values_60k"))]
            rows.append(base + [", ".join(f["cells"])])
        heads = ["hypothesis", "window", "rho", "column", "sign", "mean", "lo (95%)", "hi (95%)",
                 "status"]
        if checked:
            heads += ["n seeds (60,000)", "mean (60,000)", "lo (60,000)", "hi (60,000)",
                      "verdict (60,000)", "lo adjusted (60,000, H3)", "all 12 equivalent at 60,000",
                      "seed values (60,000)"]
        w(_table(heads + ["cells to re-train"], rows))
        if H.get("resolved_decisions"):
            w("")
            w("Decisions after the check: " + "; ".join(
                f"{_key_label(k)}: {DECISION_TEXT.get(v, v)}" for k, v in H["resolved_decisions"].items()
                if v is not None) + ".")
    w("")
    w("## Cells not run")
    w("")
    w(f"Not run by design (Plan A): {_compress(cov['not_run_by_design'])}.")
    w("")
    w(f"Not run or incomplete (planned, no complete metrics file): "
      f"{_compress(cov['missing'] + cov['incomplete'])}.")
    w("")
    w("## Deviations")
    w("")
    w("Every deviation from `docs/PREREG.md` is logged in [docs/DECISIONS.md](docs/DECISIONS.md). "
      "The grid change (deviation prereg-v1.1, Plan A) was decided before any confirmatory "
      "run existed.")
    w("")
    w("## Descriptive tables")
    w("")
    w("### Convergence (descriptive; never changes a verdict)")
    w("")
    w(_table(["family", "size", "window", "runs present", "runs converged", "runs expected",
              "cell converged"],
             [[c["family"], c["size"], str(c["window"]), str(c["n_runs"]), str(c["n_converged"]),
               str(c["expected_runs"]), "yes" if c["converged"] else "no"]
              for c in H["convergence_cells"]]))
    w("")
    w("Per run (steps present in the metrics file; converged flag):")
    w("")
    w(_table(["cell", "steps", "converged"],
             [[r["cell_id"], ", ".join(str(x) for x in r["steps"]), "yes" if r["converged"] else "no"]
              for r in H["runs"]]))
    w("")
    w("### Benchmarks only (family A, h = 1, implied persistence)")
    w("")
    b = H.get("benchmarks")
    if b:
        df = pd.DataFrame(b)
        df["col"] = df.apply(lambda r: r["bench"] if r["window"] == "-"
                             else f"{r['bench']} W={r['window']}", axis=1)
        order = ["fire", "ls", "bayes"] + [f"window_bayes W={x}" for x in (2, 4, 8, 16)]
        pv = df.pivot(index="rho", columns="col", values="implied_persistence")
        rhos = GRID_RHOS + ["prior"]
        w(_table(["rho"] + order, [[r] + [f4(pv.loc[r, c], False) if c in pv.columns and r in pv.index
                                          else "-" for c in order] for r in rhos]))
    else:
        w("Not computed in this run.")
    w("")
    w("### Human subjects (AKLMT 2023, Experiment 1)")
    w("")
    w(_table(["rho", "implied persistence", "se", "error-on-revision b", "se"],
             [[f"{r['rho']:.1f}", f4(r["implied_rho"], False), f4(r["implied_rho_se"], False),
               f4(r["b_eor"], False), f4(r["b_eor_se"], False)] for r in H["human"]]))
    w("")
    w("### AKLMT design against the Family B prior")
    w("")
    dc = H.get("design_check")
    if dc:
        w(_table(["rho", "series", "series mean", "series sd", "series min", "series max",
                  "innovation mean", "innovation sd", "innovation min", "innovation max",
                  "abs mean / innovation sd", "prior sd of mu / sigma", "prior P(sigma^2 >= observed)"],
                 [[r["rho"], str(r["n_series"]), f4(r["series_mean"], False), f4(r["series_sd"], False),
                   f4(r["series_min"], False), f4(r["series_max"], False), f4(r["innov_mean"], False),
                   f4(r["innov_sd"], False), f4(r["innov_min"], False), f4(r["innov_max"], False),
                   f4(r["abs_mean_over_sd"], False), f4(r["prior_sd_mu_over_sigma"], False),
                   f"{r['prior_p_var_ge']:.3e}"] for r in dc]))
    else:
        w("The check could not be run: the raw replication package is not available locally.")
    w("")
    w("## Figures")
    w("")
    for n, cap in (("fig1_gap_map", "Gap against window-Bayes by window and rho (size l, family A, "
                                    "h = 1, step 30000)"),
                   ("fig2_human_overlay", "Family B overlay with human subjects"),
                   ("fig3_training_traces", "Gap against training step (size l, family A, h = 1)"),
                   ("fig4_size", "Gap against model size (family A, h = 1, step 30000)")):
        w(f"![{cap}](figures/{n}.png)")
        w("")
    return "\n".join(L).rstrip() + "\n"


# ---------------------------------------------------------------- driver

def run(metrics_dir, out_dir=ROOT, *, metrics_60k_dir=None, eval_dir=None, cfg=None,
        aklmt_zip=None, benchmarks=True, grid=None) -> dict:
    out = Path(out_dir)
    (out / "results").mkdir(parents=True, exist_ok=True)
    df = load_metrics(metrics_dir)
    summarise(df).to_csv(out / "results" / "summary.csv", index=False)
    summary = pd.read_csv(out / "results" / "summary.csv", dtype={"window": str, "rho": str})
    summary["rho"] = summary["rho"].map(norm_rho)

    H = hypotheses(summary)
    H["prior_rows"] = prior_rows(summary)
    runs = run_table(df)
    H["coverage"] = coverage(runs, cfg)
    H["convergence_cells"] = _records(convergence_cells(runs))
    H["runs"] = _records(runs)
    if metrics_60k_dir is not None:
        checked = training_length_check(H["flagged"], summarise(load_metrics(metrics_60k_dir)))
        H["training_length_check"] = checked
        H["resolved_decisions"] = resolve_decisions(H, checked)
    H["benchmarks"] = (_records(bm.benchmark_rows(eval_dir=eval_dir, cfg=cfg, grid=grid))
                       if benchmarks else None)
    human = hm.human_curve()
    H["human"] = _records(human)
    dc = hm.aklmt_design_check(aklmt_zip)
    H["design_check"] = None if dc is None else _records(dc)
    (out / "results" / "hypotheses.json").write_text(json.dumps(_clean(H), indent=1, allow_nan=False))
    H = json.loads((out / "results" / "hypotheses.json").read_text())

    make_figures(summary, human, out / "figures")
    label = str(metrics_dir)
    (out / "RESULTS.md").write_text(render(summary, H, label))
    return H


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pre-registered analysis and results pack.")
    ap.add_argument("--metrics", default=str(ROOT / "results" / "metrics"))
    ap.add_argument("--out", default=str(ROOT))
    ap.add_argument("--metrics-60k", default=None, help="metrics dir of the 60,000-step re-train")
    ap.add_argument("--eval-dir", default=None)
    ap.add_argument("--no-benchmarks", action="store_true")
    a = ap.parse_args(argv)
    run(a.metrics, a.out, metrics_60k_dir=a.metrics_60k, eval_dir=a.eval_dir,
        benchmarks=not a.no_benchmarks)
    print(f"wrote {a.out}/RESULTS.md, results/summary.csv, results/hypotheses.json, figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
