"""Export the SPF panel for the illustrative survey comparison (PREREG section 8).

Runs inside the environment of the read-only companion repository
expectations-without-hindsight (EWH) and writes only into this repository:

    PYTHONDONTWRITEBYTECODE=1 uv run --project ~/Documents/expectations-without-hindsight \
        --no-sync python scripts/export_spf_panel.py --out data/spf_panel.parquet

Data provenance: everything comes from EWH's own pipeline over the raw public files of its
2026-11-11 snapshot (Philadelphia Fed SPF microdata and RTDSM vintages; FRED TB3MS and GS10
for the two interest rates, which are not revised): `ewh.data.spf.spf_pairs(var, "bgms")`,
`spf.sample`, `spf.trim(spf.load(var))` and `ewh.data.realtime`. That is EWH's "ours" path of
its SPF replication. The BGMS and CG replication packages are never opened: the export
refuses any read of a zip archive under EWH's raw `bgms/` or `cg/` directories.

One row per (variable, survey quarter t), headline variables of EWH's replication, sample
1968Q4-2016Q4 (EWH `spf_replication.HEADLINE`, `FIRST`, `LAST`). Columns (PANEL_COLUMNS):

    variable      EWH variable name (NGDP, RGDP, PGDP, CPI, UNEMP, TB3M, TN10Y)
    kind          "growth" (target is percent growth x[t+3]/x[t-1] - 1) or "level" (x[t+3])
    target        "t+3": the survey forecast made at t of quarter t+3, revised from the survey
                  at t-1 (BGMS's "f3q" error and revision)
    t             survey quarter, "YYYYQn"
    hist_first    quarter of the first element of `hist`, "YYYYQn"
    hist          real-time levels available at survey t: vintage t only, one per quarter
                  from hist_first to t-1, NaN where vintage t has no value (x[t-1] is NaN when
                  it was released late: 1996Q1 for NGDP, RGDP, PGDP), in SPF units. Empty when
                  no vintage dated t exists (CPI before 1994Q3).
    hist_source   where `hist` comes from (file and vintage)
    hist_vintage  the vintage of `hist`, "YYYYQn": always t, or "" when there is none (FRED
                  rates are not revised: their vintage t is the series cut to quarters < t)
    last_filled   always False: a late x[t-1] is never filled from a later vintage (that would
                  use a figure published after the survey deadline). Kept so that a panel
                  built under a filling rule is refused by analysis/spf.py.
    outcome       realised target EWH uses for the errors (growth from the first vintage that
                  contains x[t+3]; FRED value for rates)
    cons_err      consensus error: mean over forecasters of EWH's individual errors (after
                  BGMS's trimming and the >= 10 pairs rule), as `spf.consensus`
    cons_rev      consensus revision, likewise
    n_pairs       forecasters with both an error and a revision at t
    spf_c1..c6    cross-sectional mean of the trimmed SPF forecasts for quarters t-1..t+4
                  (raw SPF units: levels, or annualised quarterly percent rates for CPI)
    ind_id        forecaster IDs with a complete (err, rev) pair at t
    ind_err       their errors
    ind_rev       their revisions
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EWH_ROOT = Path.home() / "Documents" / "expectations-without-hindsight"
GROWTH_KIND = ("NGDP", "RGDP", "PGDP", "CPI")
LEVEL_KIND = ("UNEMP", "TB3M", "TN10Y")
TARGET = "t+3"
PANEL_COLUMNS = (["variable", "kind", "target", "t", "hist_first", "hist", "hist_source",
                  "hist_vintage", "last_filled", "outcome", "cons_err", "cons_rev", "n_pairs"]
                 + [f"spf_c{k}" for k in range(1, 7)] + ["ind_id", "ind_err", "ind_rev"])


def q(p: pd.Period) -> str:
    return f"{p.year}Q{p.quarter}"


def kind_of(var: str) -> str:
    if var in GROWTH_KIND:
        return "growth"
    if var in LEVEL_KIND:
        return "level"
    raise ValueError(f"no kind for {var!r}")


def build_panel(source, variables) -> pd.DataFrame:
    """The panel from a data source with three methods (EWHSource below, or a test fixture):

    pairs(var)     sampled individual pairs: ID, t (Period), fc, fc_prev, outcome, err, rev
    forecasts(var) trimmed raw forecasts: ID, yq (Period), c1..c6
    history(var, t) -> (levels Series on a quarterly PeriodIndex, source text, vintage
                        Period of those levels or None); only a vintage dated t is kept
    """
    rows = []
    for var in variables:
        pairs = source.pairs(var)
        fc = source.forecasts(var)
        spf_mean = fc.groupby("yq")[[f"c{k}" for k in range(1, 7)]].mean()
        cons = pairs.groupby("t")[["err", "rev"]].mean()            # = ewh spf.consensus
        out = pairs.groupby("t")["outcome"].first()
        for t in sorted(pairs.t.unique()):
            p = pairs[(pairs.t == t) & pairs.err.notna() & pairs.rev.notna()]
            levels, src, vintage = source.history(var, t)
            if vintage != t:                         # strict real time: vintage t or nothing
                levels, vintage = levels.iloc[:0], None
            levels = levels[levels.index < t].dropna()
            if len(levels):
                rng = pd.period_range(levels.index.min(), t - 1, freq="Q")
                hist = levels.reindex(rng).to_numpy(float)
                first = q(rng[0])
            else:
                hist, first = np.array([], float), ""
            c = spf_mean.loc[t] if t in spf_mean.index else pd.Series(np.nan, spf_mean.columns)
            rows.append({
                "variable": var, "kind": kind_of(var), "target": TARGET, "t": q(t),
                "hist_first": first, "hist": [float(v) for v in hist], "hist_source": src,
                "hist_vintage": q(vintage) if vintage is not None else "",
                "last_filled": False, "outcome": float(out.get(t, np.nan)),
                "cons_err": float(cons.err.get(t, np.nan)),
                "cons_rev": float(cons.rev.get(t, np.nan)), "n_pairs": int(len(p)),
                **{f"spf_c{k}": float(c[f"c{k}"]) for k in range(1, 7)},
                "ind_id": [int(v) for v in p.ID], "ind_err": [float(v) for v in p.err],
                "ind_rev": [float(v) for v in p.rev],
            })
    return pd.DataFrame(rows, columns=PANEL_COLUMNS)


def vintage_history(m: pd.DataFrame, t: pd.Period) -> tuple[pd.Series, pd.Period | None]:
    """Levels of vintage t from an (observation x vintage) matrix, observations before t only;
    (empty, None) if the matrix has no vintage dated t with data. A later vintage is never
    used, and a late x[t-1] is never filled in."""
    if t not in m.columns or not m[t].notna().any():
        return pd.Series([], dtype=np.float64, index=pd.PeriodIndex([], freq="Q")), None
    lv = m[t].dropna()
    return lv[lv.index < t].sort_index(), t


def write_panel(panel: pd.DataFrame, out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out, index=False, compression="zstd")


class EWHSource:
    """EWH's raw-file pipeline (its replication's "ours" path), imported lazily."""

    def __init__(self):
        from ewh.data import realtime, spf
        from ewh.data.spf_replication import FIRST, HEADLINE, LAST
        self.rt, self.spf = realtime, spf
        self.first, self.last, self.headline = FIRST, LAST, HEADLINE

    def pairs(self, var):
        return self.spf.sample(self.spf.spf_pairs(var, "bgms"), self.first, self.last)

    def forecasts(self, var):
        return self.spf.trim(self.spf.load(var))

    def history(self, var, t):
        rt = self.rt
        if var in rt.FRED:
            s = rt.fred_quarterly(var)
            return (s[s.index < t], f"FRED {rt.FRED[var]} (EWH snapshot {rt.SNAPSHOT}; "
                    "not revised), quarterly means before t", t)
        lv, v = vintage_history(rt.vintage_matrix(var), t)
        src = (f"RTDSM {rt.RTDSM[var][0]} vintage {q(v)}" if v is not None
               else f"RTDSM {rt.RTDSM[var][0]}: no vintage dated {q(t)}")
        return lv, src, v


def _forbid_replication_packages(raw_root: Path) -> None:
    banned = [(raw_root / "bgms").resolve(), (raw_root / "cg").resolve()]
    real = zipfile.ZipFile

    class Guarded(real):
        def __init__(self, file, *a, **kw):
            if isinstance(file, (str, Path)):
                p = Path(file).resolve()
                if any(p == b or b in p.parents for b in banned):
                    raise PermissionError(f"export must not read replication package {p}")
            super().__init__(file, *a, **kw)

    zipfile.ZipFile = Guarded


def _stale(g: pd.DataFrame) -> int:
    """Rows with a vintage-t history whose x[t-1] is missing (late release)."""
    n = 0
    for _, r in g.iterrows():
        if r.hist_vintage == "":
            continue
        end = pd.Period(r.hist_first, freq="Q") + len(r["hist"]) - 1
        n += int(end != pd.Period(r.t, freq="Q") - 1 or not np.isfinite(r["hist"][-1]))
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "data" / "spf_panel.parquet"))
    args = ap.parse_args(argv)
    from ewh.provenance.archive import RAW_ROOT, REPO_ROOT
    _forbid_replication_packages(RAW_ROOT)
    src = EWHSource()
    panel = build_panel(src, src.headline)
    write_panel(panel, Path(args.out))
    commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    print(f"EWH commit {commit}, snapshot {src.rt.SNAPSHOT}; {len(panel)} rows -> {args.out} "
          f"({Path(args.out).stat().st_size} bytes)")
    for var, g in panel.groupby("variable", sort=False):
        ok = g.cons_err.notna() & g.cons_rev.notna()
        e, r = g.cons_err[ok].to_numpy(), g.cons_rev[ok].to_numpy()
        rc = r - r.mean()
        b = float((rc * (e - e.mean())).sum() / (rc * rc).sum())
        print(f"{var:6s} {g.t.iloc[0]}-{g.t.iloc[-1]} rows {len(g):3d}  consensus b {b:+.6f} "
              f"(T = {int(ok.sum())})  no vintage t: {int((g.hist_vintage == '').sum())}  "
              f"x[t-1] missing (late release): {_stale(g)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
