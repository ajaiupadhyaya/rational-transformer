"""Illustrative survey comparison (PREREG section 8; exploratory, droppable, not a test).

Real SPF macro histories are fed to the Family B models and the error-on-revision
coefficient b of the models' forecasts is printed beside the survey's own b on the same
variable-quarters. Real series are not AR(1) and the models never saw them.

Data: `data/spf_panel.parquet`, written by `scripts/export_spf_panel.py` from the
companion repository's raw-file pipeline (see that script and `data/PROVENANCE.md`).

The series fed to a model
-------------------------
For each variable and survey quarter t the panel holds the real-time levels x[s] of
vintage t, s <= t-1. The model sees their first differences d[s]:

    growth variables (NGDP, RGDP, PGDP, CPI): d[s] = 400 (x[s] / x[s-1] - 1),
        the annualised quarterly percent growth rate (the SPF's own CPI unit);
    level variables (UNEMP, TB3M, TN10Y): d[s] = x[s] - x[s-1].

`scale_history` standardises d by the mean (loc) and standard deviation (scale) of the
first differences dated strictly before t, i.e. every observation the vintage holds; the
model sees (d - loc) / scale and its forecasts are mapped back by loc + scale * F.

Horizons (how the survey's target is matched)
---------------------------------------------
The survey regression (BGMS 2020, as replicated by the companion repository) uses the
forecast made at t of quarter t+3 and its revision from the survey at t-1:

    growth:  fc = 100 (x[t+3] / x[t-1] - 1) as forecast at t, base x[t-1] of vintage t;
             fc_prev = the t-1 survey's forecast of growth from t-1 to t+3;
    level:   fc = x[t+3] as forecast at t;  fc_prev = x[t+3] as forecast at t-1.

The model's last input at t is d[t-1], so quarter t+k is its horizon k+1. Hence

    growth:  fc      = 100 (prod_{h=1..4} (1 + F_t[h] / 400) - 1)
             fc_prev = 100 (prod_{h=2..5} (1 + F_{t-1}[h] / 400) - 1)
    level:   fc      = x[t-1] + sum_{h=1..4} F_t[h]
             fc_prev = x[t-2] + sum_{h=1..5} F_{t-1}[h]     (x[t-2] of vintage t-1)

with F_t the forecasts made from vintage t and F_{t-1} those made from vintage t-1, exactly
as the survey's two forecasts use the information of their own survey dates. Every horizon
needed is in 1..5, inside the models' 1..6. The error is the survey's realised outcome minus
fc and the revision is fc - fc_prev; b is the OLS slope (with intercept) of error on revision.

Context length: the full-window Family B model was trained on at most T_MAX = 80 inputs with
time indices 1..80, so "all values" means the last 80 scaled differences, fed with indices
1..n (n <= 80). A window-W model gets the last W, fed with indices 80-W+1..80 (a late origin,
as most training origins are).

Drop rule (PREREG section 8; plan Task 11 step 4): the arm is dropped if the survey's
horizons cannot be matched to model horizons, or if for more than 20% of the variable-quarters
the scaled history fed to a Family B model leaves the range Family B was trained on. That range
is the central 99% of standardised Family B training values (x - mu) / sigma at t = 1..80,
rho from its U[0, 1] prior (`family_b_range`, 100,000 simulated paths, fixed seed; about
[-5.72, +5.79]). A variable-quarter is out of range if any value fed to the model lies outside
it, or if its history cannot be scaled or does not end at t-1. The fraction is pooled over
variables and checked for each Family B window (full and 4) over the variable-quarters with a
survey error and revision.

`forecaster_from_checkpoint` wraps a trained Family B checkpoint for later use; this module
never loads one on its own.

Usage: uv run python -m analysis.spf [--panel data/spf_panel.parquet] [--standin bayes]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from rt.sim import HORIZONS, sample_family_b
from rt.train import T_MAX

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "data" / "spf_panel.parquet"
PANEL_COLUMNS = (["variable", "kind", "target", "t", "hist_first", "hist", "hist_source",
                  "last_filled", "outcome", "cons_err", "cons_rev", "n_pairs"]
                 + [f"spf_c{k}" for k in range(1, 7)] + ["ind_id", "ind_err", "ind_rev"])
TARGET = "t+3"
# horizons of F_t (forecast at t) and of F_{t-1} (forecast at t-1) that build fc and fc_prev
REQUIRED_HORIZONS = {"growth": ((1, 2, 3, 4), (2, 3, 4, 5)),
                     "level": ((1, 2, 3, 4), (1, 2, 3, 4, 5))}
FAMILY_B_WINDOWS = (None, 4)
OUT_OF_RANGE_MAX = 0.20
RANGE_COVERAGE = 0.99
RANGE_N = 100_000
RANGE_SEED = (30_000, 0x5BF)          # apart from training (seed, 0xA11CE, step) and eval seeds
COMPARISON_COLUMNS = ["model", "variable", "kind", "window", "horizons", "T", "first", "last",
                      "b_model", "b_survey", "b_survey_ind", "n_ind"]

Forecaster = Callable[[np.ndarray], np.ndarray]


# ---- panel ------------------------------------------------------------------

def load_panel(path=None) -> pd.DataFrame:
    """The exported panel, columns checked against PANEL_COLUMNS."""
    df = pd.read_parquet(PANEL if path is None else path)
    missing = [c for c in PANEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"panel lacks columns {missing}")
    return df


def _q(s) -> pd.Period:
    return pd.Period(s, freq="Q")


def row_levels(row) -> pd.Series:
    """The row's real-time levels on a quarterly PeriodIndex (hist_first .. t-1)."""
    h = np.asarray(row["hist"], dtype=np.float64)
    if not len(h):
        return pd.Series([], dtype=np.float64, index=pd.PeriodIndex([], freq="Q"))
    return pd.Series(h, index=pd.period_range(_q(row["hist_first"]), periods=len(h), freq="Q"))


# ---- scaling ----------------------------------------------------------------

def first_differences(levels: pd.Series, kind: str) -> pd.Series:
    """d[s] = 400 (x[s]/x[s-1] - 1) (growth) or x[s] - x[s-1] (level), on consecutive quarters."""
    if not len(levels):
        return levels.astype(np.float64)
    idx = pd.period_range(levels.index.min(), levels.index.max(), freq="Q")
    x = levels.reindex(idx).astype(np.float64)
    if kind == "growth":
        return 400.0 * (x / x.shift(1) - 1.0)
    if kind == "level":
        return x - x.shift(1)
    raise ValueError(f"unknown kind {kind!r}")


def scale_history(hist: pd.Series, as_of, kind: str = "level"
                  ) -> tuple[pd.Series, tuple[float, float]]:
    """Scaled first differences of `hist` and (loc, scale).

    Only observations dated strictly before the forecast quarter `as_of` are used, for the
    differences and for loc / scale (their mean and sample standard deviation); anything in
    `hist` dated at or after `as_of` is ignored. Returns (d - loc) / scale, NaN differences
    dropped.
    """
    as_of = _q(as_of)
    h = hist[hist.index < as_of]
    d = first_differences(h, kind).dropna()
    if len(d) < 3:
        raise ValueError(f"fewer than 3 first differences before {as_of}")
    loc, scale = float(d.mean()), float(d.std(ddof=1))
    if not (np.isfinite(loc) and np.isfinite(scale) and scale > 0):
        raise ValueError(f"cannot scale history before {as_of} (loc {loc}, scale {scale})")
    return (d - loc) / scale, (loc, scale)


def fed_values(z, window: int | None) -> np.ndarray:
    """What a Family B model sees: the last W values, or the last T_MAX for the full window."""
    z = np.asarray(z, dtype=np.float64)
    return z[-(T_MAX if window is None else window):]


@cache
def family_b_range(coverage: float = RANGE_COVERAGE, n: int = RANGE_N) -> tuple[float, float]:
    """Central `coverage` interval of standardised Family B training values (x - mu) / sigma,
    t = 1..T_MAX, rho ~ U[0, 1] (`rt.sim.sample_family_b`, fixed seed)."""
    b = sample_family_b(n, np.random.default_rng(np.random.SeedSequence(list(RANGE_SEED))))
    z = (b.x[:, 1:T_MAX + 1] - b.mu[:, None]) / b.sigma[:, None]
    lo, hi = np.quantile(z, [(1 - coverage) / 2, (1 + coverage) / 2])
    return float(lo), float(hi)


# ---- forecasts, errors and revisions ----------------------------------------

def target_forecast(F: np.ndarray, last_level: float, kind: str, hs) -> float:
    """The survey target from model forecasts F (F[h-1] forecasts d at horizon h)."""
    f = np.asarray([F[h - 1] for h in hs], dtype=np.float64)
    if kind == "growth":
        return float(100.0 * (np.prod(1.0 + f / 400.0) - 1.0))
    if kind == "level":
        return float(last_level + f.sum())
    raise ValueError(f"unknown kind {kind!r}")


def _forecast_row(forecaster: Forecaster, row, window: int | None):
    """(F in original units, x[t-1], fed scaled values) or None if the row cannot be used."""
    t = _q(row["t"])
    lv = row_levels(row)
    if not len(lv) or lv.index.max() != t - 1 or not np.isfinite(lv.iloc[-1]):
        return None
    try:
        z, (loc, scale) = scale_history(lv, t, row["kind"])
    except ValueError:
        return None
    fed = fed_values(z.to_numpy(), window)
    if not np.all(np.isfinite(fed)):
        return None
    Fz = np.asarray(forecaster(fed), dtype=np.float64).reshape(-1)
    if Fz.shape != (len(HORIZONS),):
        raise ValueError(f"forecaster must return {len(HORIZONS)} forecasts; got {Fz.shape}")
    return loc + scale * Fz, float(lv.iloc[-1]), fed


def model_quarters(forecaster: Forecaster, panel: pd.DataFrame,
                   window: int | None = None) -> pd.DataFrame:
    """Per variable-quarter: the model's fc, fc_prev, err and rev beside the survey's."""
    out = []
    for var, g in panel.groupby("variable", sort=False):
        g = g.assign(_p=g["t"].map(_q)).sort_values("_p")
        kind = g["kind"].iloc[0]
        if kind not in REQUIRED_HORIZONS:
            raise ValueError(f"{var}: unknown kind {kind!r}")
        cur, prev = REQUIRED_HORIZONS[kind]
        fits = {r["_p"]: _forecast_row(forecaster, r, window) for _, r in g.iterrows()}
        for _, r in g.iterrows():
            t, now, before = r["_p"], fits[r["_p"]], fits.get(r["_p"] - 1)
            fc = target_forecast(now[0], now[1], kind, cur) if now else np.nan
            fc_prev = target_forecast(before[0], before[1], kind, prev) if before else np.nan
            out.append({"variable": var, "kind": kind, "t": r["t"], "fc": fc,
                        "fc_prev": fc_prev, "err": r["outcome"] - fc, "rev": fc - fc_prev,
                        "cons_err": r["cons_err"], "cons_rev": r["cons_rev"],
                        "ind_err": r["ind_err"], "ind_rev": r["ind_rev"]})
    return pd.DataFrame(out)


def _slope(y, x) -> float:
    y, x = np.asarray(y, np.float64), np.asarray(x, np.float64)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    if len(y) < 3:
        return np.nan
    xc = x - x.mean()
    sxx = float((xc * xc).sum())
    return float((xc * (y - y.mean())).sum()) / sxx if sxx > 0 else np.nan


def _horizon_text(kind: str) -> str:
    cur, prev = REQUIRED_HORIZONS[kind]
    return f"t: h{cur[0]}-{cur[-1]}; t-1: h{prev[0]}-{prev[-1]}"


def spf_comparison(forecaster: Forecaster, panel: pd.DataFrame, window: int | None = None,
                   label: str = "model") -> pd.DataFrame:
    """b of the model's and of the survey's forecasts per variable, on the same quarters.

    `forecaster` maps a 1-D scaled history (the values fed, see `fed_values`) to forecasts for
    horizons 1..6 in the same scaled units. Same quarters: those where the model's error and
    revision and the survey's consensus error and revision all exist. `b_survey` is the
    consensus regression, `b_survey_ind` the pooled individual one over the same quarters.
    """
    q = model_quarters(forecaster, panel, window)
    rows = []
    for var, g in q.groupby("variable", sort=False):
        same = (g[["err", "rev", "cons_err", "cons_rev"]].notna().all(axis=1)
                & np.isfinite(g[["err", "rev", "cons_err", "cons_rev"]]).all(axis=1))
        s = g[same]
        ie = np.concatenate([np.asarray(v, np.float64) for v in s["ind_err"]] or [np.array([])])
        ir = np.concatenate([np.asarray(v, np.float64) for v in s["ind_rev"]] or [np.array([])])
        rows.append({"model": label, "variable": var, "kind": g["kind"].iloc[0],
                     "window": "full" if window is None else str(window),
                     "horizons": _horizon_text(g["kind"].iloc[0]), "T": int(len(s)),
                     "first": s["t"].iloc[0] if len(s) else "",
                     "last": s["t"].iloc[-1] if len(s) else "",
                     "b_model": _slope(s["err"], s["rev"]),
                     "b_survey": _slope(s["cons_err"], s["cons_rev"]),
                     "b_survey_ind": _slope(ie, ir), "n_ind": int(len(ie))})
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


# ---- drop rule --------------------------------------------------------------

@dataclass
class DropDecision:
    drop: bool
    reasons: list[str]
    bounds: tuple[float, float]
    in_range: dict[str, float] = field(default_factory=dict)   # window -> pooled fraction
    table: pd.DataFrame = field(default_factory=pd.DataFrame)  # per variable and window

    @property
    def text(self) -> str:
        lo, hi = self.bounds
        fr = ", ".join(f"window {w}: {100 * v:.1f}% in range" for w, v in self.in_range.items())
        if self.drop:
            return "Survey comparison dropped (PREREG section 8 drop rule): " + "; ".join(
                self.reasons) + "."
        return (f"Survey comparison kept: survey horizons matched (t+3 target; model horizons "
                f"1-5); scaled histories within the Family B range [{lo:.2f}, {hi:.2f}] for "
                f"{fr} (limit: at most {100 * OUT_OF_RANGE_MAX:.0f}% out of range).")


def drop_check(panel: pd.DataFrame, windows=FAMILY_B_WINDOWS, max_horizon: int = max(HORIZONS),
               limit: float = OUT_OF_RANGE_MAX, bounds: tuple[float, float] | None = None
               ) -> DropDecision:
    """PREREG drop rule: unmatched horizons, or > `limit` of variable-quarters out of range."""
    lo, hi = family_b_range() if bounds is None else bounds
    reasons = []
    for var, g in panel.groupby("variable", sort=False):
        kinds, targets = set(g["kind"]), set(g["target"])
        if len(kinds) != 1 or not kinds <= set(REQUIRED_HORIZONS):
            reasons.append(f"horizons cannot be matched for {var}: unknown kind {sorted(kinds)}")
            continue
        if targets != {TARGET}:
            reasons.append(f"horizons cannot be matched for {var}: survey target "
                           f"{sorted(targets)} is not {TARGET}")
            continue
        need = max(max(h) for h in REQUIRED_HORIZONS[kinds.pop()])
        if need > max_horizon:
            reasons.append(f"horizons cannot be matched for {var}: needs model horizon {need}, "
                           f"models forecast 1..{max_horizon}")
    use = panel[panel["cons_err"].notna() & panel["cons_rev"].notna()]
    rows, in_range = [], {}
    for w in windows:
        wname = "full" if w is None else str(w)
        n_out = n_bad = 0
        for var, g in use.groupby("variable", sort=False):
            out = bad = 0
            worst = 0.0
            for _, r in g.iterrows():
                t, lv = _q(r["t"]), row_levels(r)
                try:
                    if not len(lv) or lv.index.max() != t - 1:
                        raise ValueError("stale history")
                    z, _ = scale_history(lv, t, r["kind"])
                    fed = fed_values(z.to_numpy(), w)
                    if not np.all(np.isfinite(fed)):
                        raise ValueError("missing values in the fed window")
                except (ValueError, KeyError):
                    bad += 1
                    continue
                worst = max(worst, float(np.abs(fed).max()))
                out += int(fed.min() < lo or fed.max() > hi)
            rows.append({"variable": var, "window": wname, "n": int(len(g)), "out_of_range": out,
                         "unscalable": bad, "frac_out": (out + bad) / len(g) if len(g) else np.nan,
                         "max_abs_fed": worst})
            n_out, n_bad = n_out + out, n_bad + bad
        n = len(use)
        frac = (n_out + n_bad) / n if n else 1.0
        in_range[wname] = 1.0 - frac
        if not n:
            reasons.append("no variable-quarter has a survey error and revision")
        elif frac > limit:
            reasons.append(f"window {wname}: scaled histories outside the Family B range "
                           f"[{lo:.2f}, {hi:.2f}] (or unscalable) for {100 * frac:.1f}% of "
                           f"{n} variable-quarters (limit {100 * limit:.0f}%)")
    return DropDecision(drop=bool(reasons), reasons=reasons, bounds=(lo, hi),
                        in_range=in_range, table=pd.DataFrame(rows))


def spf_arm(forecasters: dict[str, tuple[Forecaster, int | None]], panel=None
            ) -> tuple[pd.DataFrame, DropDecision]:
    """The comparison table for labelled (forecaster, window) pairs, plus the drop decision.
    If the arm is dropped the table is empty and the decision says why."""
    panel = load_panel() if panel is None else panel
    decision = drop_check(panel)
    if decision.drop:
        return pd.DataFrame(columns=COMPARISON_COLUMNS), decision
    frames = [spf_comparison(f, panel, w, label) for label, (f, w) in forecasters.items()]
    return pd.concat(frames, ignore_index=True), decision


# ---- forecasters -------------------------------------------------------------

def forecaster_from_checkpoint(path, window: int | None, size: str = "l",
                               device: str = "cpu") -> Forecaster:
    """A trained Family B `Forecaster` as a callable on 1-D scaled histories.

    Loaded as `rt.evaluate._load_model` does (architecture from the checkpoint's own config).
    Full window: the last T_MAX values with time indices 1..n. Window W: the last W values with
    indices T_MAX-W+1..T_MAX. Returns the six forecasts read at the last position.
    """
    import torch

    from rt.evaluate import _load_model
    from rt.train import Cell

    model, _ = _load_model(Cell("B", size, window, 0), Path(path))
    dev = torch.device(device)
    model.to(dev).eval()

    @torch.no_grad()
    def f(hist) -> np.ndarray:
        v = fed_values(hist, window)
        n = len(v)
        times = np.arange(1, n + 1) if window is None else np.arange(T_MAX - n + 1, T_MAX + 1)
        out = model(torch.tensor(v[None], dtype=torch.float32, device=dev),
                    torch.tensor(times[None], dtype=torch.long, device=dev))[0, -1]
        return out.cpu().numpy().astype(np.float64)

    return f


def bayes_forecaster(window: int | None = None, grid: int = 401) -> Forecaster:
    """Stand-in: the exact Family B Bayes forecaster (`rt.bayes.bayes_b`), with the same
    placement of the inputs as `forecaster_from_checkpoint`."""
    from rt.bayes import bayes_b

    def f(hist) -> np.ndarray:
        v = fed_values(hist, window)
        n = len(v)
        x = np.zeros((1, T_MAX + max(HORIZONS) + 1))
        t = n if window is None else T_MAX
        x[0, t - n + 1:t + 1] = v
        return bayes_b(x, window, grid=grid)[0, t]

    return f


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Survey comparison: drop check on the real panel.")
    ap.add_argument("--panel", default=str(PANEL))
    ap.add_argument("--standin", choices=["bayes"], default=None,
                    help="also print the table for the Family B Bayes stand-in (no model)")
    args = ap.parse_args(argv)
    panel = load_panel(args.panel)
    d = drop_check(panel)
    print(d.text)
    print(d.table.to_string(index=False))
    if args.standin == "bayes":
        tab = pd.concat([spf_comparison(bayes_forecaster(w), panel, w, "bayes_b stand-in")
                         for w in FAMILY_B_WINDOWS], ignore_index=True)
        print(tab.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
