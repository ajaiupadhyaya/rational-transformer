"""H1-H3 decisions, flagged findings and the training-length check, exactly as PREREG 6-7.

Reads only the rows PREREG section 7 names: family A, h = 1, step 30000, the stated
benchmark, grid rho (the `prior` row is exposed by `prior_rows` and never enters a decision).
A (size, window) group with fewer seeds than the plan expects is "incomplete": shown, never
used in a decision. Size l is primary for H2 and H3; other sizes never enter a decision.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from analysis.summary import MARGINS, cell_ids, expected_seeds, norm_rho
from rt.sim import RHOS

GRID_RHOS = [norm_rho(r) for r in RHOS]
COLUMNS = ("excess_mse_pct", "gap")
H2_WINDOWS = ("2", "4", "8", "16")
H3_RHOS = ("0.6", "0.8")
H3_MARGIN = MARGINS["gap"]
BONFERRONI_K = 8
STEP, STEP_LONG = 30000, 60000
PRIMARY = "l"
OTHER_SIZES = ("xs", "s", "m")


def bonferroni_t(n: int) -> float:
    """t_{1 - 0.05/16, n - 1}: the multiplier of the 1 - 0.05/8 two-sided interval."""
    return float(stats.t.ppf(1 - 0.05 / (2 * BONFERRONI_K), n - 1))


class Index:
    """Row lookup over a summary frame (from `summarise`, or read back from summary.csv)."""

    def __init__(self, summary: pd.DataFrame | None):
        self.rows: dict[tuple, dict] = {}
        if summary is None or len(summary) == 0:
            return
        for r in summary.to_dict("records"):
            k = (r["family"], r["size"], str(r["window"]), int(r["step"]), norm_rho(r["rho"]),
                 int(r["h"]), r["bench"], r["column"])
            self.rows[k] = r

    def cell(self, size, window, bench, rho, column, *, step=STEP, family="A", h=1) -> dict:
        """The decision-ready view of one row; status is ok, incomplete or not run."""
        r = self.rows.get((family, size, str(window), step, norm_rho(rho), h, bench, column))
        base = {"family": family, "size": size, "window": str(window), "step": step,
                "rho": norm_rho(rho), "h": h, "bench": bench, "column": column}
        if r is None:
            return {**base, "status": "not run"}
        vals = json.loads(r["values"]) if isinstance(r["values"], str) else list(r["values"])
        exp = expected_seeds(family, size, window)
        n = int(r["n_seeds"])
        ok = n >= exp and n >= 2
        mean = float(r["mean"])
        out = {**base, "status": "ok" if ok else "incomplete", "n_seeds": n,
               "expected_seeds": exp, "values": vals, "mean": mean,
               "sd": _nan_none(r["sd"]), "lo": _nan_none(r["lo"]), "hi": _nan_none(r["hi"]),
               "verdict": r["verdict"], "sign": "+" if mean > 0 else ("-" if mean < 0 else "0")}
        if n >= 2:
            m = np.asarray(vals, dtype=float)
            out["lo_adj"] = float(m.mean() - bonferroni_t(n) * m.std(ddof=1) / np.sqrt(n))
        else:
            out["lo_adj"] = None
        return out


def _nan_none(v):
    v = float(v)
    return None if np.isnan(v) else v


def _rows(idx: Index, size: str, window: str, bench: str) -> list[dict]:
    return [idx.cell(size, window, bench, rho, col) for rho in GRID_RHOS for col in COLUMNS]


def _decide(rows: list[dict], good: str, bad: str) -> str:
    ok = [r for r in rows if r["status"] == "ok"]
    if any(r["verdict"] == "different" for r in ok):
        return bad
    if len(ok) == len(rows) and all(r["verdict"] == "equivalent" for r in ok):
        return good
    return "not_established"


def _missing(rows: list[dict]) -> list[dict]:
    return [{k: r[k] for k in ("size", "window", "rho", "column", "status")}
            for r in rows if r["status"] != "ok"]


def _h2_block(idx: Index, size: str, window: str) -> dict:
    rows = _rows(idx, size, window, "window_bayes")
    return {"decision": _decide(rows, "rational_given_information", "departure_pending_check"),
            "rows": rows, "missing": _missing(rows)}


def _combo(idx: Index, size: str, window: str, rho: str) -> dict:
    c = idx.cell(size, window, "window_bayes", rho, "gap")
    ok = c["status"] == "ok"
    c["positive_95"] = bool(ok and c["lo"] is not None and c["lo"] > H3_MARGIN)
    c["positive_adjusted"] = bool(ok and c["lo_adj"] is not None and c["lo_adj"] > H3_MARGIN)
    return c


def hypotheses(summary: pd.DataFrame) -> dict:
    idx = Index(summary)

    h1_rows = _rows(idx, PRIMARY, "full", "bayes")
    h1 = {"decision": _decide(h1_rows, "supported", "rejected_pending_check"),
          "rows": h1_rows, "missing": _missing(h1_rows)}

    h2 = {"primary_size": PRIMARY,
          "windows": {w: _h2_block(idx, PRIMARY, w) for w in H2_WINDOWS},
          "other_sizes": {s: {w: _h2_block(idx, s, w) for w in H2_WINDOWS} for s in OTHER_SIZES}}

    combos = [_combo(idx, PRIMARY, w, r) for w in H2_WINDOWS for r in H3_RHOS]
    k95 = sum(c["positive_95"] for c in combos)
    kadj = sum(c["positive_adjusted"] for c in combos)
    n_missing = sum(c["status"] != "ok" for c in combos)
    if kadj:
        decision = "supported_pending_check"
    elif n_missing:
        decision = "not_established"
    else:
        decision = "not_supported"
    other = {}
    for s in OTHER_SIZES:
        cs = [_combo(idx, s, w, r) for w in H2_WINDOWS for r in H3_RHOS]
        other[s] = {"combos": cs, "n_present": sum(c["status"] == "ok" for c in cs),
                    "k_positive_95": sum(c["positive_95"] for c in cs),
                    "n_missing": sum(c["status"] != "ok" for c in cs)}
    h3 = {"primary_size": PRIMARY, "denominator": BONFERRONI_K, "combos": combos,
          "k_positive_95": k95, "k_positive_adjusted": kadj, "n_missing": n_missing,
          "bonferroni_t_n5": bonferroni_t(5), "decision": decision,
          "statement": (f"{k95} of {BONFERRONI_K} positive at 95%, "
                        + (f"{kadj} after adjustment" if kadj else "none after adjustment")
                        + (f"; {n_missing} of {BONFERRONI_K} not run or incomplete"
                           if n_missing else "")),
          "other_sizes": other,
          "other_sizes_k_positive_95": sum(o["k_positive_95"] for o in other.values()),
          "other_sizes_denominator": len(OTHER_SIZES) * BONFERRONI_K}
    out = {"h1": h1, "h2": h2, "h3": h3}
    out["flagged"] = flagged_findings(out)
    return out


def prior_rows(summary: pd.DataFrame) -> dict:
    """The `prior` rows printed beside H1-H3 tables. Never part of a decision."""
    idx = Index(summary)
    out = {"h1": [idx.cell(PRIMARY, "full", "bayes", "prior", c) for c in COLUMNS]}
    for w in H2_WINDOWS:
        out[f"h2_w{w}"] = [idx.cell(PRIMARY, w, "window_bayes", "prior", c) for c in COLUMNS]
    return out


def _finding(hyp: str, row: dict, window: str, kind: str) -> dict:
    return {"id": f"{hyp}-W{window}-rho{row['rho']}-{row['column']}", "hypothesis": hyp,
            "window": window, "rho": row["rho"], "column": row["column"], "bench": row["bench"],
            "kind": kind, "sign": row["sign"], "mean": row["mean"], "lo": row["lo"],
            "hi": row["hi"], "lo_adj": row["lo_adj"], "values": row["values"],
            "cells": cell_ids("A", PRIMARY, window)}


def flagged_findings(h: dict) -> list[dict]:
    """Size-l 'different' verdicts in H1/H2 and adjusted-positive H3 combinations."""
    out = []
    for r in h["h1"]["rows"]:
        if r["status"] == "ok" and r["verdict"] == "different":
            out.append(_finding("H1", r, "full", "different"))
    for w, blk in h["h2"]["windows"].items():
        for r in blk["rows"]:
            if r["status"] == "ok" and r["verdict"] == "different":
                out.append(_finding("H2", r, w, "different"))
    for c in h["h3"]["combos"]:
        if c["positive_adjusted"]:
            out.append(_finding("H3", c, c["window"], "positive_adjusted"))
    return out


SURVIVES, NOT_ROBUST, NOT_RUN = "survives", "not robust to training length", "not run"


def training_length_check(flagged: list[dict], summary_60k: pd.DataFrame | None) -> list[dict]:
    """PREREG section 6 survival rule for each finding, from the step-60000 summary.

    H1/H2: same verdict ('different'), same column and rho, same sign at step 60000.
    H3: positive after the same Bonferroni adjustment at step 60000.
    A finding whose 60,000-step row is missing or incomplete is 'not run'.
    """
    idx = Index(summary_60k)
    out = []
    for f in flagged:
        r = idx.cell(PRIMARY, f["window"], f["bench"], f["rho"], f["column"], step=STEP_LONG)
        if r["status"] != "ok":
            outcome = NOT_RUN
        elif f["hypothesis"] == "H3":
            outcome = SURVIVES if (r["lo_adj"] is not None and r["lo_adj"] > H3_MARGIN) else NOT_ROBUST
        else:
            outcome = SURVIVES if (r["verdict"] == "different" and r["sign"] == f["sign"]) \
                else NOT_ROBUST
        rows60 = [idx.cell(PRIMARY, f["window"], f["bench"], rho, c, step=STEP_LONG)
                  for rho in GRID_RHOS for c in COLUMNS]
        all_eq = all(x["status"] == "ok" and x["verdict"] == "equivalent" for x in rows60)
        out.append({**f, "outcome": outcome, "n_60k": r.get("n_seeds"), "mean_60k": r.get("mean"), "lo_60k": r.get("lo"),
                    "hi_60k": r.get("hi"), "verdict_60k": r.get("verdict"),
                    "lo_adj_60k": r.get("lo_adj"), "values_60k": r.get("values"),
                    "all_equivalent_at_60k": bool(all_eq and f["hypothesis"] != "H3")})
    return out


def resolve_decisions(h: dict, checked: list[dict]) -> dict:
    """Decisions after the training-length check (H1, each H2 window, H3)."""
    def group(hyp, window=None):
        return [c for c in checked if c["hypothesis"] == hyp
                and (window is None or c["window"] == window)]

    def resolve(cs, pending, survive, notrun, fail):
        if not cs:
            return None
        if any(c["outcome"] == SURVIVES for c in cs):
            return survive
        if any(c["outcome"] == NOT_RUN for c in cs):
            return notrun
        return fail

    out = {}
    if h["h1"]["decision"] == "rejected_pending_check":
        out["h1"] = resolve(group("H1"), None, "rejected",
                            "different at 30,000 steps; not tested for training length",
                            "not_established")
    for w, blk in h["h2"]["windows"].items():
        if blk["decision"] == "departure_pending_check":
            out[f"h2_w{w}"] = resolve(group("H2", w), None, "departure_from_rational_use",
                                      "different at 30,000 steps; not tested for training length",
                                      "not_established")
    if h["h3"]["decision"] == "supported_pending_check":
        out["h3"] = resolve(group("H3"), None, "supported",
                            "positive after adjustment at 30,000 steps; not tested for training length",
                            "not_supported")
    return out
