"""Human (AKLMT 2023, Experiment 1) curve and a check of its design against the Family B prior."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from rt.sim import B_PRIOR, RHOS

ROOT = Path(__file__).resolve().parent.parent
HUMAN_CSV = ROOT / "data" / "aklmt_figure2.csv"
# Raw package location, from EWH `ewh/provenance/archive.py` (RAW_ROOT = <EWH>/data/raw).
EWH_ZIP = (Path.home() / "Documents" / "expectations-without-hindsight" / "data" / "raw"
           / "aklmt" / "Replication.zip")
MEMBER = "Afrouzi replication/data/Experiment1/detail.csv"


def human_curve(path=None) -> pd.DataFrame:
    """The six AKLMT Figure II rows: rho, implied_rho, implied_rho_se, b_eor, b_eor_se, ..."""
    return pd.read_csv(HUMAN_CSV if path is None else path)


def aklmt_design_check(zip_path=None) -> pd.DataFrame | None:
    """Per-rho mean, sd and range of the realised series and of its one-step innovations,
    against the Family B prior (PREREG section 2). None if the raw package is absent.

    Innovation = x_t - rho * x_{t-1} over consecutive rounds. `prior_p_var_ge` is the Family B
    prior probability that sigma^2 is at least the observed innovation variance
    (sigma^2 ~ InvGamma(a, b)); `abs_mean_over_sd` compares the series mean, in innovation sd
    units, with the prior sd of mu / sigma, which is sqrt(k).
    """
    path = Path(EWH_ZIP if zip_path is None else zip_path)
    if not path.exists():
        return None
    try:
        with zipfile.ZipFile(path) as z:
            df = pd.read_csv(io.BytesIO(z.read(MEMBER)))
    except Exception:
        return None
    df["rho"] = df["rho"].round(1)
    df = df.sort_values(["assignmentid", "round"]).reset_index(drop=True)
    g = df.groupby("assignmentid")
    lag = g["actual"].shift(1).where(g["round"].diff() == 1)
    df["innov"] = df["actual"] - df["rho"] * lag
    a, b, k = B_PRIOR["a"], B_PRIOR["b"], B_PRIOR["k"]
    rows = []
    for rho in RHOS:
        d = df[np.isclose(df["rho"], rho)]
        x, e = d["actual"].dropna().to_numpy(float), d["innov"].dropna().to_numpy(float)
        if not len(x) or not len(e):
            continue
        v = float(e.var(ddof=1))
        rows.append({"rho": f"{rho:.1f}", "n_series": int(d["assignmentid"].nunique()),
                     "series_mean": float(x.mean()), "series_sd": float(x.std(ddof=1)),
                     "series_min": float(x.min()), "series_max": float(x.max()),
                     "innov_mean": float(e.mean()), "innov_sd": float(np.sqrt(v)),
                     "innov_min": float(e.min()), "innov_max": float(e.max()),
                     "abs_mean_over_sd": abs(float(x.mean())) / float(np.sqrt(v)),
                     "prior_sd_mu_over_sigma": float(np.sqrt(k)),
                     "prior_p_var_ge": float(stats.invgamma.sf(v, a, scale=b))})
    return pd.DataFrame(rows)
