"""The gate: prove the measuring instruments work before any paid compute is used.

Three checks, written to results/gate.json as
{"passed": bool, "code_hash": str, "checks": {...}}:

  1. benchmark  Bayes `eor_b` on the prior set is within +-0.01 for both families
                (100,000 prior-drawn sequences; Family B on a 2001-point grid); the
                point-prior Bayes forecast equals `fire`; and on h = 1 squared error
                `bayes <= window_bayes` for every window and
                `window_bayes <= windowed least squares` for W in {8, 16}
                (20,000 sequences; Family B on standardised deviations).
  2. planted    the Task 4 planted `kappa` and `gap` values are recovered within 0.01.
  3. learning   results/gate/learning.json, written by `python -m rt.gate --learn` for the
                SAME code hash, records `excess_mse_pct < 5` (A-s-wfull-s0, 3000 steps,
                prior set, h = 1, against Bayes).

The sweep launcher refuses to start unless gate.json says passed and its `code_hash`
equals the current `code_hash()`, so any edit to rt/*.py or configs/*.json re-opens the gate.

Usage: uv run python -m rt.gate --learn --device mps   (once per code state, ~11 min)
       uv run python -m rt.gate        (prints each check, ends GATE PASS or GATE FAIL)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from rt.baselines import fire, ls_coefficients
from rt.bayes import bayes_a, bayes_b
from rt.diagnostics import eor_b, gap, implied_persistence, kappa
from rt.sim import T_EVAL, sample_family_a, sample_family_b
from rt.train import _atomic, load_cfg

ROOT = Path(__file__).resolve().parent.parent
N_EOR = 100_000          # prior-drawn sequences for the eor_b check, per family
N_ORDER = 20_000         # the first N_ORDER of those, for the ordering checks
GRID = 2001              # rho grid points for both families
EOR_TOL = 0.01
PLANTED_TOL = 0.01
LS_WINDOWS = (8, 16)     # windows at which a least-squares fit is well defined and compared
LEARNING_CELL = "A-s-wfull-s0"
LEARNING_STEP = 3000
LEARNING_MAX_EXCESS_PCT = 5.0
_T = np.array(T_EVAL)


def code_hash(root=None) -> str:
    """sha256 over the sorted (path, contents) of rt/*.py and configs/*.json under `root`."""
    root = ROOT if root is None else Path(root)
    files = sorted(list(root.glob("rt/*.py")) + list(root.glob("configs/*.json")),
                   key=lambda p: p.relative_to(root).as_posix())
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(root).as_posix().encode() + b"\0")
        h.update(p.read_bytes() + b"\0")
    return h.hexdigest()


def _std(batch, x, *Fs):
    """(x - mu) / sigma and (F - mu) / sigma; Family A (mu = 0, sigma = 1) passes through."""
    mu, sigma = np.asarray(batch.mu), np.asarray(batch.sigma)
    out = [(x - mu[:, None]) / sigma[:, None]]
    out += [(F - mu.reshape((-1,) + (1,) * (F.ndim - 1))) / sigma.reshape((-1,) + (1,) * (F.ndim - 1))
            for F in Fs]
    return tuple(out)


def _mse1(F, x) -> float:
    """Squared error at h = 1 pooled over t = 41..80 (F is (n, 81, >=1))."""
    return float(np.mean((x[:, _T + 1] - F[:, _T, 0]) ** 2))


def _ls_window_mse1(x, sigma, w: int, intercept: bool) -> float:
    """Standardised MSE at h = 1 of an OLS AR(1) fit on only the last `w` observations.

    The fit is affine-equivariant (with an intercept), so dividing the raw error by the
    sequence's sigma equals the error on (x - mu) / sigma; Family A has sigma = 1.
    """
    se = np.empty((x.shape[0], len(_T)))
    for j, t in enumerate(_T):
        a, b = ls_coefficients(x[:, t - w + 1:t + 1], intercept)
        se[:, j] = ((x[:, t + 1] - (a + b * x[:, t])) / sigma) ** 2
    return float(se.mean())


def _check_benchmark(cfg, batches: dict, fa_full: np.ndarray, log) -> dict:
    out = {"passed": True, "families": {}}
    for fam in ("A", "B"):
        b = batches[fam]
        t0 = time.perf_counter()
        F = fa_full if fam == "A" else bayes_b(b.x, None, grid=GRID)
        x_s, F_s = _std(b, b.x, F)
        eor = eor_b(F_s, x_s, 1)
        fam_out = {"eor_b_h1": eor, "eor_ok": abs(eor) <= EOR_TOL, "windows": {}}
        log(f"  family {fam}: bayes eor_b(h=1) = {eor:+.5f} (|.| <= {EOR_TOL})")
        n = N_ORDER
        x_o, sg_o = b.x[:n], b.sigma[:n]
        mse_bayes = _mse1(F_s[:n], x_s[:n])
        ordering_ok = True
        for w in [w for w in cfg["windows"] if w is not None]:
            Fw = bayes_a(x_o, w, grid=GRID) if fam == "A" else bayes_b(x_o, w, grid=GRID)
            xw_s, Fw_s = _std(_Slice(b, n), x_o, Fw)
            mse_w = _mse1(Fw_s, xw_s)
            entry = {"bayes": mse_bayes, "window_bayes": mse_w,
                     "bayes_le_window_bayes": mse_bayes <= mse_w}
            line = f"    W={w:>2}: bayes {mse_bayes:.5f} <= window_bayes {mse_w:.5f}"
            ok = entry["bayes_le_window_bayes"]
            if w in LS_WINDOWS:
                mse_ls = _ls_window_mse1(x_o, sg_o, w, intercept=(fam == "B"))
                entry["ls"] = mse_ls
                entry["window_bayes_le_ls"] = mse_w <= mse_ls
                ok = ok and entry["window_bayes_le_ls"]
                line += f" <= ls {mse_ls:.5f}"
            entry["ok"] = bool(ok)
            ordering_ok = ordering_ok and ok
            fam_out["windows"][str(w)] = entry
            log(line + ("" if ok else "   <-- ORDER VIOLATED"))
        fam_out["ordering_ok"] = bool(ordering_ok)
        fam_out["seconds"] = time.perf_counter() - t0
        out["families"][fam] = fam_out
        out["passed"] = bool(out["passed"] and fam_out["eor_ok"] and ordering_ok)
    # point-prior Bayes equals fire (Family A, where rho is the only unknown)
    pb = sample_family_a(500, np.random.default_rng(0), rho=0.8)
    err = float(np.nanmax(np.abs(bayes_a(pb.x, None, grid=3, prior=(0.8, 0.8)) - fire(pb.x, 0.8))))
    out["point_prior_vs_fire_max_abs"] = err
    out["point_prior_ok"] = err < 1e-9
    out["passed"] = bool(out["passed"] and out["point_prior_ok"])
    log(f"  point-prior bayes vs fire: max |diff| = {err:.2e} (< 1e-9)")
    return out


class _Slice:
    """The first n rows of a Batch (just the fields `_std` reads)."""
    def __init__(self, b, n):
        self.mu, self.sigma = b.mu[:n], b.sigma[:n]


def _check_planted(batch_a, fa_full: np.ndarray, log) -> dict:
    x, Fb = batch_a.x, fa_full
    kap = {}
    for theta in (-0.3, 0.0, 0.2, 0.5):
        Fm = Fb.copy()
        Fm[:, 2:, 0] += theta * (x[:, 2:81] - Fb[:, 1:80, 0])
        kap[str(theta)] = kappa(Fm, Fb, x)
        del Fm
    b = sample_family_a(100_000, np.random.default_rng(1), rho=0.4)
    g = gap(fire(b.x, 0.55), fire(b.x, 0.4), b.x)
    base = implied_persistence(fire(b.x, 0.4), b.x)
    ok = (all(abs(float(k) - v) < PLANTED_TOL for k, v in kap.items())
          and abs(g - 0.15) < PLANTED_TOL and abs(base - 0.4) < PLANTED_TOL)
    for k, v in kap.items():
        log(f"  planted kappa {float(k):+.1f}: recovered {v:+.5f}")
    log(f"  planted gap 0.15: recovered {g:+.5f}  (fire implied persistence {base:.5f} vs 0.4)")
    return {"passed": bool(ok), "kappa": kap, "gap": g, "gap_planted": 0.15,
            "fire_implied_persistence": base}


LEARN_CMD = "uv run python -m rt.gate --learn --device mps"


def _check_learning(root: Path, current_hash: str, log) -> dict:
    """Passes only on results/gate/learning.json written by `--learn` for THIS code."""
    path = root / "results" / "gate" / "learning.json"
    try:
        ev = json.loads(path.read_text())
        excess, ev_hash = float(ev["excess_mse_pct"]), ev["code_hash"]
    except (OSError, ValueError, KeyError, TypeError):
        msg = f"missing or unreadable {path}; produce it with: {LEARN_CMD}"
        log("  " + msg)
        return {"passed": False, "message": msg}
    if ev_hash != current_hash:
        msg = (f"{path} certifies different code (hash {str(ev_hash)[:12]}, now "
               f"{current_hash[:12]}); re-run: {LEARN_CMD}")
        log("  " + msg)
        return {"passed": False, "message": msg, "excess_mse_pct": excess}
    ok = bool(excess < LEARNING_MAX_EXCESS_PCT)
    log(f"  {LEARNING_CELL} step {LEARNING_STEP}: excess_mse_pct {excess:.3f} "
        f"(< {LEARNING_MAX_EXCESS_PCT}), code hash matches")
    return {"passed": ok, "excess_mse_pct": excess, "cell": LEARNING_CELL, "step": LEARNING_STEP}


def run_gate(cfg=None, root=None, log=print, hash_fn=None) -> dict:
    cfg = load_cfg() if cfg is None else cfg
    root = ROOT if root is None else Path(root)
    hash_fn = hash_fn or (lambda: code_hash(root))
    start_hash = hash_fn()
    rng_a, rng_b = np.random.default_rng(0), np.random.default_rng(0)
    batches = {"A": sample_family_a(N_EOR, rng_a), "B": sample_family_b(N_EOR, rng_b)}
    checks = {}
    log("check 1: benchmark")
    t0 = time.perf_counter()
    fa_full = bayes_a(batches["A"].x, None, grid=GRID)
    checks["benchmark"] = _check_benchmark(cfg, batches, fa_full, log)
    log(f"  -> {'pass' if checks['benchmark']['passed'] else 'FAIL'} ({time.perf_counter() - t0:.0f}s)")
    log("check 2: planted")
    t0 = time.perf_counter()
    checks["planted"] = _check_planted(batches["A"], fa_full, log)
    log(f"  -> {'pass' if checks['planted']['passed'] else 'FAIL'} ({time.perf_counter() - t0:.0f}s)")
    log("check 3: learning")
    checks["learning"] = _check_learning(root, start_hash, log)
    log(f"  -> {'pass' if checks['learning']['passed'] else 'FAIL'}")
    end_hash = hash_fn()
    stable = end_hash == start_hash
    if not stable:
        msg = (f"code or config changed while the gate ran (hash {start_hash[:12]} at start, "
               f"{end_hash[:12]} at end); nothing is certified, re-run the gate")
        log("code stability: FAIL\n  " + msg)
        checks["stable"] = {"passed": False, "message": msg}
    return {"passed": all(c["passed"] for c in checks.values()),
            "code_hash": start_hash, "checks": checks}


def learn(device: str, root=None, log=print, hash_fn=None, run=subprocess.run) -> int:
    """Run the learning check as subprocesses and record it against the code it ran on.

    A results/gate directory that was not produced under this same code hash is moved
    aside to results/gate.stale-<timestamp> (never deleted), so the trainer cannot resume
    or skip stale checkpoints.
    """
    root = ROOT if root is None else Path(root)
    hash_fn = hash_fn or (lambda: code_hash(root))
    h0 = hash_fn()
    gdir = root / "results" / "gate"
    marker = gdir / "learn_started.json"
    if gdir.exists() and any(gdir.iterdir()):
        same = False
        try:
            same = json.loads(marker.read_text()).get("code_hash") == h0
        except (OSError, ValueError):
            pass
        if not same:
            stale = root / "results" / f"gate.stale-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
            os.replace(gdir, stale)
            log(f"moved old learning-check outputs to {stale}")
    gdir.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"code_hash": h0}))
    py = [sys.executable, "-m"]
    steps = [py + ["rt.train", "--cell", LEARNING_CELL, "--device", device,
                   "--max-steps", str(LEARNING_STEP), "--out", str(gdir)],
             py + ["rt.evaluate", "--cell", LEARNING_CELL, "--device", device,
                   "--ckpt-dir", str(gdir), "--out", str(gdir / "metrics")]]
    for cmd in steps:
        log("$ " + " ".join(cmd[2:]))
        rc = run(cmd, cwd=root).returncode
        if rc != 0:
            log(f"learning check failed: {cmd[2]} exited {rc}")
            return 1
    h1 = hash_fn()
    if h1 != h0:
        log(f"code or config changed during the learning run ({h0[:12]} -> {h1[:12]}); "
            f"nothing recorded, re-run --learn")
        return 1
    df = pd.read_csv(gdir / "metrics" / f"{LEARNING_CELL}.csv", dtype={"rho": str})
    sel = df[(df.step == LEARNING_STEP) & (df.rho == "prior") & (df.h == 1) & (df.bench == "bayes")]
    if sel.empty:
        log(f"no step-{LEARNING_STEP} prior h=1 bayes row in the metrics CSV")
        return 1
    ev = {"code_hash": h0, "cell": LEARNING_CELL, "step": LEARNING_STEP,
          "excess_mse_pct": float(sel.excess_mse_pct.iloc[0]), "device": device,
          "finished_iso": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    atomic_write_text(gdir / "learning.json", json.dumps(ev, indent=2))
    log(f"learning check: excess_mse_pct {ev['excess_mse_pct']:.3f} at step {LEARNING_STEP} "
        f"(< {LEARNING_MAX_EXCESS_PCT} passes); wrote {gdir / 'learning.json'}")
    return 0


def atomic_write_text(path: Path, text: str) -> None:
    _atomic(Path(path), lambda p: p.write_text(text))


def write_gate(result: dict, root=None) -> Path:
    root = ROOT if root is None else Path(root)
    path = root / "results" / "gate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(result, indent=2, default=float))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the gate, or the learning check it needs.")
    ap.add_argument("--learn", action="store_true",
                    help="run the local learning check and record it against the current code")
    ap.add_argument("--device", default="mps", help="device for --learn (cpu, mps or cuda)")
    args = ap.parse_args(argv)
    if args.learn:
        return learn(args.device)
    result = run_gate()
    path = write_gate(result)
    print(f"wrote {path}")
    print("GATE PASS" if result["passed"] else "GATE FAIL")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
