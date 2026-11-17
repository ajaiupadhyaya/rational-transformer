"""Gate-bound, resumable, budget-guarded sweep over the 125 training-plus-evaluation cells.

Each cell runs as a subprocess that trains (`rt.train`, into results/checkpoints) and then
evaluates (`rt.evaluate`, into results/metrics). A cell is complete when
results/metrics/<id>.csv exists and holds every checkpoint step of the config; anything else
(missing, fewer steps, unreadable, only a .tmp file) means the cell is run, and training
resumes from its latest checkpoint by itself.

Usage:
  uv run python -m rt.sweep --device cuda --budget-usd 12 --price-per-hour P --jobs J [--only 'A-l-*']
  uv run python -m rt.sweep --dry-run
  uv run python -m rt.sweep --time-only --cell A-l-wfull-s0 --steps 500 [--device cuda]

Exit codes: 0 all planned cells finished; 1 some cell failed (listed); 3 the budget guard
stopped the sweep before every cell ran (failures, if any, are listed too). A red or stale
gate refuses with a SystemExit message.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from rt.gate import ROOT, code_hash
from rt.train import Cell, load_cfg, train_cell

FAMILY_B_SIZE = "l"
LEDGER_COLUMNS = ["cell_id", "start_iso", "end_iso", "seconds", "device", "returncode"]
EXIT_FAILED, EXIT_BUDGET = 1, 3


def grid(cfg: dict) -> list[Cell]:
    """Family A: every size x window x seed; Family B: size l x window x seed.

    Ordered smallest size first (the config's size order), Family A before Family B within
    a size, then window (config order) and seed.
    """
    sizes = list(cfg["sizes"])
    cells = [Cell("A", s, w, seed) for s in sizes for w in cfg["windows"] for seed in cfg["seeds"]]
    cells += [Cell("B", FAMILY_B_SIZE, w, seed) for w in cfg["windows"] for seed in cfg["seeds"]]
    return sorted(cells, key=lambda c: sizes.index(c.size))      # stable: keeps A before B


def checkpoint_steps(cfg: dict) -> list[int]:
    max_steps = int(cfg["max_steps"])
    return sorted({int(k) for k in cfg["checkpoints"] if 0 < k <= max_steps} | {max_steps})


def is_complete(cell: Cell, cfg: dict, metrics_dir) -> bool:
    path = Path(metrics_dir) / f"{cell.id}.csv"
    try:
        steps = set(pd.read_csv(path, usecols=["step"])["step"].astype(int))
    except Exception:                       # missing, unreadable, or malformed all mean: run it
        return False
    return set(checkpoint_steps(cfg)) <= steps


def plan(cells: list[Cell], cfg: dict, metrics_dir, only: str | None = None) -> list[Cell]:
    """The cells still to run, in grid order, after the --only filter."""
    if only:
        cells = [c for c in cells if fnmatch.fnmatchcase(c.id, only)]
    return [c for c in cells if not is_complete(c, cfg, metrics_dir)]


def require_green_gate(root: Path) -> None:
    path = root / "results" / "gate.json"
    try:
        gate = json.loads(path.read_text())
    except (OSError, ValueError):
        raise SystemExit(f"refusing to sweep: no readable {path}; run `uv run python -m rt.gate`")
    if not gate.get("passed"):
        raise SystemExit(f"refusing to sweep: {path} records a failed gate")
    if gate.get("code_hash") != code_hash(root):
        raise SystemExit(f"refusing to sweep: {path} was written for different code or config; "
                         f"re-run `uv run python -m rt.gate`")


def subprocess_runner(device: str, root: Path) -> Callable[[Cell], int]:
    """Run one cell as `rt.train` then `rt.evaluate` subprocesses; returns the return code."""
    ckpt, metrics = root / "results" / "checkpoints", root / "results" / "metrics"
    logs = root / "results" / "logs"

    def run(cell: Cell) -> int:
        logs.mkdir(parents=True, exist_ok=True)
        steps = (["-m", "rt.train", "--cell", cell.id, "--device", device, "--out", str(ckpt)],
                 ["-m", "rt.evaluate", "--cell", cell.id, "--device", device,
                  "--ckpt-dir", str(ckpt), "--out", str(metrics)])
        with open(logs / f"{cell.id}.log", "a") as log:
            for args in steps:
                rc = subprocess.run([sys.executable, *args], cwd=root, stdout=log,
                                    stderr=subprocess.STDOUT).returncode
                if rc != 0:
                    return rc
        return 0

    return run


@dataclass
class SweepResult:
    finished: list[str] = field(default_factory=list)
    failed: dict[str, int] = field(default_factory=dict)
    not_run: list[str] = field(default_factory=list)
    budget_stop: bool = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_sweep(cells: list[Cell], run_cell: Callable[[Cell], int], *, device: str, ledger_path,
              budget_usd: float, price_per_hour: float, already_spent_usd: float = 0.0,
              jobs: int = 1, clock: Callable[[], float] = time.monotonic) -> SweepResult:
    """Run `cells` in order, up to `jobs` at once, stopping launches at the budget.

    Before each launch: if already_spent + elapsed_hours * price >= budget, launch nothing
    more (running cells finish) and report the rest in `not_run`. Every finished cell, failed
    or not, appends one row to the ledger; a failure does not stop the sweep.
    """
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    t0, result = clock(), SweepResult()

    def work(cell: Cell) -> None:
        start, s0 = _now(), time.monotonic()
        try:
            rc = int(run_cell(cell))
        except Exception as exc:                       # a crashing cell must not stop the sweep
            print(f"{cell.id}: runner raised {exc!r}", file=sys.stderr)
            rc = 1
        row = [cell.id, start, _now(), f"{time.monotonic() - s0:.1f}", device, rc]
        with lock:
            new = not ledger_path.exists() or ledger_path.stat().st_size == 0
            with open(ledger_path, "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(LEDGER_COLUMNS)
                w.writerow(row)
            if rc == 0:
                result.finished.append(cell.id)
            else:
                result.failed[cell.id] = rc

    running: set = set()
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for i, cell in enumerate(cells):
            while len(running) >= max(1, jobs):
                done, running = wait(running, return_when=FIRST_COMPLETED)
            spent = already_spent_usd + (clock() - t0) / 3600.0 * price_per_hour
            if spent >= budget_usd:
                result.budget_stop = True
                result.not_run = [c.id for c in cells[i:]]
                break
            running.add(pool.submit(work, cell))
    return result


def time_only(cell_id: str, steps: int, device: str) -> str:
    """Train `cell_id` for `steps` steps in a throwaway directory; return '<id> <steps/s>'."""
    cell = Cell.parse(cell_id)
    cfg = load_cfg(max_steps=steps, checkpoints=[steps])
    with tempfile.TemporaryDirectory() as tmp:
        out = train_cell(cell, cfg, tmp, device)
        seconds = float(pd.read_csv(out / "train_log.csv").seconds.iloc[-1])
    return f"{cell.id} {steps / seconds:.2f}"


def main(argv=None, *, root=None, cfg=None, run_cell=None, clock=time.monotonic) -> int:
    ap = argparse.ArgumentParser(description="Gate-bound resumable sweep.")
    ap.add_argument("--device", default="cpu", help="cpu, mps or cuda")
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--price-per-hour", type=float, default=None)
    ap.add_argument("--already-spent-usd", type=float, default=0.0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", default=None, help="glob matched against cell ids, e.g. 'A-l-*'")
    ap.add_argument("--dry-run", action="store_true", help="print the planned cells and exit")
    ap.add_argument("--time-only", action="store_true",
                    help="print steps/second for --cell over --steps, in a temp dir; no gate")
    ap.add_argument("--cell", default=None)
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args(argv)

    if args.time_only:
        if not args.cell or not args.steps:
            ap.error("--time-only needs --cell and --steps")
        print(time_only(args.cell, args.steps, args.device))
        return 0

    root = ROOT if root is None else Path(root)
    cfg = load_cfg() if cfg is None else cfg
    require_green_gate(root)
    todo = plan(grid(cfg), cfg, root / "results" / "metrics", args.only)

    if args.dry_run:
        print(f"{len(todo)} cells to run:")
        for c in todo:
            print(c.id)
        return 0
    if args.budget_usd is None or args.price_per_hour is None:
        ap.error("a real sweep needs --budget-usd and --price-per-hour")

    run_cell = run_cell or subprocess_runner(args.device, root)
    res = run_sweep(todo, run_cell, device=args.device, ledger_path=root / "results" / "ledger.csv",
                    budget_usd=args.budget_usd, price_per_hour=args.price_per_hour,
                    already_spent_usd=args.already_spent_usd, jobs=args.jobs, clock=clock)
    print(f"finished {len(res.finished)} cells, failed {len(res.failed)}")
    if res.failed:
        print("failed cells (return code): "
              + ", ".join(f"{c} ({rc})" for c, rc in res.failed.items()))
    if res.budget_stop:
        print(f"budget guard: ${args.budget_usd:g} reached; not run ({len(res.not_run)}): "
              + ", ".join(res.not_run))
        return EXIT_BUDGET
    return EXIT_FAILED if res.failed else 0


if __name__ == "__main__":
    sys.exit(main())
