"""Train one cell (family, size, window, seed) of the grid.

Information rule: a forecast at time t with window W sees only x[max(1, t-W+1) .. t]
and those observations' true time indices; with the full window it sees x[1..t].
x[:, 0] (Family A: the constant 0; Family B: the hidden mean) is never an input.
Values are fed raw, without normalisation, because the benchmark sees raw values.

Architecture options come from the config keys "head" ("linear" | "affine") and "inputs"
("raw" | "raw+asinh"), see rt/model.py; the defaults are the original network. They are
saved with every checkpoint and evaluation rebuilds the model from them.

Usage: uv run python -m rt.train --cell A-s-wfull-s0 --device mps [--max-steps N]
         [--head linear|affine] [--inputs raw|raw+asinh] [--out DIR]
"""
import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch

from rt.model import HEADS, HORIZONS, INPUTS, SIZES, Forecaster
from rt.sim import sample_family_a, sample_family_b

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "grid.json"
T_MAX = 80                    # forecast origins t = 1..80; targets reach x[86]
SAMPLERS = {"A": sample_family_a, "B": sample_family_b}
TRAIN_SEED_TAG = 0xA11CE      # keeps training streams apart from eval seeds (10_000 + i)
# config keys that shape the training trajectory; a resume must not change them
TRAJECTORY_KEYS = ("sizes", "t_max", "horizons", "lr", "weight_decay", "betas",
                   "warmup_steps", "batch_size", "max_steps", "head", "inputs")
# architecture keys a config written before they existed implicitly had
MODEL_DEFAULTS = {"head": "linear", "inputs": "raw"}
LOG_COLUMNS = ["step", "loss", "lr", "seconds"]
_ID_RE = re.compile(r"^(A|B)-([a-z]+)-w(full|\d+)-s(\d+)$")


@dataclass(frozen=True)
class Cell:
    family: Literal["A", "B"]
    size: str
    window: int | None
    seed: int

    def __post_init__(self):
        if self.family not in SAMPLERS:
            raise ValueError(f"unknown family {self.family!r}")
        if self.size not in SIZES:
            raise ValueError(f"unknown size {self.size!r}")
        if self.window is not None and self.window < 1:
            raise ValueError(f"window must be >= 1 or None, got {self.window}")

    @property
    def id(self) -> str:
        w = "full" if self.window is None else str(self.window)
        return f"{self.family}-{self.size}-w{w}-s{self.seed}"

    @classmethod
    def parse(cls, cell_id: str) -> "Cell":
        m = _ID_RE.match(cell_id)
        if not m:
            raise ValueError(f"bad cell id {cell_id!r}; want <A|B>-<size>-w<W|full>-s<seed>")
        fam, size, w, seed = m.groups()
        return cls(fam, size, None if w == "full" else int(w), int(seed))


def load_cfg(**overrides) -> dict:
    cfg = json.loads(CONFIG_PATH.read_text())
    for k, v in overrides.items():
        if k not in cfg:
            raise KeyError(f"unknown config key {k!r}")
        cfg[k] = v
    return cfg


def build_model(cfg: dict, size: str) -> Forecaster:
    """The Forecaster a config describes (a config without "head"/"inputs" -> defaults)."""
    return Forecaster(*cfg["sizes"][size], head=cfg.get("head", MODEL_DEFAULTS["head"]),
                      inputs=cfg.get("inputs", MODEL_DEFAULTS["inputs"]))


# ---- data -------------------------------------------------------------------

def step_rng(cell: Cell, step: int) -> np.random.Generator:
    """The generator for training step `step` (0-based) of `cell`."""
    return np.random.default_rng(np.random.SeedSequence([cell.seed, TRAIN_SEED_TAG, step]))


def batch_from_x(x: np.ndarray, window: int | None, t: np.ndarray | None = None):
    """Model inputs from simulated paths x (n, 87).

    Full window: values x[:, 1..80], times 1..80, targets (n, 80, 6) with
    targets[:, t-1] = x[:, t+1..t+6], mask all True.
    Window W: for each row's origin t, inputs x[max(1, t-W+1)..t] with their time
    indices, left-padded with (value 0, time 0) to length W; targets (n, 6) =
    x[t+1..t+6]; mask marks the real inputs.
    """
    n = x.shape[0]
    assert x.shape[1] >= T_MAX + HORIZONS + 1
    if window is None:
        values = x[:, 1:T_MAX + 1]
        times = np.broadcast_to(np.arange(1, T_MAX + 1), (n, T_MAX))
        targets = np.lib.stride_tricks.sliding_window_view(
            x[:, 2:T_MAX + HORIZONS + 1], HORIZONS, axis=1)
        mask = np.ones((n, T_MAX), dtype=bool)
    else:
        t = np.asarray(t, dtype=np.int64)
        assert t.shape == (n,) and t.min() >= 1 and t.max() <= T_MAX
        rows = np.arange(n)[:, None]
        s = t[:, None] - window + 1 + np.arange(window)[None, :]       # (n, W)
        mask = s >= 1
        times = np.where(mask, s, 0)
        values = np.where(mask, x[rows, times], 0.0)
        targets = x[rows, t[:, None] + np.arange(1, HORIZONS + 1)[None, :]]
    return (torch.tensor(np.ascontiguousarray(values), dtype=torch.float32),
            torch.tensor(np.ascontiguousarray(times), dtype=torch.long),
            torch.tensor(np.ascontiguousarray(targets), dtype=torch.float32),
            torch.tensor(mask))


def make_batch(cell: Cell, rng: np.random.Generator, batch_size: int):
    x = SAMPLERS[cell.family](batch_size, rng).x
    t = None if cell.window is None else rng.integers(1, T_MAX + 1, size=batch_size)
    return batch_from_x(x, cell.window, t)


def loss_fn(pred: torch.Tensor, targets: torch.Tensor, window: int | None) -> torch.Tensor:
    """Squared error summed over horizons; every position (full) or the last (window)."""
    if window is not None:
        pred = pred[:, -1]
    return ((pred - targets) ** 2).sum(-1).mean()


# ---- training ---------------------------------------------------------------

def _lr_factor(k: int, warmup: int, max_steps: int) -> float:
    """Multiplier on the base lr for update k (0-based)."""
    if k < warmup:
        return (k + 1) / warmup
    progress = (k - warmup) / max(1, max_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _checkpoints(out: Path) -> dict[int, Path]:
    found = {}
    for p in out.glob("step*.pt"):
        m = re.fullmatch(r"step(\d+)\.pt", p.name)
        if m:
            found[int(m.group(1))] = p
    return found


def _atomic(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def train_cell(cell: Cell, cfg: dict, out_dir, device: str) -> Path:
    out = Path(out_dir) / cell.id
    out.mkdir(parents=True, exist_ok=True)
    max_steps = int(cfg["max_steps"])
    ckpt_steps = sorted({int(k) for k in cfg["checkpoints"] if 0 < k <= max_steps} | {max_steps})
    log_every = int(cfg["log_every"])
    existing = _checkpoints(out)
    if existing and max(existing) >= max_steps:
        return out                                                    # already done

    dev = torch.device(device)
    torch.manual_seed(cell.seed)
    model = build_model(cfg, cell.size).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]),
                            weight_decay=cfg["weight_decay"])
    warmup = int(cfg["warmup_steps"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: _lr_factor(k, warmup, max_steps))

    step, seconds, rows, resumed_from = 0, 0.0, [], None
    if existing:
        latest = max(existing)
        ck = torch.load(existing[latest], map_location=dev, weights_only=True)
        changed = [k for k in TRAJECTORY_KEYS
                   if ck["config"].get(k, MODEL_DEFAULTS.get(k))
                   != cfg.get(k, MODEL_DEFAULTS.get(k))]
        if changed:
            raise ValueError(f"{out}: checkpoint step{latest}.pt was trained with different "
                             f"{changed}; use a fresh output directory")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        step, seconds, resumed_from = ck["step"], ck["meta"]["seconds"], ck["step"]
        log_path = out / "train_log.csv"
        if log_path.exists():
            old = pd.read_csv(log_path)
            rows = old[old.step <= step][LOG_COLUMNS].to_dict("records")

    def meta() -> dict:
        return {"cell": cell.id, "device": str(dev), "torch_version": str(torch.__version__),
                "seconds": seconds, "resumed_from": resumed_from}

    def write_log() -> None:
        _atomic(out / "train_log.csv",
                lambda p: pd.DataFrame(rows, columns=LOG_COLUMNS).to_csv(p, index=False))

    model.train()
    acc, n_acc = torch.zeros((), device=dev), 0
    t0 = time.perf_counter()
    while step < max_steps:
        values, times, targets, _ = make_batch(cell, step_rng(cell, step), cfg["batch_size"])
        values, times, targets = values.to(dev), times.to(dev), targets.to(dev)
        lr = sched.get_last_lr()[0]
        loss = loss_fn(model(values, times), targets, cell.window)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        acc, n_acc = acc + loss.detach(), n_acc + 1
        step += 1
        is_ckpt = step in ckpt_steps
        if step % log_every == 0 or is_ckpt:
            mean_loss = (acc / n_acc).item()
            if not math.isfinite(mean_loss):
                raise RuntimeError(f"{cell.id}: non-finite training loss at step {step}")
            now = time.perf_counter()
            seconds, t0 = seconds + (now - t0), now
            rows.append({"step": step, "loss": mean_loss, "lr": lr, "seconds": seconds})
            acc, n_acc = torch.zeros((), device=dev), 0
            write_log()
        if is_ckpt:
            state = {"model": model.state_dict(), "optimizer": opt.state_dict(),
                     "scheduler": sched.state_dict(), "step": step, "config": cfg,
                     "meta": meta()}
            _atomic(out / f"step{step}.pt", lambda p: torch.save(state, p))
            _atomic(out / "run.json",
                    lambda p: p.write_text(json.dumps({**meta(), "step": step,
                                                       "max_steps": max_steps}, indent=2)))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train one cell of the grid.")
    ap.add_argument("--cell", required=True, help="e.g. A-s-wfull-s0 or B-l-w4-s3")
    ap.add_argument("--device", default="cpu", help="cpu, mps or cuda")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default="results/checkpoints")
    ap.add_argument("--head", choices=HEADS, default=None, help="override config 'head'")
    ap.add_argument("--inputs", choices=INPUTS, default=None, help="override config 'inputs'")
    args = ap.parse_args(argv)
    cell = Cell.parse(args.cell)
    over = {k: v for k, v in (("max_steps", args.max_steps), ("head", args.head),
                              ("inputs", args.inputs)) if v is not None}
    cfg = load_cfg(**over)
    out = train_cell(cell, cfg, args.out, args.device)
    log = pd.read_csv(out / "train_log.csv")
    last = log.iloc[-1]
    print(f"{cell.id} [{cfg['head']}, {cfg['inputs']}]: step {int(last.step)}  "
          f"loss {last.loss:.4f}  "
          f"{last.seconds:.0f}s on {args.device}  -> {out}")


if __name__ == "__main__":
    main()
