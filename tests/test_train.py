import numpy as np
import pandas as pd
import pytest
import torch

from rt.model import SIZES, Forecaster
from rt.train import Cell, batch_from_x, load_cfg, make_batch, step_rng, train_cell


# ---- tests from the brief -------------------------------------------------

def test_forward_shape_and_causality():
    m = Forecaster(*SIZES["xs"])
    v, t = torch.randn(4, 80), torch.arange(1, 81).repeat(4, 1)
    out = m(v, t); assert out.shape == (4, 80, 6)
    v2 = v.clone(); v2[:, 50:] += 1.0
    assert torch.allclose(m(v2, t)[:, :50], out[:, :50], atol=1e-6)


def test_window_batch_never_shows_more_than_w():
    cell = Cell("A", "xs", 4, 0)
    values, times, targets, mask = make_batch(cell, np.random.default_rng(0), 256)
    assert values.shape[1] == 4 and (mask.sum(1) <= 4).all()


def test_tiny_training_reduces_loss_and_resumes(tmp_path):
    cfg = load_cfg(max_steps=300, checkpoints=[100, 300], batch_size=256)
    p = train_cell(Cell("A", "xs", None, 0), cfg, tmp_path, "cpu")
    log = pd.read_csv(p / "train_log.csv")
    assert log.loss.iloc[-1] < 0.7 * log.loss.iloc[0]
    before = (p / "step300.pt").stat().st_mtime
    train_cell(Cell("A", "xs", None, 0), cfg, tmp_path, "cpu")     # already done
    assert (p / "step300.pt").stat().st_mtime == before


# ---- information rule ------------------------------------------------------

def test_padded_positions_do_not_influence_output():
    torch.manual_seed(0)
    m = Forecaster(*SIZES["s"])
    real = torch.arange(1, 6)                      # times 1..5
    times = torch.cat([torch.zeros(3, dtype=torch.long), real]).repeat(4, 1)
    v = torch.randn(4, 8)
    out = m(v, times)
    v2 = v.clone(); v2[:, :3] = torch.randn(4, 3) * 100.0   # garbage at pads
    out2 = m(v2, times)
    assert torch.allclose(out2[:, 3:], out[:, 3:], atol=1e-6)
    # a left-padded window gives the same forecasts as the unpadded prefix
    short = m(v[:, 3:], real.repeat(4, 1))
    assert torch.allclose(out[:, 3:], short, atol=1e-5)


def _coded_x(n):
    # x[i, j] = 1000 * i + j, so every value names its own sequence and time index
    return (1000.0 * np.arange(n)[:, None] + np.arange(87)[None, :]).astype(np.float64)


@pytest.mark.parametrize("W", [1, 2, 4, 16, 200])
def test_window_inputs_are_exactly_the_last_w_observations(W):
    n = 80
    x = _coded_x(n)
    t = np.arange(1, 81)                           # every t once
    values, times, targets, mask = batch_from_x(x, W, t)
    assert values.shape == (n, W) and times.shape == (n, W) and mask.shape == (n, W)
    assert values.dtype == torch.float32 and targets.dtype == torch.float32
    assert targets.shape == (n, 6)
    for i in range(n):
        ti = int(t[i])
        lo = max(1, ti - W + 1)
        k = ti - lo + 1                            # number of real inputs
        assert int(mask[i].sum()) == k and bool(mask[i, -k:].all())
        assert not bool(mask[i, :W - k].any())
        assert times[i, -k:].tolist() == list(range(lo, ti + 1))
        assert values[i, -k:].tolist() == [1000.0 * i + s for s in range(lo, ti + 1)]
        assert (times[i, :W - k] == 0).all() and (values[i, :W - k] == 0).all()
        assert targets[i].tolist() == [1000.0 * i + s for s in range(ti + 1, ti + 7)]
    # x_0 is never an input
    assert not bool((times[mask] == 0).any())


def test_full_window_batch_is_x1_to_x80_with_targets_at_every_position():
    x = _coded_x(3)
    values, times, targets, mask = batch_from_x(x, None)
    assert values.shape == (3, 80) and bool(mask.all())
    assert torch.equal(times, torch.arange(1, 81).repeat(3, 1))
    assert torch.equal(values, torch.tensor(x[:, 1:81], dtype=torch.float32))
    assert targets.shape == (3, 80, 6)
    for t in (1, 40, 80):
        assert targets[1, t - 1].tolist() == [1000.0 + s for s in range(t + 1, t + 7)]


@pytest.mark.parametrize("family", ["A", "B"])
def test_make_batch_window_consistency(family):
    cell = Cell(family, "xs", 8, 0)
    values, times, targets, mask = make_batch(cell, np.random.default_rng(1), 512)
    assert values.shape == (512, 8) and targets.shape == (512, 6)
    assert torch.equal(mask, times > 0)
    last = times[:, -1]
    assert int(last.min()) >= 1 and int(last.max()) <= 80
    assert torch.equal(mask.sum(1), torch.clamp(last, max=8))
    # real time indices are consecutive and end at t
    d = times[:, 1:] - times[:, :-1]
    assert bool((d[mask[:, :-1]] == 1).all())
    assert len(torch.unique(last)) > 60            # t drawn across 1..80


def test_training_stream_is_deterministic_per_step():
    cell = Cell("B", "xs", None, 2)
    a = make_batch(cell, step_rng(cell, 7), 16)[0]
    b = make_batch(cell, step_rng(cell, 7), 16)[0]
    c = make_batch(cell, step_rng(cell, 8), 16)[0]
    d = make_batch(Cell("B", "xs", None, 3), step_rng(Cell("B", "xs", None, 3), 7), 16)[0]
    assert torch.equal(a, b) and not torch.equal(a, c) and not torch.equal(a, d)


# ---- config, ids, resume ---------------------------------------------------

def test_cell_id_roundtrip():
    assert Cell("A", "l", 4, 3).id == "A-l-w4-s3"
    assert Cell("B", "s", None, 0).id == "B-s-wfull-s0"
    for cid in ("A-l-w4-s3", "B-s-wfull-s0", "A-xs-w16-s4"):
        assert Cell.parse(cid).id == cid
    with pytest.raises(ValueError):
        Cell.parse("C-s-w4-s0")


def test_config_holds_the_brief_values():
    cfg = load_cfg()
    assert {k: tuple(v) for k, v in cfg["sizes"].items()} == SIZES
    assert cfg["lr"] == 1e-3 and cfg["warmup_steps"] == 500
    assert cfg["batch_size"] == 1024 and cfg["max_steps"] == 30_000
    assert cfg["checkpoints"] == [100, 300, 1000, 3000, 10000, 30000]
    assert cfg["windows"] == [2, 4, 8, 16, None] and cfg["seeds"] == [0, 1, 2, 3, 4]
    assert cfg["eval_n_per_rho"] == 20000 and cfg["eval_n_prior"] == 100000
    assert cfg["convergence"]["rel_tol"] == 0.001
    assert load_cfg(max_steps=5)["max_steps"] == 5
    with pytest.raises(KeyError):
        load_cfg(max_stpes=5)


def test_resume_continues_the_same_run(tmp_path):
    cfg = load_cfg(max_steps=120, checkpoints=[50, 120], batch_size=64)
    cell = Cell("A", "xs", 4, 1)
    p1 = train_cell(cell, cfg, tmp_path / "a", "cpu")
    p2 = train_cell(cell, cfg, tmp_path / "b", "cpu")
    (p2 / "step120.pt").unlink()                    # simulate a kill after step 50
    train_cell(cell, cfg, tmp_path / "b", "cpu")
    s1 = torch.load(p1 / "step120.pt", weights_only=True)
    s2 = torch.load(p2 / "step120.pt", weights_only=True)
    assert s2["step"] == 120 and s2["meta"]["resumed_from"] == 50
    for k in s1["model"]:
        assert torch.allclose(s1["model"][k], s2["model"][k], atol=1e-6), k
    log1, log2 = pd.read_csv(p1 / "train_log.csv"), pd.read_csv(p2 / "train_log.csv")
    assert list(log1.step) == list(log2.step) == [50, 100, 120]
    assert np.allclose(log1.loss, log2.loss, rtol=1e-5)
    assert not list(p2.glob("*.tmp"))


def test_resume_refuses_a_changed_schedule(tmp_path):
    cell = Cell("A", "xs", 2, 0)
    train_cell(cell, load_cfg(max_steps=60, checkpoints=[50], batch_size=32), tmp_path, "cpu")
    (tmp_path / cell.id / "step60.pt").unlink()
    with pytest.raises(ValueError):
        train_cell(cell, load_cfg(max_steps=90, checkpoints=[50], batch_size=32), tmp_path, "cpu")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no mps")
def test_mps_matches_cpu():
    torch.manual_seed(0)
    m = Forecaster(*SIZES["xs"])
    times = torch.cat([torch.zeros(2, dtype=torch.long), torch.arange(1, 7)]).repeat(3, 1)
    v = torch.randn(3, 8)
    out = m(v, times)
    out_mps = m.to("mps")(v.to("mps"), times.to("mps")).cpu()
    assert torch.allclose(out[:, 2:], out_mps[:, 2:], atol=1e-4)
