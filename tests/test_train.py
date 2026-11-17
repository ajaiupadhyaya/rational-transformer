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


# ---- architecture options: head and inputs ---------------------------------

VARIANTS = [(h, i) for h in ("linear", "affine") for i in ("raw", "raw+asinh")]


@pytest.mark.parametrize("head,inputs", VARIANTS)
def test_forward_shape_for_each_variant(head, inputs):
    m = Forecaster(*SIZES["xs"], head=head, inputs=inputs)
    v, t = torch.randn(4, 80) * 20.0, torch.arange(1, 81).repeat(4, 1)
    out = m(v, t)
    assert out.shape == (4, 80, 6) and torch.isfinite(out).all()
    assert m.value_in.in_features == (1 if inputs == "raw" else 2)
    assert m.head.out_features == (6 if head == "linear" else 12)


def test_default_variant_is_affine_raw_asinh():
    torch.manual_seed(0)
    a = Forecaster(*SIZES["xs"])
    torch.manual_seed(0)
    b = Forecaster(*SIZES["xs"], head="affine", inputs="raw+asinh")
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert load_cfg()["head"] == "affine" and load_cfg()["inputs"] == "raw+asinh"
    from rt.train import LEGACY_MODEL, build_model
    assert LEGACY_MODEL == {"head": "linear", "inputs": "raw"}
    bare = {"sizes": SIZES}                       # a config without the keys predates them
    assert (build_model(bare, "xs").head_kind, build_model(bare, "xs").inputs_kind) == (
        "linear", "raw")
    with pytest.raises(ValueError):
        Forecaster(*SIZES["xs"], head="quadratic")
    with pytest.raises(ValueError):
        Forecaster(*SIZES["xs"], inputs="log")


@pytest.mark.parametrize("inputs", ["raw", "raw+asinh"])
def test_affine_head_starts_with_a_zero_and_is_affine_in_the_current_value(inputs):
    torch.manual_seed(0)
    m = Forecaster(*SIZES["xs"], head="affine", inputs=inputs)
    assert torch.equal(m.head.weight[:6], torch.zeros_like(m.head.weight[:6]))
    assert torch.equal(m.head.bias[:6], torch.zeros(6))
    with torch.no_grad():                           # give a_h a value, then check the formula
        m.head.weight[:6].normal_()
        m.head.bias[:6].fill_(0.5)
    v, t = torch.randn(2, 10) * 5.0, torch.arange(1, 11).repeat(2, 1)
    h = m.time_emb(t) + m.value_in(
        v.unsqueeze(-1) if inputs == "raw" else torch.stack([v, torch.asinh(v)], -1))
    for blk in m.blocks:
        h = blk(h, None)
    ab = m.head(m.ln_f(h))
    expect = ab[..., :6] * v.unsqueeze(-1) + ab[..., 6:]
    assert torch.allclose(m(v, t), expect, atol=1e-5)


@pytest.mark.parametrize("inputs", ["raw", "raw+asinh"])
def test_affine_head_keeps_the_information_rule(inputs):
    torch.manual_seed(1)
    m = Forecaster(*SIZES["s"], head="affine", inputs=inputs)
    with torch.no_grad():                           # a_h != 0, so x_t really enters
        m.head.weight[:6].normal_(0.0, 0.1)
    # causality: later values never move earlier forecasts
    v, t = torch.randn(4, 80) * 10.0, torch.arange(1, 81).repeat(4, 1)
    out = m(v, t)
    v2 = v.clone(); v2[:, 50:] += 7.0
    assert torch.allclose(m(v2, t)[:, :50], out[:, :50], atol=1e-5)
    assert not torch.allclose(m(v2, t)[:, 50:], out[:, 50:], atol=1e-3)
    # pads: garbage at padded positions changes nothing; padded = unpadded prefix
    real = torch.arange(1, 6)
    times = torch.cat([torch.zeros(3, dtype=torch.long), real]).repeat(4, 1)
    w = torch.randn(4, 8)
    o = m(w, times)
    w2 = w.clone(); w2[:, :3] = torch.randn(4, 3) * 100.0
    assert torch.allclose(m(w2, times)[:, 3:], o[:, 3:], atol=1e-5)
    assert torch.allclose(o[:, 3:], m(w[:, 3:], real.repeat(4, 1)), atol=1e-5)


def test_cli_head_and_inputs_overrides_reach_the_checkpoint(tmp_path):
    from rt.train import main
    main(["--cell", "A-xs-w2-s0", "--max-steps", "2", "--head", "affine",
          "--inputs", "raw+asinh", "--out", str(tmp_path)])
    ck = torch.load(tmp_path / "A-xs-w2-s0" / "step2.pt", weights_only=True)
    assert ck["config"]["head"] == "affine" and ck["config"]["inputs"] == "raw+asinh"
    assert ck["model"]["head.weight"].shape[0] == 12
    assert ck["model"]["value_in.weight"].shape[1] == 2


def test_resume_refuses_a_changed_architecture_and_accepts_old_configs(tmp_path):
    cell = Cell("A", "xs", 2, 0)
    cfg = load_cfg(max_steps=60, checkpoints=[50], batch_size=32, head="linear", inputs="raw")
    train_cell(cell, cfg, tmp_path, "cpu")
    (tmp_path / cell.id / "step60.pt").unlink()
    with pytest.raises(ValueError):
        train_cell(cell, {**cfg, "head": "affine"}, tmp_path, "cpu")
    # a checkpoint written before the keys existed was the original network (linear, raw):
    # it resumes under a linear/raw config and is refused under the current affine default
    p = tmp_path / cell.id / "step50.pt"
    ck = torch.load(p, weights_only=True)
    del ck["config"]["head"], ck["config"]["inputs"]
    torch.save(ck, p)
    with pytest.raises(ValueError):
        train_cell(cell, load_cfg(max_steps=60, checkpoints=[50], batch_size=32), tmp_path, "cpu")
    out = train_cell(cell, cfg, tmp_path, "cpu")
    assert torch.load(out / "step60.pt", weights_only=True)["meta"]["resumed_from"] == 50
