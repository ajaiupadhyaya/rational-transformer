import json
import re
import threading
import types
from pathlib import Path

import pandas as pd
import pytest

from rt import gate
from rt.gate import _check_learning, code_hash, write_gate
from rt.sweep import grid, is_complete, main, plan, subprocess_runner
from rt.train import Cell, load_cfg


def make_root(tmp_path, gate="green"):
    """A tiny repo-shaped tree; `gate` is 'green', 'stale', 'red' or None (no gate.json)."""
    (tmp_path / "rt").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "rt" / "a.py").write_text("x = 1\n")
    (tmp_path / "configs" / "grid.json").write_text('{"k": 1}\n')
    if gate is not None:
        (tmp_path / "results").mkdir()
        rec = {"passed": gate != "red",
               "code_hash": "0" * 64 if gate == "stale" else code_hash(tmp_path), "checks": {}}
        (tmp_path / "results" / "gate.json").write_text(json.dumps(rec))
    return tmp_path


def small_cfg():
    cfg = load_cfg()
    cfg["sizes"] = {"xs": cfg["sizes"]["xs"], "l": cfg["sizes"]["l"]}
    cfg["windows"], cfg["seeds"] = [2, None], [0]
    return cfg


def write_metrics(root, cell, steps):
    d = root / "results" / "metrics"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cell_id": cell.id, "step": steps}).to_csv(d / f"{cell.id}.csv", index=False)


def test_sweep_refuses_without_green_gate(tmp_path):
    for state in ("missing", "stale", "red"):
        (tmp_path / state).mkdir()
        root = make_root(tmp_path / state, gate=None if state == "missing" else state)
        with pytest.raises(SystemExit):
            main(["--dry-run"], root=root, cfg=small_cfg())
    (tmp_path / "green").mkdir()
    green = make_root(tmp_path / "green", gate="green")
    assert main(["--dry-run"], root=green, cfg=small_cfg()) == 0


def test_code_hash_changes_when_config_changes(tmp_path):
    root = make_root(tmp_path, gate=None)
    h0 = code_hash(root)
    assert h0 == code_hash(root)
    (root / "configs" / "grid.json").write_text('{"k": 2}\n')
    h1 = code_hash(root)
    assert h1 != h0
    (root / "rt" / "a.py").write_text("x = 2\n")
    assert code_hash(root) not in (h0, h1)
    (root / "rt" / "notes.txt").write_text("not python")     # only rt/*.py and configs/*.json count
    assert code_hash(root) == code_hash(root)


def test_code_hash_default_is_independent_of_cwd(tmp_path, monkeypatch):
    h = code_hash()
    monkeypatch.chdir(tmp_path)
    assert code_hash() == h


def test_resume_skips_complete_and_redoes_partial(tmp_path):
    cfg, root = load_cfg(), tmp_path
    cells = grid(cfg)
    all_steps = cfg["checkpoints"]
    write_metrics(root, cells[0], all_steps)                  # complete
    write_metrics(root, cells[1], all_steps[:3])              # 3 of 6 checkpoints
    (root / "results" / "metrics" / f"{cells[1].id}.csv.tmp").write_text("stray")
    (root / "results" / "metrics" / f"{cells[2].id}.csv.tmp").write_text("only a tmp file")
    assert is_complete(cells[0], cfg, root / "results" / "metrics")
    assert not is_complete(cells[1], cfg, root / "results" / "metrics")
    assert not is_complete(cells[2], cfg, root / "results" / "metrics")
    todo = plan(cells, cfg, root / "results" / "metrics")
    assert todo == cells[1:]
    (root / "results" / "metrics" / f"{cells[3].id}.csv").write_text("garbage,\x00\n")
    assert cells[3] in plan(cells, cfg, root / "results" / "metrics")      # unreadable: rerun


def test_only_glob_matches_cell_ids(tmp_path):
    cfg = load_cfg()
    todo = plan(grid(cfg), cfg, tmp_path, only="A-l-*")
    assert len(todo) == 25 and all(c.id.startswith("A-l-") for c in todo)


def test_budget_guard_stops_launching(tmp_path, capsys):
    root = make_root(tmp_path)
    cfg, state, ran = small_cfg(), {"t": 0.0}, []

    def fake_run(cell):
        ran.append(cell.id)
        state["t"] += 2 * 3600.0                               # each cell "takes" 2 h
        return 0

    code = main(["--budget-usd", "12", "--price-per-hour", "6", "--jobs", "1"],
                root=root, cfg=cfg, run_cell=fake_run, clock=lambda: state["t"])
    out = capsys.readouterr().out
    planned = [c.id for c in grid(cfg)]
    assert code == 3
    assert ran == planned[:1]                                  # 2 h x $6 = $12 >= budget
    for cid in planned[1:]:
        assert cid in out                                      # the message names what was not run
    assert len(pd.read_csv(root / "results" / "ledger.csv")) == 1


def test_already_spent_counts_against_budget(tmp_path):
    root, ran = make_root(tmp_path), []
    code = main(["--budget-usd", "12", "--price-per-hour", "6", "--already-spent-usd", "12"],
                root=root, cfg=small_cfg(), run_cell=lambda c: ran.append(c.id) or 0,
                clock=lambda: 0.0)
    assert code == 3 and ran == []


def test_failed_cell_is_recorded_and_does_not_stop_sweep(tmp_path, capsys):
    root, cfg = make_root(tmp_path), small_cfg()
    planned = grid(cfg)
    bad = planned[1].id
    ran = []

    def fake_run(cell):
        ran.append(cell.id)
        if cell.id == bad:
            return 7
        write_metrics(root, cell, cfg["checkpoints"])
        return 0

    code = main(["--budget-usd", "100", "--price-per-hour", "1", "--device", "cuda", "--jobs", "2"],
                root=root, cfg=cfg, run_cell=fake_run)
    out = capsys.readouterr().out
    assert code == 1
    assert sorted(ran) == sorted(c.id for c in planned)        # everything was attempted
    assert f"{bad} (7)" in out
    ledger = pd.read_csv(root / "results" / "ledger.csv")
    assert list(ledger.columns) == ["cell_id", "start_iso", "end_iso", "seconds", "device",
                                    "returncode"]
    assert len(ledger) == len(planned) and set(ledger.device) == {"cuda"}
    assert int(ledger[ledger.cell_id == bad].returncode.iloc[0]) == 7
    # the next run redoes only the failed cell
    assert [c.id for c in plan(planned, cfg, root / "results" / "metrics")] == [bad]


def test_real_sweep_needs_budget_flags(tmp_path):
    with pytest.raises(SystemExit):
        main([], root=make_root(tmp_path), cfg=small_cfg(), run_cell=lambda c: 0)


def test_time_only_needs_no_gate_and_prints_one_line(tmp_path, capsys):
    root = make_root(tmp_path, gate=None)
    code = main(["--time-only", "--cell", "A-xs-wfull-s0", "--steps", "3", "--device", "cpu"],
                root=root)
    out = capsys.readouterr().out
    assert code == 0
    assert re.fullmatch(r"A-xs-wfull-s0 \d+\.\d+\n", out)
    assert not (root / "results").exists()


def test_grid_counts():
    cells = grid(load_cfg())
    assert len(cells) == 125
    assert sum(c.family == "B" for c in cells) == 25
    assert {c.size for c in cells if c.family == "B"} == {"l"}
    assert len({c.id for c in cells}) == 125
    sizes = list(load_cfg()["sizes"])
    ranks = [sizes.index(c.size) for c in cells]
    assert ranks == sorted(ranks)                              # smallest size first
    assert cells[0] == Cell.parse("A-xs-w2-s0")


def _learning_json(root, **kw):
    d = root / "results" / "gate"
    d.mkdir(parents=True, exist_ok=True)
    rec = {"code_hash": "h1", "cell": "A-s-wfull-s0", "step": 3000, "excess_mse_pct": 2.29,
           "device": "mps", "finished_iso": "2026-11-17T00:00:00+00:00", **kw}
    (d / "learning.json").write_text(json.dumps(rec))


def test_learning_check_missing_file_names_the_command(tmp_path):
    r = _check_learning(tmp_path, "h1", lambda *_: None)
    assert not r["passed"] and "rt.gate --learn" in r["message"]


def test_learning_check_stale_hash_fails(tmp_path):
    _learning_json(tmp_path, code_hash="old")
    r = _check_learning(tmp_path, "h1", lambda *_: None)
    assert not r["passed"] and "--learn" in r["message"]


@pytest.mark.parametrize("excess,passed", [(2.29, True), (4.99, True), (5.0, False), (7.5, False)])
def test_learning_check_value_threshold(tmp_path, excess, passed):
    _learning_json(tmp_path, excess_mse_pct=excess)
    assert _check_learning(tmp_path, "h1", lambda *_: None)["passed"] is passed


def _stub_heavy(monkeypatch):
    monkeypatch.setattr(gate, "sample_family_a", lambda *a, **k: types.SimpleNamespace(x=None))
    monkeypatch.setattr(gate, "sample_family_b", lambda *a, **k: types.SimpleNamespace(x=None))
    monkeypatch.setattr(gate, "bayes_a", lambda *a, **k: None)
    monkeypatch.setattr(gate, "_check_benchmark", lambda *a, **k: {"passed": True})
    monkeypatch.setattr(gate, "_check_planted", lambda *a, **k: {"passed": True})


def test_run_gate_records_start_hash_and_passes_when_stable(tmp_path, monkeypatch):
    _stub_heavy(monkeypatch)
    _learning_json(tmp_path)
    r = gate.run_gate(small_cfg(), root=tmp_path, log=lambda *_: None, hash_fn=lambda: "h1")
    assert r["passed"] and r["code_hash"] == "h1"


def test_run_gate_fails_if_code_changes_during_the_run(tmp_path, monkeypatch):
    _stub_heavy(monkeypatch)
    _learning_json(tmp_path)
    hashes = iter(["h1", "h2"])
    msgs = []
    r = gate.run_gate(small_cfg(), root=tmp_path, log=msgs.append, hash_fn=lambda: next(hashes))
    assert not r["passed"] and r["code_hash"] == "h1"
    assert not r["checks"]["stable"]["passed"]
    assert any("changed while the gate ran" in m for m in msgs)


def _fake_learn_run(root, calls, excess=2.29, on_run=None):
    def run(cmd, cwd=None):
        calls.append(cmd[2:])
        gdir = root / "results" / "gate"
        if cmd[2] == "rt.train":
            (gdir / "A-s-wfull-s0").mkdir(parents=True, exist_ok=True)
            (gdir / "A-s-wfull-s0" / "step3000.pt").write_text("ck")
        else:
            (gdir / "metrics").mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"step": 3000, "rho": "prior", "h": 1, "bench": "bayes",
                           "excess_mse_pct": excess}]).to_csv(
                gdir / "metrics" / "A-s-wfull-s0.csv", index=False)
        if on_run:
            on_run()
        return types.SimpleNamespace(returncode=0)
    return run


def test_learn_moves_stale_dir_aside_and_records_evidence(tmp_path):
    old = tmp_path / "results" / "gate"
    old.mkdir(parents=True)
    (old / "keep-me.txt").write_text("old evidence")
    calls = []
    rc = gate.learn("cpu", root=tmp_path, log=lambda *_: None, hash_fn=lambda: "h1",
                    run=_fake_learn_run(tmp_path, calls))
    assert rc == 0
    stale = list((tmp_path / "results").glob("gate.stale-*"))
    assert len(stale) == 1 and (stale[0] / "keep-me.txt").read_text() == "old evidence"
    ev = json.loads((tmp_path / "results" / "gate" / "learning.json").read_text())
    assert ev["code_hash"] == "h1" and ev["excess_mse_pct"] == 2.29 and ev["device"] == "cpu"
    assert ev["cell"] == "A-s-wfull-s0" and ev["step"] == 3000 and ev["finished_iso"]
    assert calls[0][:5] == ["rt.train", "--cell", "A-s-wfull-s0", "--device", "cpu"]
    assert "--max-steps" in calls[0] and "3000" in calls[0]
    assert "rt.evaluate" in calls[1] and "--ckpt-dir" in calls[1] and "--out" in calls[1]
    assert _check_learning(tmp_path, "h1", lambda *_: None)["passed"]
    assert not _check_learning(tmp_path, "h2", lambda *_: None)["passed"]


def test_learn_resumes_a_same_code_run_without_moving_it(tmp_path):
    gdir = tmp_path / "results" / "gate"
    gdir.mkdir(parents=True)
    (gdir / "learn_started.json").write_text(json.dumps({"code_hash": "h1"}))
    rc = gate.learn("cpu", root=tmp_path, log=lambda *_: None, hash_fn=lambda: "h1",
                    run=_fake_learn_run(tmp_path, []))
    assert rc == 0 and not list((tmp_path / "results").glob("gate.stale-*"))


def test_learn_fails_and_records_nothing_if_code_changes_during_run(tmp_path):
    hashes = iter(["h1", "h2"])
    rc = gate.learn("cpu", root=tmp_path, log=lambda *_: None, hash_fn=lambda: next(hashes),
                    run=_fake_learn_run(tmp_path, []))
    assert rc == 1 and not (tmp_path / "results" / "gate" / "learning.json").exists()


def test_learn_fails_when_a_subprocess_fails(tmp_path):
    rc = gate.learn("cpu", root=tmp_path, log=lambda *_: None, hash_fn=lambda: "h1",
                    run=lambda cmd, cwd=None: types.SimpleNamespace(returncode=5))
    assert rc == 1 and not (tmp_path / "results" / "gate" / "learning.json").exists()


def test_concurrent_cache_writers_publish_an_intact_file(tmp_path):
    import numpy as np
    from rt.evaluate import _save_atomic_npy
    arr = np.arange(2_000_000, dtype=np.float64).reshape(1000, 2000)
    target = tmp_path / "bench.npy"
    barrier = threading.Barrier(4)

    def writer():
        barrier.wait()
        for _ in range(3):
            _save_atomic_npy(target, arr)

    threads = [threading.Thread(target=writer) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert np.array_equal(np.load(target), arr)
    assert [p.name for p in tmp_path.iterdir()] == ["bench.npy"]       # no stray temp files


def test_stray_temp_files_do_not_make_a_cell_complete(tmp_path):
    cfg, cell = load_cfg(), grid(load_cfg())[0]
    d = tmp_path / "metrics"
    d.mkdir()
    pd.DataFrame({"step": cfg["checkpoints"]}).to_csv(d / f"{cell.id}.csv.k3j2h.tmp", index=False)
    assert not is_complete(cell, cfg, d)


def test_subprocess_runner_end_to_end_on_a_tiny_cell(tmp_path, monkeypatch):
    repo = Path(gate.ROOT)
    cfg = json.loads((repo / "configs" / "grid.json").read_text())
    cfg.update(max_steps=20, checkpoints=[10, 20], log_every=10, warmup_steps=5, batch_size=16,
               eval_n_per_rho=40, eval_n_prior=40)
    (tmp_path / "grid.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("RT_CONFIG", str(tmp_path / "grid.json"))
    monkeypatch.setenv("RT_EVAL_DIR", str(tmp_path / "eval"))
    monkeypatch.setenv("PYTHONPATH", str(repo))
    cell = Cell.parse("A-xs-wfull-s0")
    rc = subprocess_runner("cpu", tmp_path)(cell)
    assert rc == 0
    assert is_complete(cell, cfg, tmp_path / "results" / "metrics")
    assert (tmp_path / "results" / "checkpoints" / cell.id / "step20.pt").exists()


def test_write_gate_is_atomic_json(tmp_path):
    path = write_gate({"passed": True, "code_hash": "h", "checks": {}}, root=tmp_path)
    assert json.loads(path.read_text())["passed"] is True
    assert not list(path.parent.glob("*.tmp"))
