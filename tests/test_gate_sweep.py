import json
import re

import pandas as pd
import pytest

from rt.gate import _check_learning, code_hash, write_gate
from rt.sweep import grid, is_complete, main, plan
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


def test_learning_check_missing_file_names_the_commands(tmp_path):
    r = _check_learning(tmp_path, lambda *_: None)
    assert not r["passed"]
    assert "rt.train --cell A-s-wfull-s0" in r["message"] and "rt.evaluate" in r["message"]


@pytest.mark.parametrize("excess,passed", [(2.29, True), (5.0, False), (7.5, False)])
def test_learning_check_reads_the_prior_row(tmp_path, excess, passed):
    d = tmp_path / "results" / "gate" / "metrics"
    d.mkdir(parents=True)
    rows = [
        {"step": 3000, "rho": "prior", "h": 1, "bench": "bayes", "excess_mse_pct": excess},
        {"step": 3000, "rho": "0.0", "h": 1, "bench": "bayes", "excess_mse_pct": 99.0},
        {"step": 3000, "rho": "prior", "h": 1, "bench": "window_bayes", "excess_mse_pct": 99.0},
        {"step": 1000, "rho": "prior", "h": 1, "bench": "bayes", "excess_mse_pct": 99.0},
    ]
    pd.DataFrame(rows).to_csv(d / "A-s-wfull-s0.csv", index=False)
    assert _check_learning(tmp_path, lambda *_: None)["passed"] is passed


def test_write_gate_is_atomic_json(tmp_path):
    path = write_gate({"passed": True, "code_hash": "h", "checks": {}}, root=tmp_path)
    assert json.loads(path.read_text())["passed"] is True
    assert not list(path.parent.glob("*.tmp"))
