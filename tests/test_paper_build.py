"""Tests for analysis/paper_build.py and the paper sources in docs/paper/.

Read-only: they check the generated files that are in the repository against the results
files, and rebuild only the variables (in memory).
"""
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from analysis import paper_build as pb

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "docs" / "paper"
VAR = re.compile(r"\{\{<\s*var\s+([^\s>]+)\s*>\}\}")
MINUS = str.maketrans({"−": "-"})


def num(text: str) -> float:
    return float(text.translate(MINUS).replace(",", ""))


@pytest.fixture(scope="module")
def variables() -> dict:
    return json.loads((PAPER / "_variables.yml").read_text())


@pytest.fixture(scope="module")
def inp() -> pb.Inputs:
    return pb.Inputs(ROOT)


@pytest.fixture(scope="module")
def hyp() -> dict:
    return json.loads((ROOT / "results" / "hypotheses.json").read_text())


def lookup(tree: dict, key: str):
    node = tree
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, str) else None


def sources() -> list[Path]:
    return sorted(PAPER.glob("*.qmd")) + sorted(PAPER.glob("sections/*.qmd")) + sorted(PAPER.glob("appendix/*.qmd"))


def table_rows(tid: str) -> list[list[str]]:
    lines = [ln for ln in (PAPER / "_generated" / "tables" / f"{tid}.md").read_text().splitlines()
             if ln.startswith("|")]
    return [[c.strip() for c in ln.strip("|").split("|")] for ln in lines[2:]]


# ---- every {{< var >}} resolves ---------------------------------------------------------

def test_every_var_resolves(variables):
    missing = [(p.name, k) for p in sources() for k in VAR.findall(p.read_text())
               if lookup(variables, k) is None]
    assert missing == []
    assert sum(len(VAR.findall(p.read_text())) for p in sources()) > 400


def test_variables_file_is_current(variables, inp):
    assert pb.build_variables(inp).tree == variables


def test_variables_map_covers_every_key(variables):
    keys = set(pd.read_csv(PAPER / "variables_map.csv")["key"])

    def leaves(node, prefix=""):
        for k, v in node.items():
            yield from leaves(v, prefix + k + ".") if isinstance(v, dict) else [prefix + k]
    assert set(leaves(variables)) == keys


def test_every_include_exists():
    inc = re.compile(r"\{\{<\s*include\s+([^\s>]+)\s*>\}\}")
    for p in sources():
        for target in inc.findall(p.read_text()):
            assert (PAPER / target).exists(), (p.name, target)


def test_every_cross_reference_has_a_target():
    text = "\n".join(p.read_text() for p in sources())
    text += "\n".join(p.read_text() for p in (PAPER / "_generated").rglob("*.md"))
    labels = set(re.findall(r"\{#((?:sec|tbl|fig|eq|prp)-[\w-]+)", text))
    refs = set(re.findall(r"@((?:sec|tbl|fig|eq|prp)-[\w-]+)", text))
    assert refs - labels == set()


def test_every_citation_is_in_the_bibliography():
    bib = set(re.findall(r"@\w+\{(\w+),", (PAPER / "references.bib").read_text()))
    text = "\n".join(p.read_text() for p in sources())
    cited = {k for k in re.findall(r"@([a-z]+\d{4}[a-z]+)", text)}
    assert cited and cited <= bib


# ---- decisions match results/hypotheses.json ---------------------------------------------

def test_decision_variables_match_hypotheses(variables, hyp):
    assert lookup(variables, "h1.decision") == pb.DECISION_TEXT[hyp["h1"]["decision"]] == "supported"
    for w in ("2", "4", "8", "16"):
        d = hyp["h2"]["windows"][w]
        assert lookup(variables, f"h2.w{w}.decision") == pb.DECISION_TEXT[d["decision"]] == "rational given its information"
        assert int(lookup(variables, f"h2.w{w}.n_equivalent")) == sum(r["verdict"] == "equivalent" for r in d["rows"]) == 12
    assert lookup(variables, "h3.decision") == pb.DECISION_TEXT[hyp["h3"]["decision"]] == "not supported"
    assert lookup(variables, "h3.statement") == hyp["h3"]["statement"] == "0 of 8 positive at 95%, none after adjustment"
    assert int(lookup(variables, "h1.n_equivalent")) == sum(r["verdict"] == "equivalent" for r in hyp["h1"]["rows"]) == 12
    assert int(lookup(variables, "h3.k95")) == hyp["h3"]["k_positive_95"] == 0
    assert int(lookup(variables, "h3.k_adjusted")) == hyp["h3"]["k_positive_adjusted"] == 0
    assert int(lookup(variables, "flagged.n")) == len(hyp["flagged"]) == 0


def test_h3_tightness_variables(variables, hyp):
    combos = hyp["h3"]["combos"]
    assert num(lookup(variables, "h3.max_hi")) == pytest.approx(max(c["hi"] for c in combos), abs=5e-5)
    assert num(lookup(variables, "h3.max_hi")) < 0.001 < 0.02 == num(lookup(variables, "design.h3.threshold"))
    assert num(lookup(variables, "h3.max_lo_adj")) == pytest.approx(max(c["lo_adj"] for c in combos), abs=5e-5)


def test_decision_wording_is_the_preregistered_wording():
    prereg = (ROOT / "docs" / "PREREG.md").read_text()
    for phrase in ("supported", "rejected", "not established", "rational given its information",
                   "a departure from rational use of the window"):
        assert f"**{phrase}" in prereg


# ---- generated tables reproduce results/summary.csv --------------------------------------

def check_conf_rows(rows, inp, window_col: bool):
    for r in rows:
        w, rho = (r[0], r[1]) if window_col else ("full", r[0])
        if rho.startswith("prior"):
            rho = "prior"
        measure, est, sd, interval, df, status = (r[i + (2 if window_col else 1)] for i in (0, 2, 3, 4, 5, 8))
        col = {"gap": "gap", "excess error": "excess_mse_pct"}[measure]
        s = inp.row("l", w, rho, col)
        lo, hi = (num(x) for x in interval.strip("[]").split(", "))
        assert num(est) == pytest.approx(s.mean, abs=5.1e-5)
        assert num(sd) == pytest.approx(s.sd, abs=5.1e-5)
        assert lo == pytest.approx(s.lo, abs=5.1e-5) and hi == pytest.approx(s.hi, abs=5.1e-5)
        assert int(df) == s.n_seeds - 1 and status == s.verdict


def test_h1_table_reproduces_summary(inp):
    rows = table_rows("h1-confirmatory")
    assert len(rows) == 14
    check_conf_rows(rows, inp, window_col=False)


def test_h2_table_reproduces_summary(inp):
    rows = table_rows("h2-confirmatory")
    assert len(rows) == 48
    check_conf_rows(rows, inp, window_col=True)


def test_h3_table_reproduces_summary_and_hypotheses(inp, hyp):
    rows = table_rows("h3-confirmatory")
    assert len(rows) == 8
    for r, c in zip(rows, hyp["h3"]["combos"]):
        s = inp.row("l", r[0], r[1], "gap")
        assert (r[0], r[1]) == (c["window"], c["rho"])
        assert num(r[3]) == pytest.approx(s.mean, abs=5.1e-5)
        assert num(r[6]) == pytest.approx(c["lo_adj"], abs=5.1e-5)
        assert r[9] == r[10] == "no"


def test_verdict_appendix_prints_every_summary_row(inp):
    text = (PAPER / "_generated" / "appendix" / "verdicts.md").read_text()
    assert text.count(pb.NOTE) == 22
    body = [ln for ln in text.splitlines() if ln.startswith("| ") and "| excess error |" in ln or "| gap |" in ln]
    full_dupes = len(inp.S[(inp.S.window == "full") & (inp.S.bench == "window_bayes")])
    assert len(body) == len(inp.S) - full_dupes
    # the first verdict table is A-l-w2, step 3,000, against bayes: compare it row by row
    first = [[c.strip() for c in ln.strip("|").split("|")] for ln in body[:84]]
    for r in first:
        col = {"gap": "gap", "excess error": "excess_mse_pct"}[r[2]]
        s = inp.idx[("A", "l", "2", 3000, r[0], int(r[1]), "bayes", col)]
        assert num(r[4]) == pytest.approx(s.mean, abs=5.1e-5)
        assert r[8] == s.verdict
        assert [num(v) for v in r[9].split(", ")] == pytest.approx(list(s.values), abs=5.1e-5)


def test_size_counts_match_summary(inp):
    for r in table_rows("size-counts"):
        size = r[0].split(" ")[0]
        for w, cell in zip(pb.WINDOWS, r[2:]):
            assert int(cell.split(" ")[0]) == pb.counts(inp, size, w)["equivalent"]


# ---- hygiene: no typed numbers in the prose ----------------------------------------------

ALLOWED = [
    r"ρ\s*(=|≤|≥|∈|~)\s*\{?\d(\.\d)?((,| and| or| to|–|,\s*…,)\s*\d(\.\d)?)*\}?",   # grid labels
    r"W\s*=\s*\d+",                                                                   # window labels
    r"1\.96",                                                                         # normal quantile
    r"0\.00005", r"−0\.0000",                                                         # the rounding note
    r"ρ up to \d\.\d",                                                                 # a grid label
    r"TF32", r"RTX A6000", r"SHA-256",                                                # names
]


def strip_prose(text: str) -> str:
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
    text = re.sub(r"(?<!\\)\$[^$\n]*\$", " ", text)
    text = re.sub(r"`[^`\n]*`", " ", text)
    text = re.sub(r"\{\{<.*?>\}\}", " ", text)
    text = re.sub(r"\{#[^}]*\}", " ", text)
    text = re.sub(r"@[\w-]+", " ", text)
    for pat in ALLOWED:
        text = re.sub(pat, " ", text)
    return text


def test_no_typed_numbers_in_sections():
    """Decimals and numbers of two or more digits must come from _variables.yml."""
    offenders = []
    for p in sorted(PAPER.glob("sections/*.qmd")) + [PAPER / "brief_body.qmd"]:
        for m in re.finditer(r"\d+\.\d+|\d{2,}", strip_prose(p.read_text())):
            offenders.append((p.name, m.group(0)))
    assert offenders == []


def test_escape_md_protects_pandoc_specials():
    out = pb.escape_md("cost ~$2.11, sigma^2 ~ x and `a~b^c$d`\ntext\n- item")
    assert "\\~\\$2.11" in out and "sigma\\^2" in out and "`a~b^c$d`" in out
    assert "text\n\n- item" in out


def test_every_decisions_entry_is_classified(inp):
    for title, _ in inp.decision_entries():
        assert title.split(": ", 1)[1] in pb.RESULTS_SEEN
