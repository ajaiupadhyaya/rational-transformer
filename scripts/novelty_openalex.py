"""Novelty-check OpenAlex queries for the Rational Transformer (docs/novelty_check.md).

Provenance: adapted from expectations-without-hindsight, file
`scripts/novelty_openalex.py` at commit 57e9084. Changes from the original:
- the queries (transformers / in-context learning / sequence models crossed with the
  expectations-formation literature, and the known neighbours by title);
- every query prints its OpenAlex hit count (`meta.count`) and the date of the run;
- the search is over title and abstract (`title_and_abstract.search`), not full text,
  so hit counts measure topical overlap rather than passing mentions;
- standard library `urllib` instead of `httpx`, so the repo gains no dependency;
- a generic User-Agent. No contact address, name or `mailto` is sent.
The SSRN-only pass of the original is kept.

Usage: uv run python scripts/novelty_openalex.py [from_date] [--titles N]
       (default from_date 2017-01-01: the transformer era; N titles per query, default 10)
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import date

# Boolean syntax (AND / OR, quoted phrases) is OpenAlex's own.
QUERIES = [
    # transformers / ICL x the expectations-formation literature
    '(transformer OR "in-context learning") AND overreaction',
    '(transformer OR "in-context learning") AND underreaction',
    '(transformer OR "in-context learning") AND "expectations formation"',
    '(transformer OR "neural network") AND "Coibion" AND "Gorodnichenko"',
    '(transformer OR "neural network" OR "machine learning") AND "forecast revision" AND "forecast error"',
    '"error-on-revision" OR "errors on revisions"',
    '(transformer OR "neural network") AND "diagnostic expectations"',
    '(transformer OR "in-context learning") AND "working memory" AND overreaction',
    '"implied persistence" AND (forecast OR expectations)',
    # transformers x AR(1) / Bayes-optimal forecasting
    '(transformer OR "in-context learning") AND "AR(1)" AND forecast',
    '(transformer OR "in-context learning") AND autoregressive AND "Bayes-optimal"',
    '"in-context learning" AND ("Kalman filter" OR "optimal filtering")',
    '(transformer OR "in-context learning") AND "linear dynamical system"',
    '"in-context learning" AND (Bayesian OR "Bayes-optimal") AND "time series"',
    '(transformer OR "neural network") AND "rational expectations" AND forecast AND bias',
    '"behavioral machine learning" OR ("machine learning" AND "forecast bias" AND regularization)',
    # known neighbours by title (to confirm they are indexed and see what cites near them)
    '"Overreaction in Expectations: Evidence and Theory"',
    '"Overreaction and Working Memory"',
    '"Overreaction in Macroeconomic Expectations"',
    '"Information Rigidity and the Expectations Formation Process"',
    '"Transformers learn to implement" AND "filtering"',
    '"transformers" AND "implicit state estimators"',
]
# Supplementary queries added after reading the first pass; run on all sources with no
# date filter (2026-11-16 these were run with the same parameters from a scratch helper).
EXTRA_QUERIES = [
    '"in-context learning" AND "linear dynamical systems"',
    '"in-context" AND "autoregressive process" AND transformer',
    '(transformer OR transformers) AND Kalman AND ("in-context" OR "in context")',
    '"in-context" AND ("noisy linear regression" OR "linear regression") AND ("Bayes-optimal" OR "Bayes optimal") AND transformer',
    '("large language model" OR LLM OR GPT) AND overreaction AND (expectations OR forecasts)',
    '("neural network" OR "deep learning" OR "recurrent") AND overreaction AND expectations AND forecast',
    '("neural network" OR "deep learning" OR transformer) AND "expectations formation" AND (Bayesian OR rational)',
    '(transformer OR "neural network") AND "trained from scratch" AND forecast AND Bayes',
    '"in-context learning" AND (expectations OR beliefs) AND (overreaction OR extrapolation OR "belief updating")',
]
SSRN = "S4210172589"
UA = "rational-transformer/0.1 (academic research)"
API = "https://api.openalex.org/works"


def get(params: dict) -> dict:
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params),
                                 headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main(since: str, n_titles: int) -> None:
    print(f"# OpenAlex run {date.today().isoformat()}, from_publication_date:{since}, "
          f"title_and_abstract.search")
    seen = set()
    for q in QUERIES:
        for label, extra in (("all", ""), ("ssrn", f",primary_location.source.id:{SSRN}")):
            body = get({
                "filter": f"title_and_abstract.search:{q},from_publication_date:{since}{extra}",
                "per-page": n_titles,
                "sort": "relevance_score:desc",
                "select": "title,publication_date,doi,ids,cited_by_count",
            })
            print(f"\n## [{label}] {q}  -> {body['meta']['count']} hits")
            for w in body["results"]:
                key = w.get("doi") or w.get("title")
                mark = " " if key in seen else "*"
                seen.add(key)
                print(f"{mark} {w.get('publication_date')} | {(w.get('title') or '')[:110]} "
                      f"| {w.get('doi') or w['ids'].get('openalex')} | cites {w.get('cited_by_count')}")
            time.sleep(1)
    for q in EXTRA_QUERIES:
        body = get({"filter": f"title_and_abstract.search:{q}", "per-page": n_titles,
                    "sort": "relevance_score:desc",
                    "select": "title,publication_date,doi,ids,cited_by_count"})
        print(f"\n## [extra, all years] {q}  -> {body['meta']['count']} hits")
        for w in body["results"]:
            key = w.get("doi") or w.get("title")
            mark = " " if key in seen else "*"
            seen.add(key)
            print(f"{mark} {w.get('publication_date')} | {(w.get('title') or '')[:110]} "
                  f"| {w.get('doi') or w['ids'].get('openalex')} | cites {w.get('cited_by_count')}")
        time.sleep(1)
    print(f"\n# {len(seen)} distinct works listed ('*' marks first appearance)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = 10
    if "--titles" in sys.argv:
        n = int(sys.argv[sys.argv.index("--titles") + 1])
        args = [a for a in args if a != str(n)]
    main(args[0] if args else "2017-01-01", n)
