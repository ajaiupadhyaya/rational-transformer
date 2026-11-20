# Paper build: progress log

Academic paper for "The Rational Transformer" (Quarto → xelatex), mirroring
`~/Documents/ai-delegation/docs/paper/`. Updated after each step so the work can be resumed.

## Done
- [x] Studied the reference paper (layout, `_quarto.yml`, sections, generated appendix, variables pipeline).
- [x] Read PREREG.md, DECISIONS.md, the house report text, README, PROVENANCE, results schemas, rt/ and analysis/ code.
- [x] `analysis/paper_build.py` (`uv run python -m analysis.paper_build`): writes `_variables.yml`
      (about 800 keys), `variables_map.csv`, `tables.csv`, `figures.csv`, `_generated/tables` (26),
      `_generated/figures` (12 vector PDFs + includes), `_generated/appendix`
      (preregistration, deviations, disclosures, verdicts, data, hypotheses).
- [x] `upstream_sources.csv` (URLs, retrieval times, hashes of the upstream raw files, copied once
      from the companion repository's manifest with `--refresh-upstream`).

## Next
- [ ] `_quarto.yml`, `paper.qmd`, `references.bib`.
- [ ] Sections 00 (abstract), brief, 01–13, statements; appendix qmd files A–F.
- [ ] Render, inspect pages, fix; `tests/test_paper_build.py`.

## Decisions
- `_variables.yml` is written as JSON (valid YAML), nested keys, as the reference does.
- Everything numeric in prose is a `{{< var >}}`; design constants too. Grid labels (ρ = 0.6,
  W = 4, size l, horizon 1, H1–H3) are typed as labels. A hygiene test enforces this.
- Decision words come from `results/hypotheses.json` through `DECISION_TEXT` (PREREG section 7 wording).
- Negative numbers use a true minus sign; a printed −0.0000 is a negative value below 0.00005.
- Appendix D: one subsection per (family, size, window) cell, 22 cells; per cell and checkpoint,
  two verdict tables (against Bayes and window-Bayes, all ρ and horizons 1–6, with seed values)
  and one table of diagnostic means. Landscape, scriptsize, no captions (bold lead lines).
- Error-on-revision table is computed seed by seed from `results/metrics/` (summary.csv holds
  only seed means of b). Exploratory; no margin was pre-registered, so no verdict is attached.
- "Results seen" column of Appendix B is a classification in code (`RESULTS_SEEN`) that quotes
  the timing DECISIONS records; the build fails if DECISIONS gains an entry with no classification.
- The DECISIONS sign-off entry records no statement in the owner's own words; Appendix A says so
  and quotes the entry verbatim.
- References: only works listed in the house report or docs/novelty_check.md. Works known only by
  surname there are entered with surnames only.

## Update 2026-11-20 (coordinator relay of two author instructions; applied, not restarted)
- Sections 00–11 and the brief are written. Remaining at this point: 12, 13, statements, appendix qmd files, render, tests.
- Appendix A signature: quote the author's own statement "I sign off yeah", given on 2026-11-20
  AFTER the results were known, as a confirmation of the 2026-11-17 sign-off recorded in
  DECISIONS (quoted verbatim too). Stored in `docs/paper/signature.json`, read by the generator.
  DECISIONS.md itself is not edited (it is reproduced verbatim in Appendix B).
- Layout: the reference paper is a guide, not a template. Working-paper title page (title,
  author with footnote for links, "Working paper" status and version date, abstract, JEL codes,
  keywords), TOC on the following page, running headers (fancyhdr), microtype, muted link colour,
  right-aligned numeric columns (fixed decimals and explicit signs give decimal alignment),
  "Notes:" under tables, figures set in Pagella to match the text. Pagella + Pagella Math kept.

## Status 2026-11-20: complete
- All sections (00–14), brief, appendices A–F written; `quarto render docs/paper` succeeds:
  `_output/paper.pdf`, 424 pages (front matter, body, references, statements: pp. 1–49;
  appendices: pp. 50–424; Appendix D alone pp. 67–401). `paper.tex` is kept beside `paper.qmd` for arXiv.
- `tests/test_paper_build.py` (17 tests): every var resolves; variables file current; decisions
  match `results/hypotheses.json`; H1/H2/H3 tables and Appendix D reproduce `results/summary.csv`;
  cross-references, includes and citations resolve; no typed numbers in the sections.
- Appendix D prints the full-window window-Bayes tables once (they equal the Bayes tables; the
  generator asserts it), and tables are set single-spaced: 335 pages instead of 442.
- Remaining LaTeX warnings: about 350 overfull hboxes, all but ~25 under 5pt, in narrow columns
  of the scriptsize appendix tables; no missing glyphs; no undefined references or citations.
- Left for the author: `<!-- AUTHOR: ... -->` placeholders (affiliation and contact e-mail on the
  title page; confirm the Declaration of interests and Funding statements).
- Not done (not needed for submission): a separate two-page `brief.qmd`, `PROSE_OBJECTS.md`.
- To rebuild: `uv run python -m analysis.paper_build && quarto render docs/paper`.

## Revision round (2026-11-20, coordinator): in progress
- Applying the senior review (`academic_review.md`) and the fact-check overrides B1–B9.
- Rulings: S1 accepted (six references, added after the pre-freeze search, said so in 2.4);
  T4 NOT applied (Appendix D stays whole) but P1 and P5 applied in full; S12 run-in sentence;
  AUTHOR markers stay. DECISIONS.md, novelty_check.md and paper/the-rational-transformer/ untouched.

## Revision round: complete (2026-11-20)
- Applied the senior review and fact-check B1–B9. PDF 417 pp: body to p. 48 (Statements now
  precede References), appendices pp. 49–417, Appendix D pp. 66–396 with an index table
  (page refs), the cell in the running head, Needspace before lead lines, Bayes/window-Bayes labels;
  appendix subsections are out of the TOC (TOC is 3 pages).
- Abstract 1,898 characters as rendered; plain text in `_output/abstract.txt`.
- Not applied: T4 (ruling: D stays whole); T7 (no edit proposed); P9 "cite published versions"
  (not verified); P11(c) float placement left at 'H'; P2 optional removal of constant columns.
  P9's `@unpublished` dropped the arXiv numbers under the Chicago style, so preprints are
  `@article` with `journal = {arXiv preprint arXiv:…}` (title in quotation marks, identifier kept).
  P3 sweep done for Section 3 only (proposition, proof and the running-text symbols).
- T6: the Limitations "Cost" paragraph is deleted; cost now lives in Section 4.6, Statements,
  Appendix C item 9 and the cost table of Appendix E, all with the $14.35–$15.06 range.
- Captions must not end in "pp" (pandoc reads "pp." as an abbreviation and breaks the label).
- Tests: 45 passed. Remaining LaTeX warnings: 170 overfull hboxes, 14 of them over 5pt (max 15pt).
