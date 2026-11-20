# Submission package: arXiv and SSRN

**Status (2026-11-20):** Zenodo published: version 2, record 24400796, DOI 10.5281/zenodo.24400796, with the corrected title page; all versions resolve through DOI 10.5281/zenodo.24387528 (version 1 is 24387529). Replace the SSRN file through Revise once SSRN approves it. SSRN submitted, abstract 7733526, under review. arXiv submission 8401422 is open and waits for a cs.LG endorsement.

Everything below is ready to paste. Posting is the author's action. Do not post until the
preconditions are met.

## Preconditions

- [x] Repository public with the `prereg-v1` to `prereg-v1.4` tags pushed (20 November 2026):
      https://github.com/ajaiupadhyaya/rational-transformer. The URL is in section 05 of the paper.
- [x] Cover numbers set: Report Esn-0015, Publications No. 33 (next in the esnupi series after Esn-0014 / No. 32).
- [ ] Cloud spend confirmed in the Thunder Compute console. If it differs from about $14.35,
      update the summary card and the Cost bullet in section 10.
- [ ] Final PDF read once, start to finish.

## Shared metadata

- **Title:** The Rational Transformer
- **Subtitle (SSRN allows one):** Small transformers trained only to minimise forecast error,
  measured against exact Bayes forecasters that see what the network sees
- **Author:** Ajai Upadhyaya
- **Affiliation:** fill in, e.g. school or "Independent researcher"
- **Keywords:** transformers; in-context learning; Bayesian forecasting; expectations;
  overreaction; AR(1); pre-registration
- **JEL codes (SSRN):** C11 (Bayesian analysis), C45 (neural networks), C53 (forecasting),
  D83 (learning), D84 (expectations), G41 (behavioural finance)
- **File to submit:** the academic paper, `docs/paper/_output/paper.pdf` (Quarto/LaTeX, modelled on *Delegating to the Machine*). The house-style report `paper/the-rational-transformer/the-rational-transformer.pdf` is the site/BASIS edition only and is **never** uploaded to SSRN, arXiv or Zenodo.

## Abstract (1,569 characters; fits arXiv's 1,920 limit)

In forecasting experiments on AR(1) series, people overreact to news, and more so when the
series is less persistent. A rational forecaster who has to learn persistence also departs from
the truth at any fixed persistence, because its estimate is pulled toward the prior. So a
deviation from the full-information forecast is not evidence of a bias. We train causal
transformers from scratch on AR(1) sequences with persistence drawn from U[0, 1], at four sizes
and five memory lengths, and compare each with the exact Bayes forecaster that holds the same
prior and the same window of data. Hypotheses, margins and decision rules were frozen in a
tagged pre-registration before any confirmatory run. The largest model with full memory is
equivalent to Bayes on all 12 pre-registered verdicts. With its memory cut to 2, 4, 8 or 16
observations it is equivalent to window-Bayes on all 12 verdicts at every window, so what it
gets wrong relative to full Bayes is accounted for, within the margins, by the information it
lacks. In none of the 8 window-persistence combinations tested does it overreact where
window-Bayes underreacts, at 95% or after adjustment; no mean gap there is above zero. These
verdicts hold at the last checkpoint, after the learning rate has decayed to zero; at 10,000 of
30,000 steps the windowed models are not yet equivalent. The Bayes forecast is the squared-error
optimum, so this is a result about training: by the end of its schedule this network has
reached the information-limited Bayesian forecast, with no residual in the human direction.

## arXiv

- **Primary category:** cs.LG
- **Cross-lists:** econ.EM (econometrics), stat.ML
- **Comments field:** 14 pages, 4 figures. Pre-registered (tags prereg-v1 to v1.4). Code, data
  and every verdict with its seed values: https://github.com/ajaiupadhyaya/rational-transformer
- **Licence:** CC BY 4.0 lets anyone reuse the work with credit. The arXiv default licence is
  the more restrictive option.
- **Format:** PDF-only upload is accepted for PDFs not produced from TeX. This one is built
  from HTML with headless Chrome.
- **Endorsement:** a first submission to cs.LG may need an endorsement from an existing cs.LG
  author. arXiv says so during submission, if it applies.

## SSRN

- **Account:** a free SSRN account is needed. Upload the same PDF.
- **Classification:** pick the networks SSRN offers for forecasting, expectations or
  behavioural finance, and for AI or machine learning. The topics the paper speaks to are
  expectations formation (Afrouzi et al. 2023; Bordalo et al. 2020; Coibion and Gorodnichenko
  2015) and in-context learning.
- **Abstract:** same as above.
- **After posting:** SSRN screens papers before they appear, usually within a few days.
  Posting on both SSRN and arXiv is allowed. Add the arXiv identifier to SSRN once it exists,
  and the reverse.
