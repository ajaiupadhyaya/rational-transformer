# Novelty check (before the `prereg-v1` tag)

**Date:** 2026-11-16 (local; the APIs stamped their responses 2026-11-17 UTC).
**Question:** has anyone already applied the expectations-formation diagnostics
(error-on-revision, implied persistence by rho, horizon) to transformers trained from scratch,
against an exact Bayes benchmark and an information-matched window-Bayes benchmark, mapped
against context length, with a human overlay? Stop rule: if yes, the study is BLOCKED.

**Verdict: not found. Proceed.** No paper found combines any two of the three defining
pieces: (i) the expectations diagnostics of Coibion–Gorodnichenko / BGMS / AKLMT, (ii) a
small network trained from scratch on a known process with an exact Bayes reference, and
(iii) an information-matched benchmark swept across context length. The two literatures this
study joins are each well populated on their own side: transformers that match the Kalman
filter or the Bayes predictor *in mean squared error*, and language models or ML forecasters
that fail *expectations tests* on human-style data. Neither side measures bias against a
Bayes forecaster with the same information.

## Claimed gap, stated precisely

The study computes, for transformers trained from scratch only to minimise squared forecast
error on AR(1) paths with rho ~ U[0, 1]:

1. the implied persistence (slope of the forecast on the current value) by rho on the AKLMT
   grid {0, 0.2, ..., 1}, by horizon 1–6, **minus** that of an exact grid-posterior Bayes
   forecaster with the same prior;
2. the same against **window-Bayes**, the exact posterior from only the last W observations and
   their true time indices, the best possible forecaster with exactly the network's information,
   with W in {2, 4, 8, 16, full} imposed as a hard input truncation;
3. the BGMS error-on-revision coefficient for the network next to each benchmark's own
   coefficient (which is not zero for window-Bayes);
4. a Family B (unknown mean and scale) AKLMT curve drawn over the human-subject curve.

What is new is the combination: the expectations diagnostics, measured *in excess of* an exact
and an information-matched Bayes benchmark, as a function of context length, on networks whose
training objective is known to be the conditional mean. MSE-parity with Kalman or Bayes (the
neighbours below) is close to our H1 and is treated in PREREG.md as a near-replication.

## How the search was done

| Source | Method | Coverage note |
|---|---|---|
| OpenAlex | `scripts/novelty_openalex.py` (adapted from EWH @ 57e9084): `title_and_abstract.search`, boolean queries, from 2017-01-01, all sources and SSRN-only (`S4210172589`); then nine supplementary queries with no date filter | many SSRN records in OpenAlex have no abstract, so title-and-abstract search can miss them |
| OpenAlex | lookup by DOI for every economics reference and every candidate neighbour | — |
| arXiv | export API by id (every arXiv identifier in this file) and by keyword | arXiv keyword search is coarse (terms are effectively OR-ed); used for verification more than discovery |
| Web | two extended web searches (transformers on AR(1) vs a Bayes benchmark with implied persistence; the "fixed-window autoregression" line on linear dynamical systems) and one exact-title search | — |
| ICLR proceedings | direct page, to verify a paper with no arXiv record found | — |
| OpenReview | API | **blocked** by a challenge page (HTTP 403); not bypassed |
| Google Scholar, SSRN | not searched directly | EWH found both blocked by bot checks; covered through OpenAlex |

Privacy: requests carried a generic User-Agent and no name, e-mail or `mailto`. An OpenAlex
response suggested registering an API key with the user's e-mail address; that was not done.
The OpenAlex free tier is metered per IP per day, which is why the supplementary queries were
not re-run inside the full script.

### Queries and hit counts (OpenAlex, 2026-11-16)

Main pass: `title_and_abstract.search`, `from_publication_date:2017-01-01`.

| Query | All | SSRN |
|---|---|---|
| `(transformer OR "in-context learning") AND overreaction` | 5 | 1 |
| `(transformer OR "in-context learning") AND underreaction` | 2 | 0 |
| `(transformer OR "in-context learning") AND "expectations formation"` | 0 | 0 |
| `(transformer OR "neural network") AND "Coibion" AND "Gorodnichenko"` | 0 | 0 |
| `(transformer OR "neural network" OR "machine learning") AND "forecast revision" AND "forecast error"` | 3 | 1 |
| `"error-on-revision" OR "errors on revisions"` | 2 | 2 |
| `(transformer OR "neural network") AND "diagnostic expectations"` | 1 | 0 |
| `(transformer OR "in-context learning") AND "working memory" AND overreaction` | 0 | 0 |
| `"implied persistence" AND (forecast OR expectations)` | 3 | 1 |
| `(transformer OR "in-context learning") AND "AR(1)" AND forecast` | 12 | 3 |
| `(transformer OR "in-context learning") AND autoregressive AND "Bayes-optimal"` | 2 | 0 |
| `"in-context learning" AND ("Kalman filter" OR "optimal filtering")` | 2 | 0 |
| `(transformer OR "in-context learning") AND "linear dynamical system"` | 3 | 0 |
| `"in-context learning" AND (Bayesian OR "Bayes-optimal") AND "time series"` | 3 | 0 |
| `(transformer OR "neural network") AND "rational expectations" AND forecast AND bias` | 2 | 0 |
| `"behavioral machine learning" OR ("machine learning" AND "forecast bias" AND regularization)` | 247 | 4 |
| `"Overreaction in Expectations: Evidence and Theory"` | 3 | 0 |
| `"Overreaction and Working Memory"` | 2 | 1 |
| `"Overreaction in Macroeconomic Expectations"` | 1 | 0 |
| `"Information Rigidity and the Expectations Formation Process"` | 3 | 2 |
| `"Transformers learn to implement" AND "filtering"` | 0 | 0 |
| `"transformers" AND "implicit state estimators"` | 1 | 0 |

The 247 hits of the "behavioral machine learning" query are almost all cybersecurity and
health uses of the phrase; its top hit is the Frank–Gao–Yang paper below. The
Coibion–Gorodnichenko title query returns later papers only because the main pass starts in
2017; the 2015 paper itself was verified by DOI.

Supplementary pass: same field, no date filter, all sources.

| Query | Hits |
|---|---|
| `"in-context learning" AND "linear dynamical systems"` | 2 |
| `"in-context" AND "autoregressive process" AND transformer` | 2 |
| `(transformer OR transformers) AND Kalman AND ("in-context" OR "in context")` | 2 |
| `"in-context" AND ("noisy linear regression" OR "linear regression") AND ("Bayes-optimal" OR "Bayes optimal") AND transformer` | 8 |
| `("large language model" OR LLM OR GPT) AND overreaction AND (expectations OR forecasts)` | 4 |
| `("neural network" OR "deep learning" OR "recurrent") AND overreaction AND expectations AND forecast` | 2 |
| `("neural network" OR "deep learning" OR transformer) AND "expectations formation" AND (Bayesian OR rational)` | 0 |
| `(transformer OR "neural network") AND "trained from scratch" AND forecast AND Bayes` | 0 |
| `"in-context learning" AND (expectations OR beliefs) AND (overreaction OR extrapolation OR "belief updating")` | 3 |

Every listed title was read. Apart from the papers in the tables below, the hits are off topic
(momentum trading, LEO routing, retinal imaging, hydrology, ransomware detection and the like).

## Closest papers and how this study differs

Verification column: **arXiv** = identifier, title and authors confirmed through the arXiv
export API; **OA** = confirmed through OpenAlex by DOI; **ICLR** = confirmed on the ICLR
proceedings page.

### Transformers against Kalman / Bayes (MSE side)

| Paper | Verified | What it does | How this study differs |
|---|---|---|---|
| Balim, Du, Oymak, Ozay, "Can Transformers Learn Optimal Filtering for Unknown Systems?", arXiv:2308.08536 | arXiv | Trains a transformer across many linear systems; on unseen systems it matches the Kalman filter's output-prediction error; sample-complexity guarantee; identifies failure classes. | Accuracy only. No slope, revision or bias diagnostics, no information-matched benchmark, no context-length map. Our H1 is close to a replication of its headline in a scalar AR(1). |
| Akram, Vikalo, "Transformers as Implicit State Estimators: In-Context Learning in Dynamical Systems", arXiv:2410.16546 (v1 title "Can Transformers In-Context Learn Behavior of a Linear Dynamical System?") | arXiv | In-context predictions match the Kalman filter in linear-Gaussian systems and approach EKF / particle filters in nonlinear ones; degrades gracefully when parameters are withheld. | Same as above: MSE closeness to the optimal filter. Does not ask whether residual deviations are over- or underreaction, or separate information limits from learning failures. |
| Vlădărean, Zhang, Sra, "On learning linear dynamical systems in context with attention layers", ICLR 2026 | ICLR (no arXiv record found through the API; OpenReview blocked) | The "improper learner" line: the optimal single linear-attention layer equals one gradient step on a window-size-one autoregression objective; for larger windows, a link to preconditioned conjugate gradient; offered as an explanation of Kalman-level performance. | Expressivity theory of *what* attention computes (a fixed-window autoregression rather than system recovery). Our window-Bayes benchmark is the behavioural counterpart: it is the best any fixed-window forecaster can do, so the gap from it measures learning failure, not the window. No expectations diagnostics. |
| Cole, Zhao, Lu, Zhang, "In-Context Learning of Linear Dynamical Systems with Transformers: Approximation Bounds and Depth-Separation", arXiv:2502.08136 | arXiv | Log-depth transformers reach least-squares error on noisy LDS; single-layer linear transformers have a non-vanishing lower bound. | Approximation theory; no trained-network bias measurements. Relevant to our size axis (xs = 1 layer). |
| Sander, Giryes, Suzuki, Blondel, Peyré, "How do Transformers perform In-Context Autoregressive Learning?", arXiv:2402.05787 | arXiv | Trained transformers on noiseless s_{t+1} = W s_t learn W in context; one-layer linear transformer = one GD step. | Mechanism, deterministic AR, no noise, no Bayes benchmark, no bias diagnostics. |
| Raventós, Paul, Chen, Ganguli, "Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression", arXiv:2306.15063 | arXiv | Transformers trained from scratch on linear regression behave like the Bayes estimator for the pretraining prior below a task-diversity threshold and like ridge (non-Bayesian, prior-free) above it. | The closest "is the trained network Bayes?" test, but for i.i.d. regression, not time-series forecasting; MSE against estimators, not expectations diagnostics. Our training prior is continuous, so their discrete-task regime does not arise. |
| Acero, "What Do Foundation Models for Market Microstructure Learn? A Controlled Study with Synthetic Limit Order Books", ICAIF 2026, doi:10.1145/3858063.3858749 (dated 2026-11-13) | OA | GPT-style models trained on simulator order flow with known latent parameters; the learned predictive matches the exact Bayes optimum to within a millinat per token; regime decodable from activations. | **Methodologically the closest** (from-scratch training on a known simulator, exact Bayes reference). It measures next-token log loss and representations, not forecast-slope or revision biases, has no information-matched benchmark and no context-length sweep, and makes no link to the expectations literature. |
| Chib, Tan, Zhang, "Learning the Macroeconomic Language", arXiv:2512.21031 | arXiv | Transformer trained on DSGE-simulated panels mixed with real data; compared with a VAR on accuracy and likelihood. | Accuracy against a VAR; no exact Bayes, no bias diagnostics. |

### Expectations tests on machine forecasters (bias side)

| Paper | Verified | What it does | How this study differs |
|---|---|---|---|
| Frank, Gao, Yang, "Behavioral Machine Learning? Regularization and Forecast Bias", arXiv:2303.16158 (SSRN 4395903) | arXiv, OA | Rational forecasters with optimal regularisation fail forecast-efficiency (CG-type) tests; ML earnings forecasts show near-zero bias at one year and overreaction at two; analysts shift toward overreaction after 2013. | Real analyst data with an unknown true process, so "bias" is measured against FIRE, not against a Bayes forecaster with the same information. Our design makes that comparison exact and is in the same spirit: violations of FIRE tests need not be irrationality. |
| Gao, Jiang, Yan, "Debiasing LLMs by Fine-tuning", arXiv:2604.02921 | arXiv | Off-the-shelf LLMs show extrapolation bias on AKLMT-style AR(1) tasks; LoRA fine-tuning on rational-forecast instructions reduces it. | Pretrained language models prompted with numbers, measured against FIRE. Not trained from scratch, no exact or window-Bayes benchmark. |
| Bini, Cong, Huang, Jin, "Behavioral Economics of AI: LLM Biases and Corrections", arXiv:2602.09362 (NBER w34745 per EWH) | arXiv | Runs many human-bias experiments, including AKLMT-style forecasting, on LLM families. | Same: LLMs, FIRE reference. |
| Chen, Green, Gulen, Zhou, "What Does ChatGPT Make of Historical Stock Returns?", arXiv:2409.11540 | arXiv | LLM return forecasts over-extrapolate, like humans. | LLM, real returns, no Bayes benchmark. |
| Nagel, Tseng, Xiu, "Completions, Not Convictions: Can LLM Probabilities Proxy for Human Beliefs?", SSRN 7265538 | OA (title, authors; no abstract in OpenAlex) | Matched by the error-on-revision query; asks whether LLM probabilities behave like human beliefs. | LLM output probabilities, not trained forecasters on a known process. |
| Wang, Liu, "Bridging Expectation Signals: LLM-Based Experiments and a Behavioral Kalman Filter Framework", arXiv:2601.17527 | arXiv | LLM agents' weights on priors and signals in a behavioural Kalman framework. | LLM agents; no from-scratch training, no exact Bayes. |
| Chen, Huang, "Memory and Beliefs in Financial Markets: A Machine Learning Approach", SSRN 5013622 | OA (title, authors; no abstract) | Matched by the forecast-revision query; by its title, ML used to study memory in human beliefs. | Human beliefs; ML as a measurement tool. Abstract not seen, so the description is from metadata only. |

### The economics anchors (not neighbours; verified for citation)

| Paper | Verified | Role here |
|---|---|---|
| Afrouzi, Kwon, Landier, Ma, Thesmar, "Overreaction in Expectations: Evidence and Theory", QJE 138(3), 2023, doi:10.1093/qje/qjad009 | OA (OpenAlex dates the record 2020, the working-paper year) | The experiment (AR(1), rho grid, rounds 41–80), implied persistence, and the human curve for the overlay. |
| Afrouzi, Kwon, Landier, Ma, Thesmar, "Overreaction and Working Memory", NBER w27947, 2020, doi:10.3386/w27947 (also SSRN 3714439) | OA | The working-paper version of the QJE paper (same five authors); its theory ties overreaction to imperfect use of past information, which our window W makes literal. |
| Bordalo, Gennaioli, Ma, Shleifer, "Overreaction in Macroeconomic Expectations", AER 110(9), 2020, doi:10.1257/aer.20181219 | OA | Individual-level error-on-revision regression (our D3). |
| Coibion, Gorodnichenko, "Information Rigidity and the Expectations Formation Process: A Simple Framework and New Facts", AER 105(8), 2015, doi:10.1257/aer.20110306 | OA | The error-on-revision test. |

## References that could not be fully verified

- **Vlădărean, Zhang, Sra (ICLR 2026):** title, authors and abstract confirmed on the ICLR
  proceedings page; no arXiv identifier could be found through the arXiv API, and the OpenReview
  API was blocked. Cite it by venue.
- **Bini et al. as NBER w34745:** the arXiv record was verified; the NBER number is carried
  over from EWH's check and was not re-verified here.
- **Chen–Huang and Nagel–Tseng–Xiu:** OpenAlex holds no abstract, and SSRN was not read, so
  their overlap is judged from title and metadata. Neither is about networks trained on a
  known process. Read both before the paper's literature section.
- "Transformers as Statisticians" (NeurIPS 2023), cited in the design, was not re-checked here;
  it is context for H1 and nothing more.

## Residual blind spot

An SSRN-only paper with no abstract in OpenAlex, or a preprint indexed in the last few days.
Re-run `uv run python scripts/novelty_openalex.py 2026-11-01` immediately before the
`prereg-v1` tag and add any new hit here.
