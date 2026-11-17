# The Rational Transformer — design

Date: 2026-11-16. Status: awaiting AJ's approval. Nothing is built.

## Intent

Human forecasters make predictable errors: they overreact to recent news,
more so when the series is less persistent and the horizon is longer
(Afrouzi, Kwon, Landier, Ma, Thesmar 2023, "AKLMT"; Bordalo, Gennaioli, Ma,
Shleifer 2020, "BGMS"). The question here: **does a transformer trained only
to minimise forecast error end up rational, or does it show the same
biases, and if so, under what constraints?**

Why now, and why this: AJ has $20 of Thunder Compute credit and wants a
quick, thorough, publishable piece for BASIS. The models are tiny, the right
answer is known exactly, and nothing is licensed, so the whole study fits the
credit and the repo can be public. It is also the controlled companion to
*Expectations Without Hindsight* (EWH): EWH measures these biases in real
language models, at high inference cost and with sampling noise; this study
shows what pure error-minimisation produces when the truth is known. A point
forecast is deterministic, so the sampling-noise problem EWH fights does not
exist here.

## What is already known (positioning)

- Transformers trained on many linear systems approach Kalman/Bayes accuracy
  in mean squared error (Balim et al. arXiv 2308.08536; arXiv 2410.16546;
  "Transformers as Statisticians", NeurIPS 2023). One analysis argues they
  do it by fitting a fixed-window autoregression, not by recovering the
  system.
- "Behavioral Machine Learning?" (arXiv 2303.16158) shows regularised ML
  forecasters fail the Coibion–Gorodnichenko test on real analyst data.
- AKLMT's own theory ties overreaction to limited working memory.
- Four searches found no paper that runs the expectations diagnostics
  (error-on-revision, AKLMT implied persistence, horizon) on transformers
  trained from scratch against an exact Bayes benchmark, or that maps them
  against context length. That is the claimed gap. H1 below is therefore
  close to a replication; the contribution is H2, H3 and the human
  comparison. The builder repeats the novelty check with OpenAlex (as EWH
  did) before anything is tagged, and stops to report if it finds the gap
  closed.

## Goals

1. A map of where trained transformers sit between rational and human-like,
   as a function of attention window, model size and training time.
2. A clean split of any bias into "the model lacks the information" versus
   "the model failed to learn".
3. A direct overlay of the transformer's AKLMT curve on the human subjects'
   curve.
4. A public repo, a results pack, and a short house-style paper PDF.

## Non-goals

- No language models, no ChronoGPT, no text.
- No changes to the EWH repo (its code is bound to a tag). Needed pieces
  (AKLMT replication, SPF replication, least-squares learner) are copied in
  with provenance from a pinned EWH commit.
- No reinforcement learning, no market or trading claims.

## Anti-goals

- Saying "transformers are irrational" when the bias is entirely an
  information limit. The window-Bayes benchmark exists to prevent this.
- Changing hypotheses or analysis after seeing sweep results.
- Spending past the cap, or leaving an instance running.

## Constraints

- Thunder Compute credit: $20 total. **Hard cap $15 of spend, $5 reserve.**
  Price is read from the live pricing page and quoted to AJ before an
  instance is created. AJ creates the account and token; Claude never
  handles the password. Instance deleted once results are copied off and
  verified.
- Everything that can run on the Mac runs on the Mac first (≤4 processes,
  battery guard, resumable).
- uv for Python. Commits authored as the GitHub noreply address, set before
  the first commit.

## Approaches considered

1. **Abliterate the future** (remove post-cutoff knowledge from Qwen 7B).
   Dropped: the effect being removed was small in BASIS 008, the closest
   prior work found ablation does nothing, and the data is licensed.
2. **Goodhart's Sharpe** (GRPO-train a model to write alphas). Dropped: tight
   on $20, crowded, and the result is the expected one.
3. **The Rational Transformer** (this document). **Picked by AJ on
   2026-11-16**: large clean effects, cheap, public, informative either way.

AJ was told that the core evidence is simulated, against the usual
real-data preference, and chose to go ahead. Real data appears as a
comparison arm only.

## Design

### A point the whole design rests on

A rational Bayesian who has to learn ρ is unbiased only on average over
the prior. At any fixed ρ its estimate is pulled toward the prior, so its
implied persistence sits above ρ when ρ is low and below ρ when ρ is high.
Part of the "overreaction at low persistence" pattern is therefore what a
rational learner does anyway. Humans in AKLMT differ from this at high ρ:
their implied persistence is at or above the truth for every ρ below 1,
where a Bayesian's is below it. So nothing here is called a bias unless it
is measured **in excess of the matching Bayes benchmark at the same ρ**,
and the sharp question is whether a trained network ever overreacts where
a Bayesian underreacts.

### Data (two families)

**Family A, primary, for the hypotheses.** AR(1):
`x_t = ρ·x_{t-1} + ε_t`, `ε ~ N(0,1)`, `x_0 = 0`, mean zero and noise
scale known. Each sequence draws ρ ~ Uniform[0,1]. 80 steps.

**Family B, for the human overlay and the real-data arm.** Same process
with an unknown mean and noise scale drawn per sequence from priors set to
cover AKLMT's design; forecasts scored only at rounds 41–80, as the
subjects' were. Descriptive, not part of H1–H3.

Evaluation: per-ρ statistics on the AKLMT grid
ρ ∈ {0, 0.2, 0.4, 0.6, 0.8, 1}; anything whose rational value is zero only
on average (see gate 1) uses ρ drawn from the prior. Evaluation seeds are
never used in training.

### Models

Transformer over real-valued inputs. Outputs point forecasts for horizons
1 to 6 (a revision at horizon h needs the horizon h+1 forecast from the
period before), trained with squared error against realised values, so
the target is the conditional mean.

- Sizes: 1 layer/d32, 2/d64, 4/d128, 8/d256.
- Window W ∈ {2, 4, 8, 16, full}. **W is a total information limit: the
  input is truncated to the last W observations**, not a per-layer mask
  (a mask would let deep models see further than W). The model is also
  given the true time index t, and so is the benchmark, so the two hold
  exactly the same information.
- 5 seeds per cell → 100 runs in the main grid for Family A; Family B
  trains the largest size only, at each window.
- Checkpoints at log-spaced steps.

### Benchmarks (all exact, by grid posterior)

- **Full-information rational**: `ρ^h · x_t` with the true ρ.
- **Bayes**: posterior over ρ under the training prior from all
  observations so far; forecast `E[ρ^h | data] · x_t`.
- **Window-Bayes**: posterior from the last W observations and t,
  including the likelihood of the first observation in the window given
  (ρ, t). This is the best any forecaster with that information can do.
  For Family B the grid also covers mean and scale.
- **Least-squares learner**: from EWH, for continuity with that paper.

### Diagnostics

- D1. Excess squared error over the matching benchmark, in percent.
- D2. **Implied-persistence gap (headline)**: slope of the forecast on the
  current value, per ρ and horizon, minus the same slope for the matching
  benchmark. Positive means overreaction beyond what a rational forecaster
  with the same information shows.
- D3. Error-on-revision coefficient b (BGMS individual-level form), per ρ,
  horizons 1 to 5, **reported next to the matching benchmark's own b**
  (window-Bayes forecasts are not a martingale, so its b is not zero).
- D4. κ against full Bayes only: slope of (model forecast − Bayes forecast)
  on the Bayes one-step surprise `x_t − F_{t-1}x_t`.
- D5. Raw AKLMT curve for Family B (implied persistence against ρ) drawn
  with the human-subject curve from EWH and the Family B Bayes curve.

Uncertainty and verdicts: seeds are the unit. "Equivalent to the
benchmark" means the 95% interval of the seed mean lies wholly inside a
margin fixed in advance: excess error under 1% and |implied-persistence
gap| under 0.02. A cell whose training loss has not plateaued (criterion
fixed in advance by the builder) is reported as "not converged", never as
bias. Evaluation samples are sized so within-seed error is negligible.

### Hypotheses, frozen in a tagged commit before the cloud sweep

- **H1 (near-replication).** Largest model, full window, end of training,
  is equivalent to Bayes at every ρ on the grid.
- **H2 (information or learning).** For each W below full, is the model
  equivalent to window-Bayes at every ρ? Two-sided. If yes, everything a
  window-limited transformer gets "wrong" is the information limit and it
  is rational given what it sees. If no, the sign and the ρ where the gap
  appears are the finding.
- **H3 (the human signature).** At ρ = 0.6 and 0.8, where humans overreact
  and window-Bayes underreacts, is the model's implied-persistence gap
  positive in any converged cell? A yes would be human-like behaviour that
  rational learning cannot produce.

Exploratory, labelled as such: training-time traces of the gap (do
networks pass through an overreacting phase before converging), model
size, a mismatched prior (train on ρ ~ U[0.5,1], test on low ρ), a
mid-sequence break in ρ.

### Real-data comparison (illustrative, not a test)

Feed histories of the SPF variables already replicated in EWH into the
Family B models, scaled using only data before each forecast date, compute
b at the horizons that match the survey, and print it beside the SPF's b
for the same variable-quarters. If horizons or scaling cannot be matched
cleanly the arm is dropped and the write-up says so. Real series are not
AR(1) and the models never saw them.

### Gate before any spend (all local, $0)

1. Bayes benchmark: its own error-on-revision b is zero within interval by
   Monte Carlo when ρ is drawn from the prior (not at a fixed ρ), and it
   collapses to the full-information forecast under a point prior.
   Window-Bayes beats or ties every other forecaster given the same inputs.
2. Planted answer: a simulated forecaster equal to Bayes plus θ times the
   Bayes one-step surprise is recovered as κ = θ within interval, and a
   forecaster with a planted persistence offset is recovered by D2, for
   several values.
3. A small model trained on the Mac reaches under 5% excess error.

The sweep launcher refuses to run unless the gate is green at the current
code hash (same pattern as Scar Tissue).

### Compute

1. Local: gate, plus as much of the small-model grid as is practical.
2. Cloud pilot, about 15 minutes on one A100: measure minutes per run and
   how many runs share the GPU; project the total and report it to AJ.
3. Cloud sweep only if the projection fits the $15 cap; otherwise cut seeds
   or the largest size first and say so. Each run is independent,
   resumable, and synced off the instance as it finishes.

Working estimate, to be replaced by the pilot's number: 5–8 A100 hours.

### Deliverables

- Repo `~/Documents/rational-transformer`, public on GitHub (Claude's call:
  nothing licensed; ask AJ again at the moment of making it public).
- `RESULTS.md` with every number traceable to a CSV, and figures: the
  implied-persistence-gap map by window and ρ, the AKLMT overlay with humans, training-time traces, the
  SPF comparison table.
- A short paper PDF in the house style used for lookahead-probe.
- The BASIS text is AJ's to write; Claude supplies numbers and figures
  (Claude's call, following lookahead-probe; AJ can change it).
- An honest venue verdict after the results are in.

## Left to the builder

Optimiser and schedule, step counts, batch size, evaluation sample size,
grid resolution for the posterior, plotting, test layout, Thunder Compute
CLI specifics (including whether a stopped instance still bills), and the
exact files copied from EWH.
