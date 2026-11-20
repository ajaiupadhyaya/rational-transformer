# BASIS hand-off: figures and numbers

Source of every number: `RESULTS.md` and `results/hypotheses.json`, which match the paper.
Figures are in `figures/`.

## The story in three sentences

When people forecast simple time series, they overreact to the latest number, and more so when
the series is noisy. A rational forecaster who has to learn how persistent a series is also
looks "biased" against the truth, so the fair comparison is with a perfect Bayesian who sees
exactly the same data. Small AI models trained only to minimise forecast error end up matching
that Bayesian almost exactly, and they never show the human-style overreaction.

## Headline numbers

| What | Number |
|---|---|
| Largest model with full memory, verdicts matching the exact Bayesian (H1) | 12 of 12 |
| Memory lengths (2, 4, 8, 16) where the model matches a Bayesian with the same memory (H2) | 4 of 4 |
| Places where the model overreacts beyond that Bayesian, at persistence 0.6 and 0.8 (H3) | 0 of 8 |
| How close: largest gap between model and Bayesian at 0.6–0.8, upper end of the 95% interval | +0.0008, against a +0.02 threshold set in advance |
| Training runs, all finished | 96 of 96 |
| Model sizes and memory lengths | 4 sizes, 5 memory lengths, 3 to 5 random seeds each |
| Cloud GPU cost | about $14.35 (about 41 GPU-hours); small models trained on a laptop |
| Rules fixed before any result was seen | yes, pre-registration tagged 17 November 2026 |

## The caveats that keep it honest

- **Training length.** The match only appears at the end of training, when the learning rate
  has decayed to zero. At a third of the way through, the models were still clearly off.
- **Size.** The smallest model (1 layer) does not match. It pulls low-persistence series
  up and mid-persistence series down, toward a middling value, which is the opposite of the human pattern. Every size from 2 layers up
  matches.
- **The human comparison.** People's answers sit close to a Bayesian with a short memory and an
  unknown mean. The paper does not claim people are irrational beyond that.
- **Scope.** This is simulated data and one model design. It says nothing about ChatGPT-style
  models or about real people.

## Figures

| File | Use it for | One-line caption |
|---|---|---|
| `fig2_human_overlay.png` | the lead image | People (red) against the AI models and the matching Bayesians: the models sit on their Bayesian curves. |
| `fig1_gap_map.png` | "how close is close" | At every memory length the model stays far inside the ±0.02 band fixed in advance. |
| `fig4_size.png` | the size story | Only the smallest model leaves the band, and on the underreaction side. |
| `fig3_training_traces.png` | the training-length caveat | Wide disagreement early in training collapses onto the Bayesian by the end. |

The paper numbers these figures by order of appearance: fig1 is Figure 1, fig4 is Figure 2,
fig3 is Figure 3 and fig2 is Figure 4.
