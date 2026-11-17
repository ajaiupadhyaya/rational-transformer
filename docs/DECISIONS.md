# Decisions

A running log of choices fixed during the project.

## 2026-11-16: Family B prior constants

Family B draws sigma^2 ~ InvGamma(a=3, b=2) (shape a, scale b; mean b/(a-1) = 1)
and mu | sigma^2 ~ N(0, 25 * sigma^2), i.e. `B_PRIOR = dict(a=3.0, b=2.0, k=25.0)`
in `rt/sim.py`. The series starts at x_0 = mu.
