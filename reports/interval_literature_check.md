# Checking two claims about O'Connor et al. (2025)

Reproduced by `scripts/verify_interval_literature.py` from the paper's own linked
repository and from the reference implementation it calls.

## The conformal bands are nominally 90% and 70%, not 80% and 40%

- The paper calls the reference implementation with alpha = 0.1, 0.3
  (`EnbPI_SPCI_DAM.py`), having defined the interval as `[q_alpha, q_(1-alpha)]` with
  confidence `1 - 2*alpha` — so alpha = 0.1 is labelled 80%.
- The reference implementation builds the bounds at residual percentiles
  `['beta_hat_bin', '(1 - alpha + beta_hat_bin']`, a probability width of `1 - alpha`.
- So alpha is the total miscoverage: the bands are 90% and 70% intervals, compared
  against quantile regression's genuine 80% and 40%.

## On its published random-forest forecasts, the forest's quantiles score better

Day-ahead market, 365 days x 24 hours. Interval score is width plus `2/alpha` per unit
of price outside the band, at the labelled level; lower is better.

| Method | Coverage (labelled 0.80) | Mean width | Interval score |
|---|---|---|---|
| QR | 0.860 | 33.1 | 47.3 |
| EnbPI | 0.748 | 21.9 | 55.4 |
| SPCI | 0.714 | 20.5 | 55.7 |

The conformal bands are narrower than quantile regression's and cover less, despite
being nominally wider — the mislabelling is not what flatters them.

**A caveat carried on the methodology page.** The paper's Table 3 ranks the same three
the other way for this model and market: QR 33.7, EnbPI 32.14, SPCI 31.65. None of the usual scoring
conventions reproduces those values from the published forecasts. The ranking does
appear if the miscoverage penalty is left unscaled by `1/alpha`, which charges a band
little for missing the price:

| Method | Interval score, penalty unscaled |
|---|---|
| QR | 36.0 |
| EnbPI | 28.6 |
| SPCI | 27.5 |

That is a reading of their table, not a claim about their code.
