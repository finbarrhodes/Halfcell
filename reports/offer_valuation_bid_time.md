# Offer valuation: formula against the day-ahead plan

£k per MW per year, net of cycling costs, split into availability (FR) and trading. Selection is before 2025-01-01; confirmation from it. Engine `817fbfd91c2f`.

| Run | Net (sel) | FR (sel) | Trading (sel) | Net (conf) | FR (conf) | Trading (conf) | Breach periods |
|---|---|---|---|---|---|---|---|
| pf_formula | 119.4 | 58.9 | 63.2 | 73.3 | 41.8 | 34.0 | 355 |
| pf_lp | 133.1 | 57.1 | 79.3 | 81.6 | 38.0 | 46.5 | 317 |
| naive_formula_bid | 91.1 | 58.3 | 35.5 | 60.1 | 39.8 | 22.8 | 442 |
| naive_lp_bid | 96.7 | 56.7 | 43.1 | 64.3 | 38.7 | 28.4 | 394 |
| ml_formula_bid | 97.2 | 64.9 | 34.4 | 62.7 | 42.0 | 23.2 | 555 |
| ml_lp_bid | 104.2 | 62.9 | 43.9 | 67.2 | 39.4 | 30.7 | 357 |
| naive_lp_shrink0.5_bid | 102.1 | 64.6 | 40.0 | 66.3 | 43.1 | 25.6 | 525 |
| ml_lp_shrink0.5_bid | 105.4 | 67.3 | 40.2 | 67.8 | 44.2 | 26.1 | 490 |

| Valuation | Foresight ratio (sel) | Foresight ratio (conf) |
|---|---|---|
| formula_bid | 21.5% | 20.0% |
| lp_bid | 20.8% | 16.8% |
| lp_shrink0.5_bid | 10.6% | 10.3% |
