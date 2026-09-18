# Offer valuation: formula against the day-ahead plan

£k per MW per year, net of cycling costs, split into availability (FR) and trading. Selection is before 2025-01-01; confirmation from it. Engine `47e7ef659a1e`.

| Run | Net (sel) | FR (sel) | Trading (sel) | Net (conf) | FR (conf) | Trading (conf) | Breach periods |
|---|---|---|---|---|---|---|---|
| pf_formula | 119.4 | 58.9 | 63.2 | 73.3 | 41.8 | 34.0 | 355 |
| pf_formula_vint | 119.4 | 58.9 | 63.2 | 73.3 | 41.8 | 34.0 | 353 |
| naive_formula_bid | 91.1 | 58.3 | 35.5 | 60.1 | 39.8 | 22.8 | 442 |
| naive_formula_bid_vint | 91.6 | 58.8 | 35.5 | 59.9 | 39.8 | 22.7 | 367 |
| ml_formula_bid | 97.2 | 64.9 | 34.4 | 62.7 | 42.0 | 23.2 | 555 |
| ml_formula_bid_vint | 97.7 | 64.9 | 35.0 | 62.7 | 42.0 | 23.3 | 570 |
| pf_lp_vint | 133.1 | 57.1 | 79.3 | 81.6 | 38.0 | 46.5 | 317 |
| naive_lp_bid_vint | 96.5 | 56.8 | 42.9 | 64.0 | 38.7 | 28.2 | 379 |
| naive_lp_shrink0.25_bid_vint | 101.4 | 68.3 | 35.1 | 65.8 | 45.7 | 22.4 | 631 |
| naive_lp_shrink0.5_bid_vint | 102.1 | 64.7 | 40.0 | 66.1 | 43.1 | 25.5 | 503 |
| naive_lp_shrink0.75_bid_vint | 99.7 | 60.5 | 42.2 | 65.5 | 40.4 | 27.7 | 419 |
| ml_lp_bid_vint | 104.6 | 62.9 | 44.3 | 67.3 | 39.4 | 30.8 | 357 |
| ml_lp_shrink0.25_bid_vint | 100.8 | 66.9 | 35.8 | 66.9 | 46.0 | 23.2 | 1105 |
| ml_lp_shrink0.5_bid_vint | 105.9 | 67.3 | 40.8 | 67.9 | 44.2 | 26.2 | 514 |
| ml_lp_shrink0.75_bid_vint | 105.6 | 64.7 | 43.3 | 68.0 | 41.6 | 29.1 | 451 |

| Valuation | Foresight ratio (sel) | Foresight ratio (conf) |
|---|---|---|
| formula_bid | 21.5% | 20.0% |
| formula_bid_vint | 22.0% | 20.8% |
| lp_bid_vint | 22.2% | 18.6% |
| lp_shrink0.25_bid_vint | -1.6% | 7.0% |
| lp_shrink0.5_bid_vint | 12.3% | 11.3% |
| lp_shrink0.75_bid_vint | 17.7% | 16.0% |
