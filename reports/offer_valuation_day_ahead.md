# Offer valuation: formula against the day-ahead plan

£k per MW per year, net of cycling costs, split into availability (FR) and trading. Selection is before 2025-01-01; confirmation from it. Engine `90bb8362b5e0`.

| Run | Net (sel) | FR (sel) | Trading (sel) | Net (conf) | FR (conf) | Trading (conf) | Breach periods |
|---|---|---|---|---|---|---|---|
| pf_formula | 119.4 | 58.9 | 63.2 | 73.3 | 41.8 | 34.0 | 355 |
| pf_lp | 133.1 | 57.1 | 79.3 | 81.6 | 38.0 | 46.5 | 317 |
| naive_formula | 93.6 | 59.6 | 36.8 | 62.2 | 41.6 | 23.3 | 471 |
| naive_lp | 98.3 | 56.9 | 44.8 | 65.7 | 38.9 | 29.8 | 422 |
| ml_formula | 98.8 | 66.0 | 35.0 | 64.2 | 42.4 | 24.4 | 494 |
| ml_lp | 104.6 | 63.3 | 43.9 | 67.7 | 40.2 | 30.5 | 402 |

| Valuation | Foresight ratio (sel) | Foresight ratio (conf) |
|---|---|---|
| formula | 20.1% | 17.7% |
| lp | 18.2% | 12.5% |
