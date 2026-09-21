# Offer valuation: formula against the day-ahead plan

£k per MW per year, net of cycling costs, split into availability (FR) and trading. Selection is before 2025-01-01; confirmation from it. Engine `a2434c73551a`.

| Run | Net (sel) | FR (sel) | Trading (sel) | Net (conf) | FR (conf) | Trading (conf) | Breach periods |
|---|---|---|---|---|---|---|---|
| pf_lp_vint | 133.1 | 57.1 | 79.3 | 81.6 | 38.0 | 46.5 | 317 |
| naive_lp_shrink0.5_bid_vint | 102.1 | 64.7 | 40.0 | 66.1 | 43.1 | 25.5 | 503 |
| ml_lp_shrink0.5_bid_vint | 105.9 | 67.3 | 40.8 | 67.9 | 44.2 | 26.2 | 514 |
| naive_lp_dyn1_bid_vint | 100.9 | 66.3 | 36.9 | 66.2 | 43.8 | 24.9 | 666 |
| naive_lp_dyn0.75_bid_vint | 99.6 | 66.5 | 35.3 | 66.3 | 44.9 | 23.8 | 835 |
| naive_lp_dyn0.5_bid_vint | 98.5 | 66.7 | 33.7 | 65.9 | 45.5 | 22.7 | 1016 |
| ml_lp_dyn1_bid_vint | 104.7 | 64.8 | 42.4 | 67.8 | 41.2 | 29.4 | 407 |
| ml_lp_dyn0.75_bid_vint | 104.9 | 65.5 | 41.7 | 68.3 | 43.2 | 27.7 | 579 |
| ml_lp_dyn0.5_bid_vint | 103.8 | 66.6 | 39.3 | 67.7 | 44.8 | 25.4 | 752 |

| Valuation | Foresight ratio (sel) | Foresight ratio (conf) |
|---|---|---|
| lp_dyn0.5_bid_vint | 15.4% | 11.6% |
| lp_dyn0.75_bid_vint | 15.8% | 12.7% |
| lp_dyn1_bid_vint | 11.6% | 10.8% |
| lp_shrink0.5_bid_vint | 12.3% | 11.3% |
