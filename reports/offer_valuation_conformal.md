# Offer valuation: formula against the day-ahead plan

£k per MW per year, net of cycling costs, split into availability (FR) and trading. Selection is before 2025-01-01; confirmation from it. Engine `9e9ef40025b8`.

| Run | Net (sel) | FR (sel) | Trading (sel) | Net (conf) | FR (conf) | Trading (conf) | Breach periods |
|---|---|---|---|---|---|---|---|
| pf_lp_vint | 133.1 | 57.1 | 79.3 | 81.6 | 38.0 | 46.5 | 317 |
| naive_lp_shrink0.5_bid_vint | 102.1 | 64.7 | 40.0 | 66.1 | 43.1 | 25.5 | 503 |
| ml_lp_shrink0.5_bid_vint | 105.9 | 67.3 | 40.8 | 67.9 | 44.2 | 26.2 | 514 |
| ml_lp_cp0.1per_bid_vint | 100.9 | 67.3 | 35.4 | 67.0 | 43.0 | 26.2 | 924 |
| ml_lp_cp0.2per_bid_vint | 102.1 | 66.7 | 37.3 | 67.2 | 41.8 | 27.8 | 840 |
| ml_lp_cp0.3per_bid_vint | 103.6 | 65.5 | 40.1 | 67.5 | 41.1 | 28.8 | 638 |
| ml_lp_cp0.2blo_bid_vint | 102.3 | 66.9 | 37.4 | 67.2 | 42.0 | 27.6 | 757 |
| ml_lp_shrink0.5_cp0.2per_bid_vint | 100.9 | 68.3 | 34.4 | 66.8 | 43.8 | 25.3 | 1018 |
| naive_lp_cp0.1per_bid_vint | 100.4 | 68.8 | 33.6 | 64.9 | 45.4 | 21.8 | 495 |
| naive_lp_cp0.2per_bid_vint | 101.2 | 66.4 | 37.1 | 65.2 | 43.7 | 23.9 | 497 |
| naive_lp_cp0.2blo_bid_vint | 101.3 | 66.4 | 37.2 | 65.4 | 43.7 | 24.0 | 502 |
| naive_lp_shrink0.5_cp0.2per_bid_vint | 99.2 | 69.0 | 32.1 | 64.7 | 46.6 | 20.3 | 748 |

| Valuation | Foresight ratio (sel) | Foresight ratio (conf) |
|---|---|---|
| lp_cp0.1per_bid_vint | 1.6% | 12.6% |
| lp_cp0.2blo_bid_vint | 3.1% | 11.2% |
| lp_cp0.2per_bid_vint | 2.5% | 12.2% |
| lp_cp0.3per_bid_vint | — | — |
| lp_shrink0.5_bid_vint | 12.3% | 11.3% |
| lp_shrink0.5_cp0.2per_bid_vint | 5.0% | 12.8% |
