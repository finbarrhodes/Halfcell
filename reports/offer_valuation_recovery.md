# Offer valuation: formula against the day-ahead plan

£k per MW per year, net of cycling costs, split into availability (FR) and trading. Selection is before 2025-01-01; confirmation from it. Engine `2baaca73b85e`.

| Run | Net (sel) | FR (sel) | Trading (sel) | Net (conf) | FR (conf) | Trading (conf) | Breach periods |
|---|---|---|---|---|---|---|---|
| pf_lp_vint | 133.1 | 57.1 | 79.3 | 81.6 | 38.0 | 46.5 | 317 |
| naive_lp_shrink0.5_bid_vint | 102.1 | 64.7 | 39.9 | 66.1 | 43.1 | 25.5 | 503 |
| ml_lp_shrink0.5_bid_vint | 105.9 | 67.3 | 40.8 | 67.9 | 44.2 | 26.2 | 514 |
| pf_lp_rec_vint | 133.7 | 57.2 | 79.9 | 84.6 | 38.2 | 49.5 | 74 |
| naive_lp_shrink0.5_rec_bid_vint | 103.1 | 64.8 | 40.9 | 68.3 | 43.3 | 27.8 | 168 |
| ml_lp_shrink0.5_rec_bid_vint | 106.9 | 67.3 | 41.8 | 70.7 | 44.4 | 29.1 | 250 |
| pf_lp_recany_vint | 133.3 | 57.2 | 79.5 | 83.4 | 38.2 | 48.3 | 71 |
| naive_lp_shrink0.5_recany_bid_vint | 102.8 | 64.8 | 40.5 | 67.9 | 43.3 | 27.3 | 163 |
| ml_lp_shrink0.5_recany_bid_vint | 106.5 | 67.3 | 41.4 | 70.1 | 44.4 | 28.3 | 227 |

| Valuation | Foresight ratio (sel) | Foresight ratio (conf) |
|---|---|---|
| lp_shrink0.5_bid_vint | 12.3% | 11.3% |
| lp_shrink0.5_rec_bid_vint | 12.3% | 14.7% |
| lp_shrink0.5_recany_bid_vint | 12.2% | 14.2% |

Against the published signals: net £k/MW/yr less the shipped ML run, and the foresight ratio with the shipped naive run as the floor.

| Run | Δ net vs shipped (sel) | Δ net vs shipped (conf) | Foresight (sel) | Foresight (conf) |
|---|---|---|---|---|
| ml_lp_shrink0.5_bid_vint | +0.00 | +0.00 | 12.3% | 11.3% |
