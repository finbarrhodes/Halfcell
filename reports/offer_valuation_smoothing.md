# Offer valuation: formula against the day-ahead plan

£k per MW per year, net of cycling costs, split into availability (FR) and trading. Selection is before 2025-01-01; confirmation from it. Engine `ec5033e61596`.

| Run | Net (sel) | FR (sel) | Trading (sel) | Net (conf) | FR (conf) | Trading (conf) | Breach periods |
|---|---|---|---|---|---|---|---|
| pf_lp_vint | 133.1 | 57.1 | 79.3 | 81.6 | 38.0 | 46.5 | 317 |
| naive_lp_shrink0.5_bid_vint | 102.1 | 64.7 | 40.0 | 66.1 | 43.1 | 25.5 | 503 |
| ml_lp_shrink0.5_bid_vint | 105.9 | 67.3 | 40.8 | 67.9 | 44.2 | 26.2 | 514 |
| ml_lp_shrink0.5_sm3_bid_vint | 105.8 | 67.3 | 40.6 | 68.0 | 44.2 | 26.3 | 538 |
| ml_lp_shrink0.5_sm5_bid_vint | 105.7 | 67.1 | 40.5 | 68.1 | 44.2 | 26.3 | 560 |
| ml_lp_shrink0.5_sm9_bid_vint | 103.2 | 67.2 | 38.0 | 67.7 | 44.3 | 25.9 | 533 |
| naive_lp_shrink0.5_sm3_bid_vint | 102.5 | 64.7 | 40.2 | 66.3 | 43.1 | 25.7 | 523 |
| naive_lp_shrink0.5_sm5_bid_vint | 102.9 | 64.7 | 40.5 | 66.5 | 43.1 | 25.7 | 485 |
| naive_lp_shrink0.5_sm9_bid_vint | 100.6 | 64.6 | 38.1 | 66.0 | 43.1 | 25.2 | 491 |
| ml_lp_shrink0.5_osm5_bid_vint | 105.1 | 67.5 | 39.6 | 67.6 | 44.7 | 25.3 | 652 |
| ml_lp_sm5_bid_vint | 104.8 | 62.9 | 44.2 | 67.6 | 39.4 | 31.0 | 352 |

| Valuation | Foresight ratio (sel) | Foresight ratio (conf) |
|---|---|---|
| lp_shrink0.5_bid_vint | 12.3% | 11.3% |
| lp_shrink0.5_osm5_bid_vint | — | — |
| lp_shrink0.5_sm3_bid_vint | 11.0% | 11.1% |
| lp_shrink0.5_sm5_bid_vint | 9.2% | 11.0% |
| lp_shrink0.5_sm9_bid_vint | 8.1% | 11.0% |
| lp_sm5_bid_vint | — | — |
