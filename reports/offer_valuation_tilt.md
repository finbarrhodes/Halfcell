# Offer valuation: formula against the day-ahead plan

£k per MW per year, net of cycling costs, split into availability (FR) and trading. Selection is before 2025-01-01; confirmation from it. Engine `fb8cce78b49d`.

| Run | Net (sel) | FR (sel) | Trading (sel) | Net (conf) | FR (conf) | Trading (conf) | Breach periods |
|---|---|---|---|---|---|---|---|
| pf_lp_vint | 133.1 | 57.1 | 79.3 | 81.6 | 38.0 | 46.5 | 317 |
| naive_lp_shrink0.5_bid_vint | 102.1 | 64.7 | 40.0 | 66.1 | 43.1 | 25.5 | 503 |
| ml_lp_shrink0.5_bid_vint | 105.9 | 67.3 | 40.8 | 67.9 | 44.2 | 26.2 | 514 |
| naive_lp_a0.5b+0.3_bid_vint | 101.2 | 63.0 | 40.8 | 66.0 | 42.5 | 25.9 | 466 |
| naive_lp_a0.5b+0.6_bid_vint | 100.2 | 61.2 | 41.7 | 65.6 | 42.0 | 26.0 | 446 |
| naive_lp_a0.5b-0.3_bid_vint | 102.3 | 66.3 | 38.4 | 66.2 | 43.9 | 24.8 | 551 |
| naive_lp_a0.6b+0.4_bid_vint | 99.9 | 60.8 | 41.9 | 65.4 | 41.4 | 26.6 | 436 |
| ml_lp_a0.5b+0.3_bid_vint | 105.5 | 67.1 | 40.6 | 67.8 | 44.0 | 26.3 | 523 |
| ml_lp_a0.5b+0.6_bid_vint | 105.0 | 66.9 | 40.2 | 67.6 | 43.6 | 26.5 | 510 |
| ml_lp_a0.5b-0.3_bid_vint | 105.4 | 66.7 | 40.8 | 68.0 | 44.2 | 26.2 | 590 |
| ml_lp_a0.6b+0.4_bid_vint | 105.3 | 65.9 | 41.6 | 67.8 | 42.9 | 27.5 | 480 |

| Valuation | Foresight ratio (sel) | Foresight ratio (conf) |
|---|---|---|
| lp_a0.5b+0.3_bid_vint | 13.6% | 11.7% |
| lp_a0.5b+0.6_bid_vint | 14.6% | 12.8% |
| lp_a0.5b-0.3_bid_vint | 9.9% | 11.6% |
| lp_a0.6b+0.4_bid_vint | 16.4% | 14.6% |
| lp_shrink0.5_bid_vint | 12.3% | 11.3% |
