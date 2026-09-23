# Offer valuation: formula against the day-ahead plan

£k per MW per year, net of cycling costs, split into availability (FR) and trading. Selection is before 2025-01-01; confirmation from it. Engine `ff9b71bb613c`.

| Run | Net (sel) | FR (sel) | Trading (sel) | Net (conf) | FR (conf) | Trading (conf) | Breach periods |
|---|---|---|---|---|---|---|---|
| pf_lp_vint | 133.1 | 57.1 | 79.3 | 81.6 | 38.0 | 46.5 | 317 |
| naive_lp_shrink0.5_bid_vint | 102.1 | 64.7 | 39.9 | 66.1 | 43.1 | 25.5 | 503 |
| ml_lp_shrink0.5_bid_vint | 105.9 | 67.3 | 40.8 | 67.9 | 44.2 | 26.2 | 514 |
| ml_lp_cp0.3per_bid_vint | 103.6 | 65.5 | 40.1 | 67.5 | 41.1 | 28.8 | 638 |
| ml_lp_qr0.2_bid_vint | 101.1 | 67.0 | 35.9 | 66.9 | 42.6 | 26.6 | 994 |
| ml_lp_qr0.3_bid_vint | 102.5 | 66.6 | 37.8 | 67.1 | 42.0 | 27.4 | 781 |
| ml_lp_cqr0.2_bid_vint | 101.6 | 65.4 | 38.0 | 67.0 | 42.0 | 27.2 | 942 |
| ml_lp_cqr0.3_bid_vint | 102.1 | 65.0 | 39.1 | 67.1 | 41.8 | 27.6 | 919 |
| ml_lp_spci0.2_bid_vint | 102.5 | 67.2 | 37.1 | 67.1 | 42.9 | 26.6 | 706 |
| ml_lp_spci0.3_bid_vint | 103.7 | 65.3 | 40.4 | 67.4 | 41.2 | 28.6 | 645 |
| ml_lp_ens0.2_bid_vint | 101.4 | 66.3 | 37.0 | 67.2 | 42.5 | 27.0 | 980 |
| ml_lp_shrink0.5_qr0.3_bid_vint | 100.8 | 68.4 | 34.1 | 66.7 | 44.4 | 24.5 | 832 |
| ml_lp_shrink0.5_spci0.3_bid_vint | 101.4 | 67.9 | 35.3 | 67.0 | 43.6 | 25.7 | 891 |

| Valuation | Foresight ratio (sel) | Foresight ratio (conf) |
|---|---|---|
| lp_cp0.3per_bid_vint | — | — |
| lp_cqr0.2_bid_vint | — | — |
| lp_cqr0.3_bid_vint | — | — |
| lp_ens0.2_bid_vint | — | — |
| lp_qr0.2_bid_vint | — | — |
| lp_qr0.3_bid_vint | — | — |
| lp_shrink0.5_bid_vint | 12.3% | 11.3% |
| lp_shrink0.5_qr0.3_bid_vint | — | — |
| lp_shrink0.5_spci0.3_bid_vint | — | — |
| lp_spci0.2_bid_vint | — | — |
| lp_spci0.3_bid_vint | — | — |

Against the published signals: net £k/MW/yr less the shipped ML run, and the foresight ratio with the shipped naive run as the floor.

| Run | Δ net vs shipped (sel) | Δ net vs shipped (conf) | Foresight (sel) | Foresight (conf) |
|---|---|---|---|---|
| ml_lp_shrink0.5_bid_vint | +0.00 | +0.00 | 12.3% | 11.3% |
| ml_lp_cp0.3per_bid_vint | -2.33 | -0.43 | 4.8% | 8.5% |
| ml_lp_qr0.2_bid_vint | -4.79 | -0.98 | -3.2% | 5.0% |
| ml_lp_qr0.3_bid_vint | -3.39 | -0.83 | 1.4% | 5.9% |
| ml_lp_cqr0.2_bid_vint | -4.36 | -0.90 | -1.8% | 5.5% |
| ml_lp_cqr0.3_bid_vint | -3.78 | -0.79 | 0.1% | 6.2% |
| ml_lp_spci0.2_bid_vint | -3.47 | -0.76 | 1.1% | 6.4% |
| ml_lp_spci0.3_bid_vint | -2.22 | -0.48 | 5.1% | 8.2% |
| ml_lp_ens0.2_bid_vint | -4.53 | -0.74 | -2.3% | 6.5% |
| ml_lp_shrink0.5_qr0.3_bid_vint | -5.11 | -1.23 | -4.2% | 3.4% |
| ml_lp_shrink0.5_spci0.3_bid_vint | -4.49 | -0.87 | -2.2% | 5.7% |
