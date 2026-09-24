# Offer valuation: formula against the day-ahead plan

£k per MW per year, net of cycling costs, split into availability (FR) and trading. Selection is before 2025-01-01; confirmation from it. Engine `aa57ca6b2248`.

| Run | Net (sel) | FR (sel) | Trading (sel) | Net (conf) | FR (conf) | Trading (conf) | Breach periods |
|---|---|---|---|---|---|---|---|
| pf_lp_margin_vint | 132.8 | 57.2 | 78.9 | 81.0 | 38.2 | 45.7 | 63 |
| naive_lp_shrink0.5_margin_bid_vint | 102.0 | 64.8 | 39.7 | 65.5 | 43.2 | 24.7 | 146 |
| ml_lp_shrink0.5_margin_bid_vint | 105.6 | 67.3 | 40.4 | 67.1 | 44.4 | 25.1 | 205 |

| Valuation | Foresight ratio (sel) | Foresight ratio (conf) |
|---|---|---|
| lp_shrink0.5_margin_bid_vint | 11.9% | 10.0% |
