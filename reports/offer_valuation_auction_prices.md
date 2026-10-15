# Offer valuation: formula against the day-ahead plan

_Written by `python scripts/compare_offer_valuation.py --runs pf:lp:recany:vint,naive:lp:0.5:recany:bid:vint,ml:lp:0.5:recany:bid:vint,ml:lp:0.5:recany:bid:vint:dacost,ml:lp:0.5:recany:bid:vint:daplan,ml:lp:0.5:recany:bid:vint:daboth,naive:lp:0.5:recany:bid:vint:dacost,naive:lp:0.5:recany:bid:vint:daplan,naive:lp:0.5:recany:bid:vint:daboth,ml:lp:0.5:recany:bid:vint:fcstcost,naive:lp:0.5:recany:bid:vint:fcstcost,pf:lp:recany:vint:dacost,pf:lp:recany:vint:daboth --jobs 4 --report offer_valuation_auction_prices` on 2026-10-15, at commit `29a08f4f185c` with uncommitted changes; engine `8ac0d5b10dd0`._

£k per MW per year, net of cycling costs, split into availability (FR) and trading. Selection is before 2025-01-01; confirmation from it. Engine `8ac0d5b10dd0`.

| Run | Net (sel) | FR (sel) | Trading (sel) | Net (conf) | FR (conf) | Trading (conf) | Breach periods |
|---|---|---|---|---|---|---|---|
| pf_lp_recany_vint | 133.3 | 57.2 | 79.4 | 87.2 | 39.5 | 50.8 | 73 |
| naive_lp_shrink0.5_recany_bid_vint | 102.8 | 64.8 | 40.6 | 70.5 | 44.8 | 28.3 | 168 |
| ml_lp_shrink0.5_recany_bid_vint | 106.5 | 67.3 | 41.4 | 72.7 | 46.2 | 29.1 | 235 |
| ml_lp_shrink0.5_recany_bid_vint_dacost | 107.5 | 64.7 | 45.0 | 76.1 | 30.4 | 48.2 | 267 |
| ml_lp_shrink0.5_recany_bid_vint_daplan | 109.4 | 64.9 | 46.7 | 74.5 | 45.6 | 31.4 | 177 |
| ml_lp_shrink0.5_recany_bid_vint_daboth | 111.0 | 62.0 | 51.3 | 78.2 | 29.7 | 51.0 | 151 |
| naive_lp_shrink0.5_recany_bid_vint_dacost | 104.0 | 62.3 | 44.3 | 74.2 | 30.0 | 46.8 | 180 |
| naive_lp_shrink0.5_recany_bid_vint_daplan | 106.8 | 64.7 | 44.6 | 72.9 | 45.5 | 29.9 | 176 |
| naive_lp_shrink0.5_recany_bid_vint_daboth | 108.3 | 61.9 | 48.9 | 76.7 | 29.6 | 49.6 | 143 |
| ml_lp_shrink0.5_recany_bid_vint_fcstcost | 103.6 | 76.9 | 28.8 | 68.5 | 75.5 | -4.4 | 365 |
| naive_lp_shrink0.5_recany_bid_vint_fcstcost | 101.7 | 63.8 | 40.5 | 69.2 | 40.6 | 31.2 | 197 |
| pf_lp_recany_vint_dacost | 134.7 | 54.7 | 83.4 | 90.0 | 25.1 | 67.9 | 79 |
| pf_lp_recany_vint_daboth | 128.3 | 54.3 | 77.2 | 86.0 | 24.5 | 64.5 | 54 |

| Valuation | Foresight ratio (sel) | Foresight ratio (conf) |
|---|---|---|
| lp_shrink0.5_recany_bid_vint | 12.1% | 13.1% |
| lp_shrink0.5_recany_bid_vint_daboth | 10.8% | 14.2% |
| lp_shrink0.5_recany_bid_vint_dacost | 12.0% | 14.1% |
| lp_shrink0.5_recany_bid_vint_daplan | 9.7% | 11.0% |
| lp_shrink0.5_recany_bid_vint_fcstcost | 6.1% | -3.6% |

Against the published signals: net £k/MW/yr less the shipped run for the same signal, and the foresight ratio with the shipped naive run as the floor.

| Run | Δ net vs shipped (sel) | Δ net vs shipped (conf) | Foresight (sel) | Foresight (conf) |
|---|---|---|---|---|
| naive_lp_shrink0.5_recany_bid_vint | +0.00 | +0.00 | 0.0% | 0.0% |
| ml_lp_shrink0.5_recany_bid_vint | +0.00 | +0.00 | 12.1% | 13.0% |
| ml_lp_shrink0.5_recany_bid_vint_dacost | +1.02 | +3.37 | 15.5% | 33.2% |
| ml_lp_shrink0.5_recany_bid_vint_daplan | +2.91 | +1.78 | 21.7% | 23.7% |
| ml_lp_shrink0.5_recany_bid_vint_daboth | +4.54 | +5.48 | 27.0% | 45.8% |
| naive_lp_shrink0.5_recany_bid_vint_dacost | +1.19 | +3.71 | 3.9% | 22.2% |
| naive_lp_shrink0.5_recany_bid_vint_daplan | +4.05 | +2.39 | 13.3% | 14.3% |
| naive_lp_shrink0.5_recany_bid_vint_daboth | +5.53 | +6.16 | 18.2% | 36.9% |
| ml_lp_shrink0.5_recany_bid_vint_fcstcost | -2.91 | -4.16 | 2.6% | -11.8% |
| naive_lp_shrink0.5_recany_bid_vint_fcstcost | -1.15 | -1.33 | -3.8% | -8.0% |
