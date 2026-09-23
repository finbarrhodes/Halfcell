# Interval benchmark: guard bands scored as forecasts

Bid-time Random Forest forecast, APXMIDP, walk-forward. Every method is scored on the days all of them band. Alpha is per tail: the target coverage is 1 − 2α. Pinball is the mean over the two bounds and Winkler the interval score; for both, lower is better. Built by `scripts/interval_benchmark.py`.

## α = 0.2 (60% interval), 1631 days, 2022-03-01 to 2026-08-17

| Method | Coverage | Below | Above | Mean width | Pinball | Winkler |
|---|---|---|---|---|---|---|
| scp | 0.630 | 0.175 | 0.195 | 65.2 | 12.17 | 121.7 |
| qr | 0.652 | 0.155 | 0.193 | 65.6 | 11.06 | 110.6 |
| spci | 0.688 | 0.141 | 0.171 | 69.8 | 11.92 | 119.2 |
| spci_w | 0.688 | 0.140 | 0.171 | 69.7 | 11.85 | 118.5 |

Winkler by year:

| Method | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| scp | 253.6 | 108.8 | 79.3 | 83.7 | 94.1 |
| qr | 253.2 | 84.6 | 69.4 | 74.2 | 85.4 |
| spci | 242.0 | 102.1 | 82.9 | 85.7 | 94.2 |
| spci_w | 239.1 | 101.3 | 82.6 | 85.6 | 94.1 |
