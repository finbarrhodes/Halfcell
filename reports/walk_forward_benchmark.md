# Model benchmark on walk-forward folds

Refit every 3 months, each origin predicting only the days that follow. Folds before 2025-01-01 are the selection set; the rest are held back as confirmation.

Spread bias is reported beside the accuracy metrics because it is the only one that charges a forecast for spread it invents: LEAR won every other column here and earned £19k/MW/yr less than reusing yesterday's prices.

| Model | RMSE (sel) | RMSE (conf) | Spearman (sel) | Spearman (conf) | Spike RMSE (conf) | Spread bias (conf) | Spread MAE (conf) | Fit time |
|---|---|---|---|---|---|---|---|---|
| rf | 53.09 | 35.24 | 0.841 | 0.587 | 52.49 | -12.55 | 29.53 | 0.0 min |
| rf-residual | 56.98 | 27.45 | 0.832 | 0.673 | 38.09 | -18.9 | 29.76 | 0.0 min |
| hgb | 53.05 | 33.64 | 0.835 | 0.605 | 49.15 | -2.01 | 27.75 | 0.0 min |
| hgb-pinball | 51.68 | 27.39 | 0.852 | 0.669 | 42.38 | -22.95 | 30.61 | 0.0 min |

| Model | £k/MW/yr | Foresight ratio | Unavailable periods |
|---|---|---|---|
| rf | 87.3 | -11.0% | 494 |
| rf-residual | 86.7 | -13.4% | 543 |
| hgb | 86.3 | -14.7% | 402 |
| hgb-pinball | 89.4 | -2.8% | 599 |
