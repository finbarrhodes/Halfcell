# Model benchmark on walk-forward folds

Refit every 3 months, each origin predicting only the days that follow. Folds before 2025-01-01 are the selection set; the rest are held back as confirmation.

| Model | RMSE (sel) | RMSE (conf) | Spearman (sel) | Spearman (conf) | Spike RMSE (conf) | Fit time |
|---|---|---|---|---|---|---|
| rf | 53.09 | 35.24 | 0.841 | 0.587 | 52.49 | 0.0 min |
| lgb | 54.52 | 34.41 | 0.836 | 0.588 | 51.37 | 1.8 min |
| xgb | 56.63 | 36.32 | 0.833 | 0.559 | 53.52 | 0.5 min |
| lear | 698.58 | 33.24 | 0.794 | 0.647 | 46.39 | 9.3 min |
