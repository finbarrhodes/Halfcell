# Feature ablation on walk-forward folds

Spread bias and MAE first: they are the statistics that decide whether a forecast
helps a battery, and the only ones that charge it for spread it invents. Figures are
the confirmation folds (from 2025-01-01); the selection folds drove nothing here
but are in the JSON.

| Model | Features | Spread bias | Spread MAE | RMSE | Spearman | Spike RMSE |
|---|---|---|---|---|---|---|
| rf | baseline | -12.55 | 29.53 | 35.24 | 0.587 | 52.49 |
| rf | wind | -16.6 | 28.89 | 29.67 | 0.699 | 45.9 |

| Model | Features | £k/MW/yr | Foresight ratio |
|---|---|---|---|
| rf | baseline | 87.3 | 19.7% |
| rf | wind | 87.1 | 18.8% |
