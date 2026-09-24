# Error bars on the forecast's value

Paired daily comparisons of the shipped Random Forest against reusing an older day's prices, on the engine as published. Blocks of 28 days, 20,000 resamples.

## Is it more accurate? Diebold-Mariano on paired daily losses

| Stage | Loss | Mean gain | t | p |
|---|---|---|---|---|
| dispatch | squared error | 227.3 | 1.14 | 2.56e-01 |
| dispatch | spread error | -2.6 | -1.32 | 1.88e-01 |
| offer | squared error | 41.5 | 0.12 | 9.03e-01 |
| offer | spread error | 0.4 | 0.16 | 8.69e-01 |

A positive mean gain means the model loses less than naive. The dispatch row uses the day-ahead vintage each strategy dispatches on; the offer row uses the bid-time vintage offers see.

## Is it worth more money? Block bootstrap of the daily revenue difference

| Window | ML − naive (£k/MW/yr) | 95% interval | P(≤ 0) | Days |
|---|---|---|---|---|
| all | 3.26 | [+2.13, +4.45] | 0.000 | 1797 |
| selection | 3.76 | [+2.16, +5.38] | 0.000 | 1203 |
| confirmation | 2.26 | [+1.23, +3.45] | 0.000 | 594 |
