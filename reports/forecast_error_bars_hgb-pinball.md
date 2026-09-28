# Error bars on the forecast's value

Paired daily comparisons of the candidate model `hgb-pinball` against reusing an older day's prices, on the engine as published. Blocks of 28 days, 20,000 resamples.

## Is it more accurate? Diebold-Mariano on paired daily losses

| Stage | Loss | Mean gain | t | p |
|---|---|---|---|---|
| dispatch | squared error | 474.8 | 2.30 | 2.14e-02 |
| dispatch | spread error | -5.4 | -2.69 | 7.29e-03 |
| offer | squared error | 269.9 | 0.61 | 5.41e-01 |
| offer | spread error | -4.2 | -1.72 | 8.64e-02 |

A positive mean gain means the model loses less than naive. The dispatch row uses the day-ahead vintage each strategy dispatches on; the offer row uses the bid-time vintage offers see.

## Is it worth more money? Block bootstrap of the daily revenue difference

| Window | ML − naive (£k/MW/yr) | 95% interval | P(≤ 0) | Days |
|---|---|---|---|---|
| all | 2.12 | [-0.08, +4.03] | 0.029 | 1797 |
| selection | 2.39 | [-0.82, +5.18] | 0.064 | 1203 |
| confirmation | 1.58 | [+0.56, +2.61] | 0.001 | 594 |
