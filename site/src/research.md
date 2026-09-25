# Research Experiments

This page collects the experiments behind the model used on the rest of the site: what each one
tried, what it measured, and why most of them were not adopted. Every run uses the same engine over
the same backtest, so the numbers here can be compared with each other and with the headline
figures elsewhere. The code and the full reports are in the
[repository](https://github.com/finbarrhodes/Halfcell).

## What the forecast is worth

Measuring what a forecast is worth needs a scale, and the natural one runs between the two
strategies that bracket it. The **foresight ratio** is the share of that gap the model closes,
`(ML − Naive) / (Perfect Foresight − Naive)` on net revenue. It comes out at about 13%: the model
is worth around £3k/MW/yr over the floor, in every year of the backtest, against a ceiling £26k
above it.

The industry more often quotes Percent of Perfect, which is revenue as a share of perfect foresight
with no floor subtracted, and on that measure the model scores 80.9%. On a stacked battery that
number says very little, because simply reusing the last complete day's prices already scores
78.2%. Most of the revenue is response availability, and no forecast moves that. Subtracting the
floor asks the narrower and more useful question of how much of the *capturable* gap a forecast
closes. On arbitrage alone, which is the footing closest to a price-forecasting study, Percent of
Perfect is 50.1% against the floor's 37.9%.

| £k / MW / yr | Frequency response | Trading | Wear | Net |
|---|---|---|---|---|
| Perfect foresight | 51.7 | 70.2 | −3.3 | 118.5 |
| Naive | 58.6 | 36.7 | −2.6 | 92.7 |
| ML model | 60.6 | 37.6 | −2.4 | 95.9 |

**The model and the ceiling earn through different channels.** All of perfect foresight's advantage
comes from trading, but the model earns most of its lead from response instead: +£2.1k against
+£0.9k at trading. It values each block's arbitrage more accurately at the moment the offers are
made, and so holds better positions. Beating persistence at trading is a harder problem, because it
needs a forecast that identifies *which* half-hours will be extreme, and lowering average error
across all of them does not do that. The offer stage only ever sees a block-level summary, which a
sharper half-hourly curve barely changes. That pattern is behind two of the results below, where
[a much more accurate model earned far less](#why-the-model-is-not-chosen-by-accuracy) and
[a much better forecast earned nothing more](#a-better-forecast-that-earned-nothing).

**The lead holds up against its own uncertainty.** Resampling the paired daily revenue differences
in four-week blocks, rather than single days, because a dispatch decision carries state into the
next one ([Künsch, 1989](https://doi.org/10.1214/aos/1176347265)), puts the model's lead at
£3.26k/MW/yr with a 95% interval of £2.13k to £4.45k, and £2.26k [£1.23k, £3.45k] from 2025 alone.
Its *accuracy* edge over persistence does not reach significance, though: a
[Diebold-Mariano test](https://doi.org/10.1080/07350015.1995.10524599) on paired daily losses gives
p = 0.26 on squared error and p = 0.19 on the error in the day's spread. The two findings sit
together comfortably. Spike days dominate the squared-error comparison, while the revenue
difference is a small gain repeated across many days, earned by holding better response positions
rather than by predicting prices more precisely.

## Why the model is not chosen by accuracy

<p class="muted">Random Forest kept · 17 Sep 2026</p>

Four forecasters were benchmarked on the same walk-forward folds, with the folds from 2025 held
back so that the choice could not be made on the same evidence used to report it. The most accurate
of the four turned out to be the worst earner, and by a wide margin. Accuracy below is scored on
the held-back folds, and revenue over the full backtest under the per-block offer rule in force at
the time:

| Model | RMSE | Spearman ρ | Spike RMSE | £k/MW/yr | Foresight ratio |
|---|---|---|---|---|---|
| Random Forest | 35.2 | 0.587 | 52.5 | **87.3** | **19.7%** |
| LightGBM | 34.4 | 0.588 | 51.4 | 87.0 | 18.2% |
| XGBoost | 36.3 | 0.559 | 53.5 | — | — |
| LEAR | **33.2** | **0.647** | **46.4** | 64.1 | −91.5% |

LEAR wins every accuracy column and still earns £19k/MW/yr *less than reusing yesterday's prices*.
The reason is calibration, in the one dimension the decisions actually depend on. LEAR
over-predicts the daily spread by £272/MWh across the backtest, and by £29.5 even in the calm
recent market, while the trees under-predict it, the Random Forest by £31.5. For a price-taker,
under-predicting is the safer error to make. A spread that fails to arrive does its damage at the
offer stage, where an inflated arbitrage value leads the battery to decline response contracts
worth having. Dispatch is barely affected, because it trades on the *ordering* of periods and
settles at the realised price (see
[why only the offer](#discounting-the-forecast-at-the-offer)), so over-prediction there costs only
the extra cycling the optimiser is talked into. From 2025 LEAR held 23 MW of Low products against
the forest's 30, sat out 30% of EFA blocks against 12%, and gave up £0.71M of availability revenue
to gain £0.07M of trading, while cycling 48% more energy. Clipping its forecasts to the price range
seen before each origin recovered almost nothing, which points to systematic bias rather than to a
handful of extreme days.

**Since then.** Spike RMSE is blind to this failure, because it scores only the spikes that
actually happened, so spread calibration is now reported beside the accuracy metrics on the
[Forecasting & Dispatch](./backtester) page. **Revisit if** a candidate's spread bias is corrected,
as its accuracy lead would then be worth testing again.

## A better forecast that earned nothing

<p class="muted">Not adopted · the wind collector and features stay in the repository · 17 Sep 2026</p>

Adding NESO's day-ahead wind forecast cut RMSE on the held-back folds by 16%, from 35.2 to 29.7,
lifted rank correlation from 0.587 to 0.699 and improved error on spikes by 13%. Revenue moved from
£87.3k to £87.1k per MW per year. At the time the model traded £0.8k/MW/yr *worse* than naive and
earned its whole lead through response, so a sharper half-hourly curve had nothing to act on.
Adopting it would have added a monthly data dependency for no measured gain. **Revisit if** dispatch
learns to abstain, trading only where the forecast is confident enough to beat persistence. That
work is queued, and sharpness may begin to matter once the forecast is allowed to decline a trade.

## Discounting the forecast at the offer

<p class="muted">A constant discount adopted; four refinements not · 18–21 Sep 2026</p>

A plan built on a forecast does more than inherit the forecast's errors: it actively selects them.
The plan commits capacity to the half-hours where the forecast shows the widest spread, and those
are disproportionately the half-hours the forecast flattered, so its estimate of what free capacity
is worth comes out biased upward even when the forecast itself is unbiased. This is the
[optimiser's curse](https://doi.org/10.1287/mnsc.1050.0451) (Smith & Winkler, 2006), and the
standard remedy is to discount before choosing. The cost here is lopsided as well, because a spread
that fails to arrive loses the trade and also the response contract that was declined to keep the
capacity free. Halving the forecast's deviations is worth about £4k/MW/yr to naive and £1k to the
model, with the weight chosen on the folds before 2025.

Four attempts to do better than a single constant all lost to it: a weight fitted per day by the
[Mincer & Zarnowitz (1969)](https://www.nber.org/books-and-chapters/economic-forecasts-and-expectations-analysis-forecasting-behavior-and-performance/evaluation-economic-forecasts)
slope, a weight that leaned on how volatile the day looked, conformal
[guard bands](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6598009) per settlement period,
and spreading trades across neighbouring half-hours. Revenue turns out to be sensitive to *how
much* trading value is discounted and nearly flat in how the discount is shaped. The forecast's
weakness is in timing rather than size: it gets a day's magnitude roughly right, but picks the peak
half-hour to within one period only 30–40% of the time.

**Why only the offer.** Dispatch is prone to the same bias, but its mistakes are revisable, since
it re-solves every half-hour, and they cost the trade alone. Dispatch also turns on the *ordering*
of periods, which a uniform discount leaves untouched. At the bid deadline the same discount does
real work, rescaling trading value against clearing prices and so changing which side wins the
capacity, and the offer stage is working from the weaker forecast in any case (RMSE 57.7 against
48.5). What dispatch needs instead is the right *hour*. The device that suits it, spreading a trade
across hours the forecast cannot tell apart, is worth £0.64k/MW/yr [£0.32k, £0.97k] to naive, whose
shape is two days stale, and nothing to the model (−£0.10k [−£0.46k, £0.23k]). It is not shipped,
because the gain accrues to the floor, and the floor is more useful left plain.

**Revisit if** the offer stops treating the forecast as certain. A plan that optimises over
scenarios could shrink hard on some days and barely on others, which one constant cannot.

## Interval forecasts as guard bands

<p class="muted">Not adopted ·
<a href="https://github.com/finbarrhodes/Halfcell/blob/main/reports/interval_benchmark.md">interval scores</a> ·
<a href="https://github.com/finbarrhodes/Halfcell/blob/main/reports/offer_valuation_quantile.md">revenue runs</a> ·
23 Sep 2026</p>

[O'Connor et al. (2025)](https://arxiv.org/abs/2502.04935) benchmark probabilistic forecasts of
Irish day-ahead and balancing prices — quantile regression, split conformal prediction, EnbPI,
[SPCI](https://arxiv.org/abs/2212.03463) and an average of them — and find that the conformal
methods give more reliable intervals and more trading profit. The offer plan can already trade
against a band, selling against its lower edge and buying against its upper one, so the question
was whether a better band earns more than the constant discount does. Four were built walk-forward
on bid-time information: the forest's own quantiles (a
[quantile regression forest](https://www.jmlr.org/papers/v7/meinshausen06a.html) with the shipped
forest's settings), those quantiles conformalised
([CQR](https://arxiv.org/abs/1905.03222)), SPCI on the forecast's residuals, and the paper's
average. Each was scored as a forecast on 1,631 days, then run through the full engine.

| 60% band | Winkler score | Coverage | Revenue, to 2024 | Revenue, 2025 on |
|---|---|---|---|---|
| Split conformal (the earlier attempt) | 121.7 | 0.630 | −3.8 | −0.7 |
| Quantile forest | **110.6** | 0.652 | −4.8 | −1.0 |
| CQR | 111.8 | **0.597** | −4.4 | −0.9 |
| SPCI | 119.2 | 0.688 | −3.5 | −0.8 |
| Average (the paper's ensemble) | 113.9 | 0.673 | −4.5 | −0.7 |

*The Winkler score is an interval's width plus a penalty for every price outside it; lower is
better. Coverage should be 0.60. Revenue is £k/MW/yr against the shipped model.* Every band lost to
the constant discount, at 40% bands as well as 60%, and the band that scored best as a forecast
lost the most revenue. The quantile forest is right that volatile days carry more uncertainty, and
its band is twice as wide on the fifth of days with the biggest realised spreads. But those are
exactly the days whose spread turns out to be real, so a wider band there declines the best trading
of the year. SPCI behaves differently again: its forests split to predict the error's mean, so they
follow a forecast that keeps missing in one direction. That made SPCI the best band in 2022 and in
no other year, which says more about that market than about the band being noisier.

Two cautions on the paper itself, both reproduced from its own linked repository by
`scripts/verify_interval_literature.py`. Its EnbPI and SPCI "0.1–0.9" bounds are nominally *90%*
intervals: the reference implementation it calls builds them from the residual percentiles
[β, 1−α+β], so its `alpha` is the total miscoverage, while the paper defines α per tail (α = 0.1
→ 80%). They are then compared against quantile regression's genuine 80%. And on the
random-forest day-ahead forecasts it publishes, the forest's own quantiles score better than
either conformal method — coverage 0.86 against 0.75 and 0.71 for a labelled 0.80, and an
interval score of 47 against 55 and 56, the score being width plus 2/α per unit of price outside
the band. The paper's own table ranks them the other way (33.7 against 32.1 and 31.7). Those
values do not reproduce from the published forecasts under the usual conventions, and that
ranking appears only if the miscoverage penalty is left unscaled by 1/α, which charges a band
very little for missing the price. This is a reading of their published table rather than a claim
about their code.

**Revisit if** the plan becomes scenario-based, using the quantiles as a distribution to optimise
over rather than as a discount, or if a band is built around the decision itself — how likely a
block's spread is to beat its clearing prices — rather than around the price. The quantile forest,
CQR and SPCI are in `src/analysis/`, ready to reuse.

## Timing recovery through the reserve

<p class="muted">Adopted: recovery credited on the strict reading, with a margin at block starts · 24 Sep 2026</p>

After EAC, when Low holdings and the reserve held for High products take up the whole discharge
rating, the reserve becomes the only route out for the energy DR High absorbs. The plan used to
count what flows through the reserve cost and never what they earned, so it sold that energy
whenever headroom ran short, whatever the price happened to be. Over the backtest the reserve sold
at £65–67/MWh while ordinary trading sold at £129–146, and it carried 3–11% of all discharge.

Letting recovery earn at the price fixes the timing, but it also invites the reserve to start
trading, so it earns only on energy that delivery has put in play, tracked in an account per side.
Which particular MWh leaves the store cannot be known, so the account is kept two ways that bracket
the answer: spent only by the reserve's own flows (loose), or by every trade first (strict).
Recovery that waits for a good price has a side effect of its own. It runs the store up towards the
headroom a new block restores, and a burst of DR delivery can then start that block outside its
requirement; tried on its own over one quarter, the credit doubled breaches. Every plan therefore
now arrives at each later block's start with a margin inside the requirement. The two changes,
measured separately over the backtest, in £k/MW/yr and breach periods:

| | Before | Margin alone | Margin + credit, strict | Margin + credit, loose |
|---|---|---|---|---|
| Perfect foresight | 117.8 · 317 | 117.4 · 63 | 118.5 · 71 | 119.2 · 74 |
| Naive | 91.6 · 503 | 91.3 · 146 | 92.7 · 163 | 93.0 · 168 |
| ML model | 94.8 · 514 | 94.3 · 205 | 95.9 · 227 | 96.4 · 250 |

The margin is there for compliance rather than revenue: it removes 60–80% of breach periods for
under £0.5k/MW/yr of trading, and the revenue gain belongs to the credit. Neither change moves what
the forecast is worth beyond its interval, as the model's lead is £3.00k with the margin alone and
£3.26k with both, because the untimed reserve had been costing all three signals about equally. The
strict reading is the one that ships, so the reserve never earns on energy that ordinary trading
could have brought in. **Revisit if** a real operator's recovery accounting becomes known, since
the loose reading is worth another £0.3k–0.7k/MW/yr, or if a delivery forecast can replace the
recent 90th percentile that currently sets the margin.

## What moved the headline

The foresight ratio has moved further on corrections to the engine than on anything done to the
forecast itself. The changes below are grouped by what each one was really about.

**What the contracts require**

| Correction | Effect |
|---|---|
| Allocation follows NESO's rules, replacing a proportional split that sold the full rating into all six products | FR availability had been overstated about 1.5× |
| Response delivery modelled from one-second frequency | Arbitrage revenue fell for every strategy, narrowing the gap between floor and ceiling |
| Recovery through the reserve earns at the price, within what delivery put in play, and every plan keeps a margin at each new block's start ([timing recovery](#timing-recovery-through-the-reserve)) | +£0.8k perfect foresight, +£1.1k naive, +£1.1k the model; breach periods down 55–78% |

**What each decision may know**

| Correction | Effect |
|---|---|
| Walk-forward retraining replaced one fixed train/test split | The forecast had looked worth £18.8k/MW/yr on the 42 months it trained on, against £2.3k on the 18 held out; the ratio fell from about 66% to 20% |
| Offers and tomorrow's dispatch see only forecasts that existed at the time | Offers: −£2.4k naive, −£1.6k the model; dispatch: no measurable change |

**What free capacity is worth**

| Correction | Effect |
|---|---|
| Offers priced from a day-ahead trading plan rather than one cycle per block (from 2025 a block's own spread averaged £15.5/MWh, the day's £68.9) | +£11.9k/MW/yr for perfect foresight, +£9.1k naive, +£7.3k the model |
| The plan's forecast shrunk halfway to its daily mean ([discounting the forecast](#discounting-the-forecast-at-the-offer)) | +£4k naive, +£1k the model; the model's lead narrowed from £4.1k to £3.2k as the ceiling pulled away |

Taken together, these corrections made the forecast matter less without making the forecast any
worse, because the engine around it began getting more of the revenue right on its own. All of them
landed between 15 and 24 September 2026, and the
[repository](https://github.com/finbarrhodes/Halfcell) has the order they landed in.
