# Research Experiments

What the forecast is actually worth, and the experiments measured against it — most of which did
not ship. Each entry says what was tried, what happened, why, and what would change the answer.
All of them run the same engine over the same backtest, so the runs compare directly; code and
reports are in the [repository](https://github.com/finbarrhodes/Halfcell).

## What the forecast is worth

The **foresight ratio** is the share of the gap between floor and ceiling a forecast closes,
`(ML − Naive) / (Perfect Foresight − Naive)` on net revenue. Here it is about 13%: the model is
worth about £3k/MW/yr over the floor, in every year of the backtest, against a ceiling £26k above
it. The industry's usual measure is a different one: Percent of Perfect, revenue as a share of
perfect foresight with no floor subtracted, on which the model scores 80.9%. On a stacked battery
that says little, because reusing the last complete day's prices already scores 78.2%: most of the
revenue is response availability that no forecast moves. Subtracting the floor asks the narrower
question of how much of the *capturable* gap a forecast closes. On arbitrage alone, the footing
closest to a price-forecasting study, Percent of Perfect is 50.1% against the floor's 37.9%.

| £k / MW / yr | Frequency response | Trading | Wear | Net |
|---|---|---|---|---|
| Perfect foresight | 51.7 | 70.2 | −3.3 | 118.5 |
| Naive | 58.6 | 36.7 | −2.6 | 92.7 |
| ML model | 60.6 | 37.6 | −2.4 | 95.9 |

**The model and the ceiling earn through different channels.** The whole of perfect foresight's
advantage is trading, while the model earns most of its lead through response (+£2.1k, against
+£0.9k at trading): it values each block's arbitrage more accurately when the offers are made, and
holds better positions. Beating persistence at trading needs a forecast that identifies *which*
half-hours will be extreme; lowering average error across all of them does not do that, and the
offer stage consumes a block-level summary that a sharper half-hourly curve barely moves. That is
why [a much more accurate model earned far less](#why-the-model-is-not-chosen-by-accuracy), and
[a much better forecast earned nothing more](#a-better-forecast-that-earned-nothing).

**How much of it is noise.** Not much. Resampling the paired daily revenue differences in
four-week blocks, because a dispatch decision carries state into the next day
([Künsch, 1989](https://doi.org/10.1214/aos/1176347265)), puts the model's lead at £3.26k/MW/yr
(95% interval £2.13k to £4.45k), and £2.26k [£1.23k, £3.45k] from 2025 alone. Its *accuracy* edge
over persistence, though, is not significant: a
[Diebold-Mariano test](https://doi.org/10.1080/07350015.1995.10524599) on paired daily losses gives
p = 0.26 on squared error and p = 0.19 on the error in the day's spread. Spike days dominate
squared-error differences, while the revenue difference is a small, repeated gain from holding
better response positions: the forecast earns through allocation rather than precision.

## Why the model is not chosen by accuracy

<p class="muted">Random Forest kept · 17 Sep 2026</p>

Four forecasters were benchmarked on the same walk-forward folds, with the folds from 2025 held
back so the choice could not be made on the evidence used to report it. The most accurate was the
worst earner, and not marginally. Accuracy is scored on the held-back folds, revenue over the full
backtest under the per-block offer rule of the time:

| Model | RMSE | Spearman ρ | Spike RMSE | £k/MW/yr | Foresight ratio |
|---|---|---|---|---|---|
| Random Forest | 35.2 | 0.587 | 52.5 | **87.3** | **19.7%** |
| LightGBM | 34.4 | 0.588 | 51.4 | 87.0 | 18.2% |
| XGBoost | 36.3 | 0.559 | 53.5 | — | — |
| LEAR | **33.2** | **0.647** | **46.4** | 64.1 | −91.5% |

LEAR wins every accuracy column and earns £19k/MW/yr *less than reusing yesterday's prices*. The
cause is calibration in the one dimension the decisions consume: LEAR over-predicts the daily
spread by £272/MWh across the backtest, and by £29.5 even in the calm recent market, while the
trees under-predict it (Random Forest by £31.5). For a price-taker that asymmetry is protective. A
spread that fails to arrive does its damage at the offer stage, where an inflated arbitrage value
declines response contracts worth having. Dispatch is barely touched: it trades on the *ordering* of
periods and settles at the realised price (see
[why only the offer](#discounting-the-forecast-at-the-offer)), so there over-prediction costs only
the extra cycling it talks the optimiser into. From 2025 LEAR held 23 MW of Low
products against the forest's 30, sat out 30% of EFA blocks against 12%, and gave up £0.71M of
availability revenue to gain £0.07M of trading while cycling 48% more energy. Clipping its
forecasts to the price range seen before each origin recovered almost nothing: the problem is
systematic bias, not the tail.

**Since then.** Spike RMSE is blind to this failure, since it scores only spikes that happened, so
spread calibration is reported beside the accuracy metrics on the
[Forecasting & Dispatch](./backtester) page. **Revisit if** a candidate's spread bias is corrected;
its accuracy lead would then be worth testing again.

## A better forecast that earned nothing

<p class="muted">Not adopted · the wind collector and features stay in the repository · 17 Sep 2026</p>

Adding NESO's day-ahead wind forecast cut RMSE on the held-back folds by 16% (35.2 to 29.7), lifted
rank correlation from 0.587 to 0.699 and improved error on spikes by 13%; revenue moved from £87.3k
to £87.1k per MW per year. At the time the model traded £0.8k/MW/yr *worse* than naive and earned
its whole lead through response, so a sharper half-hourly curve had nothing to act on. Adopting it
would add a monthly data dependency for no measured gain. **Revisit if** dispatch learns to abstain
— trading only where the forecast is confident enough to beat persistence, which is queued — since
sharpness may matter once the forecast is allowed to decline a trade.

## Discounting the forecast at the offer

<p class="muted">A constant discount adopted; four refinements not · 18–21 Sep 2026</p>

A plan built on a forecast does not merely inherit its errors, it *selects* them: it commits
capacity to the half-hours where the forecast shows the widest spread, which are disproportionately
the ones the forecast flattered, so its estimate of what free capacity is worth is biased upward
even when the forecast is unbiased — the
[optimiser's curse](https://doi.org/10.1287/mnsc.1050.0451) (Smith & Winkler, 2006), whose remedy is
to discount before choosing. The cost is lopsided too: a spread that fails to arrive costs the trade
*and* the response contract declined to keep capacity free. Halving the forecast's deviations is
worth about £4k/MW/yr to naive and £1k to the model, chosen on the folds before 2025.

Four ways to do better than one constant all lost to it: a weight fitted per day by the
[Mincer & Zarnowitz (1969)](https://www.nber.org/books-and-chapters/economic-forecasts-and-expectations-analysis-forecasting-behavior-and-performance/evaluation-economic-forecasts)
slope, a weight leaning on how loud the day looked, conformal
[guard bands](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6598009) per settlement period,
and spreading trades across neighbouring half-hours. Revenue is sensitive to *how much* trading
value is discounted and nearly flat in how the discount is shaped. The forecast's weakness is timing
rather than size: it gets a day's magnitude roughly right, and picks the peak half-hour within one
period only 30–40% of the time.

**Why only the offer.** Dispatch is prone to the same bias, but its mistakes are revisable — it
re-solves every half-hour — and cost the trade alone, and it turns on the *ordering* of periods,
which a uniform discount leaves untouched. At the deadline the same discount rescales trading value
against clearing prices, and so changes which side wins the capacity; the offer also works from the
weaker forecast (RMSE 57.7 against 48.5). Dispatch needs the *hour* right instead. The device that
suits it, spreading a trade across hours the forecast cannot tell apart, is worth £0.64k/MW/yr
[£0.32k, £0.97k] to naive, whose shape is two days stale, and nothing to the model (−£0.10k
[−£0.46k, £0.23k]). It is not shipped: the gain accrues to the floor, which is more useful left
plain.

**Revisit if** the offer stops treating the forecast as certain. A plan that optimises over
scenarios could shrink hard on some days and barely on others, which one constant cannot.

## Interval forecasts as guard bands

<p class="muted">Not adopted ·
<a href="https://github.com/finbarrhodes/Halfcell/blob/main/reports/interval_benchmark.md">interval scores</a> ·
<a href="https://github.com/finbarrhodes/Halfcell/blob/main/reports/offer_valuation_quantile.md">revenue runs</a> ·
23 Sep 2026</p>

[O'Connor et al. (2025)](https://arxiv.org/abs/2502.04935) benchmark probabilistic forecasts of
Irish day-ahead and balancing prices — quantile regression, split conformal prediction, EnbPI,
[SPCI](https://arxiv.org/abs/2212.03463) and an average of them — and find the conformal methods
give more reliable intervals and more trading profit. The offer plan can already trade against a
band, selling against its lower edge and buying against its upper one, so the question was whether
a better band earns more than the constant discount. Four were built walk-forward on bid-time
information: the forest's own quantiles (a
[quantile regression forest](https://www.jmlr.org/papers/v7/meinshausen06a.html) with the shipped
forest's settings), those quantiles conformalised
([CQR](https://arxiv.org/abs/1905.03222)), SPCI on the forecast's residuals, and the paper's
average. Each was scored as a forecast on 1,631 days, and run through the full engine.

| 60% band | Winkler score | Coverage | Revenue, to 2024 | Revenue, 2025 on |
|---|---|---|---|---|
| Split conformal (the earlier attempt) | 121.7 | 0.630 | −3.8 | −0.7 |
| Quantile forest | **110.6** | 0.652 | −4.8 | −1.0 |
| CQR | 111.8 | **0.597** | −4.4 | −0.9 |
| SPCI | 119.2 | 0.688 | −3.5 | −0.8 |
| Average (the paper's ensemble) | 113.9 | 0.673 | −4.5 | −0.7 |

*The Winkler score is an interval's width plus a penalty for every price outside it; lower is
better. Coverage should be 0.60. Revenue is £k/MW/yr against the shipped model.* Every band
lost to the constant, at 40% bands too, and the best-scored lost the most. The quantile forest is right that loud days are uncertain: its band is
twice as wide on the fifth of days with the biggest realised spreads. Those are the days whose
spread is real, so the better the band, the more of the best trading it declines. SPCI's forests
split to predict the error's mean, so they follow a forecast that keeps missing one way, which
made it the best band in 2022 and in no other year, and not one that is merely noisier.

Two cautions on the paper itself, both reproduced from its own linked repository by
`scripts/verify_interval_literature.py`. Its EnbPI and SPCI "0.1–0.9" bounds are nominally *90%*
intervals: the reference implementation it calls builds them from the residual percentiles
[β, 1−α+β], so its `alpha` is the total miscoverage, while the paper defines α per tail (α = 0.1
→ 80%). They are then compared against quantile regression's genuine 80%. And on the
random-forest day-ahead forecasts it publishes, the forest's own quantiles score better than
either conformal method — coverage 0.86 against 0.75 and 0.71 for a labelled 0.80, and an
interval score of 47 against 55 and 56, the score being width plus 2/α per unit of price outside
the band. The paper's own table ranks them the other way (33.7 against 32.1 and 31.7); those
values do not reproduce from the published forecasts under the usual conventions, and that
ranking appears only if the miscoverage penalty is left unscaled by 1/α, which charges a band
little for missing the price. That is a reading of their table, not a claim about their code.

**Revisit if** the plan becomes scenario-based, using the quantiles as a distribution to optimise
over rather than as a discount, or if a band is built around the decision itself — how likely a
block's spread is to beat its clearing prices — rather than around the price. The quantile forest,
CQR and SPCI are in `src/analysis/`, ready to reuse.

## Timing recovery through the reserve

<p class="muted">Adopted: recovery credited on the strict reading, with a margin at block starts · 24 Sep 2026</p>

After EAC, where Low holdings and the reserve for High ones take the whole discharge rating, the
reserve is the only way out for energy DR High absorbs. The plan used to count what reserve flows
cost and never what they earned, so it sold that energy whenever headroom ran short, whatever the
price: over the backtest the reserve sold at £65-67/MWh while trading sold at £129-146, and it
carried 3-11% of all discharge.

Letting recovery earn at the price fixes the timing but invites the reserve to trade, so it earns
only on energy delivery has put in play, kept in an account per side. Which MWh leaves the store
cannot be known, so the account has two readings that bracket the answer: spent only by the
reserve's own flows (loose), or by every trade first (strict). Recovery that waits for a price also
runs the store up to the headroom a new block restores, and a burst of DR delivery then starts the
block outside its requirement; tried alone on one quarter, the credit doubled breaches. So every plan
now meets each later block's start a margin inside it. Measured apart over the backtest, in
£k/MW/yr and breach periods:

| | Before | Margin alone | Margin + credit, strict | Margin + credit, loose |
|---|---|---|---|---|
| Perfect foresight | 117.8 · 317 | 117.4 · 63 | 118.5 · 71 | 119.2 · 74 |
| Naive | 91.6 · 503 | 91.3 · 146 | 92.7 · 163 | 93.0 · 168 |
| ML model | 94.8 · 514 | 94.3 · 205 | 95.9 · 227 | 96.4 · 250 |

The margin is a compliance device: 60-80% fewer breach periods for under £0.5k/MW/yr of trading.
The revenue is the credit's. Neither moves what the forecast is worth outside its interval (the
model's lead is £3.00k with the margin alone and £3.26k with both), because the untimed reserve
cost every signal about alike. The strict reading ships, so the reserve never earns on energy
trading could have brought in. **Revisit if** a real operator's recovery accounting is known, since
the loose reading is worth another £0.3k-0.7k/MW/yr, or if a delivery forecast can replace the
recent 90th percentile that sets the margin.

## What moved the headline

The foresight ratio has moved more from corrections to the engine than from anything done to the
forecast. Grouped by what each correction was about:

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

A better engine made the forecast matter less; it did not make the forecast worse. All of it landed
between 15 and 24 September 2026; the [repository](https://github.com/finbarrhodes/Halfcell) has the
order it landed in.
