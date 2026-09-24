# Methodology & Data Sources

How the backtest works, what its price forecast is worth, what has been tried and set aside,
and where the data comes from. [The model](#the-model) and [the price forecast](#the-price-forecast)
describe what runs; [what the forecast is worth](#what-the-forecast-is-worth) reads the results;
the [case studies](#case-studies) keep the experiments that were measured and not adopted, with
enough detail to pick them up again.

```js
const coverage = await FileAttachment("data/coverage.json").json();
const manifest = await FileAttachment("data/manifest.json").json();
const p = manifest.ml_mpc.params;
const fmt = (d) => new Date(d).toLocaleDateString("en-GB", {year: "numeric", month: "short"});
const pct = (x) => `${(x * 100).toFixed(0)}%`;
```

## The model

Every day the battery makes two decisions on the same price forecast: at the bid deadline, what
frequency response (FR) to offer for tomorrow, and every half-hour, how to trade the power its
contracts leave free. Revenue is always settled at actual prices.

| Reference asset | |
|---|---|
| Size | ${p.power_mw} MW / ${p.power_mw * p.duration_h} MWh, starting half full |
| Round-trip efficiency | ${pct(p.efficiency_rt)}, applied on the charge side |
| Wear | £${p.cycling_cost_per_mwh} per MWh discharged, whether by a trade or by delivery |
| Availability | ${pct(p.availability_factor)}, applied to every revenue stream and cost |
| Backtest | ${fmt(p.start_date)} – ${fmt(p.end_date)} |
| Wholesale price | APXMIDP market index |

### Stage 1: what to offer

Offers for all six EFA blocks of day D close the afternoon before: 14:00 on D-1 since the
Enduring Auction Capability (EAC) went live on 2 November 2023, 14:30 before it. At that moment
the model decides how many MW to hold in each of the six products (DC, DM and DR, each High and
Low) in each block.

It offers every product at its **opportunity cost**, the trading the same capacity would
otherwise do, and works that out the way a dispatcher would: by planning. At the deadline it
chooses the six blocks' holdings together with a half-hourly trading plan to the end of day D,
on the forecast that existed at that moment, so a holding costs whatever it takes out of that
plan. Three things follow that one price per block cannot capture:

- **Direction depends on the hour.** Overnight the plan charges, which High products get in the
  way of; at the evening peak it sells, which Low products get in the way of.
- **State of charge matters.** Holding High overnight costs nothing if the store is already
  nearly full, and a good deal if it is empty.
- **Values do not add.** Two evening peaks are substitutes, so keeping both free is worth one
  sale of the store rather than two. A product's first megawatts are often free and its last
  dear: the stepped price-quantity shape EAC orders are built for.

Each offer also carries the expected cost of the energy the product will deliver (see
[response delivery](#response-delivery)). NESO accepts an order when the clearing price covers
its offer, so a battery bidding at cost ends up holding whichever permitted combination earns
most at the clearing prices. Solving the plan against them reproduces cost-reflective bidding
without assuming foresight, for a price-taker whose offers do not move clearing prices. The plan
is a linear programme of about 270 variables, solved in two milliseconds by
[HiGHS](https://doi.org/10.1007/s12532-017-0130-5), a simplex solver, because holdings often tie
and only a vertex solution breaks the ties cleanly.

Two refinements. The plan's forecast is pulled halfway towards its daily mean first, because a
plan built on a forecast overstates what keeping capacity free is worth (see
[discounting the forecast at the offer](#discounting-the-forecast-at-the-offer)); the weight was
chosen from 0.25, 0.5, 0.75 and 1 on the years before 2025, and held up from 2025. And energy
left at the end of day D is valued at the day's mean forecast price less wear. Valued at nothing,
the plan would empty the store in the last block and make holding Low response there look
expensive for no real reason.

### NESO's rules

`src/analysis/neso_rules.py` holds each rule with its clause. They decide which combinations are
permitted:

- **Capacity in each direction.** MW offered into Low products, plus the Reserved Capacity held
  for any High products, must fit within the discharge rating; High MW plus the reserve for Low
  products must fit within the charge rating
  ([Procurement Rules](https://www.neso.energy/document/378246/download) 8.3.3.2 and Schedule 1).
  The reserve is 10% of the offered MW for DC, 20% for DM and 40% for DR, held in the opposite
  direction for energy recovery. (The Service Terms' 20% is a different figure, the Energy
  Recovery volume, for all three services.) So a MW sold into DC Low cannot also be sold into
  DR Low, though it can back a High product at the same time.
- **Energy.** Each Low contract needs its delivery energy in store, and each High contract the
  same again as headroom: MW × 15 minutes for DC, 30 for DM, 60 for DR
  ([Service Terms](https://www.neso.energy/document/384606/download) 6.11). Both must fit in the
  battery together.
- **Reachability.** All six blocks are solved together from the battery's actual state of energy
  at the deadline and the commitments it still holds for the rest of D-1. Every block's
  required range must be reachable using only the power the contracts leave free; some
  combinations use the whole rating in both directions, so state of energy cannot move at all
  while they run.
- **Maximum Sell Size.** No more than 100 MW in any one product.
- **Auction size — a modelling limit, not a NESO rule.** No more than 20% of any auction's
  cleared volume. Before EAC the DM and DR auctions were small (median DM Low cleared 4 MW, DR
  Low 55 MW), and without a limit a 50 MW battery would have held more than NESO bought in the
  whole auction in 14–45% of the blocks where it held them. Beyond about a fifth of an auction,
  one unit's offer could plausibly set the price, which a price-taker cannot. Since EAC the
  limit rarely binds. Re-running NESO's clearing with the battery's offers inserted into the
  published order books would measure this directly; that is planned.

Two of these rules changed during the backtest:

| Service days from | Rule |
|---|---|
| Start of data (Sep 2021) | One service per unit per EFA block: DC, DM or DR, with High and Low of that service allowed together ([DR Auction Rules](https://www.neso.energy/document/246746/download) 7.3.1) |
| 2 Nov 2023, EAC go-live | Capacity can be split across all three services in the same block. The model holds Reserved Capacity from here |
| 15 Nov 2024 | Reserved Capacity becomes a rule ([Procurement Rules v2.0](https://www.neso.energy/document/347456/download)) |

**Reserved Capacity before it was a rule.** NESO introduced the reserve together with the
running energy requirement described under [response delivery](#response-delivery): proposed
in June 2024, binding from 15 November 2024. The model applies that requirement throughout, and
a stack across services can use the whole rating in both directions, leaving no power to
recover the energy it delivers: without the reserve the battery would have missed its
requirement in about a third of half-hours between EAC go-live and November 2024. A careful
operator would not hold such a stack, so the model holds the reserve from EAC go-live. Before
EAC a unit held one service per block and stayed within its requirement without it.

### Pre-EAC service choice

Before 2 November 2023 a unit could offer only one of DC, DM and DR into each EFA block, and had
to pick before the auction cleared. For each block the model picks the service that would have
earned most at the **previous day's clearing prices** for the same block, the latest a bidder
had at the 14:30 deadline, and is paid the day's actual prices for what it offered. That uses
only what a bidder knew.

Operators behaved differently. Across 247,893 unit-blocks of NESO's legacy order data (March 2022
– October 2023, from the [NESO Data Portal](https://www.neso.energy/data-portal)), no unit offered
two services into one block and 86.5% of orders went in after the previous day's results; but 91%
of unit-blocks repeated the previous day's service, 82% were DC, and units picked yesterday's best
payer no more often (33%) than the day's eventual best (32%). A plausible reason is duration: the
fleet averaged about 1.1 hours in 2022
([Modo Energy](https://modoenergy.com/research/modo-battery-energy-storage-year-review-2023-capacity-revenues-frequency-response)),
and DR's 60-minute requirement binds hard on a one-hour battery, while the 2-hour reference battery
holds DR comfortably. A sensitivity run holding every pre-EAC block in DC was dropped: a full-power
DC stack in both directions cannot recover the energy it delivers, and spent a fifth of the pre-EAC
half-hours unavailable.

### Stage 2: dispatch

Dispatch is a rolling **Model Predictive Control (MPC) linear programme**, re-solved every
settlement period. At each period it plans to the end of tomorrow, the furthest any forecast yet
exists for (between 49 and ${p.horizon} periods), executes only the first period's decision,
and re-solves: dispatch must be committed before future prices are known.

```
maximise  Σ price[t] × (p_dis[t] − p_chg[t]) × 0.5h  −  cycling_cost × Σ p_dis[t] × 0.5h
```

subject to the state-of-charge equation, the physical store, the power the contracts leave free
on each side, and the state-of-energy range the contracts require at the start of every period,
which makes the battery pre-position for upcoming blocks. That range is soft: missing it costs
£5,000 per MWh, about 2.5× the highest price in the data, so the LP always moves towards
compliance, never breaches a contract to capture a spread, and stays solvable when a requirement
genuinely cannot be reached. A larger penalty would buy no safety; at £50,000 the solver
returned 1% of solves as inaccurate. Charging and discharging at once is never optimal at a
positive spread, so no binary variables are needed. Solved with the Clarabel interior-point
solver through cvxpy ([Diamond & Boyd, 2016](https://www.jmlr.org/papers/v17/15-408.html)).

A day's commitments enter dispatch at its bid deadline: an operator must be able to deliver
everything it offered, so it positions for its offers before results publish, and for a
price-taker the offers are exactly what clears. A settlement period that starts outside the
requirement counts as unavailable
([Service Terms](https://www.neso.energy/document/384606/download) 6.12) and loses an eighth of
the block's availability payment; the [Forecasting & Dispatch](./backtester) page reports how many
periods each strategy missed. Trades execute at actual prices, so a forecast error can lose money
on the day; that is intended.

### Response delivery

A contract is not only a promise to stand ready. Whenever frequency leaves the deadband, a
battery holding response has to deliver, and the energy that moves changes its state of charge.
The model works that energy out from GB frequency itself.

**From frequency to energy.** NESO publishes
[system frequency](https://www.neso.energy/data-portal/system-frequency-data) at one-second
resolution. Each second goes through each service's response curve
([Service Terms](https://www.neso.energy/document/384606/download), Table 1) and is summed into
MWh per MW contracted for every settlement period, service and direction. Low products answer
frequency below 50 Hz by discharging; High products answer frequency above it by charging.

| Service | Nothing within | 5% at | 100% at |
|---|---|---|---|
| DC | ±0.015 Hz | ±0.2 Hz | ±0.5 Hz |
| DM | ±0.015 Hz | ±0.1 Hz | ±0.2 Hz |
| DR | ±0.015 Hz | — (a straight line) | ±0.2 Hz |

Frequency sits outside DR's deadband most of the time, so DR moves far more energy than DC: about
2.5 MWh per MW per day in each direction in 2021, rising to 3.6 in 2026, against 0.13–0.18 for DC.

**Who pays for that energy.** Nobody. Delivery volumes adjust the unit's imbalance position
(Service Terms 16), so energy a Low product gives away is gone and has to be bought back, and
energy a High product absorbs arrives free and can be sold on. Every MWh discharged in delivery
also wears the battery.

**The requirement moves with delivery.** The Minimum State of Energy Requirement (Service Terms
6.11) starts each block at the Contracted Response Energy Volume, falls by the energy delivered
in each settlement period, and climbs back by the Energy Recovery Adjustment Volume: the
shortfall three periods earlier, at most 20% of the volume per period. The Service Terms' own
example, which the tests reproduce: a 50 MWh contract that delivers 2 MWh in its first half-hour
needs 48 MWh until the sixth period, then 50 again. The Reserved Capacity exists to recover that
energy ([SOE Monitoring Guidance](https://www.neso.energy/document/347241/download)), so dispatch
may use it in any period to keep a requirement reachable, including ahead of a block that
restores the full requirement. The plan counts what energy moved through the reserve costs, never
what it earns, so the reserve is not used to trade. A requirement at or below zero counts as
allowed unavailability.

**Planning without knowing frequency.** Within a block, delivery lowers state of charge and the
requirement together, so the LP plans on no further delivery. Where a later block restores the
full requirement, it plans on delivery continuing at the previous day's average.

**Pricing delivery into offers.** Each product's expected delivery is its average over the
previous 28 days for that EFA block, valued at the block's average price over the previous 7
days, *p̄*. Holding a MW of a Low product costs `E_low × (p̄ / η + wear)`, the energy bought back
plus wear; holding a MW of a High product earns `η × E_high × (p̄ − wear)`, the free energy sold
on less wear. That is why the model will hold DR High at a negative price when the energy it
absorbs is worth more than the price it pays, and why DR Low is worth much less than its clearing
price suggests.

**A site with no interest in arbitrage.** The FR-only scenario follows every rule above, gives
arbitrage no value when choosing offers, and trades only to keep its contracts deliverable,
moving as little energy as its requirement needs. Those trades still settle at market prices,
so its net revenue is availability payments less the cost of recovering delivered energy,
losses and wear.

### Revenue accounting

- **Availability.** `clearing price (£/MW/h) × MW held × 4 hours` per EFA block, less an eighth
  of the block's payment for each unavailable settlement period. High and Low name the frequency
  excursion, not the power direction: a Low product (DCL, DML, DRL) discharges when frequency
  falls, a High product (DCH, DMH, DRH) charges when it rises. The same MW can back a High and a
  Low product at once, never two in the same direction. Availability revenue differs slightly
  between strategies, because the forecast sets what each product is offered at and dispatch
  sets the state of energy each day's offers start from.
- **Wholesale trading.** Settled each period at the APXMIDP market index (APX Power UK, via
  Elexon): `revenue[t] = price[t] × (e_dis[t] − e_chg[t])`. It is the GB spot reference, and
  gives a more realistic spread than the imbalance price (SSP), which can reach extreme negative
  values in high-renewable periods and would inflate arbitrage.
- **Delivery.** Energy moved when a contract is called on is neither paid nor charged;
  recovering it shows up in trading and wear.
- **Wear.** Every MWh discharged costs £${p.cycling_cost_per_mwh}, by an arbitrage trade, a
  recovery trade or delivery (for a battery holding DR, delivery is most of it). It enters the
  LP objective, so the optimiser avoids unprofitable cycles. It is a financial proxy for cycle
  ageing, SEI growth and lithium plating, which consume a finite cycle life
  ([Xu et al., 2018](https://arxiv.org/abs/1703.07968);
  [Lee & Kim, 2022](https://doi.org/10.1016/j.ijepes.2021.107795)).
- **Availability factor.** A uniform ${pct(p.availability_factor)} on every stream and cost,
  standing in for outages and faults at the rate the GB fleet is generally reported to run at.
  It is an assumption: NESO sets no minimum availability percentage, and reduces the
  Availability Payment for declared unavailability and under-delivery
  ([Service Terms](https://www.neso.energy/document/384606/download) 5 and 7). Missed
  state-of-energy requirements are deducted separately, period by period, before it applies.

### Negative clearing prices

**13.5% of auction records (8,134 of 60,054) clear below zero**, concentrated in DR High (5,205)
and DM High (2,816), and all of them after EAC went live: the legacy auctions never did. The
model holds a negative-priced product only when its delivery pays for it. Low products carry a
positive delivery cost, so they are never held below zero; DR High absorbs energy the battery can
sell on, and when that energy is worth more than the negative price the model holds it. Its
revenue breakdown can therefore show negative DR High revenue, with the matching income in
wholesale trading.

| Year | DRH mean | DRL mean | DRH negative | DR pair (H+L) |
|---|---|---|---|---|
| 2022 | +£11.53 | +£13.00 | 0% | +£24.52 |
| 2023 | +£1.10 | +£10.66 | 14% | +£11.75 |
| 2024 | −£4.80 | +£8.18 | 88% | +£3.38 |
| 2025 | −£2.42 | +£13.21 | 77% | +£10.79 |
| 2026 | −£8.31 | +£16.41 | 92% | +£8.10 |

*£/MW/h.* The pair stays reliably positive: fewer than 11% of blocks price negative as a pair.

**Why DR High went negative.** The break is sharp: no DR High block cleared below zero in October
2023, 87% did in November, the month EAC went live, and 83–89% have since (DM High broke the same
way, to about half). EAC allows prices below zero, and lets a provider put several products in one
order at a single price, accepted all-or-nothing when the order's total surplus is non-negative
([EAC Detailed Market Design](https://www.neso.energy/document/276866/download)). A provider wanting
DR Low can pair it with DR High, whose leg then clears below zero while the order as a whole pays,
which fits the pair staying positive. The free energy DR High absorbs adds to the case. And its
prices rise with the daily renewable share (ρ = +0.43, against −0.18 for DR Low), as the charge
leg's role predicts: surplus renewable output pushes frequency up, which is when it is called.

## The price forecast

### Three price signals

All three run the identical allocation and dispatch engine; only the forecast differs. It sets
the opportunity cost FR is offered at as well as driving dispatch.

| Strategy | Price signal fed to the LP | What it represents |
|---|---|---|
| **Perfect Foresight** | Actual day-D wholesale prices | The ceiling; needs knowledge of the future |
| **Naive** | The last complete day's 48 half-hourly prices | The floor; any real model must beat this |
| **ML Model** | Random Forest forecast for day D | The realistic case, using only data that existed at each decision |

### The Random Forest

A Random Forest regressor predicts the 48 half-hourly APXMIDP prices for day D. The features are
tabular rather than sequential, trees need no scaling, and tree methods are competitive with deep
learning on short-horizon day-ahead tasks ([Lago et al., 2021](https://doi.org/10.1016/j.apenergy.2021.116983);
[Weron, 2014](https://doi.org/10.1016/j.ijforecast.2014.08.008)); it was kept over more accurate
models for a reason the [case studies](#why-the-model-is-not-chosen-by-accuracy) explain.
Features, all from the last complete day or earlier:

- Same-period lagged prices 1, 2, 7 and 14 days prior, and the previous day's mean, standard
  deviation, maximum and minimum
- Generation mix: total generation, renewable and fossil fractions, and each fuel
- Settlement period, day of week and day of year as sin/cos pairs, so period 48 and period 1 are
  adjacent; weekend and UK bank holiday flags
- **GB BESS fleet capacity** (`bess_fleet_mw`, monthly, from the DESNZ Renewable Energy Planning
  Database), for the structural shift as a growing fleet competes for the same spreads, and
  **spread suppression** (`bess_fleet_mw / gen_total`), the fleet's penetration: as it grows,
  batteries flatten the merit order and compress spreads

<div class="note">
<b>Reading the REPD fleet series.</b> REPD is a <i>planning</i> database published
quarterly, with two consequences. Recent months are revised upward as projects are
confirmed operational — the Q4 2025 extract put March 2026 at 3.4 GW, while the Q2 2026
extract puts the same month at 5.0 GW, so figures for the most recent year are
provisional. And the extract always lags the half-hourly price data, so trailing months
are projected from a 12-month linear trend anchored at the last measured value, keeping
the cumulative series monotonic. Projected months are flagged
<code>is_extrapolated</code>: currently
${coverage.bess_fleet.n_extrapolated} of ${coverage.bess_fleet.rows} months
(measured through ${fmt(coverage.bess_fleet.measured_end)}, projected to
${fmt(coverage.bess_fleet.end)}). Months before the first REPD entry are zero; months
after the last carry the most recent measured capacity forward rather than dropping to
zero.
</div>

### What each forecast is allowed to know

A forecast of day D built from all of D-1 exists only once D-1 has ended. That is right for
dispatch on day D and too late for two other uses. Offers for D close at 14:00 on D-1, so they are
priced on a forecast built from data to D-2: D-2's own prices for naive, and for the model a second
walk-forward table whose features all sit one day further back. The same early forecast drives
tomorrow's periods in each dispatch plan, and the plan stops there, because no honest forecast yet
exists for the day after. Perfect foresight runs the same engine and horizon on actual prices.

The rule is stricter than reality. At 14:00 on D-1 a real operator has also seen D-1's morning,
and the hourly day-ahead auctions for D have already cleared (results by 10:00). The model uses
neither, so its offer-stage information is conservative.

### Walk-forward validation

Every forecast behind the revenue figures is out-of-sample. The model is refit at quarterly
origins on the history available at each and used only for the days until the next, so no day is
ever predicted by a model that trained on it. The feature matrix begins in January 2019 and the
backtest in September 2021, so the first origin already has two and a half years to learn from.
Before this, one fixed split trained on everything before ${p.test_start}, which left 42 of the 60
months forecast by a model fitted on those very days; it survives only as a diagnostic on the
[Forecasting & Dispatch](./backtester) page, and as the fit that supplies feature importances.

## What the forecast is worth

The **foresight ratio** is the share of the gap between floor and ceiling a forecast closes,
`(ML − Naive) / (Perfect Foresight − Naive)` on net revenue. Here it is about 12%: the model is
worth about £3k/MW/yr over the floor, in every year of the backtest, against a ceiling £26k above
it. Published GB and European forecasting studies treat 70–85% as strong, but they forecast
day-ahead auction prices over shorter, calmer windows, usually scored on pure arbitrage rather than
a co-optimisation against frequency response contracts.

| £k / MW / yr | Frequency response | Trading | Wear | Net |
|---|---|---|---|---|
| Perfect foresight | 50.7 | 68.4 | −3.2 | 115.9 |
| Naive | 57.5 | 35.2 | −2.5 | 90.1 |
| ML model | 59.6 | 35.9 | −2.2 | 93.3 |

**The model and the ceiling earn through different channels.** The whole of perfect foresight's
advantage is trading, while the model earns most of its lead through response (+£2.1k, against
+£0.7k at trading): it values each block's arbitrage more accurately when the offers are made, and
holds better positions. Beating persistence at trading needs a forecast that identifies *which*
half-hours will be extreme; lowering average error across all of them does not do that, and the
offer stage consumes a block-level summary that a sharper half-hourly curve barely moves. That is
why a much more accurate model earned far less, and a much better forecast earned nothing more
(see the [case studies](#case-studies)).

**How much of it is noise.** Not much. Resampling the paired daily revenue differences in
four-week blocks, because a dispatch decision carries state into the next day
([Künsch, 1989](https://doi.org/10.1214/aos/1176347265)), puts the model's lead at £3.18k/MW/yr
(95% interval £2.05k to £4.37k), and £1.79k [£0.88k, £2.76k] from 2025 alone. Its *accuracy* edge
over persistence, though, is not significant: a
[Diebold-Mariano test](https://doi.org/10.1080/07350015.1995.10524599) on paired daily losses gives
p = 0.26 on squared error and p = 0.19 on the error in the day's spread. Spike days dominate
squared-error differences, while the revenue difference is a small, repeated gain from holding
better response positions: the forecast earns through allocation rather than precision.

### How the headline has moved

Each change below made the model more honest or more capable, and several moved the foresight
ratio without the forecast changing at all.

| 2026 | Change | Effect |
|---|---|---|
| 15 Sep | Allocation follows NESO's rules, replacing a proportional split that sold the full rating into all six products | FR availability had been overstated about 1.5× |
| 16 Sep | Response delivery modelled from one-second frequency | Arbitrage revenue fell for every strategy, narrowing the gap between floor and ceiling |
| 17 Sep | Walk-forward retraining replaced one fixed train/test split | The forecast had looked worth £18.8k/MW/yr on the 42 months it trained on, against £2.3k on the 18 held out; the ratio fell from about 66% to 20% |
| 18 Sep | Offers priced from a day-ahead trading plan rather than one cycle per block (from 2025 a block's own spread averaged £15.5/MWh, the day's £68.9) | +£11.9k/MW/yr for perfect foresight, +£9.1k naive, +£7.3k the model |
| 18 Sep | Offers and tomorrow's dispatch see only forecasts that existed at the time | Offers: −£2.4k naive, −£1.6k the model; dispatch: no measurable change |
| 18 Sep | The plan's forecast shrunk halfway to its daily mean | +£4k naive, +£1k the model; the model's lead narrowed from £4.1k to £3.2k as the ceiling pulled away, and the ratio fell to about 12% |

A better engine made the forecast matter less; it did not make the forecast worse.

## Case studies

Experiments measured against the shipped engine, kept so they can be revisited: what was tried,
what happened, why, and what would change the answer. Code and reports are in the
[repository](https://github.com/finbarrhodes/Halfcell).

### Why the model is not chosen by accuracy

<p class="muted">17 Sep 2026 · Random Forest kept</p>

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
spread that fails to arrive costs twice, in the trade and at the offer stage, where an inflated
arbitrage value declines response contracts worth having. From 2025 LEAR held 23 MW of Low
products against the forest's 30, sat out 30% of EFA blocks against 12%, and gave up £0.71M of
availability revenue to gain £0.07M of trading while cycling 48% more energy. Clipping its
forecasts to the price range seen before each origin recovered almost nothing: the problem is
systematic bias, not the tail.

**Since then.** Spike RMSE is blind to this failure, since it scores only spikes that happened, so
spread calibration is reported beside the accuracy metrics on the
[Forecasting & Dispatch](./backtester) page. **Revisit if** a candidate's spread bias is corrected;
its accuracy lead would then be worth testing again.

### A better forecast that earned nothing

<p class="muted">17 Sep 2026 · not adopted · the wind collector and features stay in the repository</p>

Adding NESO's day-ahead wind forecast cut RMSE on the held-back folds by 16% (35.2 to 29.7), lifted
rank correlation from 0.587 to 0.699 and improved error on spikes by 13%; revenue moved from £87.3k
to £87.1k per MW per year. At the time the model traded £0.8k/MW/yr *worse* than naive and earned
its whole lead through response, so a sharper half-hourly curve had nothing to act on. Adopting it
would add a monthly data dependency for no measured gain. **Revisit if** dispatch learns to abstain
— trading only where the forecast is confident enough to beat persistence, which is queued — since
sharpness may matter once the forecast is allowed to decline a trade.

### Discounting the forecast at the offer

<p class="muted">18–21 Sep 2026 · a constant discount adopted; four refinements not</p>

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

### Interval forecasts as guard bands

<p class="muted">23 Sep 2026 · not adopted ·
<a href="https://github.com/finbarrhodes/Halfcell/blob/main/reports/interval_benchmark.md">interval scores</a> ·
<a href="https://github.com/finbarrhodes/Halfcell/blob/main/reports/offer_valuation_quantile.md">revenue runs</a></p>

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

## Known limitations

**Not modelled**

- Intraday and day-ahead auction trading (APXMIDP stands in for them), and the Balancing Mechanism
- Grid connection limits and real-time dispatch constraints
- Battery degradation beyond the flat wear cost: calendar and cycle ageing, which depend on depth of
  discharge, C-rate, temperature and chemistry (NMC and LFP differ materially), would need a
  state-of-health track and a degradation-aware dispatch policy

**Approximations**

- *Rolling horizon, not a global optimum.* One LP over the whole backtest would earn more, but only
  with prices no operator has when committing.
- *No terminal value in dispatch.* Energy left when a plan ends is worth nothing to it, so each plan
  sells down towards its end; the end is always at least 24 hours away and only the first period is
  executed. The offer plan values leftover energy instead.
- *Offers and dispatch are solved in sequence, each treating the forecast as certain,* with one
  discount for every day. See [Swierczynski et al. (2021)](https://doi.org/10.3390/en14248365) and
  [Bai et al. (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0306261924015149) for
  joint formulations.
- *Conservative information at the offer stage:* data only to the end of D-2.
- *Delivery follows frequency instantly, product by product.* The Service Terms allow up to 10
  seconds to reach full delivery, which moves little energy over half an hour.
- *Baselines change within the half-hour;* NESO's baseline submission timings are not modelled.
- *Recovery through the reserve is planned at cost, not value,* so where little trading power is
  left, the free energy DR High absorbs is sold when recovery needs it rather than at the best price.
- *NESO's discretion is not modelled.* It may waive penalties during extended deviations beyond
  0.1 Hz (Service Terms 6.11 vi), and may treat a non-compliant unit as unavailable until satisfied
  it has recovered (6.12). The model always counts a missed requirement, but only in the periods that
  start outside it, so it understates that exposure.
- *Expected delivery costs are recent averages:* four weeks of delivery and a week of prices, not
  forecasts of either.
- *Walk-forward, but not nested.* Hyperparameters were chosen once rather than inside each fold, a
  rolling training window is untested against the expanding one, and quarterly refits are a
  modelling choice. Trees cannot extrapolate beyond the prices they have seen.
- *Price-taker,* backed by the 20% auction-size limit, which is why results scale linearly with power
  and the dispatch page stops at 100 MW.
- *Continuous MW, unlimited orders.* EAC trades whole MW and caps each unit's orders per day.

## Data sources

```js
const rows = [
  ["Frequency response auction results (DC/DR/DM)", "NESO Data Portal", coverage.auctions],
  ["APXMIDP market index price", "Elexon Insights Solution API", coverage.market_index],
  ["System buy/sell prices (SBP/SSP)", "Elexon Insights Solution API", coverage.system_prices],
  ["Generation by fuel type (daily)", "Elexon Insights Solution API", coverage.generation],
  ["GB BESS fleet capacity (monthly)", "DESNZ REPD", coverage.bess_fleet],
].map(([dataset, source, c]) => ({
  Dataset: dataset, Source: source,
  Coverage: `${fmt(c.start)} – ${fmt(c.end)}`,
  Records: d3.format(",")(c.rows),
}));

display(Inputs.table(rows, {rows: 6, width: {Dataset: 300, Source: 210}}));
```

Coverage is read from the processed datasets at build time, so this table cannot drift out of date
when the pipeline is re-run. Both NESO and Elexon APIs are public and need no key.
[NESO Data Portal](https://www.neso.energy/data-portal) ·
[Elexon Insights](https://developer.data.elexon.co.uk/) ·
[DESNZ REPD](https://www.gov.uk/government/publications/renewable-energy-planning-database-monthly-extract)

- **Auction results span two feeds.** Legacy DC/DR/DM auctions run from September 2021 to November
  2023 and EAC results from then on. NESO publishes EAC results as one resource per fiscal year plus
  a live feed it rotates into a new archive each April; collection stitches the segments together
  and de-duplicates the day where adjacent segments overlap.
- **Overlapping pulls resolve to the most recent one.** Data is collected in overlapping date-ranged
  batches, and the trailing days of any batch are provisional: Elexon moves system prices through
  several settlement runs, and NESO revises embedded solar and wind after the fact. Where two batches
  disagree about a settlement period, the one collected later is kept, so a settled value always
  displaces the estimate it replaces.

## Literature & references

**NESO frequency response rules and market evidence**

- NESO. *Response Services Service Terms*.
  [neso.energy](https://www.neso.energy/document/384606/download)
- NESO. *Response Services Procurement Rules*, version 5
  ([neso.energy](https://www.neso.energy/document/378246/download)); version 2.0, effective
  15 November 2024 ([neso.energy](https://www.neso.energy/document/347456/download)).
- NESO. (2025). *SOE Monitoring Guidance for Energy Limited DC/DM/DR Providers*, V2.
  [neso.energy](https://www.neso.energy/document/347241/download)
- National Grid ESO. (2023). *Enduring Auction Capability: Detailed Market Design*.
  [neso.energy](https://www.neso.energy/document/276866/download)
- NESO. *System Frequency*, one-second resolution, monthly files.
  [neso.energy](https://www.neso.energy/data-portal/system-frequency-data)
- National Grid ESO. *Dynamic Regulation Auction Rules* (legacy auctions).
  [neso.energy](https://www.neso.energy/document/246746/download)
- Modo Energy. *Battery Energy Storage Year in Review: 2023*.
  [modoenergy.com](https://modoenergy.com/research/modo-battery-energy-storage-year-review-2023-capacity-revenues-frequency-response)

**Electricity price forecasting**

- Lago, J., Marcjasz, G., De Schutter, B., & Weron, R. (2021). Forecasting day-ahead
  electricity prices: A review of state-of-the-art algorithms, best practices and an
  open-access benchmark. *Applied Energy*, 293, 116983.
  [doi:10.1016/j.apenergy.2021.116983](https://doi.org/10.1016/j.apenergy.2021.116983)
- Weron, R. (2014). Electricity price forecasting: A review of the fundamental and
  econometric approaches. *International Journal of Forecasting*, 30(4), 1030–1081.
  [doi:10.1016/j.ijforecast.2014.08.008](https://doi.org/10.1016/j.ijforecast.2014.08.008)

**Probabilistic forecasting and conformal prediction**

- O'Connor, C., Bahloul, M., Rossi, R., Prestwich, S., & Visentin, A. (2025). Conformal prediction
  for electricity price forecasting in the day-ahead and real-time balancing market.
  [arXiv:2502.04935](https://arxiv.org/abs/2502.04935)
- Xu, C., & Xie, Y. (2023). Sequential predictive conformal inference for time series.
  *Proceedings of the 40th International Conference on Machine Learning*, PMLR 202.
  [arXiv:2212.03463](https://arxiv.org/abs/2212.03463)
- Romano, Y., Patterson, E., & Candès, E. (2019). Conformalized quantile regression.
  *Advances in Neural Information Processing Systems*, 32.
  [arXiv:1905.03222](https://arxiv.org/abs/1905.03222)
- Meinshausen, N. (2006). Quantile regression forests. *Journal of Machine Learning Research*, 7,
  983–999. [jmlr.org](https://www.jmlr.org/papers/v7/meinshausen06a.html)
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and
  estimation. *Journal of the American Statistical Association*, 102(477), 359–378.
  [doi:10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437)

**MPC dispatch engine & LP solver**

- Diamond, S., & Boyd, S. (2016). CVXPY: A Python-Embedded Modeling Language for Convex
  Optimization. *JMLR*, 17(83), 1–5.
  [jmlr.org/papers/v17/15-408](https://www.jmlr.org/papers/v17/15-408.html)
- Goulart, P., & Chen, Y. (2024). Clarabel: An interior-point solver for conic programs with
  quadratic objectives. *IEEE TAC*.
  [doi:10.1109/TAC.2024.3457633](https://doi.org/10.1109/TAC.2024.3457633)
- Huangfu, Q., & Hall, J. A. J. (2018). Parallelizing the dual revised simplex method.
  *Mathematical Programming Computation*, 10, 119–142. The HiGHS solver behind the offer plan.
  [doi:10.1007/s12532-017-0130-5](https://doi.org/10.1007/s12532-017-0130-5)
- Smith, J. E., & Winkler, R. L. (2006). The optimizer's curse: Skepticism and postdecision
  surprise in decision analysis. *Management Science*, 52(3).
  [doi:10.1287/mnsc.1050.0451](https://doi.org/10.1287/mnsc.1050.0451)

**BESS dispatch optimisation & co-optimisation**

- Swierczynski, M., et al. (2021). Co-Optimizing Battery Storage for Energy Arbitrage and
  Frequency Regulation in the GB Market. *Energies*, 14(24), 8365.
  [doi:10.3390/en14248365](https://doi.org/10.3390/en14248365)
- Bai, X., et al. (2024). Smart optimization in battery energy storage systems: An overview.
  [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0306261924015149)
- Lee, J.-O., & Kim, Y.-S. (2022). Novel battery degradation cost formulation for optimal
  scheduling of battery energy storage systems. *IJEPES*, 137, 107795.
  [doi:10.1016/j.ijepes.2021.107795](https://doi.org/10.1016/j.ijepes.2021.107795)

**Battery degradation modelling**

- Xu, B., et al. (2018). Modeling of lithium-ion battery degradation for cell life
  assessment. *IEEE Transactions on Smart Grid*, 9(2), 1131–1140.
  [arXiv:1703.07968](https://arxiv.org/abs/1703.07968)
- Reniers, J. M., Mulder, G., & Howey, D. A. (2021). Economic MPC of Li-ion battery cyclic
  aging via online rainflow analysis. *Journal of Energy Storage*.
  [doi:10.1002/est2.228](https://doi.org/10.1002/est2.228)

**GB BESS market context**

- Modo Energy. (2024). *GB Battery Storage Report*.
  [modoenergy.com](https://modoenergy.com/research/future-of-battery-energy-storage-buildout-in-great-britain)
- Timera Energy. (2023). Battery investors confront revenue shift in 2023.
  [timera-energy.com](https://timera-energy.com/blog/battery-investors-confront-revenue-shift-in-2023/)
