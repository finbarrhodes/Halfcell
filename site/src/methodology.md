# Methodology & Data Sources

This page documents the modelling approach, assumptions, and data sources used by the
forecasting and dispatch model.

```js
const coverage = await FileAttachment("data/coverage.json").json();
const manifest = await FileAttachment("data/manifest.json").json();
const p = manifest.ml_mpc.params;
const scenarios = (await FileAttachment("data/revenue-scenarios.parquet").parquet()).toArray()
  .map((d) => ({...d, month_dt: new Date(d.month_dt)}));
const fmt = (d) => new Date(d).toLocaleDateString("en-GB", {year: "numeric", month: "short"});
```

## Two-stage participation model

The backtester separates frequency response (FR) availability revenue from energy arbitrage
without counting the same capacity twice, and within the rules NESO sets for Dynamic
Services providers.

**Stage 1: what to offer, at the bid deadline.** Offers for all six EFA blocks of day D close
the afternoon before: 14:00 on D-1 since the Enduring Auction Capability (EAC) went live on
2 November 2023, and 14:30 before it. At that moment the model decides how many MW to hold
in each of the six products (DC, DM and DR, each High and Low) in each block.

It offers every product at its **opportunity cost**: the trading the same capacity would
otherwise do. The model works that out the way a dispatcher would, by planning. At the
deadline it chooses the six blocks' holdings together with a half-hourly trading plan from
the deadline to the end of day D, on the forecast that existed at that moment (see
[what each forecast is allowed to know](#what-each-forecast-is-allowed-to-know)), so a holding
costs whatever it takes out of that plan. Three things follow that one price per block cannot
capture:

- **Direction depends on the hour.** Overnight the plan charges, which High products get in
  the way of; at the evening peak it sells, which Low products get in the way of.
- **State of charge matters.** Holding High overnight costs nothing if the store is already
  nearly full, and a good deal if it is empty.
- **Values do not add.** Two evening peaks are substitutes, so keeping both free is worth one
  sale of the store rather than two. A product's first megawatts are often free and its last
  dear: the stepped price-quantity shape EAC orders are built for.

Each offer also carries the expected cost of the energy the product will deliver when called
on (see [response delivery](#response-delivery)). NESO accepts an order when the clearing
price covers its offer, so a battery bidding at cost ends up holding whichever permitted
combination earns most at the clearing prices. Solving the plan against them reproduces the
outcome of cost-reflective bidding rather than assuming foresight, on the assumption that the
battery is a price-taker whose offers do not move clearing prices. The plan is a linear
programme of about 270 variables, solved in two milliseconds by
[HiGHS](https://doi.org/10.1007/s12532-017-0130-5), a simplex solver, because holdings often
tie and only a vertex solution breaks those ties cleanly.

**Two refinements.** The plan's forecast is pulled halfway towards its daily mean before
planning. A plan chases the hours its forecast shows as most extreme, so forecast errors
select themselves into it and it overstates what keeping capacity free is worth — the
optimiser's curse ([Smith & Winkler, 2006](https://doi.org/10.1287/mnsc.1050.0451)) — and the
error is lopsided: over-valuing headroom costs a contract and then a trade, under-valuing it
costs only some trading. The weight was chosen on the years before 2025 from 0.25, 0.5, 0.75
and 1, and held up on 2025 onward. And energy left at the end of day D is valued at the day's
mean forecast price less wear; valued at nothing, the plan would empty the store in the last
block and make holding Low response there look expensive for no real reason.

<div class="note">
<b>Until 18 September offers were priced block by block.</b> Each block's free capacity was
valued at one cycle between its own cheapest and dearest half-hours,
<code>(avg_discharge − avg_charge / η − cycling_cost) × duration_h</code>, the same for High and
Low. The battery's real trade crosses blocks: from 2025 the spread inside a block averaged
£15.5/MWh against £68.9 across the day, so the old rule valued the wrong trade and valued it
small. On the same information, planning raised perfect foresight by £11.9k/MW/yr, naive by
£9.1k and the model by £7.3k, and beat the old rule in every calendar year for all three.
</div>

NESO's rules set which combinations are permitted:

- **Capacity in each direction.** MW offered into Low products, plus the Reserved Capacity
  held for any High products, must fit within the battery's discharge rating. High MW plus
  the reserve for Low products must fit within its charge rating
  ([Procurement Rules](https://www.neso.energy/document/378246/download) 8.3.3.2, with the
  shares defined in that document's Schedule 1). Reserved Capacity is at least 10% of the
  offered MW for DC, 20% for DM and 40% for DR, held in the opposite direction for energy
  recovery; the model holds exactly those shares. Note the Service Terms' own 20% is a
  different figure — the Energy Recovery volume, which is 20% for all three services. So a MW sold into DC Low cannot also be sold into
  DR Low, though it can back a High product at the same time.
- **Energy.** Each Low contract needs its delivery energy in store, and each High contract
  the same again as headroom: MW × 15 minutes for DC, 30 for DM, 60 for DR
  ([Service Terms](https://www.neso.energy/document/384606/download) 6.11). Both must fit in
  the battery together.
- **Reachability.** All six blocks are solved together, starting from the battery's actual
  state of energy at the deadline and the commitments it still holds for the rest of D-1.
  Every block's required range must be reachable using only the power the contracts leave
  free. Some combinations use the whole rating in both directions, so state of energy cannot
  move at all while they run.
- **Maximum Sell Size.** No more than 100 MW in any one product.
- **Auction size — a modelling limit, not a NESO rule.** No more than 20% of any auction's
  cleared volume. Before EAC the DM and DR auctions were small (median DM Low cleared 4 MW,
  DR Low 55 MW), and without a limit a 50 MW battery would have held more than NESO bought in
  the whole auction in 14–45% of the blocks where it held them. Beyond about a fifth of an
  auction, one unit's offer could plausibly set the price, which a price-taker cannot. Since
  EAC the limit rarely binds. Inserting the battery's offers into NESO's published order
  books and re-running the clearing would measure this directly; that is planned.

Two of these rules changed during the backtest:

| Service days from | Rule |
|---|---|
| Start of data (Sep 2021) | One service per unit per EFA block: DC, DM or DR, with High and Low of that service allowed together ([DR Auction Rules](https://www.neso.energy/document/246746/download) 7.3.1) |
| 2 Nov 2023, EAC go-live | Capacity can be split across all three services in the same block. The model holds Reserved Capacity from here |
| 15 Nov 2024 | Reserved Capacity becomes a rule ([Procurement Rules v2.0](https://www.neso.energy/document/347456/download)) |

**Reserved Capacity before it was a rule.** NESO introduced the reserve together with the
running energy requirement described under [response delivery](#response-delivery):
proposed in June 2024, binding from 15 November 2024. The model applies that requirement
throughout, and a stack across services can use the whole rating in both directions, leaving
no power to recover the energy it delivers. Without the reserve, the battery would have
missed its requirement in about a third of half-hours between EAC go-live and November 2024.
A careful operator would not hold such a stack, so the model holds the reserve from EAC
go-live. Before EAC a unit held one service per block and stayed within its requirement
without it.

Before EAC the unit has to choose its service before the auction clears; see
[pre-EAC service choice](#pre-eac-service-choice).

<div class="note">
<b>Earlier versions of this model got this wrong.</b> They sold the full rating into all six
products at once, which the per-direction rule forbids, and split capacity between FR and
arbitrage with a proportional heuristic. On FR availability alone that overstated a
rule-compliant allocation by about 1.5× over the backtest. Every revenue figure on the site
now comes from the rule-compliant model.
</div>

**Stage 2: dispatch around the commitments.** The rolling MPC below trades only the power
the contracts leave free on each side, and holds state of energy where NESO's running
requirement needs it as the contracts are called on (see
[response delivery](#response-delivery)). A day's commitments enter dispatch at its bid deadline: an operator must be able to
deliver everything it offered, so it positions for its offers before results publish, and in
a price-taker model the offers are exactly what clears. A settlement period that starts
outside the requirement counts as unavailability
([Service Terms](https://www.neso.energy/document/384606/download) 6.12), and the block
loses an eighth of its availability payment for each one. The dispatch page reports how many
periods each strategy missed.

## Pre-EAC service choice

Before 2 November 2023, a unit could offer only one of DC, DM and DR into each EFA block, and
had to pick before the auction cleared. For each block the model picks the service that
would have earned most at the **previous day's clearing prices** for the same block, the
latest a bidder had at the 14:30 deadline. It is then paid the day's actual prices for what
it offered.

The fleet's own orders show how operators actually chose. Across 247,893 unit-blocks of
NESO's legacy order data (March 2022 – October 2023, from the
[NESO Data Portal](https://www.neso.energy/data-portal)):

- no unit offered two services into the same block, confirming the rule;
- 86.5% of orders went in after the previous day's results had published, so those prices
  were available;
- 91% of unit-blocks repeated the previous day's service, and 82% were DC;
- units picked the previous day's best-paying service 33% of the time, no more often than
  the service that turned out best on the day (32%).

Operators were not chasing yesterday's prices; most stayed in DC. A plausible reason is
duration. The GB fleet averaged about 1.1 hours in 2022
([Modo Energy](https://modoenergy.com/research/modo-battery-energy-storage-year-review-2023-capacity-revenues-frequency-response)),
and DR's 60-minute delivery requirement binds hard on a one-hour battery. The model's
2-hour reference battery can hold DR comfortably, so it picks DR whenever DR paid more the
day before.

The D-1 rule is kept because it uses only information a bidder had at the deadline. A
sensitivity run that held every pre-EAC block in DC instead has been dropped: a full-power DC
stack in both directions leaves no power to recover the energy it delivers, so its store
drained and it spent a fifth of the pre-EAC half-hours unavailable, which says more about that
stack than about the service choice.

## Response delivery

A contract is not only a promise to stand ready. Whenever frequency leaves the deadband, a
battery holding response has to deliver it, and the energy that moves changes its state of
charge. The model works that energy out from GB frequency itself.

**From frequency to energy.** NESO publishes system frequency at one-second resolution
([System Frequency](https://www.neso.energy/data-portal/system-frequency-data) dataset). Each second's reading goes through each service's response curve
([Service Terms](https://www.neso.energy/document/384606/download), Table 1), and the result
is summed into MWh per MW contracted for every settlement period, for each service and
direction. Low products answer frequency below 50 Hz by discharging; High products answer
frequency above it by charging.

| Service | Nothing within | 5% at | 100% at |
|---|---|---|---|
| DC | ±0.015 Hz | ±0.2 Hz | ±0.5 Hz |
| DM | ±0.015 Hz | ±0.1 Hz | ±0.2 Hz |
| DR | ±0.015 Hz | — (a straight line) | ±0.2 Hz |

Frequency sits outside DR's deadband most of the time, so DR moves far more energy than DC:
in this data about 2.5 MWh per MW per day in each direction in 2021, rising to 3.6 in 2026,
against 0.13–0.18 for DC.

**Who pays for that energy.** Nobody. NESO passes delivery volumes to Elexon and the unit's
imbalance position is adjusted by them (Service Terms 16). Energy a Low product gives away is
simply gone and has to be bought back; energy a High product absorbs arrives for free and can
be sold on. Every MWh discharged in delivery also wears the battery.

**The requirement moves with delivery.** NESO does not ask a battery to hold its full
response energy while it is delivering. The Minimum State of Energy Requirement (Service
Terms 6.11) starts each block at the Contracted Response Energy Volume, falls by the energy
delivered in each settlement period, and then climbs back by the Energy Recovery Adjustment
Volume: the shortfall three periods earlier, at most 20% of the volume per period. The
Service Terms' own example: a 50 MWh contract that delivers 2 MWh in its first half-hour needs
48 MWh until the sixth period, then 50 again. The Reserved Capacity exists so the battery can
recover that energy
([SOE Monitoring Guidance](https://www.neso.energy/document/347241/download)), so dispatch may
use it in any period of its plan to keep a requirement reachable, including ahead of a block
that restores the full requirement. The plan counts what energy moved through the reserve
costs at the price, never what it earns, so the reserve is not used to trade. A period that
starts below the running requirement counts as unavailable; a requirement at or below zero is
allowed unavailability.

**Planning without knowing frequency.** Within a block, delivery lowers state of charge and
the requirement together, so the LP plans on no further delivery. Where a later block will
restore the full requirement, it plans on delivery carrying on at its average over the
previous day, which the operator has already seen.

**Pricing delivery into offers.** At the bid deadline, each product's expected delivery is
its average over the previous 28 days for that EFA block, and the energy is valued at the
block's average price over the previous 7 days, *p̄*. Holding a MW of a Low product for the
block costs `E_low × (p̄ / η + wear)`: the energy bought back, plus wear. Holding a MW of a
High product earns `η × E_high × (p̄ − wear)`: the free energy sold on, less wear. That is why
the model will hold DR High at a negative price when the energy it absorbs is worth more than
the price it pays, and why DR Low is worth much less than its clearing price suggests.

**A site with no interest in arbitrage.** The FR-only scenario follows every rule above, gives
arbitrage no value when choosing what to offer, and trades only to keep its contracts
deliverable: its LP ignores prices and moves as little energy as its requirement needs. Those
trades still settle at market prices, so its net revenue is availability payments less the
cost of recovering delivered energy, losses and wear.

## Dispatch strategies

Intraday dispatch is driven by a **rolling Model Predictive Control (MPC) linear programme**,
re-solved at every 30-minute settlement period. At each period *t* the LP plans to the end of
tomorrow — the furthest any forecast yet exists for, between 49 and ${p.horizon} periods —
returns only the first period's decision, then re-solves: a receding-horizon approach
reflecting the real constraint that dispatch must be committed before future prices are
known.

**LP formulation.** Decision variables are charge power *p_chg[t]*, discharge power
*p_dis[t]*, and state of charge *SoC[t+1]* over the horizon. The objective maximises net
arbitrage revenue minus cycling degradation cost:

```
maximise  Σ price[t] × (p_dis[t] − p_chg[t]) × 0.5h  −  cycling_cost × Σ p_dis[t] × 0.5h
```

subject to:

- SoC state equation with round-trip efficiency applied on the charge side
- State of energy within the physical store at every point
- State of energy inside the range the FR contracts require at the start of every period,
  which forces the battery to pre-position for upcoming blocks. The range is soft: missing
  it costs £5,000 per MWh, about 2.5× the highest price in the data. The LP therefore always
  moves towards compliance, never breaches a contract to capture a spread, and stays
  solvable when a requirement genuinely cannot be reached. A larger penalty would buy no
  safety: it shares the objective with prices, and at £50,000 the solver needed more
  iterations and returned 1% of solves as inaccurate
- Discharge and charge power each bounded by what the contracts leave free on that side

Mutual exclusion of simultaneous charge and discharge is handled by LP relaxation: because
the objective penalises cycling, simultaneous charge and discharge is never optimal at a
positive spread, so no binary variables are required. Solved with the **CLARABEL**
interior-point solver bundled with cvxpy
([Diamond & Boyd, 2016](https://www.jmlr.org/papers/v17/15-408.html)).

**Three price signals are benchmarked.** All three run the identical allocation and dispatch
engine; only the forecast differs. It sets the opportunity cost FR is offered at as well as
driving dispatch.

| Strategy | Price signal fed to LP | What it represents |
|---|---|---|
| **Perfect Foresight** | Actual day-D wholesale prices | Theoretical ceiling — needs advance knowledge of the future |
| **Naive** | The last complete day's 48 half-hourly prices | The floor; any real model must beat this |
| **ML Model** | Random Forest forecast for day D | Realistic best case, using only data that existed at each decision |

Dispatch decisions execute unconditionally at actual prices. Per-period revenue can be
negative when forecast error causes an unfavourable trade — that is the realistic
operational outcome and is intentional.

### What each forecast is allowed to know

A forecast of day D built from all of D-1 exists only once D-1 has ended. That is right for
dispatch on day D and too late for two other uses. Offers for D close at 14:00 on D-1, so they
are priced on a forecast built from data to D-2: D-2's own prices for naive, and for the model
a second walk-forward table whose features all sit one day further back. The same early
forecast drives tomorrow's periods in each dispatch plan, and the plan stops there, because no
honest forecast yet exists for the day after. Perfect foresight runs the same engine and
horizon on actual prices.

Until 18 September both uses read forecasts built from all of the previous day before it had
ended — for naive, tomorrow's slot in the dispatch plan held today's actual prices. At the
offer stage that was worth £2.4k/MW/yr to naive and £1.6k to the model. In dispatch it was
worth nothing measurable in the full stack (removing it moved revenue by +£0.2k and +£0.3k),
but £1.6k to naive in the trading-only scenario, where the whole battery trades and seeing
today's real prices in tomorrow's slot helps most; £0.1k to the model.

The rule is still stricter than reality. At 14:00 on D-1 a real operator has also seen D-1's
morning, and the hourly day-ahead auctions for D have already cleared (results by 10:00). The
model uses neither, so its offer-stage information is conservative.

The **foresight ratio** measures how much of the gap between those two bounds a forecast
closes: `(ML − Naive) / (Perfect Foresight − Naive)`, on net revenue. Published GB and
European price-forecasting literature treats 70–85% as strong performance.

Three things have moved this number, and all are worth knowing about.

The largest was leakage. Until walk-forward retraining replaced the fixed split, most of the
backtest was forecast by a model that had trained on those same days, which put the ratio
near 66%. With every forecast out-of-sample it fell to about 20%, and stayed there in every
sub-period, so the old figure was leakage rather than a kind market.

The second is the denominator. The ratio is a share of the *capturable* headroom, so anything
that moves the gap between floor and ceiling moves it without the forecast changing at all.
Modelling response delivery narrowed the gap: once a Low contract has to buy back the energy it
gives away, arbitrage-driven revenue falls for every strategy.

The third is the engine. Pricing offers from a trading plan earns more for every signal, but
most for perfect foresight, which can exploit the day's real shape. And shrinking the plan's
forecast gives naive the caution the Random Forest's conservative spreads already supplied —
worth about £4k/MW/yr to naive and £1k to the model. The model's lead over naive narrowed from
£4.1k to £3.2k while the ceiling pulled away, and the ratio fell to about 12%. A better engine
made the forecast matter less; it did not make the forecast worse.

Published figures of 70–85% come from studies forecasting day-ahead auction prices over
shorter, calmer windows, usually scored on pure arbitrage rather than a co-optimisation
against frequency response contracts. The honest headline here is that the forecast is worth
about £3k/MW/yr over the floor, in every year of the backtest, against a ceiling £26k above it.

For LP-based joint co-optimisation of arbitrage and frequency response in GB, see
[Swierczynski et al. (2021)](https://doi.org/10.3390/en14248365).

## Ancillary service availability revenue

- Revenue = `clearing_price (£/MW/h) × MW held in that product × 4 hours per EFA block`,
  less an eighth of the block's payment for each settlement period the battery was
  unavailable.
- **High and Low name the frequency excursion, not the battery's power direction.** A Low
  service (DCL, DRL, DML) answers a *low*-frequency event by injecting power — the battery
  discharges. A High service (DCH, DRH, DMH) answers a *high*-frequency event by absorbing
  power — the battery charges.
- The same MW can back a High and a Low product at once, provided the store has energy for
  the Low contracts and headroom for the High ones. It cannot back two products in the same
  direction: DC Low, DM Low and DR Low share the discharge rating
  (see [Stage 1](#two-stage-participation-model)).
- Clearing prices from the NESO Data Portal (legacy DC/DR/DM auctions Sep 2021 – Nov 2023,
  EAC service Nov 2023 – present). NESO publishes EAC results as one resource per fiscal
  year plus a live current-year feed, rotating the live feed into a new archive each April;
  collection stitches these segments together, de-duplicating the one-day overlap where
  adjacent segments meet.
- Energy delivered when a contract is called on earns and costs nothing itself (see
  [response delivery](#response-delivery)); recovering it shows up in wholesale trading and
  wear.
- Ancillary revenue differs slightly between strategies. The forecast sets the arbitrage
  opportunity cost each product is offered at, and dispatch determines the state of energy
  each day's offers start from.

## Wholesale energy arbitrage revenue

- Computed period-by-period as the LP dispatches:
  `revenue[t] = actual_price[t] × (e_dis[t] − e_chg[t])`, summed across all 48 settlement
  periods in the day.
- Power in each period is bounded on each side by what the block's FR contracts leave free:
  the discharge rating less Low MW and the reserve for High MW, and the charge rating less
  High MW and the reserve for Low MW. The reserve itself may be used to recover energy after
  delivery, not to trade. Round-trip efficiency
  (${(p.efficiency_rt * 100).toFixed(0)}%) is applied to the charge side of the SoC state
  equation.
- The cycling wear cost (£${p.cycling_cost_per_mwh}/MWh discharged) is deducted each period
  and enters the LP objective, so the optimiser naturally avoids unprofitable cycles.
- **Price reference: APXMIDP market index** (APX Power UK) from Elexon Insights. This is the
  actual GB spot settlement reference, giving a materially more realistic daily spread than
  the imbalance settlement price (SSP), which can reach extreme negative values during
  high-renewable periods and would otherwise inflate arbitrage revenue.

## Negative clearing prices

GB frequency response auctions clear below zero more often than is widely appreciated:
**13.5% of auction records in this dataset (8,134 of 60,054) have a negative clearing
price**, concentrated in DR High (5,205 records) and DM High (2,816). As the storage fleet
has grown, procurement volumes have been outpaced and the High-side services in particular
have tipped into oversupply.

**The model holds a negative-priced product only when its delivery pays for it.** Each
product is offered at its opportunity cost plus the expected cost of the energy it delivers.
For Low products that cost is positive, so they are never held below zero. For High products
it can be negative: DR High absorbs energy the battery can sell on, and when that energy is
worth more than the negative price the model holds it. A block where nothing pays after
delivery gets no FR commitment and leaves the whole battery free to trade.

Every negative price in the data comes after EAC went live in November 2023: the legacy
auctions never cleared below zero.

## Why DR High clears negative

DR High is the clearest example of why negative prices had to be included rather than
floored. Taken alone it looks like a loss-making service; in context it is not.

| Year | DRH mean | DRL mean | DRH negative | DR pair (H+L) |
|---|---|---|---|---|
| 2022 | +£11.53 | +£13.00 | 0% | +£24.52 |
| 2023 | +£1.10 | +£10.66 | 14% | +£11.75 |
| 2024 | −£4.80 | +£8.18 | 88% | +£3.38 |
| 2025 | −£2.42 | +£13.21 | 77% | +£10.79 |
| 2026 | −£8.31 | +£16.41 | 92% | +£8.10 |

*All figures £/MW/h.*

DRH has cleared negative in the large majority of blocks since 2024, reaching 92% in 2026 —
while DRL has strengthened over the same period. **The pair, however, remains reliably
positive.** Adding the two legs gives +£8.10/MW/h in 2026, and fewer than 11% of blocks
price negative as a pair.

**When it started is the clearest clue.** No DR High block cleared negative in October 2023;
87% did in November, the month EAC went live, and 83–89% have since. The legacy auctions
never cleared below zero. DM High shows the same break, from never to around half of blocks.

EAC changed two things that make a negative High leg possible. Prices may go below zero, and
a provider can put several products in one order at a single price, accepted all-or-nothing
when the order's total surplus, `Σ MW × (clearing price − offer price)`, is non-negative
([EAC Detailed Market Design](https://www.neso.energy/document/276866/download)). A provider
wanting DR Low can offer DR High alongside it in one order, and the High leg can then clear
below zero while the order as a whole still pays. That fits the pair staying positive even as
DR High fell.

Two other explanations fit less well:

- **Matched cleared volumes.** DR High and DR Low have cleared within 5% of each other on 73%
  of blocks since November 2024, against 5% in EAC's first year. But NESO sets how much it
  buys in each direction, so matched volumes say more about its requirement than about how
  providers bid.
- **A rule tying the legs together.** There is none. High and Low are separate products, and
  a provider can hold either alone. The links are physical: a MW of DR Low needs an hour of
  energy in store, a MW of DR High an hour of headroom, and each reserves 40% of its MW on the
  opposite side (a rule since November 2024, held in the model from EAC go-live). The model
  applies those constraints directly (see
  [Stage 1](#two-stage-participation-model)).

**A further reason, now in the model.** Energy absorbed while delivering DR High is neither
paid for nor charged (Service Terms 16), and DR delivers a lot of it (see
[response delivery](#response-delivery)). A provider can accept a negative DR High price and
still come out ahead by selling that energy on, and the model does exactly that when the
numbers work. Its breakdown can therefore show negative DR High revenue, with the matching
income in wholesale trading.

The correlation with renewable output fits the charge leg's role. DRH prices are
*positively* correlated with the daily renewable share (ρ = +0.43) while DRL is weakly
negative (ρ = −0.18). DRH is the charge leg — the service that answers an over-frequency
excursion — and heavy renewable output is precisely what pushes the system toward surplus
and over-frequency, so the charge leg gains value on windy days even while clearing from a
negative base. Demand for the discharge leg does not track wind the same way, which is what
DRL's mild negative correlation reflects.

## Availability factor

- Applied as a uniform multiplier to all revenue streams and cycling costs.
- Missed state-of-energy requirements are deducted separately, period by period, before this
  factor applies (see [Stage 2](#two-stage-participation-model)).
- Models periods where the asset is unavailable through planned maintenance, unplanned
  faults, grid curtailment, or service delivery failures.
- The default of ${(p.availability_factor * 100).toFixed(0)}% is an assumption, not a NESO
  figure. The Response documents set no minimum availability percentage: a unit is expected to
  be available throughout each Contracted Service Period, and its Availability Payment is
  reduced for declared unavailability and for under-delivery
  ([Service Terms](https://www.neso.energy/document/384606/download) 5 and 7). The 95% stands
  in for outages and faults at the rate the GB fleet is generally reported to run at.

## Cycling wear cost and battery degradation

- Applied to every MWh discharged: `cycling wear cost (£/MWh) × MWh discharged`, whether by
  an arbitrage trade, a recovery trade, or delivery under an FR contract. For a battery
  holding DR, delivery is most of it.
- *Why cycling matters beyond cost:* lithium-ion cells degrade through two primary
  mechanisms that accelerate with use — SEI layer growth, which irreversibly consumes
  cyclable lithium, and lithium plating at the anode, which increases with deeper discharge
  and higher charge rates. Each MWh cycled consumes a small fraction of finite cycle life.
  The cycling wear cost is a financial proxy for that physical degradation: aggressive
  dispatch earns more in the short run but consumes cycle life faster, reducing useful life
  and residual value. For rigorous treatments of cycle-based degradation cost, see
  [Xu et al. (2018)](https://arxiv.org/abs/1703.07968) and
  [Lee & Kim (2022)](https://doi.org/10.1016/j.ijepes.2021.107795).

## ML price forecast model

A **Random Forest regressor** predicts the 48 half-hourly APXMIDP prices for day D from the
last complete day's data: D-1 for dispatch on day D, and D-2 for offers and for tomorrow's
periods in each plan (see [what each forecast is allowed to know](#what-each-forecast-is-allowed-to-know)).

*Why Random Forest?* The feature set is tabular (lagged prices, generation-mix ratios,
temporal encodings) rather than sequential; trees need no feature scaling, are robust at
this data size, and give interpretable importances. This is consistent with the electricity
price forecasting literature, which finds tree-based methods competitive against deep
learning on short-horizon day-ahead tasks
([Lago et al., 2021](https://doi.org/10.1016/j.apenergy.2021.116983);
[Weron, 2014](https://doi.org/10.1016/j.ijforecast.2014.08.008)).

**Features (all from the last complete day or earlier):**

- Same-period lagged prices: 1, 2, 7 and 14 days prior
- Previous-day price statistics: mean, standard deviation, max, min across all 48 periods
- Generation mix (daily, from D-1): total generation, renewable and fossil fractions, and
  per-fuel breakdown
- Cyclical temporal encodings: settlement period, day-of-week and day-of-year as sin/cos
  pairs to preserve circularity (period 48 and period 1 are adjacent)
- Weekend and UK bank holiday flags
- **GB BESS fleet capacity** (`bess_fleet_mw`, monthly MW) from the DESNZ Renewable Energy
  Planning Database — captures the structural shift as a growing fleet competes for the
  same arbitrage spreads.
- **BESS spread suppression** (`bess_fleet_mw / gen_total`) — penetration as a fraction of
  total system generation, encoding the mechanism directly: as penetration grows, batteries
  charge cheap and discharge expensive, flattening the merit order and compressing spreads.

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

**Walk-forward validation.** Every forecast behind the revenue figures is out-of-sample. The
model is refit at quarterly origins on the history available at that point and used only for
the days until the next origin, so no day is ever predicted by a model that trained on it.
What makes this possible is that the feature matrix begins in January 2019 while the backtest
begins in September 2021: the first origin already has two and a half years of history to
learn from.

This replaced a single fixed train/test split, and the change was not cosmetic. Under that
arrangement the model trained on everything before ${p.test_start} while the revenue backtest
still ran from 2021, so 42 of its 60 months were forecast by a model fitted on those very
days. In-sample the Random Forest ranks a day's 48 periods almost perfectly, and ranking is
exactly what dispatch consumes — so the apparent value of forecasting was £18.8k/MW/yr across
that stretch against £2.3k in the 18 genuinely held-out months. The fixed split survives only
as a diagnostic on the [Forecasting & Dispatch](./backtester) page, and as the fit that
supplies feature importances.

### Why the model is not chosen by accuracy

All four candidates were benchmarked on the same
walk-forward folds, with the folds from 2025 held back so the choice could not be made on the
evidence used to report it. The most accurate forecaster was the worst earner, and not
marginally. Revenue here was measured on 17 September, under the per-block offer rule that
the trading plan has since replaced.

| Model | RMSE | Spearman ρ | Spike RMSE | £k/MW/yr | Foresight ratio |
|---|---|---|---|---|---|
| Random Forest | 35.2 | 0.587 | 52.5 | **87.3** | **19.7%** |
| LightGBM | 34.4 | 0.588 | 51.4 | 87.0 | 18.2% |
| XGBoost | 36.3 | 0.559 | 53.5 | — | — |
| LEAR | **33.2** | **0.647** | **46.4** | 64.1 | −91.5% |

Accuracy columns are the held-back folds; revenue is the full backtest. LEAR wins every
accuracy column and earns £19k/MW/yr *less than reusing yesterday's prices*.

The cause is calibration in the one dimension dispatch consumes. LEAR over-predicts the daily
price spread by £272/MWh across the backtest and by £29.5 even in the calm recent market, while
the trees under-predict it — Random Forest by £31.5. For a price-taker battery that asymmetry
is protective. A spread that fails to materialise costs money twice: once in the trade itself,
and again at the offer stage, where an inflated shadow arbitrage value makes the model decline
frequency response contracts worth having. Over 2025 onward, LEAR held 23 MW of Low products
against Random Forest's 30, sat out 30% of EFA blocks against 12%, and gave up £0.71M of
availability revenue to gain £0.07M of trading revenue while cycling 48% more energy.

Clipping LEAR's forecasts to the price range observed before each origin — a guardrail any
operator would have — cut its worst prediction from £67,062/MWh to £1,984 and recovered almost
nothing (£64.9k), because in the recent folds nothing needed clipping. The problem is the
systematic bias, not the tail.

So Random Forest ships for robustness rather than accuracy, and the reported metrics are known
to be blind to the failure that decided this: spike-RMSE scores error on spikes that *happened*,
so a forecast inventing spikes is never charged for it. Spread calibration, the signed error in
each day's predicted spread, is now reported beside them on the
[Forecasting & Dispatch](./backtester) page for that reason.

### Why a better forecast stopped helping

Adding NESO's day-ahead wind forecast to the feature set produces a much better forecast and
exactly no more money. Across the held-back folds it cuts RMSE by 16% (35.2 to 29.7), lifts
rank correlation from 0.587 to 0.699 and improves error on price spikes by 13% — and revenue
moves from £87.3k to £87.1k per MW per year (measured on 17 September, under the per-block
offer rule).

Decomposing the headroom shows why. Perfect foresight beats the naive floor by £20.9k/MW/yr,
and every penny of that is wholesale trading: £21.2k of trading advantage against £0.4k *less*
availability revenue. The realistic model captures none of it. It trades £0.8k/MW/yr worse
than simply reusing yesterday's prices, and the £4.1k it does earn comes entirely from holding
better frequency response positions — £4.5k of extra availability revenue bought by valuing
each block's arbitrage opportunity more accurately when the offers are made.

| £k / MW / yr, 17 Sep, per-block offers | Frequency response | Trading | Wear | Net |
|---|---|---|---|---|
| Perfect foresight | 53.2 | 53.5 | −2.7 | 104.1 |
| Naive (D-1 prices) | 53.6 | 32.3 | −2.7 | 83.2 |
| ML model | 58.1 | 31.5 | −2.3 | 87.3 |

Under the current engine — offers priced from a plan, on bid-time information — the same
split reads:

| £k / MW / yr, current engine | Frequency response | Trading | Wear | Net |
|---|---|---|---|---|
| Perfect foresight | 50.7 | 68.4 | −3.2 | 115.9 |
| Naive | 57.5 | 35.2 | −2.5 | 90.1 |
| ML model | 59.6 | 35.9 | −2.2 | 93.3 |

The model now edges naive at trading (+£0.7k) but still earns most of its lead through
response (+£2.1k), and the whole of perfect foresight's advantage is still trading.

So the model earns through one channel while the ceiling is built on another, and the two
respond to different things. Beating persistence at trading needs a forecast that identifies
*which* half-hours will be extreme; lowering average error across all of them does not do
that, and a 30 £/MWh error is still wide enough to trade the wrong periods. The allocation
channel, meanwhile, consumes a block-level summary of the forecast, which a sharper
half-hourly curve barely moves.

The useful conclusion is about where to spend effort. More features are not the lever — this
is the second time a large accuracy gain has produced no revenue, after a more accurate model
produced considerably less. The lever is dispatch that knows what it does not know: trading
only where the forecast is confident enough to beat persistence, and leaving the rest alone.
That is queued rather than done.

The wind collector and features are in the repository but not in the shipped model, because
adopting them would add a monthly data dependency for no measured gain. They stay available
for the confidence work, where a forecast's sharpness may matter once it is allowed to abstain.

**Known limitations.** Tree-based models cannot extrapolate beyond price ranges seen in
training; electricity price forecasting is inherently noisy; and the model improves dispatch
quality on average without eliminating error on individual days. Current metrics and feature
importances are on the [Forecasting & Dispatch](./backtester) page.

## Known limitations

**Not yet modelled**

- Intraday / day-ahead market trading (APXMIDP used as a proxy; DA auction data not integrated)
- Balancing Mechanism direct trading
- Real-time dispatch constraints or grid connection limits

**Approximations**

- *Rolling horizon is not globally optimal.* A single LP over the full backtest would yield
  more revenue in theory, but the rolling approach reflects the real constraint that
  dispatch must be committed before future prices are known.
- *Stored energy has no terminal value in dispatch.* Energy left in store when a plan ends,
  after tomorrow, is worth nothing to the LP, so each plan sells it down towards its end.
  Only the first period of each plan is executed, and the end is always at least 24 hours
  away, which keeps this from dominating; the offer-stage plan values leftover energy at the
  day's mean forecast less wear instead.
- *LP relaxation of charge/discharge mutual exclusion.* No binary variables prohibit
  simultaneous charge and discharge; because the objective penalises cycling, this is never
  optimal at a positive spread, so it is not binding in practice.
- *Offers and dispatch are solved in sequence.* Offers are fixed at the bid deadline together
  with a trading plan, and dispatch then re-optimises around them as prices and delivery
  arrive, as an operator would. Treating the forecast as certain in both is the larger
  simplification: one shrink applies to every day, where a forecast that knew how uncertain
  each day was could shrink hard on some and barely on others. See
  [Swierczynski et al. (2021)](https://doi.org/10.3390/en14248365) and
  [Bai et al. (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0306261924015149)
  for joint formulations.
- *Conservative information at the offer stage.* Offers see data only to the end of D-2, not
  D-1's morning or the day-ahead auction results a real operator has at 14:00.
- *Delivery follows frequency instantly, product by product.* The Service Terms allow up to
  10 seconds to reach full delivery, which moves little energy over half an hour, and each
  product follows its own curve, which sums to the same stacked curve NESO uses.
- *Baselines change within the half-hour.* Recovery trades take effect in the settlement
  period they are planned for; NESO's baseline submission timings are not modelled.
- *Recovery through the reserve is planned at cost, not value.* The plan counts what energy
  moved through Reserved Capacity costs at the price but never what it earns, so the reserve
  is not used to trade. Where little trading power is left, the free energy DR High absorbs
  is sold whenever recovery needs it rather than at the best price.
- *No NESO discretion.* NESO may choose not to penalise a unit during extended deviations
  beyond 0.1 Hz or multiple events (Service Terms 6.11 vi); the model always counts a missed
  requirement.
- *Unavailability can run longer in practice.* Where NESO judges a unit's state of energy
  non-compliant, it may treat the unit as unavailable not only in that settlement period but
  in every one after it until satisfied that compliance is restored
  ([Service Terms](https://www.neso.energy/document/384606/download) 6.12). The model counts
  only the periods that start outside the requirement, so it understates that exposure.
- *Expected delivery costs are recent averages.* Offers price delivery on the previous four
  weeks and the previous week's prices, not on a forecast of either.
- *Walk-forward, but not nested.* Every prediction is out-of-sample, refit quarterly on an
  expanding window, which is what lets the five-year revenue comparison mean what it says.
  Three things remain: hyperparameters were chosen once rather than re-selected inside each
  fold, a rolling training window is untested against the expanding one, and quarterly is a
  modelling choice where an operator might retrain monthly. Tree models also still cannot
  extrapolate beyond the price range they have seen, which is a real limit rather than noise.
- *Price-taker.* The battery's offers are assumed not to move clearing prices, backed by the
  20% auction-size limit. That is why results scale linearly with power, and why the dispatch
  page stops at 100 MW.
- *Continuous MW, unlimited orders.* EAC trades whole MW and caps the number of orders per
  unit per day; the LP uses continuous MW and any combination of products.

**Battery degradation (not yet modelled)**

A real asset degrades through calendar ageing (capacity fade at rest) and cycle ageing
(accelerated by depth of discharge, C-rate and temperature). A more complete model would
track state-of-health across the backtest, apply a degradation-aware dispatch policy trading
short-term revenue against cycle-life consumption, and incorporate chemistry-specific
degradation curves (NMC, LFP), which differ materially. The cycling wear cost is a
simplified financial proxy and does not capture the compounding, path-dependent nature of
real degradation.

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

Coverage is read from the processed datasets at build time rather than hardcoded, so this
table cannot drift out of date when the pipeline is re-run. Both NESO and Elexon APIs are
fully public and require no key.

<b>Overlapping pulls resolve to the most recent one.</b> Data is collected in date-ranged
batches that overlap, and the trailing days of any batch are provisional — Elexon moves
system prices through several settlement runs, and NESO revises embedded solar and wind
after the fact. Where two batches cover the same settlement period and disagree, the batch
collected later is kept, so a settled value always displaces the estimate it replaces.
[NESO Data Portal](https://www.neso.energy/data-portal) ·
[Elexon Insights](https://developer.data.elexon.co.uk/) ·
[DESNZ REPD](https://www.gov.uk/government/publications/renewable-energy-planning-database-monthly-extract)

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
