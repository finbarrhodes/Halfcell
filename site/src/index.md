# Halfcell

Welcome to Halfcell, an interactive analytics tool for grid-scale battery storage
markets in Great Britain. Battery energy storage systems (BESS) are a complex part
of Great Britain's energy transition; this tool is designed to help unpack how they
operate, how they make money, and how market conditions and data science are
impacting BESS' role in the grid.

```js
// Kept in its own cell, and ahead of the parquet load below. Every variable a
// Framework cell declares resolves together, so bundling these 6 KB of JSON with
// the parquet made the Market snapshot cards wait on parquet-wasm — a 6 MB WASM
// module they do not need. Split, the cards paint as soon as the JSON lands.
const [kpis, manifest] = await Promise.all([
  FileAttachment("data/kpis.json").json(),
  FileAttachment("data/manifest.json").json(),
]);
```

```js
import {STRATEGY_COLOURS, rangeFor} from "./components/theme.js";

const revenue = await FileAttachment("data/revenue-monthly.parquet").parquet();
```

```js
// Dates are UTC calendar days; formatting them in local time would shift them west of GMT.
const ml = manifest.ml_mpc;
const fmtGbp = (v) =>
  Math.abs(v) >= 1e6 ? `£${(v / 1e6).toFixed(2)}M` : `£${(v / 1e3).toFixed(0)}k`;
const fmtDay = (iso) => new Date(iso).toLocaleDateString("en-GB", {day: "numeric", month: "long", year: "numeric", timeZone: "UTC"});
const fmtDayShort = (iso) => new Date(iso).toLocaleDateString("en-GB", {day: "numeric", month: "short", year: "numeric", timeZone: "UTC"});
const fmtMonth = (ym) => new Date(`${ym}-01`).toLocaleDateString("en-GB", {month: "long", year: "numeric", timeZone: "UTC"});
const fmtChange = (now, then) => `${now >= then ? "+" : "−"}${Math.abs((now / then - 1) * 100).toFixed(0)}%`;
```

## What's in this tool

<div class="grid grid-cols-2">
  <div class="card nav">
    <h2><a href="./dashboard">Market Overview →</a></h2>
    <p>What markets do batteries operate in? A look into frequency response auctions (DC, DR, DM),
    wholesale and system prices, and generation mix trends.</p>
  </div>
  <div class="card nav">
    <h2><a href="./backtester">Forecasting & Dispatch →</a></h2>
    <p>A day-ahead modelling framework for FR/arbitrage capacity allocation and MPC
    dispatch, benchmarking three price forecasting strategies.</p>
  </div>
  <div class="card nav">
    <h2><a href="./research">Research Experiments →</a></h2>
    <p>What the forecast is actually worth, and the experiments measured against it: model choice,
    features, offer discounting, guard bands and reserve timing — most of which did not ship.</p>
  </div>
  <div class="card nav">
    <h2><a href="./methodology">Methodology & Data →</a></h2>
    <p>How the model works, the NESO rules and settlement mechanics it runs under, its known
    limitations, and the data behind it.</p>
  </div>
</div>

## Where data science meets the clean energy transition

Halfcell asks what data science can actually contribute to clean tech, using a concrete
case: a grid-scale battery deciding, every day, how to divide its capacity between
frequency response and wholesale arbitrage. This capacity allocation decision rests on a
forecast of tomorrow's prices, introducing a modelling problem and an opportunity for data
science methods to provide real value in BESS operations.

The chart below runs three different strategies through the same dispatch engine on the
same asset. When modelling battery revenues in a price forecasting setting, it is useful to
compare any machine learning implementation to a reasonable floor & ceiling.
**Perfect Foresight** knows tomorrow's prices and marks the ceiling, and the **Naive** model
takes the predictions out of the question and uses today's price as the prediction for
tomorrow, marking the floor and a bar any real model has to clear. A **Random Forest**
trained on lagged prices, generation mix and cyclical time features sits between the two,
and the gap it closes is the value the modelling adds.

```js
const cumulative = (() => {
  const rows = revenue.toArray().map((d) => ({...d}));
  const byStrategy = d3.group(rows, (d) => d.strategy);
  const out = [];
  for (const [strategy, rs] of byStrategy) {
    let total = 0;
    for (const r of d3.sort(rs, (d) => d.month_dt)) {
      const services = ["DCH", "DCL", "DMH", "DML", "DRH", "DRL"];
      const gross =
        d3.sum(services, (s) => r[`${s}_rev`] ?? 0) + (r.imbalance_revenue_gbp ?? 0);
      // Wear on every MWh discharged, trades and response delivery alike, as in the
      // published totals: without the delivery term the chart ran 0.3-0.4% high
      total += gross - (r.cycling_cost_gbp ?? 0) - (r.delivery_cycling_cost_gbp ?? 0);
      out.push({strategy, month: new Date(r.month_dt), total});
    }
  }
  return out;
})();

const labels = {pf_mpc: "Perfect Foresight", naive_mpc: "Naive (D-1)", ml_mpc: "ML Model"};
const strategies = Object.keys(labels);
```

```js
display(resize((width) => Plot.plot({
  width,
  height: 340, marginBottom: 36,
  marginLeft: 60,
  x: {label: null},
  y: {label: "Cumulative net revenue (£M)", transform: (d) => d / 1e6, grid: true},
  color: {
    legend: true,
    domain: strategies,
    range: rangeFor(STRATEGY_COLOURS, strategies),
    tickFormat: (d) => labels[d],
  },
  marks: [
    Plot.ruleY([0]),
    Plot.line(cumulative, {x: "month", y: "total", stroke: "strategy", strokeWidth: 2}),
  ],
})));
```

## How the market got here

Five years in which frequency response went from the battery fleet's main income to a minor
one, and the fleet turned to wholesale trading and the Balancing Mechanism.

| Year | What changed |
|---|---|
| **2020** | National Grid ESO, now NESO, launches Dynamic Containment on 1 October: sub-second response that only fast assets can provide, and batteries are its first providers ([NESO](https://www.neso.energy/news/national-grid-eso-debuts-dynamic-containment-frequency-response-service)) |
| **2021** | DC moves to day-ahead auctions for each EFA block; the auction results on this site begin on 16 September |
| **2022** | The revenue peak. DM and DR join DC on 26 March, though a unit can still offer only one service per block. DC Low averages £17.51/MW/h over the year, and the fleet a record £156k/MW, 63% of it from DC ([Modo](https://modoenergy.com/research/modo-2022-review-part-2-battery-energy-storage)) |
| **Late 2022** | New capacity outruns what NESO buys: DC Low falls from £37.04/MW/h in June to £6.34 in December |
| **2023** | DC Low averages £2.70/MW/h, and fleet revenue falls to £51k/MW, or £65k with the Capacity Market, whose share reaches 30% by December ([Modo](https://modoenergy.com/research/modo-battery-energy-storage-year-review-2023-capacity-revenues-frequency-response)). On 2 November the Enduring Auction Capability lets a unit split capacity across DC, DM and DR in one block, and prices can go below zero |
| **2024** | Revenue moves to wholesale trading and the Balancing Mechanism, where battery dispatch reaches a record 141 GWh; the fleet averages £50k/MW, and two-thirds of new capacity is two-hour ([Modo](https://modoenergy.com/research/en/gb-battery-energy-storage-markets-2024-year-in-review-great-britain-wholesale-balancing-mechanism-frequency-response-reserve)) |
| **2025–26** | Response prices stay near their floor, and DR High clears below zero in most blocks (77% in 2025, 92% in 2026 to date), paid for by the energy it absorbs ([methodology](./methodology#negative-clearing-prices)) |

Clearing prices, dates and shares of blocks come from the auction data behind this site. Fleet
revenues are Modo Energy's benchmark, which measures a different set of assets from anything
modelled here.

## Market snapshot

Where the market stood at the end of the data, ${fmtDay(kpis.spread_window_end)}, against a
year earlier. The figures update with each monthly refresh.

<div class="grid grid-cols-4">
  <div class="card kpi">
    <h2>GB battery fleet</h2>
    <span class="big">${(kpis.fleet_mw / 1e3).toFixed(1)} GW</span>
    <div class="muted">operational in ${fmtMonth(kpis.fleet_month)} · ${fmtChange(kpis.fleet_mw, kpis.fleet_mw_year_earlier)} on a year earlier · REPD, recent months provisional</div>
  </div>
  <div class="card kpi">
    <h2>Wholesale spread, last ${kpis.spread_window_days} days</h2>
    <span class="big">£${kpis.spread_30d_avg.toFixed(0)}/MWh</span>
    <div class="muted">average daily peak-to-trough · £${kpis.spread_30d_avg_year_earlier.toFixed(0)} a year earlier (${fmtChange(kpis.spread_30d_avg, kpis.spread_30d_avg_year_earlier)})</div>
  </div>
  <div class="card kpi">
    <h2>Modelled revenue — ML strategy</h2>
    <span class="big">£${(ml.summary.annualised_per_mw / 1e3).toFixed(0)}k</span>
    <div class="muted">per MW per year · ${ml.params.power_mw} MW / ${ml.params.duration_h}h reference asset</div>
  </div>
  <div class="card kpi">
    <h2>Data through</h2>
    <span class="big">${fmtDayShort(ml.params.end_date)}</span>
    <div class="muted">${ml.summary.years_covered} years backtested</div>
  </div>
</div>
