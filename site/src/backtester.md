# Day-Ahead Forecasting & Dispatch Model

Grid-scale batteries earn from several markets at once. This model covers the **day-ahead
decision layer**: given yesterday's auction results and price data, how should an operator
split capacity between frequency response & wholesale arbitrage, and how much does forecast
quality actually change the outcome?

The walkthrough below follows a single winter day — 8 January 2026 — from the raw price
curve through to the dispatch the model settles on. Scroll, or select any step directly.

```js
import {SERVICE_COLOURS, SERVICE_LABELS, STRATEGY_COLOURS, STRATEGY_LABELS, gbp} from "./components/theme.js";
import {choiceGroup, controlPanel, dateRange, dayViews, slider} from "./components/controls.js";

const manifest = await FileAttachment("data/manifest.json").json();
const revenueAll = (await FileAttachment("data/revenue-monthly.parquet").parquet())
  .toArray().map((d) => ({...d, month_dt: new Date(d.month_dt)}));
const scenarioAll = (await FileAttachment("data/revenue-scenarios.parquet").parquet())
  .toArray().map((d) => ({...d, month_dt: new Date(d.month_dt)}));
const socAll = (await FileAttachment("data/soc-week.parquet").parquet())
  .toArray().map((d) => ({...d, month_dt: new Date(d.month_dt)}));
```

```js
const ALL_SERVICES = ["DCH", "DCL", "DMH", "DML", "DRH", "DRL"];
const BASE_POWER_MW = manifest.ml_mpc.params.power_mw;   // cache is computed at this rating
const MAX_POWER_MW = 100;                                // NESO Maximum Sell Size per product
const DURATION_H = manifest.ml_mpc.params.duration_h;
const EFF = manifest.ml_mpc.params.efficiency_rt;
const BASE_CYCLING = manifest.ml_mpc.params.cycling_cost_per_mwh;

const SCENARIO_LABELS = {full: "FR + arbitrage", fr_only: "FR only", arb_only: "Arbitrage only"};

const bounds = (() => {
  const starts = Object.values(manifest).map((m) => m.params.start_date).filter(Boolean);
  const ends = Object.values(manifest).map((m) => m.params.end_date).filter(Boolean);
  return [new Date(d3.min(starts)), new Date(d3.max(ends))];
})();
```


```js
import {watchSteps} from "./components/scrolly.js";
const sampleDay = (await FileAttachment("data/sample-day.parquet").parquet()).toArray();
```

<div class="scrolly" id="walkthrough">
<div class="scrolly-steps">

<div class="step"><div class="step-inner">
<span class="step-num">Step 1</span>

### A day of wholesale prices

To the right are half-hourly APXMIDP prices for a random GB winter day (January 8th 2026 here). Prices
run from £73 to £291/MWh, defined by an overnight trough, a morning rise, and the evening peak when
demand is highest, wind may ease, and there is no solar generation.

That spread is the raw arbitrage opportunity: buy low, sell high.
</div></div>

<div class="step"><div class="step-inner">
<span class="step-num">Step 2</span>

### Charge cheap, discharge at the peak

The obvious arbitrage value for battery sites: Charge through the overnight trough, discharge into the evening peak.

The catch is that a 2-hour battery can only shift so much energy, and every cycle costs
something in degradation — so the model has to pick *which* periods are worth trading, not
simply trade the extremes.
</div></div>

<div class="step"><div class="step-inner">
<span class="step-num">Step 3</span>

### But frequency response pays for sitting still

NESO pays a **£/MW/h availability fee** for capacity held ready to respond within seconds,
whether or not it is ever called. That income is contracted a day ahead: offers for day D
close at 14:00 on D-1.

Each contract constrains the asset. A **Low** contract needs enough energy in store to
discharge for its full delivery window — 15 minutes for DC, 30 for DM, 60 for DR — and a
**High** contract needs the same again as headroom to absorb. The power a contract uses is
also unavailable for trading.

And contracts get called on. As frequency wanders, a battery holding DR moves energy in and
out almost all the time. That energy is neither paid for nor charged, so whatever a Low
contract gives away has to be bought back, and whatever a High contract absorbs can be sold
on.
</div></div>

<div class="step"><div class="step-inner">
<span class="step-num">Step 4</span>

### Stage 1 — decide what to offer

For each of the six EFA blocks, the model offers every product at its opportunity cost: the
trading that capacity would otherwise do, plus the expected cost of the energy the product
will deliver. It prices the trading the way a dispatcher would, by **planning**: the day's
holdings are chosen together with a half-hourly trading plan from 14:00 to the end of the
next day, so a holding costs whatever it takes out of that plan. Holding a High product
overnight, when the plan needs to charge, costs more than holding it at the evening peak, when
the plan is selling. The plan uses only the forecast that existed at 14:00, pulled halfway
towards its daily mean so that it does not pay to keep capacity free for spreads the forecast
merely imagines. It keeps the combination that earns most at the clearing prices, within
NESO's rules:

- the MW it sells in each direction, plus the share NESO reserves on the other side for
  recovering delivered energy, fit within the battery's power rating
- there is enough energy in store to deliver every Low contract for its full window, and
  enough empty space to absorb every High one
- starting from how full the battery actually is at 14:00, it can get into the charge range
  each block needs in time, using only the power its contracts leave free
- it holds no more than a fifth of what any auction cleared, small enough that its own
  offers would not have set the price

Before EAC went live in November 2023, a unit could offer only one service per block, chosen
here on the previous day's prices.
</div></div>

<div class="step"><div class="step-inner">
<span class="step-num">Step 5</span>

### Stage 2 — dispatch under constraint

With the offers set, a **linear programme** plans charge and discharge at half-hourly
resolution, re-solving every period and executing only the first — model predictive control.
Each plan runs to the end of tomorrow, the furthest any forecast yet exists for: today on the
day-ahead forecast, tomorrow on the one made a day earlier.

The shaded band is the range NESO requires: enough energy in store for the Low contracts
and enough headroom for the High ones. Each block starts at the full amount. Delivering
response lowers it, it climbs back by at most a fifth each half-hour, and the battery trades
to keep up. Starting a half-hour outside it counts as unavailability and forfeits that
period's payment.
</div></div>

<div class="step"><div class="step-inner">
<span class="step-num">Step 6</span>

### Forecast quality is the variable under test

All three strategies run the *same* allocation and dispatch engine. Only the price signal
differs: **Perfect Foresight** sees actual day-D prices, **Naive** reuses the last complete
day's, and the **ML model** — a Random Forest on lagged prices, generation mix and cyclical
time features — predicts them from data that already existed when each decision was made.

Where the traces diverge is the cost of forecast error. The analysis below quantifies it.
</div></div>

</div>
<div class="scrolly-graphic">
<div class="scrolly-rail" id="walkthrough-rail"></div>
<div id="walkthrough-figure"></div>
</div>
</div>

```js
const dayPrices = sampleDay.filter((d) => d.strategy === "ml_mpc")
  .map((d) => ({sp: d.sp, price: d.price}));

const cheap = [...dayPrices].sort((a, b) => a.price - b.price).slice(0, 8).map((d) => d.sp);
const dear  = [...dayPrices].sort((a, b) => b.price - a.price).slice(0, 8).map((d) => d.sp);

const traceFor = (key) => sampleDay.filter((d) => d.strategy === key)
  .map((d) => ({sp: d.sp, soc: d.soc_frac, strategy: STRATEGY_LABELS[key]}));
const bandFor = (key) => sampleDay.filter((d) => d.strategy === key)
  .map((d) => ({sp: d.sp, lo: d.soc_min_frac, hi: d.soc_max_frac}));

const spAxis = {label: "Settlement period", ticks: [1, 12, 24, 36, 48], domain: [1, 48]};
```

```js
// The walkthrough graphic is rendered imperatively rather than reactively: the
// step observer calls renderStep directly. Routing it through a reactive value
// meant the figure cell did not re-evaluate on change, and an explicit call is
// easier to follow than the dependency it replaced.
function buildFigure(s) {
  if (s <= 0) {
    return Plot.plot({height: 380, marginLeft: 55, x: spAxis,
      y: {label: "£/MWh", grid: true},
      marks: [Plot.ruleY([0]),
              Plot.line(dayPrices, {x: "sp", y: "price", stroke: "#0D7680", strokeWidth: 2})]});
  }

  if (s === 1) {
    return Plot.plot({height: 380, marginLeft: 55, x: spAxis,
      y: {label: "£/MWh", grid: true},
      marks: [
        Plot.ruleY([0]),
        Plot.line(dayPrices, {x: "sp", y: "price", stroke: "#33302E", strokeWidth: 1.5}),
        Plot.dot(dayPrices.filter((d) => cheap.includes(d.sp)),
          {x: "sp", y: "price", fill: "#0D7680", r: 5, symbol: "square"}),
        Plot.dot(dayPrices.filter((d) => dear.includes(d.sp)),
          {x: "sp", y: "price", fill: "#C9400A", r: 5}),
        Plot.text([{sp: cheap[0], price: d3.min(dayPrices, (d) => d.price)}],
          {x: "sp", y: "price", text: ["charge"], dy: 20, fill: "#0D7680", fontWeight: 600}),
        Plot.text([{sp: dear[0], price: d3.max(dayPrices, (d) => d.price)}],
          {x: "sp", y: "price", text: ["discharge"], dy: -14, fill: "#C9400A", fontWeight: 600}),
      ]});
  }

  if (s === 2 || s === 3) {
    // FR availability is flat within a block and known a day ahead; the contrast
    // with the volatile spot curve is the point.
    return Plot.plot({height: 380, marginLeft: 55, x: spAxis,
      y: {label: "£/MWh (spot)", grid: true},
      marks: [
        Plot.ruleY([0]),
        Plot.line(dayPrices, {x: "sp", y: "price", stroke: "#33302E",
                              strokeWidth: 1.2, strokeOpacity: 0.45}),
        Plot.ruleX([8.5, 16.5, 24.5, 32.5, 40.5],
          {stroke: "#9C948E", strokeDasharray: "3 3"}),
        Plot.text(d3.range(6).map((i) => ({x: i * 8 + 4.5, label: `EFA ${i + 1}`})),
          {x: "x", y: d3.max(dayPrices, (d) => d.price) * 0.96,
           text: "label", fill: "#66605C", fontSize: 10}),
      ]});
  }

  const traces = s >= 5
    ? ["pf_mpc", "naive_mpc", "ml_mpc"].flatMap(traceFor)
    : traceFor("ml_mpc");
  // The required range depends on what each strategy offered, so it is only
  // drawn while a single strategy is on screen.
  const band = s === 4 ? bandFor("ml_mpc") : [];

  return Plot.plot({
    height: 380, marginLeft: 55, x: spAxis,
    y: {label: "State of charge", domain: [0, 1], tickFormat: ".0%", grid: true},
    color: {legend: s >= 5, domain: Object.values(STRATEGY_LABELS),
            range: ["#4E8A3C", "#C9400A", "#0D7680"]},
    marks: [
      Plot.areaY(band, {x: "sp", y1: "lo", y2: "hi", curve: "step-after",
                        fill: "#0D7680", fillOpacity: 0.1}),
      Plot.line(band, {x: "sp", y: "lo", curve: "step-after", stroke: "#0D7680", strokeDasharray: "4 3"}),
      Plot.line(band, {x: "sp", y: "hi", curve: "step-after", stroke: "#0D7680", strokeDasharray: "4 3"}),
      Plot.line(traces, {x: "sp", y: "soc",
                         stroke: s >= 5 ? "strategy" : () => "#0D7680", strokeWidth: 2}),
    ],
  });
}

{
  const root = document.getElementById("walkthrough");
  const target = document.getElementById("walkthrough-figure");
  const rail = document.getElementById("walkthrough-rail");
  if (root && target) {
    const stop = watchSteps(root, (i) => target.replaceChildren(buildFigure(i)), {rail});
    invalidation.then(stop);
  }
}
```

## The model in full

Everything below runs that same engine across the whole backtest window rather than one day.
Set the asset, the price signal it trades on and the markets it may bid into; every figure on
the rest of the page follows the selection.

```js
// Built here and observed in the next block rather than through view(), which
// would display each control where its cell sits — five stacked form rows
// instead of one panel.
const powerInput = slider({
  min: 1, max: MAX_POWER_MW, step: 1, value: BASE_POWER_MW, unit: "MW", label: "Asset power",
});
const strategyInput = choiceGroup(Object.keys(STRATEGY_LABELS), {
  value: "ml_mpc", format: (k) => STRATEGY_LABELS[k], label: "Price signal",
});
const scenarioInput = choiceGroup(Object.keys(SCENARIO_LABELS), {
  value: "full", format: (k) => SCENARIO_LABELS[k], label: "Markets",
});
const dateInputs = dateRange({value: bounds, min: bounds[0], max: bounds[1]});

display(controlPanel([
  {label: "Asset power", input: powerInput},
  {label: "Price signal", input: strategyInput},
  {label: "Markets", input: scenarioInput},
  {label: "Date range", input: dateInputs},
], {title: "Controls"}));
```

```js
const powerMw = Generators.input(powerInput);
const strategyPick = Generators.input(strategyInput);
const scenarioPick = Generators.input(scenarioInput);
const fromPick = Generators.input(dateInputs.from);
const toPick = Generators.input(dateInputs.to);
```

<div class="muted controls-note">
Every figure comes from a precomputed run for a ${BASE_POWER_MW} MW / ${BASE_POWER_MW * DURATION_H} MWh
reference asset and scales linearly with power at fixed duration. That holds while the battery
is a price-taker, too small for its offers to move clearing prices. The slider stops at
${MAX_POWER_MW} MW, NESO's Maximum Sell Size for a single product. Which products the battery
holds is decided by the model under NESO's rules rather than chosen here; the revenue breakdown
below shows what it held.
</div>

```js
// Revenue scales linearly with power at fixed duration, so scaling the cached monthly
// table by the power ratio is exact rather than an approximation.
const scale = powerMw / BASE_POWER_MW;

// The cached table is monthly, and the cache bounds are mid-month dates
// (2021-09-16 / 2026-08-17). Comparing a month-start against a mid-month bound
// would silently drop the first and last months, so widen to whole months.
const fromMonth = d3.utcMonth.floor(fromPick);
const toMonth = d3.utcMonth.floor(toPick);
const inRange = (d) => d.month_dt >= fromMonth && d.month_dt <= toMonth;

const COLUMNS = [...ALL_SERVICES.map((s) => `${s}_rev`), "imbalance_revenue_gbp",
                 "cycling_cost_gbp", "mwh_cycled", "delivery_mwh", "delivery_cycling_cost_gbp"];
const TRADING = "Wholesale trading";

// Each scenario is its own backtest. FR-only needs no price forecast, so one
// run (strategy "all") serves every price signal.
function rowsFor(strategy, scenario) {
  const source = scenario === "full"
    ? revenueAll.filter((d) => d.strategy === strategy)
    : scenarioAll.filter((d) => d.scenario === scenario
                              && (d.strategy === strategy || d.strategy === "all"));
  return source
    .filter(inRange)
    .map((d) => {
      const row = {month_dt: d.month_dt};
      for (const c of COLUMNS) row[c] = (d[c] ?? 0) * scale;
      return row;
    })
    .sort((a, b) => a.month_dt - b.month_dt);
}

function summarise(rows, mw) {
  if (!rows.length) return null;
  const svc = {};
  for (const s of ALL_SERVICES) svc[s] = d3.sum(rows, (d) => d[`${s}_rev`]);
  const arb = d3.sum(rows, (d) => d.imbalance_revenue_gbp);
  // Wear on every MWh discharged: trades and energy delivered under FR contracts alike
  const cyc = d3.sum(rows, (d) => d.cycling_cost_gbp + d.delivery_cycling_cost_gbp);
  const gross = d3.sum(Object.values(svc)) + arb;
  const net = gross - cyc;
  // Days each month actually contributes, clipped to the backtest window. The first
  // and last months are partial (the window runs 2021-09-16 to 2026-08-17), so
  // counting whole months divides part-month revenue by a full month of time.
  const DAY = 86400000;
  const days = d3.sum(rows, (d) => {
    const lo = Math.max(d.month_dt, bounds[0]);
    const hi = Math.min(d3.utcMonth.offset(d.month_dt, 1) - DAY, bounds[1]);
    return Math.max(0, (hi - lo) / DAY + 1);
  });
  const years = days / 365.25;
  // Negative streams are kept: FR services can clear below zero, and filtering
  // them out would hide that from the breakdown table and the revenue stack.
  const breakdown = Object.fromEntries(
    Object.entries({...svc, [TRADING]: arb}).filter(([, v]) => v !== 0)
  );
  return {
    gross, cyc, net, years,
    annualised: years > 0 ? net / years : 0,
    perMw: years > 0 && mw > 0 ? net / years / mw : 0,
    mwhCycled: d3.sum(rows, (d) => d.mwh_cycled + d.delivery_mwh),
    breakdown,
    top: d3.greatest(Object.entries(breakdown), (d) => d[1])?.[0] ?? "—",
  };
}

const monthly = rowsFor(strategyPick, scenarioPick);
const summary = summarise(monthly, powerMw);
const breachPeriods = manifest[strategyPick].summary.soe_breach_periods;
```

## Results

<div class="grid grid-cols-4">
<div class="card kpi"><h2>Total net revenue</h2><span class="big">${summary ? gbp(summary.net) : "—"}</span></div>
<div class="card kpi"><h2>Annualised net</h2><span class="big">${summary ? gbp(summary.annualised) : "—"}</span><div class="muted">per year</div></div>
<div class="card kpi"><h2>Revenue per MW</h2><span class="big">${summary ? "£" + (summary.perMw / 1e3).toFixed(1) + "k" : "—"}</span><div class="muted">per MW per year</div></div>
<div class="card kpi"><h2>Top revenue stream</h2><span class="big">${summary ? (SERVICE_LABELS[summary.top] ?? summary.top) : "—"}</span></div>
</div>

Modelling a **${powerMw} MW / ${(powerMw * DURATION_H).toFixed(0)} MWh** asset
(${DURATION_H}h duration, ${(EFF * 100).toFixed(0)}% round-trip efficiency) in
**${SCENARIO_LABELS[scenarioPick]}**, using **${STRATEGY_LABELS[strategyPick]}** price signals
and MPC dispatch over a rolling horizon to the end of tomorrow.

```js
if (breachPeriods != null) display(html`<div class="muted">
Over the full backtest, this strategy started ${d3.format(",")(breachPeriods)} committed
half-hour${breachPeriods === 1 ? "" : "s"} outside the state-of-charge range its contracts
required. Each counts as unavailability and forfeits that period's payment, which is already
deducted above.
</div>`);
```

### Monthly revenue stack

```js
const streams = [...ALL_SERVICES.map((s) => ({key: `${s}_rev`, label: s})),
                 {key: "imbalance_revenue_gbp", label: TRADING}];

const stacked = monthly.flatMap((d) => [
  ...streams
    .filter((s) => (d[s.key] ?? 0) !== 0)
    .map((s) => ({month: d.month_dt, stream: SERVICE_LABELS[s.label] ?? s.label,
                  colourKey: s.label, value: d[s.key] / 1e3})),
  ...(d.cycling_cost_gbp + d.delivery_cycling_cost_gbp > 0
    ? [{month: d.month_dt, stream: "Cycling wear cost", colourKey: "Cycling cost",
        value: -(d.cycling_cost_gbp + d.delivery_cycling_cost_gbp) / 1e3}]
    : []),
]);

const streamDomain = [...ALL_SERVICES.map((s) => SERVICE_LABELS[s]), TRADING, "Cycling wear cost"];
const streamRange = [...ALL_SERVICES.map((s) => SERVICE_COLOURS[s]),
                     SERVICE_COLOURS.Arbitrage, SERVICE_COLOURS["Cycling cost"]];

display(resize((width) => Plot.plot({
  width, height: 430, marginLeft: 62, marginBottom: 36,
  x: {label: null, interval: "month"},
  y: {label: "£k", grid: true},
  color: {domain: streamDomain, range: streamRange, legend: true},
  marks: [
    Plot.ruleY([0]),
    Plot.rectY(stacked, {x: "month", y: "value", fill: "stream", interval: "month",
                         tip: true, order: streamDomain}),
  ],
})));
```

Each bar shows gross revenue by stream for that month (positive) and cycling wear cost
(negative, dark red). Net revenue is the algebraic sum of all segments — months with
heavier arbitrage dispatch carry larger cycling deductions.

### State of charge: single days, and the average week

```js
const socDaysTable = await FileAttachment("data/soc-days.parquet").parquet();
const dayPricesTable = await FileAttachment("data/day-prices.parquet").parquet();
```

```js
// Recombine the pre-aggregated sufficient statistics over the selected months.
// Summing count/total/total_sq recovers the exact mean and sd of the raw trajectory.
const socWeek = (() => {
  const rows = socAll.filter(
    (d) => d.strategy === strategyPick && inRange(d)
  );
  return Array.from(
    d3.rollup(rows, (v) => {
      const n = d3.sum(v, (d) => d.n);
      const mean = d3.sum(v, (d) => d.total) / n;
      const variance = Math.max(d3.sum(v, (d) => d.total_sq) / n - mean * mean, 0);
      const sd = Math.sqrt(variance);
      return {mean, lo: Math.max(mean - sd, 0), hi: Math.min(mean + sd, 1),
              reqLo: d3.sum(v, (d) => d.min_total) / n,
              reqHi: d3.sum(v, (d) => d.max_total) / n};
    }, (d) => d.period_in_week),
    ([p, s]) => ({period: p, ...s})
  ).sort((a, b) => a.period - b.period);
})();

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
```

```js
// The day tables, read once as typed columns. soc-days.parquet is sorted by
// strategy, day and period with exactly 48 rows per day, so the start of each
// day's block is all the index a lookup needs.
const SOC_STEPS = 250;                                   // as encoded by the loader
const STRATEGY_CODES = ["pf_mpc", "naive_mpc", "ml_mpc"];
const socCol = socDaysTable.getChild("soc").toArray();
const loCol = socDaysTable.getChild("lo").toArray();
const hiCol = socDaysTable.getChild("hi").toArray();

const dayStart = (() => {
  const strategies = socDaysTable.getChild("strategy").toArray();
  const days = socDaysTable.getChild("day").toArray();
  const index = new Map();
  for (let i = 0; i < days.length; i += 48) index.set(`${STRATEGY_CODES[strategies[i]]}|${days[i]}`, i);
  return index;
})();

const priceByDay = (() => {
  const days = dayPricesTable.getChild("day").toArray();
  const sps = dayPricesTable.getChild("sp").toArray();
  const prices = dayPricesTable.getChild("price").toArray();
  const out = new Map();
  for (let i = 0; i < days.length; i++) {
    let row = out.get(days[i]);
    if (!row) out.set(days[i], (row = new Array(48).fill(null)));
    row[sps[i] - 1] = prices[i];
  }
  return out;
})();

const dayNumber = (date) => Math.round(date.getTime() / 864e5);
const fmtDay = d3.utcFormat("%a %-d %b %Y");
// Settlement period 1 starts at 23:00 the evening before the service day.
const clockAt = (sp) => {
  const minutes = ((sp - 1) * 30 + 23 * 60) % 1440;
  return `${String(Math.floor(minutes / 60)).padStart(2, "0")}:${String(minutes % 60).padStart(2, "0")}`;
};
```

```js
// Days worth opening on. Each is picked for what it shows rather than for being
// typical, and the free date box below reaches the rest of the backtest.
const socViews = [
  {key: "solar", label: "A solar trough", date: new Date("2026-05-17"),
   note: "Sun 17 May 2026 · £55 at 14:00, £141 at 18:00"},
  {key: "spike", label: "An evening spike", date: new Date("2025-09-08"),
   note: "Mon 8 Sep 2025 · £250 at 18:00"},
  {key: "negative", label: "Paid to charge", date: new Date("2026-04-07"),
   note: "Tue 7 Apr 2026 · −£57 at midday"},
  {key: "week", label: "Average week", note: "Mean and spread across every week"},
  {key: "any", label: "Any other day", date: new Date("2026-05-17"), note: "Set the date below"},
];
const socViewPicker = dayViews(socViews, {min: bounds[0], max: bounds[1],
                                          value: socViews[0], label: "State of charge view"});
const socView = Generators.input(socViewPicker);
```

```js
function dayRows(key, date) {
  const start = dayStart.get(`${key}|${dayNumber(date)}`);
  if (start === undefined) return null;
  const prices = priceByDay.get(dayNumber(date)) ?? [];
  return d3.range(48).map((k) => ({
    sp: k + 1,
    strategy: STRATEGY_LABELS[key],
    soc: socCol[start + k] / SOC_STEPS,
    lo: loCol[start + k] / SOC_STEPS,
    hi: hiCol[start + k] / SOC_STEPS,
    price: prices[k],
  }));
}

const strategyDomain = STRATEGY_CODES.map((k) => STRATEGY_LABELS[k]);
const strategyRange = STRATEGY_CODES.map((k) => STRATEGY_COLOURS[k]);

function weekChart(width) {
  return Plot.plot({
    width, height: 340, marginLeft: 55, marginRight: 55,
    x: {label: "Day of week", ticks: d3.range(7).map((d) => d * 48),
        tickFormat: (d) => DAYS[d / 48], domain: [0, 336]},
    y: {label: "State of charge", domain: [0, 1], tickFormat: ".0%", grid: true},
    marks: [
      // Average range the FR contracts required at each point in the week
      Plot.areaY(socWeek, {x: "period", y1: "reqLo", y2: "reqHi", fill: "#0D7680", fillOpacity: 0.08}),
      Plot.line(socWeek, {x: "period", y: "reqLo", stroke: "#0D7680", strokeDasharray: "4 3", strokeWidth: 1}),
      Plot.line(socWeek, {x: "period", y: "reqHi", stroke: "#0D7680", strokeDasharray: "4 3", strokeWidth: 1}),
      Plot.ruleX(d3.range(1, 7).map((d) => d * 48), {stroke: "grey", strokeOpacity: 0.3, strokeDasharray: "2 3"}),
      Plot.areaY(socWeek, {x: "period", y1: "lo", y2: "hi", fill: "#C9400A", fillOpacity: 0.12}),
      Plot.line(socWeek, {x: "period", y: "mean", stroke: "#C9400A", strokeWidth: 2}),
      Plot.tip(socWeek, Plot.pointerX({
        x: "period", y: "mean",
        title: (d) => `${DAYS[Math.floor(d.period / 48)]} SP ${(d.period % 48) + 1}\nmean ${(d.mean * 100).toFixed(1)}%\n±1 sd ${(d.lo * 100).toFixed(1)}–${(d.hi * 100).toFixed(1)}%\nrequired ${(d.reqLo * 100).toFixed(0)}–${(d.reqHi * 100).toFixed(0)}% (avg)`,
      })),
    ],
  });
}

function dayChart(date, width) {
  const chosen = dayRows(strategyPick, date);
  if (!chosen) return html`<i>No dispatch for ${fmtDay(date)} — the backtest runs
    ${fmtDay(bounds[0])} to ${fmtDay(bounds[1])}.</i>`;
  const others = STRATEGY_CODES.filter((k) => k !== strategyPick)
    .flatMap((k) => dayRows(k, date) ?? []);

  // Price shares the state-of-charge axis, squeezed into it and read off on the
  // right. Its own scale is nice()d so the right-hand ticks land on round money.
  const shown = chosen.filter((d) => d.price != null);
  const priceScale = shown.length
    ? d3.scaleLinear(d3.extent(shown, (d) => d.price), [0.04, 0.96]).nice()
    : null;

  return Plot.plot({
    width, height: 340, marginLeft: 55, marginRight: 55,
    x: {label: "Settlement period", domain: [1, 48], ticks: [1, 12, 24, 36, 48]},
    y: {label: "State of charge", domain: [0, 1], tickFormat: ".0%", grid: true},
    color: {domain: strategyDomain, range: strategyRange, legend: true},
    marks: [
      Plot.ruleX([8.5, 16.5, 24.5, 32.5, 40.5], {stroke: "#9C948E", strokeOpacity: 0.5, strokeDasharray: "3 3"}),
      // What that day's contracts required: response energy in store, headroom above
      Plot.areaY(chosen, {x: "sp", y1: "lo", y2: "hi", curve: "step-after", fill: "#0D7680", fillOpacity: 0.1}),
      Plot.line(chosen, {x: "sp", y: "lo", curve: "step-after", stroke: "#0D7680",
                         strokeDasharray: "4 3", strokeWidth: 1, strokeOpacity: 0.6}),
      Plot.line(chosen, {x: "sp", y: "hi", curve: "step-after", stroke: "#0D7680",
                         strokeDasharray: "4 3", strokeWidth: 1, strokeOpacity: 0.6}),
      // Context, not a fourth trace: light enough to read behind the dispatch
      priceScale ? Plot.line(shown, {x: "sp", y: (d) => priceScale(d.price), stroke: "#9C948E",
                                     strokeWidth: 1, strokeOpacity: 0.9}) : null,
      // An explicit axis mark for the price suppresses Plot's implicit one, so the
      // state-of-charge axis has to be asked for by name as well.
      Plot.axisY({anchor: "left", label: "State of charge", tickFormat: ".0%"}),
      priceScale ? Plot.axisY(priceScale.ticks(5).map(priceScale), {anchor: "right", label: "£/MWh (wholesale)",
                                     tickFormat: (v) => d3.format(",.0f")(priceScale.invert(v))}) : null,
      Plot.line(others, {x: "sp", y: "soc", stroke: "strategy", strokeWidth: 1, strokeOpacity: 0.45}),
      Plot.line(chosen, {x: "sp", y: "soc", stroke: "strategy", strokeWidth: 2.6}),
      Plot.tip(chosen, Plot.pointerX({
        x: "sp", y: "soc",
        title: (d) => [`SP ${d.sp} · ${clockAt(d.sp)}`,
                       d.price == null ? "no price" : `wholesale £${d.price}/MWh`,
                       `${STRATEGY_LABELS[strategyPick]} ${(d.soc * 100).toFixed(0)}%`,
                       `required ${(d.lo * 100).toFixed(0)}–${(d.hi * 100).toFixed(0)}%`].join("\n"),
      })),
    ],
  });
}
```

<div class="soc-view">
  <div class="card">${scenarioPick !== "full"
    ? html`<i>The state-of-charge profile is shown for the FR + arbitrage run, where dispatch is simulated.</i>`
    : resize((width) => socView.date ? dayChart(socView.date, width) : weekChart(width))}</div>
  <div>${scenarioPick === "full" ? socViewPicker : ""}</div>
</div>

```js
const socOutsideRange = socView.date &&
  (d3.utcMonth.floor(socView.date) < fromMonth || d3.utcMonth.floor(socView.date) > toMonth);

display(scenarioPick !== "full" ? html`` : socView.date ? html`<p>
<b>${d3.utcFormat("%A %-d %B %Y")(socView.date)}</b>. The heavy line is the state of charge the
${STRATEGY_LABELS[strategyPick]} run held through the day; the other two strategies are drawn
faintly behind it, so where they part is where the price signal changed the dispatch. The teal
band is the range that day's frequency response contracts required — at least the Low products'
response energy in store, at least the High products' as headroom — and it steps at the EFA
block boundaries, where the allocation changes and the running requirement resets. The grey line
is the wholesale price, read on the right. Periods are numbered from the start of NESO's service
day, so period 1 is 23:00 the evening before.${socOutsideRange
  ? html` This day sits outside the date range set above, which the other figures follow.` : ""}</p>`
: html`<p>
Mean state of charge at each half-hour of an average week across the selected months. The
orange band is ±1 standard deviation across weeks. The teal band is the average range the
battery's FR contracts required at that point in the week: at least the Low products'
response energy in store, and at least the High products' as headroom. Individual days
require narrower, shifting ranges that averaging smooths out.</p>`);
```

The traces include the energy the battery delivers when its contracts are called on, worked
out from GB frequency second by second, and the trades it makes to recover that energy.
Delivery lowers the requirement as it happens, so the teal band dips where contracts are
called on most and climbs back as the battery recovers.

### Cumulative revenue by stream

```js
const cumulative = (() => {
  const out = [];
  for (const s of streams) {
    let run = 0;
    for (const d of monthly) {
      run += d[s.key] ?? 0;
      if (run !== 0) out.push({month: d.month_dt, stream: SERVICE_LABELS[s.label] ?? s.label, value: run / 1e6});
    }
  }
  return out;
})();

display(resize((width) => Plot.plot({
  width, height: 380, marginLeft: 58, marginBottom: 36,
  x: {label: null},
  y: {label: "Cumulative revenue (£M)", grid: true},
  color: {domain: streamDomain.slice(0, 7), range: streamRange.slice(0, 7), legend: true},
  marks: [
    Plot.ruleY([0]),
    Plot.line(cumulative, {x: "month", y: "value", stroke: "stream", strokeWidth: 1.8}),
  ],
})));
```

```js
display(summary ? Inputs.table(
  Object.entries(summary.breakdown)
    .map(([k, v]) => ({
      Stream: SERVICE_LABELS[k] ?? k,
      Revenue: gbp(v),
      "Share of gross": `${((v / summary.gross) * 100).toFixed(1)}%`,
    }))
    .sort((a, b) => d3.descending(
      summary.breakdown[Object.keys(SERVICE_LABELS).find((s) => SERVICE_LABELS[s] === a.Stream) ?? a.Stream],
      summary.breakdown[Object.keys(SERVICE_LABELS).find((s) => SERVICE_LABELS[s] === b.Stream) ?? b.Stream]
    )),
  {rows: 8, width: {Stream: 160}}
) : html`<i>No results for this selection.</i>`);
```

## Strategy comparison

Three price-signal strategies run the same allocation and dispatch engine on the same asset,
isolating how much *forecast quality* — not the optimiser — affects operational revenue.

| Strategy | Price signal | What it represents |
|---|---|---|
| **Perfect Foresight** | Actual day-D wholesale prices | Theoretical ceiling — needs advance knowledge of the future |
| **Naive\*** | The last complete day's prices | Zero-skill floor — any real model must beat this |
| **ML Model** | Random Forest forecast | Realistic best case, using only data that existed at each decision |

```js
// Apply the identical filter and scaling to all three strategies so the comparison
// reflects whatever selection is active above.
const allSummaries = Object.fromEntries(Object.keys(STRATEGY_LABELS).map((key) =>
  [key, summarise(rowsFor(key, scenarioPick), powerMw)]
));

const pf = allSummaries.pf_mpc, nv = allSummaries.naive_mpc, ml = allSummaries.ml_mpc;
// Mirrors compute_revenue_gap() in price_forecast.py: the denominator is the
// capturable headroom between the zero-skill floor and the ceiling, and it goes
// to zero when there is no arbitrage opportunity to capture. A bare !== check
// lets the ratio explode when the two sit within a pound of each other.
const foresightDenom = pf && nv ? pf.net - nv.net : 0;
const foresightRatio = pf && nv && ml && Math.abs(foresightDenom) >= 1
  ? (ml.net - nv.net) / foresightDenom : null;
const arbRatio = pf?.breakdown[TRADING]
  ? (ml?.breakdown[TRADING] ?? 0) / pf.breakdown[TRADING] : null;
```

<div class="grid grid-cols-2">
<div class="card">${resize((width) => Plot.plot({
  width, height: 360, marginLeft: 62, marginBottom: 42,
  x: {label: null, domain: ["naive_mpc", "ml_mpc", "pf_mpc"],
      tickFormat: (k) => ({naive_mpc: "Naive*", ml_mpc: "ML Model", pf_mpc: "Perfect Foresight"})[k]},
  y: {label: "Annualised net (£k / MW / yr)", grid: true, zero: true},
  color: {domain: ["naive_mpc", "ml_mpc", "pf_mpc"], range: ["#C9400A", "#0D7680", "#4E8A3C"]},
  marks: [
    Plot.ruleY([0]),
    Plot.barY(Object.entries(allSummaries).filter(([, s]) => s),
      {x: (d) => d[0], y: (d) => d[1].perMw / 1e3, fill: (d) => d[0]}),
    Plot.text(Object.entries(allSummaries).filter(([, s]) => s),
      {x: (d) => d[0], y: (d) => d[1].perMw / 1e3, dy: -8,
       text: (d) => `£${(d[1].perMw / 1e3).toFixed(1)}k`}),
  ],
}))}</div>
<div class="card">
<h2>Reading the chart</h2>
<p>The three bars define a range. <b>Naive*</b> sets the floor — what you would earn by
reusing the last complete day's prices. <b>Perfect Foresight</b> is the ceiling, the revenue
available if you knew the future. <b>ML Model</b> sits between them, and the question is how
close it gets to the ceiling.</p>
<p>*The floor is not quite zero-skill: its offers, like the model's, plan on a forecast pulled
halfway towards its daily mean, a setting chosen on the years before 2025. That one parameter
is worth about £4k/MW/yr to it, which makes it a stronger and fairer benchmark.</p>
<p>The <b>foresight ratio</b> quantifies this as a fraction of the capturable improvement:
<code>(ML − Naive) / (PF − Naive)</code>. The industry's usual measure, Percent of Perfect,
subtracts no floor and reads 80.9% here — but the naive floor alone reads 78.2% on it, because
most of the revenue is frequency response availability that no forecast moves. The harder
ratio is the one reported for that reason.</p>
<p>Resampled in four-week blocks, the model's lead is £3.26k/MW/yr with a 95% interval of
£2.13k to £4.45k. Its <i>accuracy</i> edge over persistence is not distinguishable from noise
(Diebold-Mariano p = 0.26 on squared error); what it is worth, it earns through allocation.</p>
<p>It is a share of the <i>capturable</i> gap, so it moves when that gap moves: charging the
model for the energy its contracts deliver lifted the floor towards the ceiling. The bigger
shift came from retraining. While a single fixed split left most of the backtest forecast by a
model that had trained on it, this read near 66%; with every forecast out-of-sample it fell
to a fifth. Then the engine improved: pricing offers from a trading plan raised the ceiling
more than either forecast, and the shrink gave the floor the caution the model's trees already
had. It now reads near an eighth — a better engine made the forecast matter less, not a
worse forecast.</p>
<p><span class="big">${foresightRatio == null ? "—" : (foresightRatio * 100).toFixed(1) + "%"}</span><br>
<span class="muted">foresight ratio${arbRatio == null ? "" : ` · ${(arbRatio * 100).toFixed(1)}% of perfect-foresight arbitrage captured`}</span></p>
</div>
</div>

```js
display(Inputs.table(
  Object.entries(allSummaries).filter(([, s]) => s).map(([key, s]) => ({
    Strategy: STRATEGY_LABELS[key],
    "Total net": gbp(s.net),
    "Annualised": gbp(s.annualised),
    "£k / MW / yr": (s.perMw / 1e3).toFixed(1),
    [TRADING]: gbp(s.breakdown[TRADING] ?? 0),
    "Cycling cost": gbp(s.cyc),
    "MWh cycled": d3.format(",.0f")(s.mwhCycled),
  })),
  {rows: 4, width: {Strategy: 170}}
));
```

### ML model detail — Random Forest

The ML strategy predicts the 48 half-hourly APXMIDP prices for day D from the last complete
day's data: D-1 for dispatch on the day, and D-2 for the offers made at 14:00 on D-1 and for
the next day in each dispatch plan, from a second walk-forward table built one day further
back. Tree-based ensembles suit this problem: the feature set is
tabular (lagged prices, generation-mix ratios, temporal encodings) rather than sequential,
they need no feature scaling, and they yield interpretable importances.

```js
const importances = (manifest.ml_mpc.feature_importances ?? []).slice(0, 12);

display(resize((width) => importances.length ? Plot.plot({
  width, height: 360, marginLeft: 165, marginRight: 52,
  x: {label: "Importance", grid: true},
  y: {label: null, domain: importances.map((d) => d.feature)},
  marks: [
    Plot.barX(importances, {x: "importance", y: "feature", fill: "#0D7680"}),
    Plot.text(importances, {x: "importance", y: "feature", dx: 4, textAnchor: "start",
                            text: (d) => d.importance.toFixed(3)}),
  ],
}) : html`<i>No feature importances in the manifest — re-run scripts/precompute_cache.py.</i>`));
```

```js
const metrics = manifest.ml_mpc.model_metrics;
const wf = metrics.walk_forward, fixed = metrics.fixed_split;
const folds = metrics.folds ?? [];

const row = (label, key, format = (v) => v) => ({
  Metric: label,
  "Walk-forward": format(wf[key]),
  "In-sample fit": format(fixed.train[key]),
  "Single held-out split": format(fixed.test[key]),
});

const metricRows = [
  row("RMSE (£/MWh)", "rmse"),
  row("MAE (£/MWh)", "mae"),
  row("Spearman ρ", "spearman"),
  row("Spike-RMSE (£/MWh)", "spike_rmse"),
];
// Added 2026-09-17; absent from caches built before then
if (wf.spread_bias != null) metricRows.push(row("Spread bias (£/MWh)", "spread_bias"),
                                            row("Spread MAE (£/MWh)", "spread_mae"));
metricRows.push(row("Observations", "n_samples", d3.format(",")));

display(Inputs.table(metricRows, {rows: 8, width: {Metric: 190}}));
```

**Every forecast behind the revenue figures is out-of-sample.** The model is refit every
${manifest.ml_mpc.params.walk_forward_cadence_months} months on the history available at
that point and predicts only the days that follow, so no day is forecast by a model that
trained on it — ${folds.length} refits across the backtest. The other two columns are
diagnostics, not the basis of anything: the in-sample fit shows how well the model
reproduces days it has already seen, and the single held-out split is the conventional
one-boundary estimate. The distance between them is why this page reports the first column.

```js
display(resize((width) => folds.length ? Plot.plot({
  width, height: 240, marginLeft: 52, marginBottom: 34,
  x: {label: null, type: "band", tickFormat: (d) => d.slice(0, 7), ticks: folds.filter((_, i) => i % 2 === 0).map((f) => f.origin)},
  y: {label: "Fold RMSE (£/MWh)", grid: true, zero: true},
  marks: [
    Plot.ruleY([0]),
    Plot.barY(folds, {x: "origin", y: "rmse", fill: "#0D7680", tip: true,
                      channels: folds[0]?.spread_bias == null
                        ? {"training rows": "train_rows", "Spearman": "spearman"}
                        : {"training rows": "train_rows", "Spearman": "spearman",
                           "spread bias": "spread_bias"}}),
    Plot.ruleY([wf.rmse], {stroke: "#C9400A", strokeDasharray: "4 3"}),
  ],
}) : html`<i>No per-fold metrics in the manifest — re-run scripts/precompute_cache.py.</i>`));
```

<p class="muted">Error per refit, against the pooled walk-forward RMSE (dashed). Spearman ρ
ranges ${d3.format(".2f")(d3.min(folds, (d) => d.spearman))}–${d3.format(".2f")(d3.max(folds, (d) => d.spearman))}
across folds, which is the spread a single split cannot show.</p>

Spike-RMSE measures error on top-decile price periods, where arbitrage revenue concentrates.
Spearman ρ matters more than RMSE for dispatch quality — the LP only needs the *ordering* of
prices to be right.

**Spread bias is the row to watch.** It is the mean signed error in each day's predicted price
spread — max minus min — which is what arbitrage actually trades on. Negative means the
forecast is conservative, under-calling how wide the day will be; positive means it invents
spread that never arrives. The asymmetry matters: a phantom spread costs a bad trade and then
makes the offer stage value trading headroom it will never use, declining frequency response
contracts worth having, while missing a real spread only forgoes upside. This is the metric that decided
the model choice, and none of the rows above it can see that failure — spike-RMSE scores error
on spikes that *happened*, so inventing them is free. See
[why the model is not chosen by accuracy](./methodology#why-the-model-is-not-chosen-by-accuracy).

Read the Spearman row with care: each column pools every period in its own window, so the
three cover different spans and are not directly comparable, and pooling across years mixes
the variation *between* days into a number meant to describe ordering *within* a day. The
per-fold range above is the better guide, and a within-day measure would be better still.

**Known limitations:** tree-based models cannot extrapolate beyond price ranges seen in
training; electricity price forecasting is inherently noisy; the model improves dispatch
quality on average without eliminating error on individual days; and hyperparameters were
chosen once rather than re-selected inside each fold.

## Sensitivity

### Cycling wear cost

Battery degradation is a real operating cost, but modelling it precisely needs a full
electrochemical model and site-specific data. A flat **£/MWh cycled** figure is used as a
financial proxy, consistent with industry practice. The NESO/Modo consensus for modern
Li-ion sits near **£${BASE_CYCLING}/MWh**, with a plausible range from under £1/MWh to
£8–10/MWh on aggressive cycling.

```js
display(summary && summary.mwhCycled > 0 ? Inputs.table(
  [0, 1, 2, 3, 5, 7.5, 10].map((c) => {
    const net = summary.gross - summary.mwhCycled * c;
    return {
      "£/MWh cycled": c.toFixed(2),
      "Total net revenue": gbp(net),
      "£k / MW / yr": summary.years > 0 && powerMw > 0
        ? (net / summary.years / powerMw / 1e3).toFixed(1) : "—",
      "": c === BASE_CYCLING ? "← base case" : "",
    };
  }), {rows: 8}
) : html`<i>No energy was cycled in this selection.</i>`);
```

```js
if (summary && summary.mwhCycled > 0) display(html`<div class="muted">
Gross revenue is held constant; only the cycling deduction changes. Total cycled across
this selection: ${d3.format(",.0f")(summary.mwhCycled)} MWh
(${d3.format(",.0f")(summary.mwhCycled / summary.years / powerMw)} MWh/MW/yr annualised).
</div>`);
```

### Service mix

How the revenue stack changes depending on which markets the asset participates in.

```js
const mixRows = [
  ["FR + arbitrage", "full"],
  ["FR only", "fr_only"],
  ["Arbitrage only", "arb_only"],
].map(([label, key]) => {
  const s = summarise(rowsFor(strategyPick, key), powerMw);
  return s ? {
    Scenario: label,
    "Total net revenue": gbp(s.net),
    "£k / MW / yr": (s.perMw / 1e3).toFixed(1),
    "Top stream": SERVICE_LABELS[s.top] ?? s.top,
  } : null;
}).filter(Boolean);

display(Inputs.table(mixRows, {rows: 3, width: {Scenario: 240}}));
```

Each scenario is its own backtest rather than the full stack with a stream removed: without
arbitrage the allocation gives FR every MW it can use, and without FR the battery trades its
whole rating. FR only is a site with no interest in wholesale arbitrage: it trades only to
recover the energy its contracts deliver, at market prices. The last row holds every block before November 2023 in DC, which
is what most of the 2022–23 fleet actually did; the main run picks each block's service on
the previous day's prices. See the [methodology](./methodology#pre-eac-service-choice) for
the evidence behind both.

The price signal matters here too. FR is offered at the arbitrage value the forecast
expects, so a forecast that overstates arbitrage holds back capacity that trading then fails
to earn back. With a weak forecast, the full stack can earn less than FR alone.
