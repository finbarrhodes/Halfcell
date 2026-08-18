# Day-Ahead Forecasting & Dispatch Model

Grid-scale batteries earn revenue by participating in multiple markets simultaneously.
This model focuses on the **day-ahead decision layer**: given yesterday's auction results
and market price data, how should a BESS operator allocate capacity between frequency
response commitment and spot arbitrage — and how much does the quality of the price
forecast actually affect the outcome?

The framework has three components: a **per-EFA-block FR/arbitrage capacity allocator**
that compares confirmed auction clearing prices against a forecast-based shadow arbitrage
value; an **MPC dispatch engine** (rolling 48-hour LP) that plans charge/discharge at
half-hourly resolution while enforcing FR SoC constraints as hard bounds; and a **price
forecasting pipeline** that benchmarks three strategies against each other.

<div class="note">
<b>Scope note:</b> this model covers the day-ahead planning layer only. Intraday
re-optimisation, Balancing Mechanism participation, and real-time dispatch are not
modelled — these require operational settlement data not available publicly, and are the
primary reason figures here differ from industry estimates. The focus is the forecasting
and optimisation methodology rather than comprehensive revenue capture.
</div>

```js
import {SERVICE_COLOURS, SERVICE_LABELS, STRATEGY_LABELS, gbp} from "./components/theme.js";

const manifest = await FileAttachment("data/manifest.json").json();
const revenueAll = (await FileAttachment("data/revenue-monthly.parquet").parquet())
  .toArray().map((d) => ({...d, month_dt: new Date(d.month_dt)}));
const socAll = (await FileAttachment("data/soc-week.parquet").parquet())
  .toArray().map((d) => ({...d, month_dt: new Date(d.month_dt)}));
```

```js
const ALL_SERVICES = ["DCH", "DCL", "DMH", "DML", "DRH", "DRL"];
const BASE_POWER_MW = manifest.ml_mpc.params.power_mw;   // cache is computed at this rating
const DURATION_H = manifest.ml_mpc.params.duration_h;
const EFF = manifest.ml_mpc.params.efficiency_rt;
const BASE_CYCLING = manifest.ml_mpc.params.cycling_cost_per_mwh;

const bounds = (() => {
  const starts = Object.values(manifest).map((m) => m.params.start_date).filter(Boolean);
  const ends = Object.values(manifest).map((m) => m.params.end_date).filter(Boolean);
  return [new Date(d3.min(starts)), new Date(d3.max(ends))];
})();
```

## Controls

```js
const powerMw = view(Inputs.range([1, 500], {
  label: "Asset power (MW)", value: BASE_POWER_MW, step: 1,
}));
const strategyPick = view(Inputs.radio(Object.keys(STRATEGY_LABELS), {
  label: "Price signal", value: "pf_mpc", format: (k) => STRATEGY_LABELS[k],
}));
const servicePick = view(Inputs.checkbox(ALL_SERVICES, {
  label: "FR services", value: ALL_SERVICES,
  format: (s) => `${s} — ${SERVICE_LABELS[s]}`,
}));
const includeArb = view(Inputs.toggle({label: "Include wholesale arbitrage", value: true}));
const fromPick = view(Inputs.date({label: "From", value: bounds[0], min: bounds[0], max: bounds[1]}));
const toPick = view(Inputs.date({label: "To", value: bounds[1], min: bounds[0], max: bounds[1]}));
```

```js
// Revenue scales linearly with power at fixed duration, so display scaling is exact —
// the same post-filter the Streamlit app applied to the cached monthly table.
const scale = powerMw / BASE_POWER_MW;
const chosen = new Set(servicePick);

// The cached table is monthly, and the cache bounds are mid-month dates
// (2021-09-16 / 2026-08-17). Comparing a month-start against a mid-month bound
// would silently drop the first and last months, so widen to whole months —
// matching the period-level filter the Streamlit app applied.
const fromMonth = d3.utcMonth.floor(fromPick);
const toMonth = d3.utcMonth.floor(toPick);
const inRange = (d) => d.month_dt >= fromMonth && d.month_dt <= toMonth;

const monthly = revenueAll
  .filter((d) => d.strategy === strategyPick && inRange(d))
  .map((d) => {
    const row = {month_dt: d.month_dt};
    for (const s of ALL_SERVICES) row[`${s}_rev`] = chosen.has(s) ? (d[`${s}_rev`] ?? 0) * scale : 0;
    row.imbalance_revenue_gbp = includeArb ? (d.imbalance_revenue_gbp ?? 0) * scale : 0;
    // No arbitrage dispatch means no cycling, so the wear cost goes with it
    row.cycling_cost_gbp = includeArb ? (d.cycling_cost_gbp ?? 0) * scale : 0;
    row.mwh_cycled = includeArb ? (d.mwh_cycled ?? 0) * scale : 0;
    return row;
  })
  .sort((a, b) => a.month_dt - b.month_dt);

function summarise(rows, mw) {
  if (!rows.length) return null;
  const svc = {};
  for (const s of ALL_SERVICES) svc[s] = d3.sum(rows, (d) => d[`${s}_rev`]);
  const arb = d3.sum(rows, (d) => d.imbalance_revenue_gbp);
  const cyc = d3.sum(rows, (d) => d.cycling_cost_gbp);
  const gross = d3.sum(Object.values(svc)) + arb;
  const net = gross - cyc;
  const years = rows.length / 12;
  const breakdown = Object.fromEntries(
    Object.entries({...svc, Arbitrage: arb}).filter(([, v]) => v > 0)
  );
  return {
    gross, cyc, net, years,
    annualised: years > 0 ? net / years : 0,
    perMw: years > 0 && mw > 0 ? net / years / mw : 0,
    mwhCycled: d3.sum(rows, (d) => d.mwh_cycled),
    breakdown,
    top: d3.greatest(Object.entries(breakdown), (d) => d[1])?.[0] ?? "—",
  };
}

const summary = summarise(monthly, powerMw);
```

## Results

<div class="grid grid-cols-4">
<div class="card"><h2>Total net revenue</h2><span class="big">${summary ? gbp(summary.net) : "—"}</span></div>
<div class="card"><h2>Annualised net</h2><span class="big">${summary ? gbp(summary.annualised) : "—"}</span><div class="muted">per year</div></div>
<div class="card"><h2>Revenue per MW</h2><span class="big">${summary ? "£" + (summary.perMw / 1e3).toFixed(1) + "k" : "—"}</span><div class="muted">per MW per year</div></div>
<div class="card"><h2>Top revenue stream</h2><span class="big">${summary ? (SERVICE_LABELS[summary.top] ?? summary.top) : "—"}</span></div>
</div>

Modelling a **${powerMw} MW / ${(powerMw * DURATION_H).toFixed(0)} MWh** asset
(${DURATION_H}h duration, ${(EFF * 100).toFixed(0)}% round-trip efficiency) using
**${STRATEGY_LABELS[strategyPick]}** price signals and MPC dispatch over a rolling 48-hour horizon.

### Monthly revenue stack

```js
const streams = [...ALL_SERVICES.map((s) => ({key: `${s}_rev`, label: s})),
                 {key: "imbalance_revenue_gbp", label: "Arbitrage"}];

const stacked = monthly.flatMap((d) => [
  ...streams
    .filter((s) => (d[s.key] ?? 0) !== 0)
    .map((s) => ({month: d.month_dt, stream: SERVICE_LABELS[s.label] ?? s.label,
                  colourKey: s.label, value: d[s.key] / 1e3})),
  ...(d.cycling_cost_gbp > 0
    ? [{month: d.month_dt, stream: "Cycling wear cost", colourKey: "Cycling cost",
        value: -d.cycling_cost_gbp / 1e3}]
    : []),
]);

const streamDomain = [...ALL_SERVICES.map((s) => SERVICE_LABELS[s]), "Arbitrage", "Cycling wear cost"];
const streamRange = [...ALL_SERVICES.map((s) => SERVICE_COLOURS[s]),
                     SERVICE_COLOURS.Arbitrage, SERVICE_COLOURS["Cycling cost"]];

display(Plot.plot({
  height: 430, marginLeft: 62,
  x: {label: null, interval: "month"},
  y: {label: "£k", grid: true},
  color: {domain: streamDomain, range: streamRange, legend: true},
  marks: [
    Plot.ruleY([0]),
    Plot.rectY(stacked, {x: "month", y: "value", fill: "stream", interval: "month",
                         tip: true, order: streamDomain}),
  ],
}));
```

Each bar shows gross revenue by stream for that month (positive) and cycling wear cost
(negative, dark red). Net revenue is the algebraic sum of all segments — months with
heavier arbitrage dispatch carry larger cycling deductions.

### Average weekly SoC profile

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
      return {mean, lo: Math.max(mean - sd, 0), hi: Math.min(mean + sd, 1)};
    }, (d) => d.period_in_week),
    ([p, s]) => ({period: p, ...s})
  ).sort((a, b) => a.period - b.period);
})();

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

display(Plot.plot({
  height: 340, marginLeft: 55, marginRight: 55,
  x: {label: "Day of week", ticks: d3.range(7).map((d) => d * 48),
      tickFormat: (d) => DAYS[d / 48], domain: [0, 336]},
  y: {label: "State of charge", domain: [0, 1], tickFormat: ".0%", grid: true},
  marks: [
    // FR feasibility band — a hard constraint in the MPC LP
    Plot.rect([{y1: 0.1, y2: 0.9}], {y1: "y1", y2: "y2", fill: "#0D7680", fillOpacity: 0.08}),
    Plot.ruleY([0.1, 0.9], {stroke: "#0D7680", strokeDasharray: "4 3", strokeWidth: 1}),
    Plot.ruleX(d3.range(1, 7).map((d) => d * 48), {stroke: "grey", strokeOpacity: 0.3, strokeDasharray: "2 3"}),
    Plot.areaY(socWeek, {x: "period", y1: "lo", y2: "hi", fill: "#C9400A", fillOpacity: 0.12}),
    Plot.line(socWeek, {x: "period", y: "mean", stroke: "#C9400A", strokeWidth: 2}),
    Plot.tip(socWeek, Plot.pointerX({
      x: "period", y: "mean",
      title: (d) => `${DAYS[Math.floor(d.period / 48)]} SP ${(d.period % 48) + 1}\nmean ${(d.mean * 100).toFixed(1)}%\n±1 sd ${(d.lo * 100).toFixed(1)}–${(d.hi * 100).toFixed(1)}%`,
    })),
  ],
}));
```

Mean state-of-charge at each half-hourly slot across the backtest, folded onto an average
week. The orange band is ±1 standard deviation across all weeks; the teal band marks the
**[10%, 90%] FR feasibility constraint** enforced as a hard bound in the rolling LP. The
pre-conditioning behaviour driven by the next block's FR obligations is visible in the shape.

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

display(Plot.plot({
  height: 380, marginLeft: 58,
  x: {label: null},
  y: {label: "Cumulative revenue (£M)", grid: true},
  color: {domain: streamDomain.slice(0, 7), range: streamRange.slice(0, 7), legend: true},
  marks: [
    Plot.ruleY([0]),
    Plot.line(cumulative, {x: "month", y: "value", stroke: "stream", strokeWidth: 1.8}),
  ],
}));
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

Three price-signal strategies run the same MPC dispatch engine on the same asset, isolating
how much *forecast quality* — not the optimiser — affects operational revenue.

| Strategy | Price signal | What it represents |
|---|---|---|
| **Perfect Foresight** | Actual day-D wholesale prices | Theoretical ceiling — needs advance knowledge of the future |
| **Naive\*** | Yesterday's prices (day D-1) | Zero-skill floor — any real model must beat this |
| **ML Model** | Random Forest forecast | Realistic best case, using features available at end of day D-1 |

```js
// Apply the identical filter and scaling to all three strategies so the comparison
// reflects whatever selection is active above.
const allSummaries = Object.fromEntries(Object.keys(STRATEGY_LABELS).map((key) => {
  const rows = revenueAll
    .filter((d) => d.strategy === key && inRange(d))
    .map((d) => {
      const row = {month_dt: d.month_dt};
      for (const s of ALL_SERVICES) row[`${s}_rev`] = chosen.has(s) ? (d[`${s}_rev`] ?? 0) * scale : 0;
      row.imbalance_revenue_gbp = includeArb ? (d.imbalance_revenue_gbp ?? 0) * scale : 0;
      row.cycling_cost_gbp = includeArb ? (d.cycling_cost_gbp ?? 0) * scale : 0;
      row.mwh_cycled = includeArb ? (d.mwh_cycled ?? 0) * scale : 0;
      return row;
    })
    .sort((a, b) => a.month_dt - b.month_dt);
  return [key, summarise(rows, powerMw)];
}));

const pf = allSummaries.pf_mpc, nv = allSummaries.naive_mpc, ml = allSummaries.ml_mpc;
const foresightRatio = pf && nv && ml && pf.net !== nv.net
  ? (ml.net - nv.net) / (pf.net - nv.net) : null;
const arbRatio = pf?.breakdown.Arbitrage
  ? (ml?.breakdown.Arbitrage ?? 0) / pf.breakdown.Arbitrage : null;
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
<p>The three bars define a range. <b>Naive*</b> sets the zero-skill floor — what you would
earn with no forecasting capability at all. <b>Perfect Foresight</b> is the ceiling, the
maximum extractable revenue if you knew the future. <b>ML Model</b> sits between them, and
the question is how close it gets to the ceiling.</p>
<p>The <b>foresight ratio</b> quantifies this as a fraction of the capturable improvement:
<code>(ML − Naive) / (PF − Naive)</code>. Published GB and European price-forecasting
literature treats 70–85% as strong performance.</p>
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
    "Arbitrage": gbp(s.breakdown.Arbitrage ?? 0),
    "Cycling cost": gbp(s.cyc),
    "MWh cycled": d3.format(",.0f")(s.mwhCycled),
  })),
  {rows: 4, width: {Strategy: 170}}
));
```

### ML model detail — Random Forest

The ML strategy predicts the 48 half-hourly APXMIDP prices for day D using features
available at the end of day D-1. Tree-based ensembles suit this problem: the feature set is
tabular (lagged prices, generation-mix ratios, temporal encodings) rather than sequential,
they need no feature scaling, and they yield interpretable importances.

```js
const importances = (manifest.ml_mpc.feature_importances ?? []).slice(0, 12);

display(importances.length ? Plot.plot({
  height: 360, marginLeft: 165,
  x: {label: "Importance", grid: true},
  y: {label: null, domain: importances.map((d) => d.feature)},
  marks: [
    Plot.barX(importances, {x: "importance", y: "feature", fill: "#0D7680"}),
    Plot.text(importances, {x: "importance", y: "feature", dx: 4, textAnchor: "start",
                            text: (d) => d.importance.toFixed(3)}),
  ],
}) : html`<i>No feature importances in the manifest — re-run scripts/precompute_cache.py.</i>`);
```

```js
const m = manifest.ml_mpc.model_metrics;
display(Inputs.table([
  {Metric: "RMSE (£/MWh)", Train: m.train.rmse, Test: m.test.rmse},
  {Metric: "MAE (£/MWh)", Train: m.train.mae, Test: m.test.mae},
  {Metric: "Spearman ρ", Train: m.train.spearman, Test: m.test.spearman},
  {Metric: "Spike-RMSE (£/MWh)", Train: m.train.spike_rmse, Test: m.test.spike_rmse},
  {Metric: "Observations", Train: d3.format(",")(m.train.n_samples), Test: d3.format(",")(m.test.n_samples)},
], {rows: 6, width: {Metric: 190}}));
```

Training uses an expanding window ending before **${manifest.ml_mpc.params.test_start}**;
everything after that date is held out. Spike-RMSE measures error on top-decile price
periods, where arbitrage revenue concentrates. Spearman ρ matters more than RMSE for
dispatch quality — the LP only needs the *ordering* of prices to be right.

**Known limitations:** tree-based models cannot extrapolate beyond price ranges seen in
training; electricity price forecasting is inherently noisy; and the model improves dispatch
quality on average without eliminating error on individual days.

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
) : html`<i>Enable wholesale arbitrage to see cycling sensitivity — with no arbitrage dispatch there is no cycling.</i>`);
```

```js
display(summary && summary.mwhCycled > 0 ? html`<div class="muted">
Gross revenue is held constant; only the cycling deduction changes. Total cycled across
this selection: ${d3.format(",.0f")(summary.mwhCycled)} MWh
(${d3.format(",.0f")(summary.mwhCycled / summary.years / powerMw)} MWh/MW/yr annualised).
</div>` : html``);
```

### Service mix

How the revenue stack changes depending on which markets the asset participates in.

```js
const mixRows = [
  ["FR only (no arbitrage)", true, false],
  ["Arbitrage only (no FR)", false, true],
  ["Full stack", true, true],
].map(([label, withFr, withArb]) => {
  const rows = monthly.map((d) => {
    const r = {month_dt: d.month_dt};
    for (const s of ALL_SERVICES) r[`${s}_rev`] = withFr ? d[`${s}_rev`] : 0;
    r.imbalance_revenue_gbp = withArb ? d.imbalance_revenue_gbp : 0;
    r.cycling_cost_gbp = withArb ? d.cycling_cost_gbp : 0;
    r.mwh_cycled = withArb ? d.mwh_cycled : 0;
    return r;
  });
  const s = summarise(rows, powerMw);
  return s ? {
    Scenario: label,
    "Total net revenue": gbp(s.net),
    "£k / MW / yr": (s.perMw / 1e3).toFixed(1),
    "Top stream": SERVICE_LABELS[s.top] ?? s.top,
  } : null;
}).filter(Boolean);

display(Inputs.table(mixRows, {rows: 4, width: {Scenario: 200}}));
```

Arbitrage-only removes all FR availability fees; cycling cost is zeroed in FR-only mode,
since in this model cycling is incurred only through arbitrage dispatch.
