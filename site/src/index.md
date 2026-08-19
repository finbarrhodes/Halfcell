# Halfcell

Welcome to Halfcell, an interactive analytics tool for grid-scale battery storage
markets in Great Britain. Battery energy storage systems (BESS) are a complex part
of Great Britain's energy transition; this tool is designed to help unpack how they
operate, how they make money, and how market conditions and data science are
impacting BESS' role in the grid.

```js
const kpis = await FileAttachment("data/kpis.json").json();
const manifest = await FileAttachment("data/manifest.json").json();
const revenue = await FileAttachment("data/revenue-monthly.parquet").parquet();
```

```js
const ml = manifest.ml_mpc;
const fmtGbp = (v) =>
  Math.abs(v) >= 1e6 ? `£${(v / 1e6).toFixed(2)}M` : `£${(v / 1e3).toFixed(0)}k`;
```

## What's in this tool

<div class="grid grid-cols-3">
  <div class="card nav">
    <h2><a href="./dashboard">Market Overview →</a></h2>
    <p>GB frequency response auction clearing prices (DC, DR, DM), High vs Low spread
    dynamics, system settlement prices, and generation mix trends.</p>
  </div>
  <div class="card nav">
    <h2><a href="./backtester">Forecasting & Dispatch →</a></h2>
    <p>A day-ahead modelling framework for FR/arbitrage capacity allocation and MPC
    dispatch, benchmarking three price forecasting strategies.</p>
  </div>
  <div class="card nav">
    <h2><a href="./methodology">Methodology & Data →</a></h2>
    <p>Modelling assumptions, data sources, and known limitations of the backtester.</p>
  </div>
</div>

## Where data science meets the clean energy transition

Halfcell asks what data science can actually contribute to clean tech, using a concrete
case: a grid-scale battery deciding, every day, how to divide its capacity between
frequency response and wholesale arbitrage. That decision rests on a forecast of tomorrow's
prices — so it is a modelling problem before it is an engineering one, and different
strategies produce measurably different outcomes.

The chart below runs three of them through the same dispatch engine on the same asset.
**Perfect Foresight** knows tomorrow's prices and marks the ceiling. **Naive** reuses
yesterday's and marks the zero-skill floor — the bar any real model has to clear. A
**Random Forest** trained on lagged prices, generation mix and cyclical time features sits
between the two, and the gap it closes is the value the modelling adds.

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
      total += gross - (r.cycling_cost_gbp ?? 0);
      out.push({strategy, month: new Date(r.month_dt), total});
    }
  }
  return out;
})();

const labels = {pf_mpc: "Perfect Foresight", naive_mpc: "Naive (D-1)", ml_mpc: "ML Model"};
```

```js
display(Plot.plot({
  height: 340,
  marginLeft: 60,
  x: {label: null},
  y: {label: "Cumulative net revenue (£M)", transform: (d) => d / 1e6, grid: true},
  color: {legend: true, domain: Object.keys(labels), tickFormat: (d) => labels[d]},
  marks: [
    Plot.ruleY([0]),
    Plot.line(cumulative, {x: "month", y: "total", stroke: "strategy", strokeWidth: 2}),
  ],
}));
```

## Market snapshot

Where the GB frequency response and wholesale markets sit right now, against their
recent averages. Figures update whenever the data pipeline is re-run.

<div class="grid grid-cols-4">
  <div class="card kpi">
    <h2>DC High — latest</h2>
    <span class="big">£${kpis.dch_latest.toFixed(2)}</span>
    <div class="muted">£/MW/h · 30d avg £${kpis.dch_30d_avg.toFixed(2)}</div>
  </div>
  <div class="card kpi">
    <h2>Wholesale spread — latest</h2>
    <span class="big">£${kpis.spread_latest.toFixed(2)}</span>
    <div class="muted">£/MWh peak-to-trough · 30d avg £${kpis.spread_30d_avg.toFixed(2)}</div>
  </div>
  <div class="card kpi">
    <h2>Modelled revenue — ML strategy</h2>
    <span class="big">£${(ml.summary.annualised_per_mw / 1e3).toFixed(0)}k</span>
    <div class="muted">per MW per year · 50 MW / 2h reference asset</div>
  </div>
  <div class="card kpi">
    <h2>Data through</h2>
    <span class="big">${ml.params.end_date}</span>
    <div class="muted">${ml.summary.years_covered} years backtested</div>
  </div>
</div>
