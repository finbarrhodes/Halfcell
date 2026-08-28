# Market Overview

The energy transition in Great Britain is well under way. Fossil fuels are on the out, 
and each year renewables make up a larger proportion of the power mix; in 2025, 44% of 
Great Britain's electricity was generated using renewables. Renewables are of course
cheap and clean, but one consequence of a more renewables-focused power mix is that 
there is less system **inertia**, the physical flywheel effect that is used to resist 
sudden changes in frequency.

Grid frequency has to stay within a whisker of 50 Hz. Matching supply to demand, all 
else equal, is slightly harder when there are less heavy turbines to provide inertia 
on the supply-side of the grid. Less inertia means things move faster when something 
breaks, and requires faster responses to avert disaster. This problem is perfectly 
suited for batteries: they can go from idle to full output in under a second, in
either direction. That capability is what the frequency response markets buy and why 
batteries' roles in the grids of the future will continue to grow. 

```js
import {MARKET_COLOURS, EFA_BLOCKS, rollingMean} from "./components/theme.js";

const auctions = (await FileAttachment("data/auctions-daily.parquet").parquet())
  .toArray().map((d) => ({...d, date: new Date(d.date)}));
const marketDaily = (await FileAttachment("data/market-daily.parquet").parquet())
  .toArray().map((d) => ({...d, date: new Date(d.date)}));
const sysPrices = (await FileAttachment("data/system-prices-daily.parquet").parquet())
  .toArray().map((d) => ({...d, date: new Date(d.date)}));
const generation = (await FileAttachment("data/generation-daily.parquet").parquet())
  .toArray().map((d) => ({...d, date: new Date(d.date)}));
```

## The shift that created the market

The generation mix is the clearest way to see why these markets exist and why they have
grown. Wind and solar rise; gas and coal fall; the system carries less and less inertia.

GB generation by fuel group. The mix matters for BESS because it shapes the underlying risk
of frequency deviation: high wind with low demand pushes frequency high, which the High-side
services answer by charging, while low wind with high demand can cause dips, which the
Low-side services answer by discharging. The High and Low in DCH, DCL and the rest name the
frequency excursion being corrected, not the direction the battery moves power.

```js
const genRolling = (() => {
  const out = [];
  for (const [fuel, rows] of d3.group(generation, (d) => d.fuel_group)) {
    const sorted = d3.sort(rows, (d) => d.date).map((d) => ({date: d.date, value: d.generation}));
    for (const r of rollingMean(sorted, 14)) out.push({...r, fuel});
  }
  return out;
})();

const fuelOrder = Array.from(
  d3.rollup(generation, (v) => d3.sum(v, (d) => d.generation), (d) => d.fuel_group)
).sort((a, b) => d3.descending(a[1], b[1])).map((d) => d[0]);

display(Plot.plot({
  height: 460, marginLeft: 60,
  x: {label: null},
  y: {label: "Avg generation (MW)", grid: true},
  color: {legend: true, domain: fuelOrder},
  marks: [
    Plot.ruleY([0], {strokeOpacity: 0.3}),
    Plot.line(genRolling, {x: "date", y: "value", stroke: "fuel", strokeWidth: 1.4}),
  ],
}));
```

14-day rolling mean, smoothing day-to-day noise while still resolving seasonal swings.
Click legend entries to isolate a fuel group.

### Average share by fuel group

```js
const shares = Array.from(
  d3.rollup(generation, (v) => d3.mean(v, (d) => d.generation), (d) => d.fuel_group),
  ([fuel, mean]) => ({fuel, mean})
).filter((d) => d.mean > 0).sort((a, b) => d3.descending(a.mean, b.mean));

const total = d3.sum(shares, (d) => d.mean);

display(Plot.plot({
  height: 320, marginLeft: 110,
  x: {label: "Share of generation (%)", grid: true},
  y: {label: null, domain: shares.map((d) => d.fuel)},
  marks: [
    Plot.barX(shares, {x: (d) => (d.mean / total) * 100, y: "fuel", fill: "#0D7680"}),
    Plot.text(shares, {
      x: (d) => (d.mean / total) * 100, y: "fuel", dx: 4, textAnchor: "start",
      text: (d) => `${((d.mean / total) * 100).toFixed(1)}%`,
    }),
  ],
}));
```

## How a battery earns from it

A grid-scale battery in GB has two main routes to revenue, and this page covers the market
data behind both.

**Frequency response — contracted availability.** NESO runs daily auctions for capacity
that must react within seconds when frequency strays from 50 Hz. Win one and you are paid
a **£/MW/h availability fee** for every hour you are committed, whether or not you are
actually called. Predictable, contracted income — but the committed capacity has to keep
enough charge and enough headroom to deliver in either direction.

There are three services, split by how fast and how long they must respond:

| Service | Frequency band | Response | Sustained for |
|---|---|---|---|
| **DC** — Dynamic Containment | ±0.2–0.5 Hz | ~1 second | 15 min |
| **DR** — Dynamic Regulation | ±0.015–0.2 Hz | continuous | 60 min |
| **DM** — Dynamic Moderation | ±0.1–0.5 Hz | ~1 second | 30 min |

Each runs as two separate auctions: **High**, which responds to *falling* frequency by
discharging, and **Low**, which responds to *rising* frequency by charging. Auctions clear
per **EFA block** — six four-hour windows covering the day — so a battery's commitment can
differ across the day.

**Wholesale arbitrage — opportunistic trading.** Separately, the battery can buy energy
when it is cheap and sell when it is expensive. The profit is the price spread less
round-trip losses and the wear cost of cycling.

The tension between the two is the subject of the
[Forecasting & Dispatch](./backtester) page: capacity committed to frequency response
cannot be freely traded, so the operator must decide each day how to split it.

```js
const allServices = d3.sort(new Set(auctions.map((d) => d.service)));
const dateExtent = d3.extent(auctions, (d) => d.date);

const servicePick = view(Inputs.checkbox(allServices, {
  label: "DC/DR/DM services", value: allServices,
}));
const fromPick = view(Inputs.date({
  label: "From", value: dateExtent[0], min: dateExtent[0], max: dateExtent[1],
}));
const toPick = view(Inputs.date({
  label: "To", value: dateExtent[1], min: dateExtent[0], max: dateExtent[1],
}));
```

```js
// All filtering happens here, in the browser — the same three keys the
// Streamlit sidebar filtered on, applied to the full-grain auction table.
const services = new Set(servicePick);
const filtered = auctions.filter(
  (d) => services.has(d.service) && d.date >= fromPick && d.date <= toPick
);
```

## Frequency Response

GB frequency response is procured through three
[**dynamic** services](https://www.neso.energy/industry-information/balancing-services/frequency-response-services/dynamic-services-dcdmdr),
each split into **High** (discharge — activated when frequency falls below 50 Hz) and
**Low** (charge — activated when frequency rises above 50 Hz) auctions.

| Service | Frequency band | Role |
|---------|---------------|------|
| **DC** – Dynamic Containment | ±0.2–0.5 Hz | Arrests large deviations within ~1 second |
| **DR** – Dynamic Regulation | ±0.015–0.2 Hz | Maintains frequency in normal operation |
| **DM** – Dynamic Moderation | ±0.1–0.5 Hz | Moderates frequency during stressed conditions |

Auctions run daily for each **EFA block** (six 4-hour windows covering the full day).
The clearing price is the marginal accepted bid for that block and service.

<details>
<summary>EFA block timings</summary>

${Inputs.table(
  Object.entries(EFA_BLOCKS).map(([k, v]) => ({"EFA Block": +k, "Time window (local clock)": v})),
  {rows: 6, height: 210}
)}

EFA Block 1 spans midnight (23:00 the previous calendar day to 03:00). All times are local GB time.
</details>

### Clearing prices — 28-day rolling average by service

Individual auction results are first averaged to a daily figure per service, then
smoothed with a 28-day rolling window, so the trend for each of the six services is
readable without daily noise obscuring the signal.

```js
const rollingByService = (() => {
  const out = [];
  for (const [service, rows] of d3.group(filtered, (d) => d.service)) {
    const daily = d3.sort(
      Array.from(d3.rollup(rows, (v) => d3.mean(v, (d) => d.clearing_price), (d) => +d.date),
        ([date, value]) => ({date: new Date(date), value})),
      (d) => d.date
    );
    for (const r of rollingMean(daily, 28)) out.push({...r, service});
  }
  return out;
})();

// Order the legend by mean level so the highest-value service reads first
const serviceOrder = Array.from(
  d3.rollup(rollingByService, (v) => d3.mean(v, (d) => d.value), (d) => d.service)
).sort((a, b) => d3.descending(a[1], b[1])).map((d) => d[0]);
```

```js
display(Plot.plot({
  height: 420, marginLeft: 55,
  x: {label: null},
  y: {label: "Rolling avg (£/MW/h)", grid: true},
  color: {legend: true, domain: serviceOrder},
  marks: [
    Plot.ruleY([0], {stroke: "currentColor", strokeOpacity: 0.3}),
    Plot.line(rollingByService, {x: "date", y: "value", stroke: "service", strokeWidth: 1.6}),
    Plot.tip(rollingByService, Plot.pointerX({
      x: "date", y: "value", stroke: "service",
      title: (d) => `${d.service}\n${d.date.toDateString()}\n£${d.value?.toFixed(2)}/MW/h`,
    })),
  ],
}));
```

<details>
<summary>Key takeaways — clearing price trends</summary>

- **2022 peak then sharp compression.** DCL clearing prices peaked at £15–20/MW/h in 2022
  as NESO expanded DC procurement ahead of renewable growth. From late 2022 a rapid wave of
  new GB BESS capacity entered the frequency response markets, outpacing NESO's procurement
  volumes and driving prices steeply lower across all services.
- **Discharge (Low) services generally clear above charge (High) services.** Fleet-wide
  charge headroom tends to be more available than discharge headroom — particularly during
  high-wind periods — so High-side auctions typically clear lower.
- **DRH and DRL behave differently from DC and DM.** DR's sustained 60-minute delivery
  requirement couples the two sides operationally, which is why the DRL spread sometimes
  inverts relative to DCL and DML.
</details>

### Price distribution

<div class="grid grid-cols-2">
  <div class="card">${
    resize((width) => Plot.plot({
      width, height: 380, marginLeft: 50,
      y: {label: "£/MW/h", grid: true},
      color: {domain: allServices, legend: false},
      marks: [
        Plot.ruleY([0], {strokeOpacity: 0.3}),
        Plot.boxY(filtered, {x: "service", y: "clearing_price", fill: "service"}),
      ],
    }))
  }</div>
  <div class="card">${
    resize((width) => Plot.plot({
      width, height: 380, marginLeft: 50, marginBottom: 45,
      x: {label: "EFA block", tickFormat: (d) => `EFA ${d}`},
      y: {label: "£/MW/h", grid: true},
      color: {domain: allServices, legend: true},
      marks: [
        Plot.ruleY([0], {strokeOpacity: 0.3}),
        Plot.boxY(filtered, {x: "efa", y: "clearing_price", fill: "service"}),
      ],
    }))
  }</div>
</div>

DCL shows the widest spread of outcomes, reflecting its role as the primary fast-discharge
service and its early-market dominance at elevated prices. Evening blocks (EFA 5–6,
15:00–23:00) attract higher premia as demand peaks and wind output often eases; the
overnight block (EFA 1) is typically cheapest to procure.

### Summary statistics

```js
display(Inputs.table(
  Array.from(d3.group(filtered, (d) => d.service), ([service, v]) => ({
    Service: service,
    "Avg price (£/MW/h)": d3.mean(v, (d) => d.clearing_price),
    "Median": d3.median(v, (d) => d.clearing_price),
    "Max": d3.max(v, (d) => d.clearing_price),
    "Avg volume (MW)": d3.mean(v, (d) => d.cleared_volume),
    "Records": v.length,
  })).sort((a, b) => d3.descending(a["Avg price (£/MW/h)"], b["Avg price (£/MW/h)"])),
  {format: {
    "Avg price (£/MW/h)": (d) => d.toFixed(2),
    "Median": (d) => d.toFixed(2),
    "Max": (d) => d.toFixed(2),
    "Avg volume (MW)": (d) => d.toFixed(1),
  }, rows: 7}
));
```

## High vs Low spread

Each service runs two separate auctions: **High** (falling frequency — BESS discharges) and
**Low** (rising frequency — BESS charges). Clearing prices differ because available discharge
and charge headroom across the fleet is rarely symmetric.

**Spread = H clearing price − L clearing price.** Positive means discharge capacity was
scarcer; negative means charge capacity was scarcer.

```js
const PAIRS = [["DC", "DCH", "DCL"], ["DR", "DRH", "DRL"], ["DM", "DMH", "DML"]];

// Join H against L on (date, EFA block) — the same inner join the Streamlit page did
const spreads = (() => {
  const out = [];
  const key = (d) => `${+d.date}|${d.efa}`;
  for (const [market, hSvc, lSvc] of PAIRS) {
    const H = new Map(auctions.filter((d) => d.service === hSvc).map((d) => [key(d), d]));
    for (const l of auctions.filter((d) => d.service === lSvc)) {
      const h = H.get(key(l));
      if (h) out.push({market, date: l.date, efa: l.efa, spread: h.clearing_price - l.clearing_price});
    }
  }
  return out;
})();

const drMean = d3.mean(spreads.filter((d) => d.market === "DR"), (d) => d.spread);
```

### Daily average H − L spread over time

```js
const dailySpread = Array.from(
  d3.rollup(spreads, (v) => d3.mean(v, (d) => d.spread), (d) => d.market, (d) => +d.date),
  ([market, m]) => Array.from(m, ([date, spread]) => ({market, date: new Date(date), spread}))
).flat();

display(Plot.plot({
  height: 400, marginLeft: 55,
  x: {label: null},
  y: {label: "£/MW/h", grid: true},
  color: {legend: true, domain: Object.keys(MARKET_COLOURS), range: Object.values(MARKET_COLOURS)},
  marks: [
    Plot.ruleY([0], {strokeDasharray: "4 3", strokeOpacity: 0.6}),
    Plot.line(dailySpread, {x: "date", y: "spread", stroke: "market", strokeWidth: 1.2}),
  ],
}));
```

```js
display(drMean < 0 ? html`<div class="note">
<p><b>Why is the DR spread consistently negative (avg ${drMean.toFixed(2)} £/MW/h)?</b></p>
<p>DR operates continuously in the normal frequency band, and NESO's energy management rules
require providers to sustain their contracted position for a full <b>60-minute</b> delivery
window — much longer than DC (15 min) or DM (30 min). Providers holding both DRH and DRL
positions need to keep SoC near the midpoint to honour either commitment for the full hour.
That longer window effectively couples the two sides in a way DC and DM do not.</p>
<p><i>DRH and DRL are technically separate auctions and can be bid independently; the coupling
is a practical consequence of the sustained delivery requirement, not an explicit rule.</i></p>
</div>` : html``);
```

<div class="grid grid-cols-2">
  <div class="card">
    <h3>Spread distribution by market</h3>
    ${resize((width) => Plot.plot({
      width, height: 360, marginLeft: 50,
      y: {label: "£/MW/h", grid: true},
      color: {domain: Object.keys(MARKET_COLOURS), range: Object.values(MARKET_COLOURS)},
      marks: [
        Plot.ruleY([0], {strokeDasharray: "4 3", strokeOpacity: 0.6}),
        Plot.boxY(spreads, {x: "market", y: "spread", fill: "market"}),
      ],
    }))}
  </div>
  <div class="card">
    <h3>Average spread by EFA block</h3>
    ${resize((width) => Plot.plot({
      width, height: 360, marginLeft: 50,
      x: {label: "EFA block", tickFormat: (d) => `EFA ${d}`},
      y: {label: "Avg £/MW/h", grid: true},
      color: {domain: Object.keys(MARKET_COLOURS), range: Object.values(MARKET_COLOURS), legend: true},
      marks: [
        Plot.ruleY([0], {strokeDasharray: "4 3", strokeOpacity: 0.6}),
        Plot.barY(spreads, Plot.groupX({y: "mean"}, {x: "efa", y: "spread", fill: "market"})),
      ],
    }))}
  </div>
</div>

DC shows the widest range of spread outcomes — its higher median reflects the relative
scarcity of fast-discharge capacity. DR sits firmly negative across both charts, confirming
the structural inversion described above. DM occupies the middle ground. Evening blocks
(EFA 5–6) show the most pronounced spreads, as demand peaks and the balance between
available charge and discharge headroom is tightest.

### H − L spread heatmap: EFA block × month

```js
const heatMarket = view(Inputs.radio(["DC", "DR", "DM"], {label: "Market", value: "DC"}));
```

```js
const heat = Array.from(
  d3.rollup(
    spreads.filter((d) => d.market === heatMarket),
    (v) => d3.mean(v, (d) => d.spread),
    (d) => d3.utcFormat("%Y-%m")(d.date),
    (d) => d.efa
  ),
  ([month, m]) => Array.from(m, ([efa, spread]) => ({month, efa, spread}))
).flat();

const lim = d3.max(heat, (d) => Math.abs(d.spread));

display(Plot.plot({
  height: 460, marginLeft: 70, marginBottom: 45,
  x: {label: "EFA block", tickFormat: (d) => `EFA ${d}`, type: "band"},
  y: {label: "Month", type: "band", tickFormat: (d) => (d.endsWith("-01") ? d.slice(0, 4) : "")},
  color: {scheme: "RdBu", domain: [lim, -lim], legend: true, label: "£/MW/h"},
  marks: [
    Plot.cell(heat, {x: "efa", y: "month", fill: "spread", inset: 0.5}),
    Plot.tip(heat, Plot.pointer({
      x: "efa", y: "month",
      title: (d) => `${heatMarket} · EFA ${d.efa} (${EFA_BLOCKS[d.efa]})\n${d.month}\n£${d.spread.toFixed(2)}/MW/h`,
    })),
  ],
}));
```

Each cell is the average H − L spread for that market, EFA block and calendar month.
Red = discharge capacity scarcer (H > L); blue = charge capacity scarcer (L > H).

```js
display(Inputs.table(
  Array.from(d3.group(spreads, (d) => d.market), ([market, v]) => ({
    Market: market,
    "Mean £/MW/h": d3.mean(v, (d) => d.spread),
    "Median": d3.median(v, (d) => d.spread),
    "Std dev": d3.deviation(v, (d) => d.spread),
    "Min": d3.min(v, (d) => d.spread),
    "Max": d3.max(v, (d) => d.spread),
    "% blocks H > L": (d3.sum(v, (d) => (d.spread > 0 ? 1 : 0)) / v.length) * 100,
  })),
  {format: {
    "Mean £/MW/h": (d) => d.toFixed(2), "Median": (d) => d.toFixed(2),
    "Std dev": (d) => d.toFixed(2), "Min": (d) => d.toFixed(2),
    "Max": (d) => d.toFixed(2), "% blocks H > L": (d) => `${d.toFixed(1)}%`,
  }, rows: 4}
));
```

## Wholesale & settlement prices

**System Buy Price (SBP)** and **System Sell Price (SSP)** are the cash-out prices used to
settle imbalance in the GB Balancing Mechanism. Parties that are *short* pay the SBP; parties
that are *long* receive the SSP. The gap between them incentivises self-balancing rather than
relying on the system operator.

```js
const spLong = sysPrices.flatMap((d) => [
  {date: d.date, series: "Avg SSP", price: d.ssp_mean},
  {date: d.date, series: "Avg SBP", price: d.sbp_mean},
]);

display(Plot.plot({
  height: 420, marginLeft: 55,
  x: {label: null},
  y: {label: "£/MWh", grid: true},
  color: {legend: true, domain: ["Avg SSP", "Avg SBP"], range: ["#C9400A", "#0D7680"]},
  marks: [
    Plot.ruleY([0], {strokeOpacity: 0.3}),
    Plot.line(spLong, {x: "date", y: "price", stroke: "series", strokeWidth: 1.1}),
  ],
}));
```

### Wholesale price spread

Daily peak-to-trough APXMIDP spread — the raw arbitrage opportunity available to a battery
on any given day, before efficiency losses and cycling cost.

```js
display(Plot.plot({
  height: 340, marginLeft: 55,
  x: {label: null},
  y: {label: "Daily peak-to-trough spread (£/MWh)", grid: true},
  marks: [
    Plot.ruleY([0], {strokeOpacity: 0.3}),
    Plot.line(marketDaily, {x: "date", y: "spread", stroke: "#C9400A", strokeOpacity: 0.35}),
    Plot.line(rollingMean(marketDaily, 28, "date", "spread"),
      {x: "date", y: "spread", stroke: "#8B2020", strokeWidth: 2}),
  ],
}));
```

Thin line is the daily spread; heavy line is a 28-day rolling average.

