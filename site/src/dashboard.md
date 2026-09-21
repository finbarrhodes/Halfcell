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
import {MARKET_COLOURS, EFA_BLOCKS, SERVICE_COLOURS, FUEL_COLOURS, rangeFor, rollingMean} from "./components/theme.js";
import {choiceGroup} from "./components/controls.js";
import {watchSteps} from "./components/scrolly.js";

const auctions = (await FileAttachment("data/auctions-daily.parquet").parquet())
  .toArray().map((d) => ({...d, date: new Date(d.date)}));
const marketDaily = (await FileAttachment("data/market-daily.parquet").parquet())
  .toArray().map((d) => ({...d, date: new Date(d.date)}));
const sysPrices = (await FileAttachment("data/system-prices-daily.parquet").parquet())
  .toArray().map((d) => ({...d, date: new Date(d.date)}));
const generation = (await FileAttachment("data/generation-daily.parquet").parquet())
  .toArray().map((d) => ({...d, date: new Date(d.date)}));

const SERVICE_ORDER = ["DCH", "DCL", "DMH", "DML", "DRH", "DRL"];
// The two auction rule changes that split the history (see Frequency Response)
const EAC_GO_LIVE = new Date("2023-11-02");
const RESERVE_RULE = new Date("2024-11-15");
```

## The shift that created the market

The generation mix is the clearest way to see why these markets exist and why they have
grown. Wind and solar rise; gas and coal fall; the system carries less and less inertia.

The relative mix matters because it shapes the underlying risk of frequency deviation and thus
the opportunity for BESS sites to step in: high wind with low demand pushes frequency high, which the
High-side services answer by charging, while low wind with high demand can cause dips, which the
Low-side services answer by discharging. The High and Low in DCH, DCL and the rest name the
frequency excursion being corrected, not the direction the battery moves power.

```js
// `generation` is the daily sum of the half-hourly MW readings, so each reading
// covers half an hour: multiplying by 0.5 h turns the sum into energy (MWh).
// Dividing by the period count would give mean MW instead, but a daily total
// reads more naturally as energy.
const MWH_PER_MW_READING = 0.5;

const genRolling = (() => {
  const out = [];
  for (const [fuel, rows] of d3.group(generation, (d) => d.fuel_group)) {
    const sorted = d3.sort(rows, (d) => d.date)
      .map((d) => ({date: d.date, value: d.generation * MWH_PER_MW_READING}));
    for (const r of rollingMean(sorted, 28)) out.push({...r, fuel});
  }
  return out;
})();

const fuelOrder = Array.from(
  d3.rollup(generation, (v) => d3.sum(v, (d) => d.generation), (d) => d.fuel_group)
).sort((a, b) => d3.descending(a[1], b[1])).map((d) => d[0]);

display(resize((width) => Plot.plot({
  width, height: 460, marginLeft: 60,
  x: {label: null},
  y: {label: "Daily generation (MWh)", grid: true},
  color: {legend: true, domain: fuelOrder, range: rangeFor(FUEL_COLOURS, fuelOrder)},
  marks: [
    Plot.ruleY([0], {strokeOpacity: 0.3}),
    Plot.line(genRolling, {x: "date", y: "value", stroke: "fuel", strokeWidth: 1.4}),
  ],
})));
```

This plot shows a 28-day rolling mean, smoothing day-to-day noise while still capturing seasonal swings.

### Average share by fuel group

```js
const shares = Array.from(
  d3.rollup(generation, (v) => d3.mean(v, (d) => d.generation), (d) => d.fuel_group),
  ([fuel, mean]) => ({fuel, mean})
).filter((d) => d.mean > 0).sort((a, b) => d3.descending(a.mean, b.mean));

const total = d3.sum(shares, (d) => d.mean);

const shareOrder = shares.map((d) => d.fuel);

display(resize((width) => Plot.plot({
  // marginRight leaves room for the value label on the longest bar; at full
  // column width it otherwise runs past the frame and gets clipped.
  width, height: 320, marginLeft: 110, marginRight: 46,
  x: {label: "Share of generation (%)", grid: true},
  y: {label: null, domain: shareOrder},
  // Same fuel, same colour as the series chart above, so the two read together.
  // No legend: the y axis already names every bar.
  color: {domain: shareOrder, range: rangeFor(FUEL_COLOURS, shareOrder), legend: false},
  marks: [
    Plot.barX(shares, {x: (d) => (d.mean / total) * 100, y: "fuel", fill: "fuel"}),
    Plot.text(shares, {
      x: (d) => (d.mean / total) * 100, y: "fuel", dx: 4, textAnchor: "start",
      text: (d) => `${((d.mean / total) * 100).toFixed(1)}%`,
    }),
  ],
})));
```

## How batteries capitalise

A grid-scale battery in GB has three main routes to revenue: frequency response (ancillary services), wholesale arbitrage, and capacity markets. As capacity markets are longer-horizon auctions (one or four years out from delivery) with more site-specific physical limitations, this page covers the market for both the shorter term markets: frequency response and wholesale arbitrage.

**Frequency response: contracted availability.** NESO runs daily auctions for capacity
that must react within seconds when frequency strays from 50 Hz. Win one and you are paid
a **£/MW/h availability fee** for every hour you are committed, whether or not you are
actually called. Predictable, contracted income, but each contract ties up part of the battery. A Low
contract needs enough energy in store to discharge for its full delivery window, a High
contract needs the same again as headroom to charge, and every MW sold in one direction
counts against the battery's rating on that side, whichever service it is sold into. 

There are three services, split by how fast and how long they must respond:

| Service | Acts on | Full response within | Sustained for | Opposite-side reserve¹ |
|---|---|---|---|---|
| **DC** — Dynamic Containment | Large deviations, ±0.2–0.5 Hz | 1 second | 15 min | 10% |
| **DM** — Dynamic Moderation | Moderate deviations, ±0.1–0.2 Hz | 1 second | 30 min | 20% |
| **DR** — Dynamic Regulation | Small, everyday deviations, ±0.015–0.2 Hz | 10 seconds | 60 min | 40% |

<p class="muted">¹ Since 15 November 2024, each MW contracted also keeps a share of the
battery's power free on the opposite side for recovering energy: a battery selling 10 MW of
DR Low must keep 4 MW of charging capacity spare. The reserve shares come from the
<a href="https://www.neso.energy/document/378246/download">Response Services Procurement
Rules</a> (Schedule 1, &ldquo;Reserved Capacity&rdquo;); response times and delivery windows
from NESO's <a href="https://www.neso.energy/document/384606/download">Response Service
Terms</a>.</p>

Each runs as two separate auctions: **High**, which responds to *rising* frequency by
charging, and **Low**, which responds to *falling* frequency by discharging. Auctions clear
per **EFA block** — six four-hour windows covering the day — so a battery's commitment can
differ across a given day.

**Wholesale arbitrage: opportunistic trading.** Spot energy prices are dependant on many factors, 
which can lead to a instances of high peaks and low (and even negative) troughs for batteries 
to capitalise on. Simply, batteries can buy energy when it is cheap and sell when it is expensive, 
taking the spread minus round-trip losses & wear cost of cycling as profit.

The tension between these two is the subject of the
[Forecasting & Dispatch](./backtester) page: capacity committed to frequency response
cannot be freely traded, so the operator must decide each day how to split it.

## Frequency Response

Each service is bought through daily
[auctions](https://www.neso.energy/industry-information/balancing-services/frequency-response-services/dynamic-services-dcdmdr),
one for every product (the High and Low of each service) in every EFA block. The clearing
price is the price of the marginal accepted offer for that product and block.

Two changes to the auctions divide the history in this data:

- **Legacy auctions**, up to 1 November 2023. A unit could offer only one of DC, DM and DR in
  a block, and prices never went below zero.
- **Enduring Auction Capability (EAC)**, from 2 November 2023. A unit can split its capacity
  across all three services in the same block and offer several products in one order, and
  prices can go negative. The opposite-side reserve has applied since 15 November 2024.

<details>
<summary>EFA block timings</summary>

${Inputs.table(
  Object.entries(EFA_BLOCKS).map(([k, v]) => ({"EFA Block": +k, "Time window (local clock)": v})),
  {rows: 6, height: 210}
)}

EFA Block 1 spans midnight (23:00 the previous calendar day to 03:00). All times are local GB time.
</details>

### Clearing prices — 28-day rolling average by product

Individual auction results are averaged to a daily figure per product, then smoothed with a
28-day rolling window. Scroll through the story, or pick an auction era on the chart to explore
it yourself.

```js
const rollingByService = (() => {
  const out = [];
  for (const [service, rows] of d3.group(auctions, (d) => d.service)) {
    const daily = d3.sort(
      Array.from(d3.rollup(rows, (v) => d3.mean(v, (d) => d.clearing_price), (d) => +d.date),
        ([date, value]) => ({date: new Date(date), value})),
      (d) => d.date
    );
    for (const r of rollingMean(daily, 28)) out.push({...r, service});
  }
  return out;
})();
```

<div class="scrolly" id="clearing-story">
<div class="scrolly-steps">

<div class="step"><div class="step-inner">
<span class="step-num">2021–22</span>

### The early peak

DC Low averaged **£17.51/MW/h** across 2022, the highest annual average of any product in
this data, with DR High not far behind at £11.53. Few batteries were yet competing for what
NESO needed to buy.
</div></div>

<div class="step"><div class="step-inner">
<span class="step-num">2023</span>

### Then DC collapsed

New battery capacity arrived faster than NESO's requirement grew. DC Low averaged £2.70/MW/h
in 2023, down 85% on 2022, and DR High fell to £1.10. The Low products of DM and DR held up far
better, and have strengthened since.
</div></div>

<div class="step"><div class="step-inner">
<span class="step-num">Throughout</span>

### Low usually clears above High

In most blocks a service's Low product clears above its High product: ${lowAbove.DC}% of DC
blocks, ${lowAbove.DM}% of DM and ${lowAbove.DR}% of DR. Across the fleet, spare capacity to
absorb power has generally been easier to find than spare capacity to inject it.
</div></div>

<div class="step"><div class="step-inner">
<span class="step-num">Since November 2023</span>

### High products go negative

EAC allowed prices below zero, and the High products went there almost immediately. Since
go-live, DR High has cleared negative in ${eacNegative.DRH}% of blocks and DM High in
${eacNegative.DMH}%, while the Low products almost never have. A provider can offer DR High and
DR Low in one order, accepted on its combined value. And DR High delivers free energy the
battery can sell on, which is worth paying a little to hold.
</div></div>

</div>
<div class="scrolly-graphic">
<div class="scrolly-rail" id="clearing-rail"></div>
<div class="chart-head"><span class="muted">28-day rolling average, £/MW/h</span><span id="clearing-era"></span></div>
<div id="clearing-figure"></div>
</div>
</div>

```js
const ERAS = {
  all: {label: "All", domain: d3.extent(rollingByService, (d) => d.date)},
  legacy: {label: "Legacy auctions", domain: [d3.min(rollingByService, (d) => d.date), EAC_GO_LIVE]},
  eac: {label: "EAC", domain: [EAC_GO_LIVE, d3.max(rollingByService, (d) => d.date)]},
};

// What each step of the story puts on the chart: an era, and the products to
// bring forward (null leaves every line at full strength)
const STORY = [
  {era: "legacy", focus: ["DCL", "DRH"]},
  {era: "all", focus: null},
  {era: "all", focus: ["DCL", "DML", "DRL"]},
  {era: "eac", focus: ["DMH", "DRH"], zero: true},
];

const RULE_MARKERS = [
  {date: EAC_GO_LIVE, label: "EAC go-live"},
  {date: RESERVE_RULE, label: "Reserve rule"},
];

function clearingChart({era, focus, zero}, width) {
  const [x0, x1] = ERAS[era].domain;
  const rows = rollingByService.filter((d) => d.date >= x0 && d.date <= x1);
  const inFocus = (d) => !focus || focus.includes(d.service);
  const markers = RULE_MARKERS.filter((d) => d.date > x0 && d.date < x1);
  return Plot.plot({
    width, height: 420, marginLeft: 45, marginRight: 10, marginTop: 24,
    x: {label: null, domain: [x0, x1]},
    y: {label: "£/MW/h", grid: true},
    color: {legend: true, domain: SERVICE_ORDER, range: SERVICE_ORDER.map((s) => SERVICE_COLOURS[s])},
    marks: [
      Plot.ruleY([0], zero
        ? {stroke: "#C9400A", strokeWidth: 1.5}
        : {stroke: "currentColor", strokeOpacity: 0.3}),
      Plot.ruleX(markers, {x: "date", stroke: "#9C948E", strokeDasharray: "3 3"}),
      Plot.text(markers, {x: "date", text: "label", frameAnchor: "top", dy: -14, dx: 4,
                          textAnchor: "start", fill: "#66605C", fontSize: 10}),
      Plot.line(rows.filter((d) => !inFocus(d)),
        {x: "date", y: "value", z: "service", stroke: "service", strokeWidth: 1, strokeOpacity: 0.18}),
      Plot.line(rows.filter(inFocus),
        {x: "date", y: "value", z: "service", stroke: "service", strokeWidth: 1.8}),
      Plot.tip(rows, Plot.pointerX({
        x: "date", y: "value", stroke: "service",
        title: (d) => `${d.service}\n${d.date.toDateString()}\n£${d.value?.toFixed(2)}/MW/h`,
      })),
    ],
  });
}

// Rendered imperatively, as on the dispatch page: the step observer and the era
// picker both drive one render function, and the picker follows the story.
{
  const root = document.getElementById("clearing-story");
  const target = document.getElementById("clearing-figure");
  const eraSlot = document.getElementById("clearing-era");
  const rail = document.getElementById("clearing-rail");
  if (root && target && eraSlot) {
    const state = {step: 0, era: STORY[0].era};
    const eraPicker = choiceGroup(Object.keys(ERAS), {
      value: state.era, format: (k) => ERAS[k].label, label: "Auction era",
    });
    eraSlot.replaceChildren(eraPicker);

    const render = () => {
      const width = Math.max(320, target.getBoundingClientRect().width || 640);
      target.replaceChildren(clearingChart({...STORY[state.step], era: state.era}, width));
    };

    eraPicker.addEventListener("input", () => {
      state.era = eraPicker.value;
      render();
    });
    const stop = watchSteps(root, (i) => {
      state.step = i;
      state.era = STORY[i].era;
      eraPicker.value = state.era;
      render();
    }, {rail});

    window.addEventListener("resize", render);
    invalidation.then(() => {
      stop();
      window.removeEventListener("resize", render);
    });
  }
}
```

### Price distribution

<div class="grid grid-cols-2">
  <div class="card">
    <h3>By product</h3>
    ${resize((width) => Plot.plot({
      width, height: 360, marginLeft: 50,
      x: {label: null, domain: SERVICE_ORDER},
      y: {label: "£/MW/h", grid: true},
      color: {domain: SERVICE_ORDER, range: SERVICE_ORDER.map((s) => SERVICE_COLOURS[s])},
      marks: [
        Plot.ruleY([0], {strokeOpacity: 0.3}),
        Plot.boxY(auctions, {x: "service", y: "clearing_price", fill: "service"}),
      ],
    }))}
    <p class="card-caption">DC Low has the widest spread of outcomes: it cleared highest in the
    early market, and near the bottom since 2023.</p>
  </div>
  <div class="card">
    <h3>By EFA block</h3>
    ${resize((width) => Plot.plot({
      width, height: 360, marginLeft: 50, marginBottom: 45,
      x: {label: "EFA block", tickFormat: (d) => `EFA ${d}`},
      y: {label: "£/MW/h", grid: true},
      color: {domain: SERVICE_ORDER, range: SERVICE_ORDER.map((s) => SERVICE_COLOURS[s]), legend: true},
      marks: [
        Plot.ruleY([0], {strokeOpacity: 0.3}),
        Plot.boxY(auctions, {x: "efa", y: "clearing_price", fill: "service"}),
      ],
    }))}
    <p class="card-caption">The Low products clear highest in EFA 5 (15:00–19:00), across the
    evening peak. Overnight EFA 1 is the cheapest block overall.</p>
  </div>
</div>

### Summary statistics

```js
display(Inputs.table(
  Array.from(d3.group(auctions, (d) => d.service), ([service, v]) => ({
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

Each service runs two separate auctions: **High** (rising frequency → BESS charges) and
**Low** (falling frequency → BESS discharges). Clearing prices differ because available
discharge and charge headroom across the fleet is rarely symmetric.

**Spread = H clearing price − L clearing price.** Positive means charge capacity was scarcer;
negative means discharge capacity was scarcer. All three markets average negative, so the
discharge leg is consistently the scarcer of the two.

```js
const PAIRS = [["DC", "DCH", "DCL"], ["DR", "DRH", "DRL"], ["DM", "DMH", "DML"]];

// Join H against L on (date, EFA block); an inner join, so a block missing either
// leg contributes no spread rather than a half-defined one
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

// Shares of blocks quoted in the text, recomputed on every refresh
const shareOf = (rows, test) => Math.round(d3.mean(rows, (d) => (test(d) ? 1 : 0)) * 100);
const lowAbove = Object.fromEntries(PAIRS.map(([m]) =>
  [m, shareOf(spreads.filter((d) => d.market === m), (d) => d.spread < 0)]));
const highAbove = Object.fromEntries(PAIRS.map(([m]) =>
  [m, shareOf(spreads.filter((d) => d.market === m), (d) => d.spread > 0)]));
const eacNegative = Object.fromEntries(SERVICE_ORDER.map((s) =>
  [s, shareOf(auctions.filter((d) => d.service === s && d.date >= EAC_GO_LIVE), (d) => d.clearing_price < 0)]));
```

### Daily average H − L spread over time

```js
const dailySpread = Array.from(
  d3.rollup(spreads, (v) => d3.mean(v, (d) => d.spread), (d) => d.market, (d) => +d.date),
  ([market, m]) => Array.from(m, ([date, spread]) => ({market, date: new Date(date), spread}))
).flat();

display(resize((width) => Plot.plot({
  width, height: 400, marginLeft: 55,
  x: {label: null},
  y: {label: "£/MW/h", grid: true},
  color: {legend: true, domain: Object.keys(MARKET_COLOURS), range: Object.values(MARKET_COLOURS)},
  marks: [
    Plot.ruleY([0], {strokeDasharray: "4 3", strokeOpacity: 0.6}),
    Plot.line(dailySpread, {x: "date", y: "spread", stroke: "market", strokeWidth: 1.2}),
  ],
})));
```

```js
if (drMean < 0) display(html`<div class="note">
<p><b>Why is the DR spread consistently negative (avg ${drMean.toFixed(2)} £/MW/h)?</b></p>
<p>Because DR High clears below zero in most blocks while DR Low does not. That started the
month the Enduring Auction Capability (EAC) went live: no DR High block cleared negative in
October 2023, and 87% did in November. The legacy auctions never cleared below zero. EAC
allows negative prices, and lets a provider offer several products in one order at a single
price, accepted when the order as a whole is in the money, so a DR High leg can clear
negative inside a package that still pays. It can pay on its own too: energy a battery
absorbs while delivering DR High is neither paid for nor charged, so it can be sold on.</p>
<p>High and Low remain separate products with separate prices, and nothing requires a
provider to hold both. The battery links them physically instead: each MW of DR Low needs an
hour of energy in store, each MW of DR High an hour of headroom, and since November 2024 each
also reserves 40% of its MW on the opposite side for energy recovery.</p>
</div>`);
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
    <p class="card-caption">DC's spreads sit closest to zero and most tightly around it, so its
    two legs are priced most symmetrically. DR sits firmly negative, with High above Low in only
    ${highAbove.DR}% of blocks; DM falls in between.</p>
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
    <p class="card-caption">Spreads are most pronounced in EFA 5 (15:00–19:00), then EFA 6, as
    demand peaks and charge and discharge headroom are least balanced.</p>
  </div>
</div>

### H − L spread by month and EFA block

```js
const heatDomain = [
  d3.min(spreads, (d) => d3.utcMonth.floor(d.date)),
  d3.utcMonth.offset(d3.max(spreads, (d) => d3.utcMonth.floor(d.date)), 1),
];

const heatStrips = PAIRS.map(([market]) => {
  const cells = Array.from(
    d3.rollup(spreads.filter((d) => d.market === market), (v) => d3.mean(v, (d) => d.spread),
      (d) => +d3.utcMonth.floor(d.date), (d) => d.efa),
    ([month, m]) => Array.from(m, ([efa, spread]) => ({month: new Date(month), efa, spread}))
  ).flat();
  // Colours saturate at the 95th percentile of |spread|, so a handful of extreme
  // months don't wash out every other cell
  const lim = Math.ceil(d3.quantile(cells, 0.95, (d) => Math.abs(d.spread)));
  return {market, cells, lim};
});

const heatColour = (lim) => ({type: "diverging", scheme: "RdBu", domain: [-lim, lim], reverse: true, clamp: true});

function heatStrip({market, cells, lim}, width) {
  return Plot.plot({
    width, height: 140, marginLeft: 50, marginRight: 10, marginTop: 4, marginBottom: 22,
    x: {type: "utc", domain: heatDomain, label: null},
    y: {domain: [0.5, 6.5], reverse: true, ticks: [1, 2, 3, 4, 5, 6], tickFormat: (d) => `EFA ${d}`,
        label: null, tickSize: 0},
    color: heatColour(lim),
    marks: [
      Plot.rect(cells, {
        x1: "month", x2: (d) => d3.utcMonth.offset(d.month, 1),
        y1: (d) => d.efa - 0.5, y2: (d) => d.efa + 0.5,
        fill: "spread", inset: 0.5,
      }),
      Plot.ruleX([EAC_GO_LIVE], {stroke: "#33302E", strokeDasharray: "3 3", strokeOpacity: 0.7}),
      Plot.tip(cells, Plot.pointer({
        x: (d) => new Date(+d.month + 14 * 864e5), y: "efa",
        title: (d) => `${market} · EFA ${d.efa} (${EFA_BLOCKS[d.efa]})\n${d3.utcFormat("%B %Y")(d.month)}\n£${d.spread.toFixed(2)}/MW/h`,
      })),
    ],
  });
}

display(resize((width) => html`<div class="heat-strips">${heatStrips.map((strip) => html`<div class="heat-strip">
  <div class="chart-head"><h4>${strip.market}</h4>${Plot.legend({color: {...heatColour(strip.lim), label: "£/MW/h"}, width: 240})}</div>
  ${heatStrip(strip, width)}
</div>`)}</div>`));
```

Each cell is the average H − L spread for a calendar month and EFA block. Red means charge
capacity was scarcer (H > L); blue means discharge capacity was scarcer (L > H). Each strip's
colours saturate at its 95th percentile (DC ±£${heatStrips[0].lim}, DR ±£${heatStrips[1].lim},
DM ±£${heatStrips[2].lim}), so a handful of extreme months don't wash out the rest; hover a cell
for its exact value. The dashed line marks EAC go-live.

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

Two prices matter for a battery trading energy. The **market index** (APXMIDP) is the
half-hourly wholesale reference that the dispatch model on this site trades against. The
**imbalance price** is what a party pays or receives for being out of balance in a
settlement period.

```js
const sysMismatches = d3.sum(sysPrices, (d) => d.sell_buy_mismatches);
const sysHalfHours = d3.sum(sysPrices, (d) => d.n);
const singlePriceNote = sysMismatches === 0
  ? `They match in all ${d3.format(",")(sysHalfHours)} half-hours of this data, so one line covers both.`
  : `They differ in ${d3.format(",")(sysMismatches)} of ${d3.format(",")(sysHalfHours)} half-hours here; the chart shows the sell price.`;
```

Since Elexon's [P305](https://www.elexon.co.uk/bsc/mod-proposal/p305/) took effect in November
2015, GB has settled imbalance at a single price: the System Sell Price and System Buy Price
are the same number. ${singlePriceNote}

### Market index and imbalance price

```js
function rollingBand(rows, series, meanKey, minKey, maxKey) {
  const sorted = d3.sort(rows, (d) => d.date);
  const mean = rollingMean(sorted, 28, "date", meanKey);
  const low = rollingMean(sorted, 28, "date", minKey);
  const high = rollingMean(sorted, 28, "date", maxKey);
  return mean.map((d, i) => ({series, date: d.date, mean: d[meanKey], low: low[i][minKey], high: high[i][maxKey]}));
}

const PRICE_SERIES = ["Market index (APXMIDP)", "Imbalance price"];
const priceBands = [
  ...rollingBand(marketDaily, PRICE_SERIES[0], "mean", "min", "max"),
  ...rollingBand(sysPrices, PRICE_SERIES[1], "price_mean", "price_min", "price_max"),
];

display(resize((width) => Plot.plot({
  width, height: 420, marginLeft: 55,
  x: {label: null},
  y: {label: "£/MWh", grid: true},
  color: {legend: true, domain: PRICE_SERIES, range: ["#0D7680", "#C9400A"]},
  marks: [
    Plot.ruleY([0], {strokeOpacity: 0.3}),
    Plot.areaY(priceBands, {x: "date", y1: "low", y2: "high", z: "series", fill: "series", fillOpacity: 0.12}),
    Plot.line(priceBands, {x: "date", y: "mean", z: "series", stroke: "series", strokeWidth: 1.6}),
    Plot.tip(priceBands, Plot.pointerX({
      x: "date", y: "mean", stroke: "series",
      title: (d) => `${d.series}\n${d.date.toDateString()}\n28-day avg £${d.mean.toFixed(0)}/MWh\ntypical day £${d.low.toFixed(0)}–£${d.high.toFixed(0)}`,
    })),
  ],
})));

const dailyRange = (rows, lo, hi) => d3.mean(rows, (d) => d[hi] - d[lo]);
const negativeShare = (rows) => (d3.sum(rows, (d) => d.negative) / d3.sum(rows, (d) => d.n)) * 100;
```

Lines are 28-day rolling averages of each day's mean price; the shaded bands run from the
average daily low to the average daily high over the same window. The two prices move together
on average, but the imbalance price swings further within a day: its daily range averages
£${dailyRange(sysPrices, "price_min", "price_max").toFixed(0)}/MWh against
£${dailyRange(marketDaily, "min", "max").toFixed(0)} for the market index, and it is negative in
${negativeShare(sysPrices).toFixed(1)}% of half-hours against ${negativeShare(marketDaily).toFixed(1)}%.

### Wholesale price spread

Daily peak-to-trough APXMIDP spread — the raw arbitrage opportunity available to a battery
on any given day, before efficiency losses and cycling cost.

```js
display(resize((width) => Plot.plot({
  width, height: 340, marginLeft: 55,
  x: {label: null},
  y: {label: "Daily peak-to-trough spread (£/MWh)", grid: true},
  marks: [
    Plot.ruleY([0], {strokeOpacity: 0.3}),
    Plot.line(marketDaily, {x: "date", y: "spread", stroke: "#C9400A", strokeOpacity: 0.35}),
    Plot.line(rollingMean(marketDaily, 28, "date", "spread"),
      {x: "date", y: "spread", stroke: "#8B2020", strokeWidth: 2}),
  ],
})));
```

Thin line is the daily spread; heavy line is a 28-day rolling average.
