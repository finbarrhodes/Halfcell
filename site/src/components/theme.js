// Shared palette and labels — mirrors src/visualization/ so the static site and
// the Python app stay visually consistent while both exist.

export const SERVICE_COLOURS = {
  DCH: "#0D7680", DCL: "#5BA8AE",
  DRH: "#4E8A3C", DRL: "#8AB87F",
  DMH: "#7B3FA0", DML: "#B08FC8",
  Arbitrage: "#C9400A",
  "Cycling cost": "#8B2020",
};

export const MARKET_COLOURS = {DC: "#0D7680", DR: "#C9400A", DM: "#4E8A3C"};

export const EFA_BLOCKS = {
  1: "23:00 – 03:00", 2: "03:00 – 07:00", 3: "07:00 – 11:00",
  4: "11:00 – 15:00", 5: "15:00 – 19:00", 6: "19:00 – 23:00",
};

export const SERVICE_LABELS = {
  DCH: "DC High", DCL: "DC Low",
  DRH: "DR High", DRL: "DR Low",
  DMH: "DM High", DML: "DM Low",
};

// Fuel groups on the generation chart. Eleven series, so this cannot lean on the
// site's three brand colours alone — but it stays on the same warm ground:
// renewables take the teal/green/amber side, fossils the orange/brown/red side,
// and every value clears 3:1 against the paper so a thin line stays legible.
//
// Without this map Plot falls back to its default `observable10`, which has ten
// colours for eleven groups — Gas and Pumped Storage came out the same blue.
export const FUEL_COLOURS = {
  Wind: "#0D7680",
  Solar: "#B87A15",
  Hydro: "#2E6FA7",
  "Pumped Storage": "#4A8FB5",
  Biomass: "#4E8A3C",
  Nuclear: "#7B3FA0",
  Interconnectors: "#8C7BA6",
  Gas: "#C9400A",
  Coal: "#5A4636",
  Oil: "#8B2020",
  Other: "#8A8078",
};

// The three backtest strategies. Previously only the backtester page named these
// colours, inline and twice; the homepage chart fell through to Plot's default
// scheme, so the same three series were a different colour on each page.
export const STRATEGY_COLOURS = {
  pf_mpc: "#4E8A3C",     // ceiling
  ml_mpc: "#0D7680",     // the model under test
  naive_mpc: "#C9400A",  // floor
};

export const STRATEGY_LABELS = {
  pf_mpc: "Perfect Foresight",
  naive_mpc: "Naive (D-1 prices)",
  ml_mpc: "ML Model",
};

// Plot wants `range` as an array lined up with `domain`. Callers order their
// domain by the data (largest series first, say), so build the range from it
// rather than from the map's own key order.
export const rangeFor = (colours, domain, fallback = "#8A8078") =>
  domain.map((key) => colours[key] ?? fallback);

export const gbp = (v) =>
  Math.abs(v) >= 1e6 ? `£${(v / 1e6).toFixed(2)}M` : `£${(v / 1e3).toFixed(0)}k`;

// Rolling mean over a sorted array of {x, y}, using a day-count window.
export function rollingMean(rows, days, xKey = "date", yKey = "value") {
  const out = [];
  const ms = days * 864e5;
  for (let i = 0; i < rows.length; i++) {
    const t = +rows[i][xKey];
    let sum = 0, n = 0;
    for (let j = i; j >= 0 && t - +rows[j][xKey] < ms; j--) {
      const v = rows[j][yKey];
      if (v != null && !isNaN(v)) { sum += v; n++; }
    }
    out.push({...rows[i], [yKey]: n ? sum / n : null});
  }
  return out;
}
