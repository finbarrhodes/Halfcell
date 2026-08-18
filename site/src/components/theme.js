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

export const STRATEGY_LABELS = {
  pf_mpc: "Perfect Foresight",
  naive_mpc: "Naive (D-1 prices)",
  ml_mpc: "ML Model",
};

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
