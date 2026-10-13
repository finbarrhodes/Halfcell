"""
Modelled revenue over the last twelve complete months, and the twelve before them.

The backtest's headline figures average five years, and 2022 alone earned more than
twice what the market has since, so they say little about where revenue stands now.
These are the same published runs, cut to recent years: complete calendar months only,
so a part-month at the end of the data never stands in for a whole one.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data/cache"
STRATEGIES = ("pf_mpc", "naive_mpc", "ml_mpc", "fr_only")

manifest = json.loads((CACHE / "manifest.json").read_text())
params = manifest["ml_mpc"]["params"]
power_mw = params["power_mw"]
end = pd.Timestamp(params["end_date"])

# The data's last month counts only if it runs to that month's last day
last = end.to_period("M") if end == end + pd.offsets.MonthEnd(0) else end.to_period("M") - 1
recent = pd.period_range(last - 11, last, freq="M")
earlier = recent - 12


def per_mw_year(net: pd.Series, months: pd.PeriodIndex, key: str) -> float:
    missing = [str(m) for m in months if m not in net.index]
    if missing:
        raise SystemExit(f"{key}: no cached revenue for {', '.join(missing)}")
    return round(float(net.loc[months].sum()) / power_mw, 0)


out = {
    "recent": {"from": str(recent[0]), "to": str(recent[-1])},
    "earlier": {"from": str(earlier[0]), "to": str(earlier[-1])},
}
for key in STRATEGIES:
    table = pd.read_parquet(CACHE / f"{key}.parquet")
    net = table.groupby(pd.to_datetime(table["month_dt"]).dt.to_period("M"))["net_revenue"].sum()
    out[key] = {"recent": per_mw_year(net, recent, key), "earlier": per_mw_year(net, earlier, key)}

print(json.dumps(out))
