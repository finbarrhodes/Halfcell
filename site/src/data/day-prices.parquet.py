"""Half-hourly wholesale price for every backtest day, in delivery order.

The day views draw the state of charge against the prices the battery was
trading on, and `soc-days.parquet` is keyed by strategy, so carrying the price
there would ship the same 48 numbers three times. It lives here instead, keyed
by day alone, and the browser joins the two.

  day    days since 1970-01-01, matching soc-days.parquet
  sp     settlement period, 1-48
  price  APX day-ahead market index price, £/MWh, to the nearest pound

A handful of (day, period) pairs have no published price; they are left out
rather than filled, so the price line breaks where the data does.
"""
import io
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

# Only the backtest window: a price with no dispatch beside it has nothing to
# draw against, and the runs all share one window.
manifest = json.loads((ROOT / "data/cache/manifest.json").read_text())
spans = [run["params"] for run in manifest.values()]
first = min(pd.Timestamp(p["start_date"]) for p in spans)
last = max(pd.Timestamp(p["end_date"]) for p in spans)

mkt = pd.read_parquet(ROOT / "data/processed/market_index.parquet")
apx = mkt[mkt["dataProvider"] == "APXMIDP"].copy()
date = pd.to_datetime(apx["settlementDate"])
apx, date = apx[(date >= first) & (date <= last)], date[(date >= first) & (date <= last)]

out = pd.DataFrame({
    "day": (date - pd.Timestamp("1970-01-01")).dt.days.astype("int32"),
    "sp": apx["settlementPeriod"].astype("uint8"),
    # Whole pounds: the chart's price axis is read to the nearest ten, and int16
    # spans every price GB has settled, negative spikes included.
    "price": apx["price"].round().astype("int16"),
}).dropna().sort_values(["day", "sp"], ignore_index=True)

buf = io.BytesIO()
out.to_parquet(buf, index=False, compression="zstd")
sys.stdout.buffer.write(buf.getvalue())
