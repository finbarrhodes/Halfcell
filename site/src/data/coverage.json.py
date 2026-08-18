"""Actual date coverage of each processed dataset.

The methodology page used to hardcode these ranges, which went stale as soon as
the pipeline was re-run — several read "Jul 2023 – present" when the underlying
series had been extended back to 2019. Deriving them keeps the page honest.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data/processed"

SPECS = {
    "auctions":    ("auctions.parquet", "EFA Date"),
    "market_index": ("market_index.parquet", "settlementDate"),
    "system_prices": ("system_prices.parquet", "settlementDate"),
    "generation":  ("generation_daily.parquet", "settlementDate"),
    "bess_fleet":  ("bess_fleet_capacity.parquet", "month"),
}

out = {}
for name, (fname, col) in SPECS.items():
    path = PROCESSED / fname
    if not path.exists():
        continue
    df = pd.read_parquet(path, columns=[col])
    s = pd.to_datetime(df[col])
    entry = {"start": s.min().strftime("%Y-%m-%d"), "end": s.max().strftime("%Y-%m-%d"),
             "rows": int(len(df))}
    if name == "bess_fleet":
        flags = pd.read_parquet(path)
        if "is_extrapolated" in flags.columns:
            entry["n_extrapolated"] = int(flags["is_extrapolated"].sum())
            measured = flags.loc[~flags["is_extrapolated"], "month"]
            entry["measured_end"] = pd.to_datetime(measured).max().strftime("%Y-%m-%d")
    out[name] = entry

print(json.dumps(out, indent=2))
