"""
scripts/compute_kpis.py
=======================
Compute market snapshot KPIs from committed processed parquets and write
to data/cache/latest_kpis.json. Only overwrites the file if all
validation checks pass — on failure the existing JSON is left untouched.

Run from the project root:
    python scripts/compute_kpis.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT     = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
CACHE    = ROOT / "data" / "cache"
OUT_FILE = CACHE / "latest_kpis.json"

# Plausibility bounds — reject values outside these before writing
BOUNDS = {
    "dch_latest":    (-10, 200),   # £/MW/h
    "dch_30d_avg":   (-10, 200),
    "dch_90d_avg":   (-10, 200),
    "spread_latest": (  0, 3000),  # £/MWh peak-to-trough
    "spread_30d_avg":(  0, 3000),
    "spread_90d_avg":(  0, 3000),
}


def _compute() -> dict:
    auctions = pd.read_parquet(PROCESSED / "auctions.parquet")
    mkt      = pd.read_parquet(PROCESSED / "market_index.parquet")

    out = {}

    # --- DCH clearing prices ---
    dch = auctions[auctions["Service"] == "DCH"]
    if dch.empty:
        raise ValueError("No DCH rows found in auctions.parquet")

    daily     = dch.groupby("EFA Date")["Clearing Price"].mean().sort_index()
    latest_dt = daily.index.max()
    cut30     = latest_dt - pd.Timedelta(days=30)
    cut90     = latest_dt - pd.Timedelta(days=90)

    out["dch_latest"]      = round(float(daily.iloc[-1]), 2)
    out["dch_latest_date"] = latest_dt.strftime("%Y-%m-%d")
    out["dch_30d_avg"]     = round(float(daily[daily.index >= cut30].mean()), 2)
    out["dch_90d_avg"]     = round(float(daily[daily.index >= cut90].mean()), 2)

    # --- APXMIDP peak-to-trough spread ---
    apx = mkt[mkt["dataProvider"] == "APXMIDP"]
    if apx.empty:
        raise ValueError("No APXMIDP rows found in market_index.parquet")

    spread = (
        apx.groupby("settlementDate")["price"]
        .agg(lambda x: x.max() - x.min())
    )
    spread      = spread[spread > 0].sort_index()
    latest_mkt  = spread.index.max()
    cut30m      = latest_mkt - pd.Timedelta(days=30)
    cut90m      = latest_mkt - pd.Timedelta(days=90)

    out["spread_latest"]      = round(float(spread.iloc[-1]), 2)
    out["spread_latest_date"] = latest_mkt.strftime("%Y-%m-%d")
    out["spread_30d_avg"]     = round(float(spread[spread.index >= cut30m].mean()), 2)
    out["spread_90d_avg"]     = round(float(spread[spread.index >= cut90m].mean()), 2)

    # --- Coverage ---
    out["data_start"]   = auctions["EFA Date"].min().strftime("%Y-%m-%d")
    out["data_end"]     = auctions["EFA Date"].max().strftime("%Y-%m-%d")
    out["computed_at"]  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return out


def _validate(kpis: dict) -> None:
    for key, (lo, hi) in BOUNDS.items():
        val = kpis.get(key)
        if val is None:
            raise ValueError(f"Missing required KPI: {key}")
        if not (lo <= val <= hi):
            raise ValueError(f"{key} = {val} outside plausible range [{lo}, {hi}]")


def main() -> None:
    print("Computing market snapshot KPIs...")

    try:
        kpis = _compute()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        _validate(kpis)
    except ValueError as exc:
        print(f"VALIDATION FAILED — {OUT_FILE} left untouched: {exc}", file=sys.stderr)
        sys.exit(1)

    CACHE.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(kpis, indent=2) + "\n")
    print(f"Written → {OUT_FILE}")
    for k, v in kpis.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
