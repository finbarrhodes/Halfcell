"""
scripts/compute_kpis.py
=======================
Compute the landing page's market snapshot from committed processed parquets
and write it to data/cache/latest_kpis.json: the GB battery fleet against a year
earlier, and the last month's wholesale spread against the same span a year
earlier. Only overwrites the file if all
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

# A full day is 48 settlement periods; 46 admits the BST spring-forward day.
MIN_PERIODS_PER_DAY = 46

# Plausibility bounds — reject values outside these before writing
BOUNDS = {
    "fleet_mw":                    (1, 100_000),  # GB battery fleet, operational MW
    "fleet_mw_year_earlier":       (1, 100_000),
    "spread_30d_avg":              (0, 3000),     # £/MWh, mean daily peak-to-trough
    "spread_30d_avg_year_earlier": (0, 3000),
}

# The spread card compares the last month of data with the same span a year earlier.
SPREAD_WINDOW_DAYS = 30
# Complete days a window needs before its average is worth showing
MIN_WINDOW_DAYS = 25


def _daily_spread(mkt: pd.DataFrame) -> pd.Series:
    """Daily APXMIDP peak-to-trough, over complete days only."""
    apx = mkt[mkt["dataProvider"] == "APXMIDP"]
    if apx.empty:
        raise ValueError("No APXMIDP rows found in market_index.parquet")

    # Only use days with a full settlement-period count. A partially-published
    # day (the current day, or a truncated API response) yields a spuriously
    # small peak-to-trough spread that still passes the plausibility bounds
    # below — so it has to be excluded here rather than caught downstream.
    # 46 rather than 48 to admit the BST spring-forward day.
    periods_per_day = apx.groupby("settlementDate")["price"].size()
    complete_days   = periods_per_day[periods_per_day >= MIN_PERIODS_PER_DAY].index

    spread = (
        apx[apx["settlementDate"].isin(complete_days)]
        .groupby("settlementDate")["price"]
        .agg(lambda x: x.max() - x.min())
    )
    spread = spread[spread > 0].sort_index()
    if spread.empty:
        raise ValueError("No complete settlement days found in market_index.parquet")
    return spread


def _compute() -> dict:
    auctions = pd.read_parquet(PROCESSED / "auctions.parquet")
    mkt      = pd.read_parquet(PROCESSED / "market_index.parquet")
    fleet    = pd.read_parquet(PROCESSED / "bess_fleet_capacity.parquet")

    out = {}

    # --- Wholesale spread: the last month of data against the same span a year earlier ---
    spread = _daily_spread(mkt)
    end = spread.index.max()
    recent = spread[spread.index > end - pd.Timedelta(days=SPREAD_WINDOW_DAYS)]
    year_end = end - pd.DateOffset(years=1)
    earlier = spread[(spread.index > year_end - pd.Timedelta(days=SPREAD_WINDOW_DAYS))
                     & (spread.index <= year_end)]
    for name, window in (("the last", recent), ("the year-earlier", earlier)):
        if len(window) < MIN_WINDOW_DAYS:
            raise ValueError(f"only {len(window)} complete days in {name} {SPREAD_WINDOW_DAYS}-day "
                             f"spread window; need {MIN_WINDOW_DAYS}")

    out["spread_30d_avg"]              = round(float(recent.mean()), 2)
    out["spread_30d_avg_year_earlier"] = round(float(earlier.mean()), 2)
    out["spread_window_end"]           = end.strftime("%Y-%m-%d")
    out["spread_window_days"]          = SPREAD_WINDOW_DAYS

    # --- GB battery fleet: the latest measured month, never a projected one ---
    # REPD trails the market, and bess_fleet_capacity projects the months since its
    # last entry. A projection on a card would read as a measurement, so the card
    # takes the latest measured month and the same month a year before it.
    measured = fleet[~fleet["is_extrapolated"]].sort_values("month")
    if measured.empty:
        raise ValueError("No measured months in bess_fleet_capacity.parquet")
    latest = measured.iloc[-1]
    before = measured[measured["month"] == latest["month"] - pd.DateOffset(years=1)]
    if before.empty:
        raise ValueError(f"No measured fleet capacity a year before {latest['month']:%Y-%m}")

    out["fleet_mw"]              = round(float(latest["bess_fleet_mw"]), 0)
    out["fleet_month"]           = latest["month"].strftime("%Y-%m")
    out["fleet_mw_year_earlier"] = round(float(before.iloc[0]["bess_fleet_mw"]), 0)

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
