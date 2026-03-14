"""
scripts/prepare_data.py
=======================
One-time local script: converts raw CSVs into small, pre-processed Parquet
files in data/processed/. Those Parquet files are committed to git and used
by the Streamlit app at runtime (locally and on Streamlit Community Cloud).

Run from the project root:
    python scripts/prepare_data.py
"""

import pandas as pd
from pathlib import Path

ROOT      = Path(__file__).parent.parent
RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

FUEL_GROUP_MAP = {
    "CCGT":          "Gas",
    "OCGT":          "Gas",
    "NUCLEAR":       "Nuclear",
    "WIND":          "Wind",
    "NPSHYD":        "Hydro",
    "BIOMASS":       "Biomass",
    "COAL":          "Coal",
    "OIL":           "Oil",
    "PS":            "Pumped Storage",
    "INTFR":         "Interconnectors",
    "INTIRL":        "Interconnectors",
    "INTNED":        "Interconnectors",
    "INTNEM":        "Interconnectors",
    "INTNSL":        "Interconnectors",
    "INTVKL":        "Interconnectors",
    "INTIFA2":       "Interconnectors",
    "INTEW":         "Interconnectors",
    "INTELEC":       "Interconnectors",
    "OTHER":         "Other",
    # Embedded generation from NESO Historic Demand Data
    "SOLAR":         "Solar",
    "EMBEDDED_WIND": "Wind",
}


def _kb(path: Path) -> int:
    return path.stat().st_size // 1024


# ---------------------------------------------------------------------------
# Auctions — merge legacy auction_results + EAC into one file
# ---------------------------------------------------------------------------
print("Processing auctions...")
frames = []
for p in sorted(RAW.glob("auction_results_*.csv")) + sorted(RAW.glob("eac_results_*.csv")):
    frames.append(pd.read_csv(p, parse_dates=["EFA Date", "Delivery Start", "Delivery End"]))

auctions = (
    pd.concat(frames, ignore_index=True)
    .drop_duplicates(subset=["Service", "EFA Date", "EFA"])
    .sort_values("EFA Date")
    .reset_index(drop=True)
)
out = PROCESSED / "auctions.parquet"
auctions.to_parquet(out, index=False)
print(f"  {len(auctions):,} rows  →  {_kb(out)} KB  ({out.name})")


# ---------------------------------------------------------------------------
# Market index — half-hourly APXMIDP + N2EX spot prices
# ---------------------------------------------------------------------------
print("Processing market index...")
frames = []
for p in sorted(RAW.glob("market_index_*.csv")):
    frames.append(pd.read_csv(p, parse_dates=["settlementDate", "startTime"]))

mkt = (
    pd.concat(frames, ignore_index=True)
    .drop_duplicates(subset=["settlementDate", "settlementPeriod", "dataProvider"])
)
out = PROCESSED / "market_index.parquet"
mkt.to_parquet(out, index=False)
print(f"  {len(mkt):,} rows  →  {_kb(out)} KB  ({out.name})")


# ---------------------------------------------------------------------------
# System prices — keep only the four columns the dashboard uses
# ---------------------------------------------------------------------------
print("Processing system prices...")
frames = []
for p in sorted(RAW.glob("system_prices_*.csv")):
    frames.append(pd.read_csv(
        p,
        parse_dates=["settlementDate"],
        usecols=["settlementDate", "settlementPeriod", "systemSellPrice", "systemBuyPrice"],
    ))

sp = (
    pd.concat(frames, ignore_index=True)
    .drop_duplicates(subset=["settlementDate", "settlementPeriod"])
)
out = PROCESSED / "system_prices.parquet"
sp.to_parquet(out, index=False)
print(f"  {len(sp):,} rows  →  {_kb(out)} KB  ({out.name})")


# ---------------------------------------------------------------------------
# Generation — pre-aggregate to daily totals by fuel group
# The dashboard only plots daily stacked areas, so half-hourly + 15-fuel-type
# granularity (69 MB CSV) can be collapsed to ~14 k rows here.
# ---------------------------------------------------------------------------
print("Processing generation (aggregating to daily by fuel group)...")
frames = []
for p in sorted(RAW.glob("generation_by_fuel_*.csv")):
    frames.append(pd.read_csv(
        p,
        parse_dates=["settlementDate"],
        usecols=["settlementDate", "fuelType", "generation"],
    ))

for p in sorted(RAW.glob("embedded_solar_wind_*.csv")):
    frames.append(pd.read_csv(
        p,
        parse_dates=["settlementDate"],
        usecols=["settlementDate", "fuelType", "generation"],
    ))

gen = pd.concat(frames, ignore_index=True)
gen["fuelGroup"] = gen["fuelType"].map(FUEL_GROUP_MAP).fillna("Other")

gen_daily = (
    gen.groupby(["settlementDate", "fuelGroup"])["generation"]
    .sum()
    .reset_index()
)
out = PROCESSED / "generation_daily.parquet"
gen_daily.to_parquet(out, index=False)
print(f"  {len(gen_daily):,} rows  →  {_kb(out)} KB  ({out.name})")

print("\nDone. Commit the files in data/processed/ to git.")
