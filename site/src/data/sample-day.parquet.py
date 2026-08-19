"""One day of half-hourly prices and MPC dispatch, for the walkthrough.

The Forecasting & Dispatch page explains the model against a single worked day.
2026-01-08 is fixed rather than chosen dynamically: it sits in the held-out test
period, has a complete 48 periods, and shows a realistic GB winter shape with an
evening peak and a ~£218/MWh peak-to-trough spread — illustrative without being
the kind of outlier that misrepresents a normal day.

Charge/discharge power is derived from the change in state of charge between
periods, which is what the dispatch LP actually decides.
"""
import io
import sys
from pathlib import Path

import pandas as pd

SAMPLE_DATE = "2026-01-08"
STRATEGIES = ("pf_mpc", "naive_mpc", "ml_mpc")


def _js_safe(df):
    for col in df.columns:
        dt = df[col].dtype
        if dt == "int64":
            lo, hi = df[col].min(), df[col].max()
            df[col] = df[col].astype("int32" if lo >= -(2**31) and hi < 2**31 else "float64")
        elif pd.api.types.is_datetime64_any_dtype(dt):
            df[col] = df[col].astype("datetime64[ms]")
    return df


ROOT = Path(__file__).resolve().parents[3]

mkt = pd.read_parquet(ROOT / "data/processed/market_index.parquet")
apx = mkt[(mkt["dataProvider"] == "APXMIDP")
          & (pd.to_datetime(mkt["settlementDate"]) == SAMPLE_DATE)]
prices = (
    apx[["settlementPeriod", "price"]]
    .rename(columns={"settlementPeriod": "sp"})
    .sort_values("sp")
    .reset_index(drop=True)
)

import json
manifest = json.loads((ROOT / "data/cache/manifest.json").read_text())
params = manifest["ml_mpc"]["params"]
energy_mwh = params["power_mw"] * params["duration_h"]

frames = []
for key in STRATEGIES:
    soc = pd.read_parquet(ROOT / f"data/cache/soc_{key}.parquet")
    soc = soc[pd.to_datetime(soc["date"]) == SAMPLE_DATE][["sp", "soc_frac"]].sort_values("sp")
    if soc.empty:
        continue
    df = prices.merge(soc, on="sp", how="left")
    # MW implied by the SoC change across a half-hour period; positive = charging
    df["power_mw"] = df["soc_frac"].diff().fillna(0) * energy_mwh / 0.5
    df.insert(0, "strategy", key)
    frames.append(df)

out = _js_safe(pd.concat(frames, ignore_index=True))
buf = io.BytesIO()
out.to_parquet(buf, index=False)
sys.stdout.buffer.write(buf.getvalue())
