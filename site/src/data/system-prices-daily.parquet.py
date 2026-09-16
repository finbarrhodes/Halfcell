"""Daily imbalance price statistics.

GB has settled imbalance at a single price since P305 (November 2015), so the
System Sell Price and System Buy Price are the same number. The daily mean, low
and high come from the sell price, and `sell_buy_mismatches` counts half-hours
where the two differ, so the page checks that claim on every refresh rather than
asserting it. `negative` counts half-hours below zero, out of `n`.
"""
import io
import sys
from pathlib import Path

import pandas as pd

def _js_safe(df):
    """Downcast int64 columns so Arrow doesn't hand JavaScript BigInts.

    parquet int64 arrives in the browser as BigInt, which Plot and d3 cannot do
    arithmetic on. int32 round-trips as a plain number.
    """
    import pandas as pd

    for col in df.columns:
        dt = df[col].dtype
        if dt == "int64":
            lo, hi = df[col].min(), df[col].max()
            df[col] = df[col].astype("int32" if lo >= -(2**31) and hi < 2**31 else "float64")
        elif pd.api.types.is_datetime64_any_dtype(dt):
            # timestamp[ns] also arrives as BigInt; millisecond precision is
            # plenty for daily/monthly series and reads as a plain number.
            df[col] = df[col].astype("datetime64[ms]")
    return df


ROOT = Path(__file__).resolve().parents[3]
sp = pd.read_parquet(ROOT / "data/processed/system_prices.parquet")

sp = sp.assign(
    mismatch=(sp["systemSellPrice"] - sp["systemBuyPrice"]).abs() > 1e-6,
    negative=sp["systemSellPrice"] < 0,
)
out = (
    sp.groupby("settlementDate")
    .agg(
        price_mean=("systemSellPrice", "mean"),
        price_min=("systemSellPrice", "min"),
        price_max=("systemSellPrice", "max"),
        n=("systemSellPrice", "size"),
        negative=("negative", "sum"),
        sell_buy_mismatches=("mismatch", "sum"),
    )
    .reset_index()
    .rename(columns={"settlementDate": "date"})
)
out["date"] = pd.to_datetime(out["date"])
out = _js_safe(out)
buf = io.BytesIO()
out.to_parquet(buf, index=False)
sys.stdout.buffer.write(buf.getvalue())
