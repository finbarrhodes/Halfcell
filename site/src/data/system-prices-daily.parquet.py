"""Daily system sell/buy price statistics (imbalance settlement)."""
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

out = (
    sp.groupby("settlementDate")
    .agg(
        ssp_mean=("systemSellPrice", "mean"),
        ssp_min=("systemSellPrice", "min"),
        ssp_max=("systemSellPrice", "max"),
        sbp_mean=("systemBuyPrice", "mean"),
    )
    .reset_index()
    .rename(columns={"settlementDate": "date"})
)
out["date"] = pd.to_datetime(out["date"])
out = _js_safe(out)
buf = io.BytesIO()
out.to_parquet(buf, index=False)
sys.stdout.buffer.write(buf.getvalue())
