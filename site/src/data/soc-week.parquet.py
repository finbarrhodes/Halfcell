"""Average-week SoC profile, pre-aggregated per strategy and month.

The raw trajectories are ~86k half-hourly rows per strategy (2.2 MB for all
three). The chart only ever shows a week-shaped average, so the sufficient
statistics for mean and standard deviation are aggregated here per
(strategy, month, period_in_week). The browser sums them over whatever date
range is selected and recovers the exact mean and sd — same numbers as
averaging the raw trajectory, a fraction of the payload.
"""
import io
import sys
from pathlib import Path

import pandas as pd


def _js_safe(df):
    """Downcast int64/timestamp[ns] so Arrow doesn't hand JavaScript BigInts."""
    for col in df.columns:
        dt = df[col].dtype
        if dt == "int64":
            lo, hi = df[col].min(), df[col].max()
            df[col] = df[col].astype("int32" if lo >= -(2**31) and hi < 2**31 else "float64")
        elif pd.api.types.is_datetime64_any_dtype(dt):
            df[col] = df[col].astype("datetime64[ms]")
    return df


ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data/cache"

frames = []
for key in ("pf_mpc", "naive_mpc", "ml_mpc"):
    df = pd.read_parquet(CACHE / f"soc_{key}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    # Fold onto a Mon–Sun week at half-hourly resolution: 7 days x 48 periods
    df["period_in_week"] = df["date"].dt.dayofweek * 48 + (df["sp"] - 1)
    df["month_dt"] = df["date"].dt.to_period("M").dt.to_timestamp()

    agg = (
        df.groupby(["month_dt", "period_in_week"])["soc_frac"]
        .agg(n="count", total="sum", total_sq=lambda s: (s**2).sum())
        .reset_index()
    )
    agg.insert(0, "strategy", key)
    frames.append(agg)

out = _js_safe(pd.concat(frames, ignore_index=True))
# float32 is ample for a 0-1 SoC fraction and halves the two largest columns;
# zstd on top brings the whole table well under the raw trajectories it replaces.
out["total"] = out["total"].astype("float32")
out["total_sq"] = out["total_sq"].astype("float32")
out["strategy"] = out["strategy"].astype("category")

buf = io.BytesIO()
out.to_parquet(buf, index=False, compression="zstd")
sys.stdout.buffer.write(buf.getvalue())
