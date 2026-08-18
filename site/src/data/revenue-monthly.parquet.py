"""Monthly revenue by strategy — the three pre-computed backtests, long format.

All figures are for the reference 50 MW asset; the browser rescales linearly.
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
CACHE = ROOT / "data/cache"

frames = []
for key in ("pf_mpc", "naive_mpc", "ml_mpc"):
    df = pd.read_parquet(CACHE / f"{key}.parquet")
    df.insert(0, "strategy", key)
    frames.append(df)

out = pd.concat(frames, ignore_index=True)
out["month_dt"] = pd.to_datetime(out["month_dt"])
# `month` is a pandas Period, which serialises to an opaque ordinal in parquet;
# month_dt carries the same information as a real timestamp.
out = out.drop(columns=["month"], errors="ignore")
out = _js_safe(out)
buf = io.BytesIO()
out.to_parquet(buf, index=False)
sys.stdout.buffer.write(buf.getvalue())
