"""Monthly revenue for the service-mix scenarios, long format.

Each scenario is a separate backtest, not the full stack with a stream deleted:
without arbitrage the allocation gives FR every MW it can use, and without FR
the battery trades its whole rating.

  arb_only           arbitrage only, one run per strategy
  fr_only            FR availability only; needs no price forecast, so one run
                     serves every strategy (strategy = "all")

The full stack is in revenue-monthly. All figures are for the reference 50 MW
asset; the browser rescales linearly.
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

sources = [(key, "arb_only", f"{key}_arb_only.parquet") for key in ("pf_mpc", "naive_mpc", "ml_mpc")]
sources.append(("all", "fr_only", "fr_only.parquet"))

frames = []
for strategy, scenario, filename in sources:
    df = pd.read_parquet(CACHE / filename)
    df.insert(0, "scenario", scenario)
    df.insert(0, "strategy", strategy)
    frames.append(df)

out = pd.concat(frames, ignore_index=True)
out["month_dt"] = pd.to_datetime(out["month_dt"])
# `month` is a pandas Period, which serialises to an opaque ordinal in parquet
out = out.drop(columns=["month"], errors="ignore")
out = _js_safe(out)
buf = io.BytesIO()
out.to_parquet(buf, index=False)
sys.stdout.buffer.write(buf.getvalue())
