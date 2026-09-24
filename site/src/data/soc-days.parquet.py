"""Half-hourly state of charge for every backtest day, compactly encoded.

The backtester draws any single day of the run beside the average week, for all
three strategies at once. That needs every day's path, though not at full
precision: state of charge and the range the FR contracts required are stored as
whole steps of 1/250 (0.4 percentage points) and the date as days since 1970,
which keeps ~260k rows to a few hundred kilobytes. The browser looks a day up by
strategy and date; `day-prices.parquet` carries the prices it traded against.

  strategy  0 = pf_mpc, 1 = naive_mpc, 2 = ml_mpc
  day       days since 1970-01-01
  sp        settlement period, 1-48
  soc, lo, hi   state of charge and its required range, in 1/250ths

Rows are sorted by strategy, day and settlement period, with exactly 48 per day.
"""
import io
import sys
from pathlib import Path

import pandas as pd

STRATEGIES = ("pf_mpc", "naive_mpc", "ml_mpc")
SCALE = 250

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data/cache"


def _steps(frac: pd.Series) -> pd.Series:
    return (frac.clip(0, 1) * SCALE).round().astype("uint8")


frames = []
for code, key in enumerate(STRATEGIES):
    df = pd.read_parquet(CACHE / f"soc_{key}.parquet")
    date = pd.to_datetime(df["date"])
    out = pd.DataFrame({
        "strategy": pd.Series(code, index=df.index, dtype="uint8"),
        "day": (date - pd.Timestamp("1970-01-01")).dt.days.astype("int32"),
        "sp": df["sp"].astype("uint8"),
        "soc": _steps(df["soc_frac"]),
        "lo": _steps(df["soc_min_frac"]),
        "hi": _steps(df["soc_max_frac"]),
    })
    complete = out.groupby("day")["sp"].transform("size") == 48
    frames.append(out[complete])

out = pd.concat(frames, ignore_index=True).sort_values(["strategy", "day", "sp"], ignore_index=True)
buf = io.BytesIO()
out.to_parquet(buf, index=False, compression="zstd")
sys.stdout.buffer.write(buf.getvalue())
