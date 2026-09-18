"""
src/analysis/wind_forecast.py
=============================
NESO's day-ahead wind forecast, tidied into the shape the feature matrix wants.

The raw resource is one row per settlement period per forecast, carrying the MW
NESO expected from transmission-connected wind (``Incentive_forecast``), the
metered wind capacity at the time (``Capacity``), and the moment the forecast was
published (``Forecast_Timestamp``).

**On Forecast_Timestamp, which is not what it appears to be.** A forecast is only
usable if it existed when the bid was made — offers for day D close at 14:00 on
D-1 under EAC, 14:30 before it — so the obvious move is to drop rows stamped
later. Doing that discards 12% of the resource and most of 2026, where the median
stamp falls about nine hours *into* the settlement day and some rows are stamped
months afterwards.

Those rows are not same-day forecasts. Checked against Elexon's independent
day-ahead wind forecast and against half-hourly wind outturn (2026-09-17), the
late-stamped rows sit no closer to outturn than a genuine day-ahead forecast
does — 370 MW against Elexon's 290 on 2026-06-10, 307 against 304 on 2026-07-15 —
and rows with proper D-1 stamps behave the same way. If they carried same-day
information they would beat a day-ahead forecast; they do not. The field is
recording when the row was published to the portal, not when the forecast was
produced.

So every row is kept, because the resource is the published day-ahead product by
construction, and each carries ``published_before_deadline`` so a caller can
choose. Where a period has several versions the latest one that beat the deadline
wins, falling back to the latest overall rather than leaving a gap.
``build_wind_forecast_table(..., enforce_deadline=True)`` drops the late rows
instead, so an ablation can be run both ways as a robustness check.

**Clock changes.** Settlement periods come from NESO's own numbering, so the 46-
and 50-period days are carried through as published rather than forced to 48.

Forecast_Timestamp is read as GMT, consistent with the resource's own
Datetime_GMT column.
"""

from datetime import time
from pathlib import Path

import pandas as pd

from src.analysis.neso_rules import EAC_BID_CLOSE, EAC_GO_LIVE, PRE_EAC_BID_CLOSE

COLUMNS = ["settlementDate", "settlementPeriod", "wind_forecast_mw", "capacity_mw",
           "forecast_made_at", "published_before_deadline"]

_RENAMES = {
    "Date": "settlementDate",
    "Settlement_period": "settlementPeriod",
    "Incentive_forecast": "wind_forecast_mw",
    "Capacity": "capacity_mw",
    "Forecast_Timestamp": "forecast_made_at",
}


def bid_deadline(service_date) -> pd.Timestamp:
    """When offers for this service day close: 14:00 on D-1 under EAC, 14:30 before it."""
    day = pd.Timestamp(service_date).normalize()
    hour, minute = EAC_BID_CLOSE if day.date() >= EAC_GO_LIVE else PRE_EAC_BID_CLOSE
    return pd.Timestamp.combine(day - pd.Timedelta(days=1), time(hour, minute))


def read_forecast_file(path) -> pd.DataFrame:
    """One NESO day-ahead wind forecast CSV, tidied to COLUMNS."""
    raw = pd.read_csv(Path(path))
    missing = set(_RENAMES) - set(raw.columns)
    if missing:
        raise ValueError(f"{Path(path).name} is missing expected columns: {sorted(missing)}")

    frame = raw.rename(columns=_RENAMES)[list(_RENAMES.values())].copy()
    frame["settlementDate"] = pd.to_datetime(frame["settlementDate"], format="mixed").dt.normalize()
    frame["settlementPeriod"] = pd.to_numeric(frame["settlementPeriod"], errors="coerce").astype("Int64")
    for column in ("wind_forecast_mw", "capacity_mw"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["forecast_made_at"] = pd.to_datetime(frame["forecast_made_at"], format="mixed", errors="coerce")
    return frame.dropna(subset=["settlementDate", "settlementPeriod"])


def flag_published_before_deadline(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Mark each row with whether its stamp proves it existed by the bid deadline.

    Rows with no stamp count as in time: the resource is published day-ahead by
    construction, and a missing field should not lose a real forecast.
    """
    deadlines = frame["settlementDate"].map(bid_deadline)
    frame = frame.copy()
    frame["published_before_deadline"] = (
        frame["forecast_made_at"].isna() | (frame["forecast_made_at"] <= deadlines)
    )
    return frame


def build_wind_forecast_table(paths, enforce_deadline: bool = False) -> pd.DataFrame:
    """
    One row per settlement period, from one or more forecast CSVs.

    Where a period appears more than once — several publications, or overlapping
    files — the latest version that beat the deadline wins, falling back to the
    latest overall so a period is never lost. enforce_deadline drops late rows
    outright instead, leaving gaps; see the module docstring for why that is the
    strict variant rather than the default.
    """
    frames = [read_forecast_file(path) for path in sorted(paths, key=str)]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)

    table = flag_published_before_deadline(pd.concat(frames, ignore_index=True))
    late = int((~table["published_before_deadline"]).sum())
    if late:
        note = "dropped" if enforce_deadline else "kept but flagged"
        print(f"  {late:,} rows stamped after their bid deadline ({note})")
    if enforce_deadline:
        table = table[table["published_before_deadline"]]

    # True sorts last, so keep="last" prefers an in-time version, then the latest stamp
    table = (table.sort_values(["settlementDate", "settlementPeriod",
                                "published_before_deadline", "forecast_made_at"])
                  .drop_duplicates(["settlementDate", "settlementPeriod"], keep="last"))
    return table[COLUMNS].sort_values(["settlementDate", "settlementPeriod"], ignore_index=True)
