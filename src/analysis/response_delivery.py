"""
src/analysis/response_delivery.py
=================================
Energy a battery delivers under Dynamic Response contracts, from GB system
frequency.

A contracted unit follows frequency along its service's response curve
(Response Service Terms, Table 1): nothing inside the ±0.015 Hz deadband, a
straight line up to 5% of the contracted MW at the knee (DC ±0.2 Hz, DM
±0.1 Hz; DR has no knee), then a straight line to 100% at saturation (DC
±0.5 Hz, DM and DR ±0.2 Hz). Low products answer frequency below 50 Hz by
discharging; High products answer frequency above it by charging.

Integrating that response over each second of NESO's frequency record gives the
energy delivered per MW contracted in each settlement period, for each service
and direction. Dispatch scales it by the MW actually held.

Approximations:
  - Delivery follows frequency instantly. The Service Terms allow 0.5-2 s to
    begin and 1-10 s to reach full delivery, which moves little energy over
    half an hour.
  - Each product follows its own curve. When products are held together, NESO
    combines them into one "general" curve with breakpoints at ±0.015, ±0.1,
    ±0.2 and ±0.5 Hz; summing the individual curves gives the same line.
"""

from pathlib import Path

import numpy as np
import pandas as pd

F0_HZ = 50.0
SECONDS_PER_PERIOD = 1800
HOURS_PER_PERIOD = 0.5
LOCAL_TZ = "Europe/London"

# (|deviation from 50 Hz|, share of contracted MW delivered), linear between points
RESPONSE_CURVES = {
    "DC": ((0.015, 0.0), (0.2, 0.05), (0.5, 1.0)),
    "DM": ((0.015, 0.0), (0.1, 0.05), (0.2, 1.0)),
    "DR": ((0.015, 0.0), (0.2, 1.0)),
}

DELIVERY_COLUMNS = [f"{service.lower()}_{side}" for service in RESPONSE_CURVES for side in ("low", "high")]


def response_share(deviation_hz, service: str) -> np.ndarray:
    """Share of the contracted MW delivered at a given deviation from 50 Hz, either side."""
    xs, ys = zip(*RESPONSE_CURVES[service])
    return np.interp(np.abs(np.asarray(deviation_hz, dtype=float)), xs, ys, left=0.0, right=1.0)


def product_shares(frequency_hz) -> dict[str, np.ndarray]:
    """Share of contracted MW each product delivers at each frequency reading."""
    deviation = np.asarray(frequency_hz, dtype=float) - F0_HZ
    shares = {}
    for service in RESPONSE_CURVES:
        shares[f"{service.lower()}_low"] = response_share(np.minimum(deviation, 0.0), service)
        shares[f"{service.lower()}_high"] = response_share(np.maximum(deviation, 0.0), service)
    return shares


def settlement_keys(timestamps) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Settlement date and period for UTC timestamps.

    Periods count half-hours elapsed since local midnight, so clock-change days
    have 46 or 50 of them, matching Elexon's numbering.
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    local_midnight = ts.tz_convert(LOCAL_TZ).normalize()
    period = (ts - local_midnight) // pd.Timedelta(minutes=30) + 1
    return local_midnight.tz_localize(None), np.asarray(period, dtype=int)


def delivery_by_settlement_period(frequency: pd.DataFrame) -> pd.DataFrame:
    """
    Energy delivered per MW contracted, by settlement period.

    Parameters
    ----------
    frequency : DataFrame with `dtm` (UTC timestamps, one per second) and `f` (Hz)

    Returns
    -------
    DataFrame with settlementDate, settlementPeriod, dc_low ... dr_high in MWh
    per MW contracted, and coverage, the share of the period's seconds with a
    reading. Missing seconds are assumed to behave like the rest of the period.
    """
    frequency = frequency.drop_duplicates("dtm", keep="last")
    dates, periods = settlement_keys(frequency["dtm"])
    table = pd.DataFrame(product_shares(frequency["f"].to_numpy()))
    table["settlementDate"] = dates
    table["settlementPeriod"] = periods
    table["observed"] = frequency["f"].notna().to_numpy()

    grouped = table.groupby(["settlementDate", "settlementPeriod"], sort=True)
    out = grouped[DELIVERY_COLUMNS].mean() * HOURS_PER_PERIOD
    out["coverage"] = grouped["observed"].sum() / SECONDS_PER_PERIOD
    return out.reset_index()


def read_frequency_file(path: Path) -> pd.DataFrame:
    """One month of NESO frequency data (CSV, or a zip holding one CSV) as dtm and f."""
    raw = pd.read_csv(path)
    raw = raw.iloc[:, :2]
    raw.columns = ["dtm", "f"]
    raw["dtm"] = pd.to_datetime(raw["dtm"], utc=True, format="ISO8601")
    raw["f"] = pd.to_numeric(raw["f"], errors="coerce")
    return raw


def build_delivery_table(paths) -> pd.DataFrame:
    """
    Delivery table across monthly files.

    A settlement period can sit in the file for the previous UTC month (the
    first hour of a British Summer Time day), so where a period appears twice
    the fuller reading wins.
    """
    frames = [delivery_by_settlement_period(read_frequency_file(Path(p))) for p in sorted(paths)]
    if not frames:
        return pd.DataFrame(columns=["settlementDate", "settlementPeriod", *DELIVERY_COLUMNS, "coverage"])
    table = pd.concat(frames, ignore_index=True)
    table = table.sort_values("coverage").drop_duplicates(["settlementDate", "settlementPeriod"], keep="last")
    return table.sort_values(["settlementDate", "settlementPeriod"], ignore_index=True)
