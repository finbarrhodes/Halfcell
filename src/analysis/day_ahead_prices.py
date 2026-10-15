"""
src/analysis/day_ahead_prices.py
================================
N2EX GB day-ahead auction prices, tidied into the settlement-period shape the
engine works in.

The raw resource is one row per delivery hour, carrying the delivery date, a
``Delivery Period`` label like ``"14:00 - 15:00"`` and the £/MWh the hour cleared
at. Two things have to be settled before it can be used: what the hour labels
mean in clock terms, and when each price became public.

**The hour labels are UTC.** Every date in the resource carries exactly 24 rows —
there are no 23- and 25-hour days at the clock changes, which a local-time series
would have — and correlating it against this repo's own UTC-stamped half-hourly
APX market index picks UTC out cleanly: over 2024-01-01..2026-09-30 a UTC reading
correlates 0.842 with the index against 0.828 for a London reading, and in BST
months, where the two readings differ by exactly one hour, 0.897 against 0.863.
So the labels are localised to UTC and converted to London time to find the
settlement period they fall in. Reading them as local time would misalign half
the year by an hour.

**When each price became public.** GB's hourly day-ahead auctions for delivery
day X clear on the morning of X-1 — EPEX at 09:20, N2EX at 09:50 — with results
published by 10:00. The resource carries no publication timestamp, and does not
need one: the delivery date *is* the publication date, offset by a day. So
``published_at`` is 10:00 on X-1, and a price is admissible for a decision taken
at time T exactly when ``published_at <= T``. For a response offer closing at
14:00 on D-1 that admits every delivery date up to and including D.

This is what makes the resource information rather than foresight, and it is the
one property the whole change rests on, so ``admissible_for_service_day`` states
it as code and the tests assert it against ``neso_rules.bid_deadline`` rather
than leaving it to the argument above.

**Clock changes.** Settlement periods are counted as absolute half-hours from
local midnight, so the 46- and 50-period days come out with 46 and 50 periods
rather than being forced to 48 — the same choice wind_forecast.py makes. A
settlement period's UTC hour never falls on a later date than its own settlement
date, because London is never behind UTC; that is what makes the publication rule
above safe for every period in a service day, and test_day_ahead_prices checks it
over the whole history rather than trusting the reasoning.
"""

from pathlib import Path

import pandas as pd

COLUMNS = ["settlementDate", "settlementPeriod", "deliveryDate", "day_ahead_price"]

# Results of the day-ahead auctions for delivery day X are published by this hour
# on X-1 (N2EX clears at 09:50). Four hours before the 14:00 EAC response deadline.
PUBLISH_HOUR_UTC = 10

_RENAMES = {"Date": "deliveryDate", "Delivery Period": "delivery_period", "Price": "day_ahead_price"}


def published_at(delivery_date) -> pd.Timestamp:
    """When the day-ahead auction for `delivery_date` published its prices: 10:00 on D-1."""
    day = pd.Timestamp(delivery_date).normalize()
    return day - pd.Timedelta(days=1) + pd.Timedelta(hours=PUBLISH_HOUR_UTC)


def admissible_for_service_day(service_day) -> pd.Timestamp:
    """
    Latest delivery date whose prices an offer for `service_day` may see.

    Offers for D close at 14:00 on D-1; the auction for delivery date X publishes
    at 10:00 on X-1. So X is admissible when 10:00 on X-1 <= 14:00 on D-1, which
    is every X up to and including D.
    """
    return pd.Timestamp(service_day).normalize()


def read_price_file(path) -> pd.DataFrame:
    """One raw CSV, with the delivery hour resolved to a UTC instant."""
    frame = pd.read_csv(path).rename(columns=_RENAMES)
    missing = {"deliveryDate", "delivery_period", "day_ahead_price"} - set(frame.columns)
    if missing:
        raise ValueError(f"{Path(path).name} is missing {sorted(missing)}")
    frame["deliveryDate"] = pd.to_datetime(frame["deliveryDate"], errors="coerce").dt.normalize()
    # "14:00 - 15:00" -> the hour starting 14:00. The label is UTC (see the module docstring).
    hour = pd.to_numeric(frame["delivery_period"].astype(str).str.slice(0, 2), errors="coerce")
    frame["day_ahead_price"] = pd.to_numeric(frame["day_ahead_price"], errors="coerce")
    frame = frame.dropna(subset=["deliveryDate", "day_ahead_price"])
    hour = hour.reindex(frame.index)
    frame = frame[hour.between(0, 23)]
    frame["hour_utc"] = (frame["deliveryDate"] + pd.to_timedelta(hour.reindex(frame.index), unit="h")
                         ).dt.tz_localize("UTC")
    return frame[["deliveryDate", "hour_utc", "day_ahead_price"]]


def hourly_prices(paths) -> pd.Series:
    """£/MWh indexed by the UTC instant each delivery hour starts, latest file winning."""
    frames = [read_price_file(path) for path in paths]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.Series(dtype=float, name="day_ahead_price",
                         index=pd.DatetimeIndex([], tz="UTC", name="hour_utc"))
    table = pd.concat(frames, ignore_index=True)
    table = table.drop_duplicates("hour_utc", keep="last").sort_values("hour_utc")
    return table.set_index("hour_utc")["day_ahead_price"]


def build_day_ahead_price_table(paths) -> pd.DataFrame:
    """
    One row per GB settlement period the auction prices cover.

    Settlement periods are half-hourly and the auction is hourly, so the two
    periods inside an hour carry that hour's price. `deliveryDate` is kept
    because it, not the settlement date, says when the price was published.
    """
    prices = hourly_prices(paths)
    if prices.empty:
        return pd.DataFrame(columns=COLUMNS)

    # A half-hourly UTC grid over the hours the auction covers, labelled with the
    # settlement period each half-hour falls in.
    grid = pd.date_range(prices.index.min(), prices.index.max() + pd.Timedelta(hours=1),
                         freq="30min", tz="UTC", inclusive="left")
    local = grid.tz_convert("Europe/London")
    midnight = local.normalize()
    table = pd.DataFrame({
        "settlementDate": midnight.tz_localize(None),
        # Absolute half-hours since local midnight, so a clock-change day gets its
        # 46 or 50 periods instead of being forced to 48.
        "settlementPeriod": ((local - midnight) // pd.Timedelta(minutes=30)).astype(int) + 1,
        "hour_utc": grid.floor("h"),
    })
    table["deliveryDate"] = table["hour_utc"].dt.tz_localize(None).dt.normalize()
    table["day_ahead_price"] = table["hour_utc"].map(prices)
    table = table.dropna(subset=["day_ahead_price"])
    return (table[COLUMNS]
            .sort_values(["settlementDate", "settlementPeriod"], ignore_index=True))
