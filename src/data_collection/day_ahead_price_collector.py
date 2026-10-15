"""
N2EX GB day-ahead auction price collector
=========================================
Downloads the hourly day-ahead auction price for the GB bidding zone — the
wholesale price at which energy for each hour of tomorrow has already cleared.

Why this signal. Offers into the response auctions close at 14:00 on D-1, and the
hourly day-ahead auctions for delivery day D clear that morning: EPEX at 09:20,
N2EX at 09:50, results published by 10:00. A real operator pricing a response
offer at 14:00 therefore *already knows* what tomorrow's energy is worth, hour by
hour, and does not have to forecast it. Halfcell priced that energy from a
forecast and a trailing mean, which made it stricter than the market it models;
this resource closes that gap at the offer stage. It is information the bid
deadline permits, not a forecast — see src/analysis/day_ahead_prices.py for the
publication rule and the leak guard built on it.

Provenance and licence. NESO Data Portal dataset "Day-Ahead Power Exchange prices
(Nord Pool)", resource "N2EX GB Day-Ahead Price", published under the NESO Open
Data Licence. Same keyless CKAN API as the other NESO loaders here, so it needs no
token and refreshes in CI exactly as they do.

Two properties make it usable without look-ahead:

  - It is an auction outturn for a *future* delivery day, published the morning
    before that day begins. Unlike the market index this repo trades at, it is
    settled before the offer deadline rather than after delivery.
  - Its delivery date says when it was published: the auction for delivery day X
    publishes at 10:00 on X-1. So "was this known at the deadline?" is a question
    about dates, not about a timestamp column that might mean something else
    (contrast wind_forecast_collector, where Forecast_Timestamp records portal
    publication rather than forecast production).

Coverage runs from 2021-09-20 22:00 UTC, which starts four days into the backtest
window; the engine falls back to the forecast for hours the resource does not
cover, so the opening days and the trailing edge degrade to current behaviour
rather than failing.

**The hour labels are UTC, not London local time.** Every date in the resource
has exactly 24 rows — there are no 23- and 25-hour clock-change days — and
correlating the series against this repo's own UTC-stamped half-hourly APX market
index picks UTC out cleanly: over 2024-01-01..2026-09-30 a UTC reading correlates
0.842 against 0.828 for a London reading, and in BST months, where the two differ
by exactly one hour, 0.897 against 0.863. Reading the labels as local time would
misalign half the year by an hour.

The whole history is one CSV, so this downloads the file rather than paging the
datastore — NESO limits datastore endpoints to 2 requests a minute, and hitting
that returns 403.

Usage:
    python -m src.data_collection.day_ahead_price_collector
    python -m src.data_collection.day_ahead_price_collector --dest /tmp/scratch
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from loguru import logger

PACKAGE_URL = ("https://api.neso.energy/api/3/action/package_show"
               "?id=day-ahead-power-exchange-prices-nordpool")
# Resource UUID of "N2EX GB Day-Ahead Price", used when the metadata lookup is
# unavailable. Discover it yourself with:
#   curl "https://api.neso.energy/api/3/action/package_show?id=day-ahead-power-exchange-prices-nordpool"
DAY_AHEAD_RESOURCE_ID = "4f27eea5-7038-4f73-9740-e3e4ad47c26a"
DAY_AHEAD_RESOURCE_NAME = "N2EX GB Day-Ahead Price"
DEST = Path(__file__).resolve().parents[2] / "data" / "raw" / "day_ahead_prices"
FILENAME = "n2ex_day_ahead_prices.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Halfcell data pipeline; +https://halfcell.uk)"}


def _open(url: str, timeout: float = 120):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout)


def resource_url() -> str:
    """
    Download URL for the day-ahead price resource.

    Resolved from the package metadata so a republished resource is followed
    rather than silently missed, falling back to the pinned UUID if the
    metadata call fails. The portal serves the resource's whole history from
    the datastore dump endpoint.
    """
    try:
        with _open(PACKAGE_URL, timeout=30) as response:
            resources = json.loads(response.read())["result"]["resources"]
        for resource in resources:
            if (resource.get("name") or "").strip() == DAY_AHEAD_RESOURCE_NAME:
                if resource["id"] != DAY_AHEAD_RESOURCE_ID:
                    logger.warning(
                        f"{DAY_AHEAD_RESOURCE_NAME} has a new resource id: {resource['id']} "
                        f"(pinned {DAY_AHEAD_RESOURCE_ID}). Update the constant."
                    )
                return resource["url"]
        logger.warning(f"No resource named {DAY_AHEAD_RESOURCE_NAME!r}; using the pinned id")
    except Exception as exc:                                    # network, JSON or schema change
        logger.warning(f"Could not read package metadata ({exc}); using the pinned id")
    return f"https://api.neso.energy/datastore/dump/{DAY_AHEAD_RESOURCE_ID}"


def download(dest: Path = DEST, force: bool = False) -> Path:
    """
    Fetch the price history to dest/n2ex_day_ahead_prices.csv.

    The file is rewritten every time because NESO appends to it daily; the
    write goes to a .part file first so an interrupted download cannot leave a
    truncated CSV behind for prepare_data to parse. The dump endpoint streams
    without a Content-Length, so the guard is a header check rather than a
    byte count: a truncated stream loses rows off the end, and the
    day-ahead table builder drops any date that is short of a full day.
    """
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / FILENAME
    url = resource_url()

    logger.info(f"Downloading N2EX GB day-ahead prices → {target}")
    part = target.with_suffix(".csv.part")
    with _open(url) as response, open(part, "wb") as handle:
        expected = response.headers.get("Content-Length")
        written = 0
        while chunk := response.read(1 << 20):
            handle.write(chunk)
            written += len(chunk)
    if expected and written != int(expected):
        part.unlink(missing_ok=True)
        raise IOError(f"Truncated download: {written} bytes of {expected}")
    header = part.open().readline().strip()
    if "Delivery Period" not in header or "Price" not in header:
        part.unlink(missing_ok=True)
        raise IOError(f"Unexpected header, the resource may have changed shape: {header!r}")
    part.replace(target)
    logger.info(f"Wrote {written / 1e6:.1f} MB")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEST,
                        help="directory to write the CSV into")
    args = parser.parse_args()
    try:
        download(args.dest)
    except Exception as exc:
        logger.error(f"Download failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
