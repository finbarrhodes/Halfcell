"""
NESO day-ahead wind forecast collector
======================================
Downloads NESO's published day-ahead wind forecast — the half-hourly MW it
expects from transmission-connected wind for the following day.

Why this signal. Every feature the price model currently uses is either a lagged
price or a *previous-day* fundamental, so nothing in it describes the day being
forecast. Tomorrow's wind is the largest single driver of GB price shape, and the
2026-09-17 benchmark showed the model's weakness is spread calibration rather
than average error — a forward-looking wind signal is the most plausible route to
improving it.

Two properties make the dataset usable without look-ahead:

  - It is a genuine forecast, published before the day it describes, not an
    outturn.
  - Each row carries Forecast_Timestamp, which lets the 14:00 D-1 bid deadline be
    checked rather than assumed. It needs care: the field records portal
    publication rather than forecast production, so 12% of rows are stamped after
    their deadline and most of 2026 is stamped inside the settlement day. Those
    rows are no closer to outturn than an independent day-ahead forecast, so
    src/analysis/wind_forecast.py keeps them and flags them instead of dropping
    them. The evidence is in that module's docstring.

Coverage runs from 2018-04-16, which spans the whole feature matrix (from
2019-01-19), so no truncated comparison is needed.

The whole history is one CSV, so this downloads the file rather than paging the
datastore — NESO limits datastore endpoints to 2 requests a minute, and hitting
that returns 403.

Usage:
    python -m src.data_collection.wind_forecast_collector
    python -m src.data_collection.wind_forecast_collector --dest /tmp/scratch
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from loguru import logger

PACKAGE_URL = "https://api.neso.energy/api/3/action/package_show?id=day-ahead-wind-forecast"
# Resource UUID of "Historic Day Ahead Wind Forecasts", used when the metadata
# lookup is unavailable. Discover it yourself with:
#   curl "https://api.neso.energy/api/3/action/package_show?id=day-ahead-wind-forecast"
HISTORIC_RESOURCE_ID = "7524ec65-f782-4258-aaf8-5b926c17b966"
HISTORIC_RESOURCE_NAME = "Historic Day Ahead Wind Forecasts"
DEST = Path(__file__).resolve().parents[2] / "data" / "raw" / "wind_forecast"
FILENAME = "day_ahead_wind_forecast.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Halfcell data pipeline; +https://halfcell.uk)"}


def _open(url: str, timeout: float = 120):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout)


def resource_url() -> str:
    """
    Download URL for the historic forecast resource.

    Resolved from the package metadata so a republished resource is followed
    rather than silently missed, falling back to the pinned UUID if the
    metadata call fails.
    """
    try:
        with _open(PACKAGE_URL, timeout=30) as response:
            resources = json.loads(response.read())["result"]["resources"]
        for resource in resources:
            if (resource.get("name") or "").strip() == HISTORIC_RESOURCE_NAME:
                if resource["id"] != HISTORIC_RESOURCE_ID:
                    logger.warning(
                        f"{HISTORIC_RESOURCE_NAME} has a new resource id: {resource['id']} "
                        f"(pinned {HISTORIC_RESOURCE_ID}). Update the constant."
                    )
                return resource["url"]
        logger.warning(f"No resource named {HISTORIC_RESOURCE_NAME!r}; using the pinned id")
    except Exception as exc:                                    # network, JSON or schema change
        logger.warning(f"Could not read package metadata ({exc}); using the pinned id")
    return (f"https://api.neso.energy/dataset/day-ahead-wind-forecast/resource/"
            f"{HISTORIC_RESOURCE_ID}/download/{FILENAME}")


def download(dest: Path = DEST, force: bool = False) -> Path:
    """
    Fetch the forecast history to dest/day_ahead_wind_forecast.csv.

    The file is rewritten every time because NESO appends to it daily; the
    write goes to a .part file first so an interrupted download cannot leave a
    truncated CSV behind for prepare_data to parse.
    """
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / FILENAME
    url = resource_url()

    logger.info(f"Downloading day-ahead wind forecast → {target}")
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
