"""
NESO system frequency collector
===============================
Downloads GB system frequency at one-second resolution, one file per month,
from NESO's "System Frequency" dataset. The response delivery calculation
(src/analysis/response_delivery.py) reduces them to the energy delivered per MW
contracted in each settlement period; only that table is committed.

Files are large (about 75 MB a month as CSV; a few early months are zipped) and
NESO's download speed varies, so each file is written to a .part file, checked
against the server's Content-Length, and retried with backoff. Complete files
are skipped, so re-running the command resumes where it stopped.

Usage:
    python -m src.data_collection.frequency_collector --start 2021-09 --end 2026-08
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from loguru import logger

PACKAGE_URL = "https://api.neso.energy/api/3/action/package_show?id=system-frequency-data"
DEST = Path(__file__).resolve().parents[2] / "data" / "raw" / "frequency"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Halfcell data pipeline; +https://halfcell.uk)"}

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]
_RESOURCE_NAME = re.compile(rf"^({'|'.join(MONTH_NAMES)})\s+(\d{{4}})")


def _open(url: str, method: str = "GET", timeout: float = 120):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS, method=method), timeout=timeout)


def list_months(start: str, end: str) -> list[tuple[str, str]]:
    """(YYYY-MM, download URL) for each published month from start to end, oldest first."""
    with _open(PACKAGE_URL, timeout=60) as resp:
        resources = json.loads(resp.read())["result"]["resources"]
    months = {}
    for resource in resources:
        match = _RESOURCE_NAME.match(resource.get("name", ""))
        if match:
            month = f"{match.group(2)}-{MONTH_NAMES.index(match.group(1)) + 1:02d}"
            if start <= month <= end:
                months[month] = resource["url"]
    return sorted(months.items())


def _remote_size(url: str) -> int | None:
    try:
        with _open(url, method="HEAD", timeout=60) as resp:
            size = resp.headers.get("Content-Length")
            return int(size) if size else None
    except Exception:
        return None


def download(month: str, url: str, dest: Path = DEST, retries: int = 5) -> Path:
    """Fetch one month to dest as frequency_YYYY-MM.csv or .zip, skipping it if already complete."""
    ext = url.rsplit(".", 1)[-1].lower()
    target = dest / f"frequency_{month}.{ext}"
    expected = _remote_size(url)
    if target.exists() and (expected is None or target.stat().st_size == expected):
        logger.info(f"{month}: already downloaded")
        return target

    part = target.with_name(target.name + ".part")
    for attempt in range(1, retries + 1):
        started, received = time.monotonic(), 0
        try:
            with _open(url) as resp, open(part, "wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
                    received += len(chunk)
            if expected is not None and received != expected:
                raise OSError(f"received {received:,} bytes, expected {expected:,}")
            part.replace(target)
            logger.info(f"{month}: {received / 1e6:.0f} MB in {time.monotonic() - started:.0f} s")
            return target
        except Exception as exc:
            wait = min(120, 10 * 2 ** (attempt - 1))
            logger.warning(f"{month}: attempt {attempt} of {retries} failed ({exc}); retrying in {wait} s")
            time.sleep(wait)
    part.unlink(missing_ok=True)
    raise RuntimeError(f"{month}: download failed after {retries} attempts")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", required=True, help="First month, YYYY-MM")
    parser.add_argument("--end", required=True, help="Last month, YYYY-MM")
    parser.add_argument("--dest", type=Path, default=DEST)
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    months = list_months(args.start, args.end)
    logger.info(f"{len(months)} months published from {args.start} to {args.end}")

    failed = []
    for month, url in months:
        try:
            download(month, url, args.dest)
        except RuntimeError as exc:
            logger.error(str(exc))
            failed.append(month)
        time.sleep(2)

    total = sum(p.stat().st_size for p in args.dest.glob("frequency_*.*") if not p.name.endswith(".part"))
    logger.info(f"done: {len(months) - len(failed)} of {len(months)} months, {total / 1e9:.2f} GB on disk")
    if failed:
        logger.error(f"failed months, re-run to retry: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
