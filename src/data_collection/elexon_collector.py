"""
Elexon BMRS Data Collection Module

Collects data from the Elexon Insights Solution API (replacement for legacy BMRS).
All endpoints are public and require no API key.

API base: https://data.elexon.co.uk/bmrs/api/v1/

Endpoints used:
- /balancing/settlement/system-prices/{date}  (was B1770)
- /balancing/pricing/market-index              (was B1780)
- /datasets/FUELHH                             (was B1620)
"""

import requests
import pandas as pd
import time
from datetime import timedelta
from typing import Optional, Dict, List, Tuple
from loguru import logger

from ..utils import (
    load_config,
    setup_logging,
    save_dataframe,
    generate_date_range,
    parse_date,
)

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"

# The Insights API rejects requests spanning more than ~8 days for
# market-index and FUELHH.  We chunk into 7-day windows to stay safe.
_CHUNK_DAYS = 7


def _date_chunks(start_date: str, end_date: str) -> List[Tuple[str, str]]:
    """Split a date range into <= 7-day windows."""
    start = parse_date(start_date)
    end = parse_date(end_date)
    chunks = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=_CHUNK_DAYS - 1), end)
        chunks.append((cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cursor = chunk_end + timedelta(days=1)
    return chunks


class ElexonBMRSCollector:
    """Collector for Elexon BMRS data via the Insights Solution REST API."""

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = load_config()

        self.config = config
        self.api_config = config["apis"]["elexon"]
        self.rate_limit = self.api_config.get("rate_limit", 60)
        self.last_request_time = 0
        self.session = requests.Session()

        setup_logging(config)
        logger.info("Elexon BMRS Collector initialized (Insights Solution API)")

    # HTTP helpers

    def _rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        min_interval = 60 / self.rate_limit
        elapsed = time.time() - self.last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        """
        GET request against the Insights Solution API.

        Returns the parsed JSON response dict, or raises on HTTP errors.
        """
        self._rate_limit_wait()
        url = f"{BASE_URL}/{path.lstrip('/')}"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # System Sell / Buy Prices  (was B1770)

    def collect_system_prices(
        self,
        start_date: str,
        end_date: str,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Collect System Sell & Buy Prices for a date range.

        The endpoint serves one settlement date at a time, so we iterate
        over each day in the range.
        """
        logger.info(f"Collecting system prices from {start_date} to {end_date}")
        frames = []

        for day in generate_date_range(start_date, end_date, freq="D"):
            date_str = day.strftime("%Y-%m-%d")
            try:
                data = self._get(f"/balancing/settlement/system-prices/{date_str}")
                records = data.get("data", [])
                if records:
                    frames.append(pd.DataFrame(records))
            except requests.RequestException as e:
                logger.warning(f"Failed for {date_str}: {e}")

        frames = [f for f in frames if not f.empty]
        if not frames:
            logger.warning("No system price data retrieved")
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df["settlementDate"] = pd.to_datetime(df["settlementDate"])
        df["startTime"] = pd.to_datetime(df["startTime"])
        logger.info(f"Collected {len(df)} system price records")

        if save:
            filename = f"system_prices_{start_date}_{end_date}"
            save_dataframe(df, filename, data_type="raw", format="csv")

        return df

    # Market Index Prices  (was B1780)

    def collect_imbalance_prices(
        self,
        start_date: str,
        end_date: str,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Collect Market Index Price & Volume (imbalance proxy).

        The endpoint caps responses at ~8 days, so we chunk the range
        into 7-day windows and concatenate.

        IMPORTANT: The API's 'from' and 'to' parameters filter by startTime (UTC),
        not by settlementDate. During BST, some settlement periods for a given date
        occur before midnight UTC (on the previous calendar day). To capture all
        settlement periods, we request from (chunk_start - 1 day) and then filter
        to the desired settlementDate range.
        """
        logger.info(f"Collecting market index prices from {start_date} to {end_date}")

        # Parse the overall start and end dates for filtering later
        overall_start = parse_date(start_date)
        overall_end = parse_date(end_date)

        # Extend end_date by 1 day for chunking to ensure we get all settlement periods
        # for the final date (which may occur on the next UTC day during BST)
        extended_end = (overall_end + timedelta(days=1)).strftime("%Y-%m-%d")

        frames = []

        for chunk_start, chunk_end in _date_chunks(start_date, extended_end):
            try:
                # Request from 1 day before chunk_start to chunk_end
                # This ensures we capture settlement periods for chunk_start that occurred
                # on (chunk_start - 1) UTC day during BST.
                # NOTE: API has a 7-day maximum range, so we can't extend both ends.
                api_start_date = parse_date(chunk_start) - timedelta(days=1)
                api_start_str = api_start_date.strftime("%Y-%m-%d")

                logger.debug(
                    f"Fetching chunk: API from={api_start_str}, to={chunk_end} "
                    f"(will filter to settlementDate {chunk_start} to {chunk_end})"
                )

                data = self._get(
                    "/balancing/pricing/market-index",
                    params={"from": api_start_str, "to": chunk_end},
                )
                records = data.get("data", [])
                if records:
                    frames.append(pd.DataFrame(records))
            except requests.RequestException as e:
                logger.warning(
                    f"Market index chunk {chunk_start}–{chunk_end} failed: {e}"
                )

        frames = [f for f in frames if not f.empty]
        if not frames:
            logger.warning("No market index data retrieved")
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df["settlementDate"] = pd.to_datetime(df["settlementDate"])
        df["startTime"] = pd.to_datetime(df["startTime"])

        # Filter to only include records within the requested settlementDate range
        # This removes the "spillover" records from the (chunk_start - 1) day
        records_before_filter = len(df)
        df = df[
            (df["settlementDate"] >= overall_start) &
            (df["settlementDate"] <= overall_end)
        ]
        records_after_filter = len(df)

        if records_before_filter > records_after_filter:
            logger.debug(
                f"Filtered out {records_before_filter - records_after_filter} records "
                f"outside requested settlementDate range"
            )

        # Deduplicate in case of overlapping chunks
        df = df.drop_duplicates(
            subset=["settlementDate", "settlementPeriod", "dataProvider"]
        )

        logger.info(f"Collected {len(df)} market index records")

        # Validate settlement period counts
        self._validate_settlement_periods(df, start_date, end_date)

        if save:
            filename = f"market_index_{start_date}_{end_date}"
            save_dataframe(df, filename, data_type="raw", format="csv")

        return df

    def _validate_settlement_periods(
        self, df: pd.DataFrame, start_date: str, end_date: str
    ):
        """
        Validate that each date has the expected number of settlement periods.

        Issues warnings for dates with incorrect counts. Expected counts:
        - Normal days: 48 SPs
        - BST spring forward (last Sunday in March): 46 SPs
        - BST fall back (last Sunday in October): 50 SPs
        """
        apx = df[df["dataProvider"] == "APXMIDP"]
        sp_counts = apx.groupby("settlementDate")["settlementPeriod"].count()

        issues = {}
        for date, count in sp_counts.items():
            expected = 48
            # BST clock changes
            if date.month == 3 and date.day >= 25 and date.day_name() == "Sunday":
                expected = 46  # Spring forward: lose 1 hour (2 SPs)
            elif date.month == 10 and date.day >= 25 and date.day_name() == "Sunday":
                expected = 50  # Fall back: gain 1 hour (2 SPs)

            if count != expected:
                issues[date] = (count, expected)

        if issues:
            logger.warning(
                f"Found {len(issues)} dates with incorrect SP counts "
                f"in range {start_date} to {end_date}:"
            )
            for date, (actual, expected) in list(issues.items())[:5]:
                logger.warning(
                    f"  {date.strftime('%Y-%m-%d %A')}: {actual} SPs (expected {expected})"
                )
            if len(issues) > 5:
                logger.warning(f"  ... and {len(issues) - 5} more")
        else:
            logger.info(
                f"Settlement period validation passed: all dates have correct SP counts"
            )

    # Generation by Fuel Type  (was B1620)

    def collect_generation_by_fuel(
        self,
        start_date: str,
        end_date: str,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Collect half-hourly generation output by fuel type (FUELHH dataset).

        The endpoint caps responses at ~8 days, so we chunk the range.
        """
        logger.info(f"Collecting generation by fuel from {start_date} to {end_date}")
        frames = []

        for chunk_start, chunk_end in _date_chunks(start_date, end_date):
            try:
                data = self._get(
                    "/datasets/FUELHH",
                    params={
                        "settlementDateFrom": chunk_start,
                        "settlementDateTo": chunk_end,
                    },
                )
                records = data.get("data", [])
                if records:
                    frames.append(pd.DataFrame(records))
            except requests.RequestException as e:
                logger.warning(
                    f"FUELHH chunk {chunk_start}–{chunk_end} failed: {e}"
                )

        frames = [f for f in frames if not f.empty]
        if not frames:
            logger.warning("No generation data retrieved")
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df["settlementDate"] = pd.to_datetime(df["settlementDate"])
        df["startTime"] = pd.to_datetime(df["startTime"])
        df["publishTime"] = pd.to_datetime(df["publishTime"])
        logger.info(f"Collected {len(df)} generation records")

        if save:
            filename = f"generation_by_fuel_{start_date}_{end_date}"
            save_dataframe(df, filename, data_type="raw", format="csv")

        return df

    # Convenience: collect everything

    def collect_all_markets(
        self,
        start_date: str,
        end_date: str,
        save: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Collect data from all available endpoints."""
        logger.info(f"Collecting all BMRS data from {start_date} to {end_date}")

        data = {
            "system_prices": self.collect_system_prices(start_date, end_date, save),
            "market_index": self.collect_imbalance_prices(start_date, end_date, save),
            "generation_by_fuel": self.collect_generation_by_fuel(
                start_date, end_date, save
            ),
        }

        total = sum(len(df) for df in data.values())
        logger.info(f"Completed BMRS collection — {total} total records")
        return data


# Quick smoke-test when run directly

if __name__ == "__main__":
    config = load_config()
    collector = ElexonBMRSCollector(config)

    start, end = "2024-01-01", "2024-01-02"
    results = collector.collect_all_markets(start, end, save=False)

    for name, df in results.items():
        print(f"\n{name.upper()}: {len(df)} records")
        if not df.empty:
            print(df.head(3).to_string())

