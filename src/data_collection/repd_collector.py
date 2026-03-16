"""
Renewable Energy Planning Database (REPD) Collector
=====================================================
Downloads and parses the DESNZ Renewable Energy Planning Database to build a
monthly time series of cumulative GB operational BESS fleet capacity (MW).

The REPD is published quarterly by the Department for Energy Security and Net
Zero (DESNZ). Each quarterly release is a static Excel file — no API key or
rate-limiting is required.

Publication page (check here for the latest quarterly URL):
  https://www.gov.uk/government/publications/renewable-energy-planning-database-monthly-extract

Usage
-----
    # Download and build the series from the latest release:
    collector = REPDCollector(
        override_url="https://assets.publishing.service.gov.uk/media/.../repd-month-march-2025.xlsx"
    )
    monthly = collector.collect(save=True)   # → data/raw/bess_fleet_capacity_raw.csv

    # Or, if you already have the file locally:
    collector = REPDCollector(local_path="data/raw/repd-march-2025.xlsx")
    monthly = collector.collect(save=True)

Notes
-----
- Technology type strings vary across quarterly releases. The BATTERY_TECH_KEYWORDS
  set handles the observed variants. If a future release introduces a new type string,
  a warning is logged with a sample of unmatched values so it can be caught promptly.
- The REPD lags the current date by roughly one quarter. Trailing months are
  linearly extrapolated using a 12-month OLS trend (numpy.polyfit degree=1).
  Extrapolation covers a ≤3 month gap in practice and is conservative relative
  to GB fleet growth rates.
"""

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from loguru import logger


# Battery storage technology type strings observed across REPD quarterly releases.
# Matching is done case-insensitively after stripping whitespace.
BATTERY_TECH_KEYWORDS = frozenset([
    "battery",
    "battery storage",
    "advanced conversion technologies",
])

# Column name candidates (lowercased, stripped) across observed REPD layouts.
# Each entry maps the canonical name we want to the set of aliases in the file.
_COL_ALIASES: dict[str, list[str]] = {
    "technology_type":       ["technology type", "technology"],
    "installed_capacity_mw": ["installed capacity (mwelec)", "installed capacity (mw)", "capacity (mw)", "mwelec"],
    "development_status":    ["development status", "status"],
    "operational_date":      ["operational", "date of operation"],
    "site_name":             ["site name", "project name", "project"],
}


def _find_col(columns: list[str], canonical: str) -> Optional[str]:
    """Return the first column in *columns* that matches one of the aliases for *canonical*."""
    aliases = _COL_ALIASES.get(canonical, [canonical])
    cols_lower = {c.strip().lower(): c for c in columns}
    for alias in aliases:
        if alias in cols_lower:
            return cols_lower[alias]
    return None


class REPDCollector:
    """
    Downloader and parser for the DESNZ Renewable Energy Planning Database.

    Parameters
    ----------
    override_url : str, optional
        Direct URL to the latest REPD Excel file. Check the GOV.UK publication
        page for the current quarterly URL — it changes each release.
    local_path : str or Path, optional
        Path to a locally-cached REPD Excel file. Skips the download step.
        Takes precedence over override_url if both are provided.
    cache_dir : str or Path, optional
        Directory to save the downloaded Excel file. Defaults to data/raw/
        relative to the project root.
    """

    def __init__(
        self,
        override_url: Optional[str] = None,
        local_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.override_url = override_url
        self.local_path = Path(local_path) if local_path else None

        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "data" / "raw"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_repd(self) -> Path:
        """
        Download the REPD Excel file to cache_dir.

        Skips the download if a local_path was provided or if a cached copy
        already exists at the expected destination. Returns the local path.

        Raises
        ------
        ValueError
            If neither override_url nor local_path is set.
        """
        if self.local_path and self.local_path.exists():
            logger.info(f"Using local REPD file: {self.local_path}")
            return self.local_path

        if not self.override_url:
            raise ValueError(
                "Provide either override_url (URL to the current quarterly REPD Excel) "
                "or local_path (path to a downloaded copy). Check:\n"
                "  https://www.gov.uk/government/publications/"
                "renewable-energy-planning-database-monthly-extract"
            )

        filename = self.override_url.split("/")[-1].split("?")[0] or "repd_latest.xlsx"
        dest = self.cache_dir / filename

        if dest.exists():
            logger.info(f"REPD already cached at {dest} — skipping download")
            return dest

        logger.info(f"Downloading REPD from {self.override_url} …")
        response = requests.get(self.override_url, timeout=120)
        response.raise_for_status()
        dest.write_bytes(response.content)
        logger.info(f"Saved REPD to {dest} ({dest.stat().st_size // 1024} KB)")
        return dest

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    def load_repd_raw(self, path: Optional[Path] = None) -> pd.DataFrame:
        """
        Read the REPD Excel workbook and return a normalised DataFrame.

        Handles the two observed header-row layouts (row 0 or row 4) by
        auto-detecting which row contains the expected column names.

        Returns a DataFrame with at least these canonical columns:
            technology_type, installed_capacity_mw, development_status, operational_date

        Additional columns are retained as-is for diagnostic purposes.
        """
        if path is None:
            path = self.download_repd()

        logger.info(f"Reading REPD workbook: {path}")

        # The workbook has a 'Definition Sheet' cover page and a 'REPD' data sheet.
        # Try the 'REPD' sheet first, then fall back to sheet index 0.
        import openpyxl as _openpyxl
        _wb = _openpyxl.load_workbook(path, read_only=True)
        _sheet_names = _wb.sheetnames
        _wb.close()
        _candidate_sheets = (
            ["REPD"] + [s for s in _sheet_names if s != "REPD"]
            if "REPD" in _sheet_names
            else _sheet_names
        )

        # Try header at row 0 first, then row 4 — the two observed REPD layouts.
        df = None
        for sheet in _candidate_sheets:
            for header_row in (0, 4):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _df = pd.read_excel(path, sheet_name=sheet, header=header_row, engine="openpyxl")
                cols = [str(c).strip() for c in _df.columns]
                found = sum(1 for key in _COL_ALIASES if _find_col(cols, key) is not None)
                if found >= 2:
                    _df.columns = cols
                    df = _df
                    logger.info(f"Using sheet '{sheet}', header row {header_row}")
                    break
            if df is not None:
                break

        if df is None:
            raise ValueError(
                f"Could not detect REPD column layout in {path}. "
                f"Sheets tried: {_candidate_sheets}. "
                f"Last columns seen: {list(_df.columns[:10])}"
            )

        # Rename to canonical names
        rename_map = {}
        for canonical in _COL_ALIASES:
            matched = _find_col(list(df.columns), canonical)
            if matched:
                rename_map[matched] = canonical
        df = df.rename(columns=rename_map)

        # Parse operational_date
        if "operational_date" in df.columns:
            df["operational_date"] = pd.to_datetime(
                df["operational_date"], dayfirst=True, errors="coerce"
            )

        # Parse capacity
        if "installed_capacity_mw" in df.columns:
            df["installed_capacity_mw"] = pd.to_numeric(
                df["installed_capacity_mw"], errors="coerce"
            )

        logger.info(f"Loaded {len(df):,} raw REPD rows")
        return df

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def filter_battery_projects(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to operational battery storage projects only.

        Criteria:
        - technology_type (lowercased, stripped) matches BATTERY_TECH_KEYWORDS
        - development_status contains "operational" (case-insensitive)
        - operational_date is not NaT
        - installed_capacity_mw > 0

        Returns a tidy DataFrame with columns:
            technology_type, installed_capacity_mw, development_status, operational_date
        """
        df = raw.copy()

        required = ["technology_type", "installed_capacity_mw", "development_status", "operational_date"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"REPD missing expected columns after parsing: {missing}")

        # Technology filter
        tech_norm = df["technology_type"].fillna("").str.strip().str.lower()
        is_battery = tech_norm.isin(BATTERY_TECH_KEYWORDS)

        # Log unmatched technology types that include "battery" or "storage" in their
        # name — catches new variant spellings in future REPD releases.
        unmatched_candidates = (
            df.loc[~is_battery, "technology_type"]
            .dropna()
            .unique()
        )
        suspicious = [
            t for t in unmatched_candidates
            if any(kw in str(t).lower() for kw in ("battery", "storage", "bess"))
        ]
        if suspicious:
            logger.warning(
                f"These technology_type strings look battery-related but didn't match "
                f"BATTERY_TECH_KEYWORDS — consider adding them: {suspicious[:10]}"
            )

        # Status filter
        status_norm = df["development_status"].fillna("").str.lower()
        is_operational = status_norm.str.contains("operational", na=False)

        # Capacity and date filters
        has_capacity = df["installed_capacity_mw"].gt(0)
        has_date = df["operational_date"].notna()

        mask = is_battery & is_operational & has_capacity & has_date
        result = df.loc[mask, ["technology_type", "installed_capacity_mw",
                                "development_status", "operational_date"]].copy()

        dropped_no_date = (is_battery & is_operational & has_capacity & ~has_date).sum()
        if dropped_no_date:
            logger.warning(
                f"Dropped {dropped_no_date} battery project(s) with no confirmed "
                "operational_date — they cannot be placed in the cumulative series."
            )

        logger.info(
            f"Filtered to {len(result):,} operational battery projects "
            f"({result['installed_capacity_mw'].sum():.0f} MW total)"
        )
        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Build monthly series
    # ------------------------------------------------------------------

    def build_monthly_capacity_series(
        self,
        projects: pd.DataFrame,
        start_date: str = "2019-01-01",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Build a monthly cumulative GB operational BESS capacity series.

        For each calendar month in [start_date, end_date], sums the
        installed_capacity_mw of all projects whose operational_date falls on
        or before the last day of that month.

        Gaps between quarterly REPD updates are forward-filled from the prior
        month (the fleet only grows). Months before the first recorded project
        are back-filled with 0.

        Months after the last REPD entry (typically the trailing ~3 months) are
        linearly extrapolated using a 12-month OLS trend. This is a conservative
        straight-line projection — GB fleet growth has been roughly linear in
        MW/month terms. Extrapolated months are flagged with is_extrapolated=True;
        that column is stripped before the series is returned.

        Returns
        -------
        DataFrame with columns:
            month (datetime64[ns], first day of each month), bess_fleet_mw (float)
        Sorted ascending. No NaNs.
        """
        if end_date is None:
            end_date = pd.Timestamp.today().to_period("M").to_timestamp()
        else:
            end_date = pd.to_datetime(end_date)

        # Full monthly index
        months = pd.period_range(start=start_date, end=end_date, freq="M")
        series = pd.DataFrame({"month": months})
        series["month_end"] = series["month"].dt.to_timestamp("M")

        # Compute cumulative capacity at end of each month
        def _cumulative_at(month_end: pd.Timestamp) -> float:
            return projects.loc[
                projects["operational_date"] <= month_end,
                "installed_capacity_mw",
            ].sum()

        series["bess_fleet_mw"] = series["month_end"].apply(_cumulative_at)

        # Identify the last month covered by actual REPD data
        last_repd_date = projects["operational_date"].max()
        last_repd_month = last_repd_date.to_period("M")

        is_extrapolated = series["month"] > last_repd_month
        n_extrap = is_extrapolated.sum()

        if n_extrap > 0:
            # Fit a linear trend on the last 12 known months
            known = series.loc[~is_extrapolated].tail(12).copy()
            x = np.arange(len(known))
            y = known["bess_fleet_mw"].values
            slope, intercept = np.polyfit(x, y, 1)

            extrap_indices = np.arange(len(known), len(known) + n_extrap)
            extrap_values = np.maximum(slope * extrap_indices + intercept, 0.0)
            series.loc[is_extrapolated, "bess_fleet_mw"] = extrap_values
            logger.info(
                f"Extrapolated {n_extrap} month(s) beyond last REPD entry "
                f"({last_repd_month}) using 12-month linear trend "
                f"(slope: +{slope:.0f} MW/month)"
            )

        # Convert Period month column to first-of-month timestamp for parquet compatibility
        series["month"] = series["month"].dt.to_timestamp()
        result = series[["month", "bess_fleet_mw"]].sort_values("month").reset_index(drop=True)

        logger.info(
            f"Monthly capacity series: {len(result)} months "
            f"({result['month'].iloc[0].date()} → {result['month'].iloc[-1].date()}), "
            f"final value: {result['bess_fleet_mw'].iloc[-1]:.0f} MW"
        )
        return result

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def collect(
        self,
        start_date: str = "2019-01-01",
        end_date: Optional[str] = None,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Full pipeline: download → parse → filter → build monthly series.

        Parameters
        ----------
        start_date : str
            ISO date — first month to include in the output series.
        end_date : str, optional
            ISO date — last month to include. Defaults to current month.
        save : bool
            If True, saves to data/raw/bess_fleet_capacity_raw.csv.

        Returns
        -------
        DataFrame with columns: month (datetime64), bess_fleet_mw (float).
        """
        path = self.download_repd()
        raw = self.load_repd_raw(path)
        projects = self.filter_battery_projects(raw)
        monthly = self.build_monthly_capacity_series(projects, start_date=start_date, end_date=end_date)

        if save:
            out = self.cache_dir / "bess_fleet_capacity_raw.csv"
            monthly.to_csv(out, index=False)
            logger.info(f"Saved monthly BESS capacity series to {out}")

        return monthly


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python repd_collector.py <repd_url_or_local_path>\n\n"
            "  Provide the URL or local path to the latest REPD quarterly Excel.\n"
            "  Find the current URL at:\n"
            "    https://www.gov.uk/government/publications/"
            "renewable-energy-planning-database-monthly-extract\n"
        )
        sys.exit(1)

    arg = sys.argv[1]
    if arg.startswith("http"):
        collector = REPDCollector(override_url=arg)
    else:
        collector = REPDCollector(local_path=arg)

    monthly = collector.collect(save=True)

    print(f"\nMonthly BESS fleet capacity ({len(monthly)} months):")
    print(monthly.to_string(index=False))

    # Cross-check against known industry milestones
    milestones = {2019: 800, 2021: 1400, 2023: 3500}
    print("\nMilestone cross-check (end-of-year values):")
    for year, expected_mw in milestones.items():
        row = monthly[monthly["month"].dt.year == year]
        if not row.empty:
            actual = row.iloc[-1]["bess_fleet_mw"]
            delta_pct = (actual - expected_mw) / expected_mw * 100
            flag = "  ⚠ >20% off" if abs(delta_pct) > 20 else ""
            print(f"  {year}: {actual:,.0f} MW  (expected ~{expected_mw:,} MW,  {delta_pct:+.0f}%){flag}")
