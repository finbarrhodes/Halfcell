"""
scripts/check_repd_freshness.py
===============================
Warn when the REPD projected tail has outgrown a quarter.

REPD is a quarterly Excel drop with no API, so the monthly data refresh cannot
update it: `bess_fleet_capacity.parquet` keeps its measured months and its tail
stays projected, widening by one month with every refresh until a new extract is
downloaded into data/raw by hand.

The projected tail is normally around five months, not one quarter. Two lags stack:
REPD is published a quarter behind, and within any published extract a project's
operational date only appears once confirmed, which trails further still. The July
2026 extract's latest confirmed operational battery project was 2026-03-23 — five
months of projection with the extract fully up to date.

So a long tail is usually upstream's business, not a missed download. The tolerance
is set to flag only a tail longer than that steady state, which is worth a look but
is still more often a slower quarter than an actionable error.

Warns rather than fails by default: a stale planning database is not a reason to
block a market data refresh. Pass --strict to exit non-zero instead.

Run from anywhere:
    python scripts/check_repd_freshness.py
    python scripts/check_repd_freshness.py --strict --max-months 3
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
DEFAULT_PARQUET = ROOT / "data" / "processed" / "bess_fleet_capacity.parquet"

# Publication lag plus operational-confirmation lag ran to 5 months in 2026 with a
# fully current extract, so 3 would warn permanently. One month of headroom past the
# observed steady state.
DEFAULT_MAX_MONTHS = 6


def trailing_extrapolated_months(df: pd.DataFrame) -> int:
    """
    Length of the unbroken run of projected months at the end of the series.

    Counted from the tail rather than as a total, because only the trailing run
    reflects a missed REPD drop. A flagged month in the middle of the series would
    mean something else entirely and should not inflate this number.

    Returns 0 when the series has no is_extrapolated column, which is how
    prepare_data.py handles an older extract that predates the flag.
    """
    if "is_extrapolated" not in df.columns or df.empty:
        return 0
    flags = df.sort_values("month")["is_extrapolated"].astype(bool).tolist()
    count = 0
    for flag in reversed(flags):
        if not flag:
            break
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Warn when the REPD projected tail exceeds a quarter."
    )
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument(
        "--max-months", type=int, default=DEFAULT_MAX_MONTHS,
        help=f"Projected months tolerated before warning (default: {DEFAULT_MAX_MONTHS})",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit 1 instead of warning, for use as a hard gate.",
    )
    args = parser.parse_args()

    if not args.parquet.exists():
        print(f"SKIP: {args.parquet} not found — nothing to check.")
        return

    df = pd.read_parquet(args.parquet)
    projected = trailing_extrapolated_months(df)
    measured_to = df.loc[~df["is_extrapolated"].astype(bool), "month"].max() \
        if "is_extrapolated" in df.columns else df["month"].max()

    print(
        f"REPD fleet series: measured through {pd.Timestamp(measured_to).date()}, "
        f"{projected} projected month(s) after it."
    )

    if projected <= args.max_months:
        print(f"OK: within the {args.max_months}-month tolerance.")
        return

    msg = (
        f"REPD projected tail is {projected} months, over the {args.max_months}-month "
        f"tolerance. Most likely upstream simply has not confirmed newer operational "
        f"dates, in which case there is nothing to do. Worth ruling out a missed "
        f"download: compare data/raw/REPD_Publication_*.xlsx against the current file on "
        f"the DESNZ page by SIZE, not name — publications are named by data quarter, so a "
        f"newer release can reuse a filename you already have."
    )
    # GitHub renders these in the job summary; harmless noise elsewhere.
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::warning title=REPD extract is stale::{msg}")
    print(f"WARNING: {msg}", file=sys.stderr)

    if args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
