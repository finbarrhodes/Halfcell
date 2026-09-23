#!/usr/bin/env python3
"""
Build the walk-forward quantile table
=====================================
Price quantiles for every day of the backtest from a quantile regression forest
(src/analysis/quantile_forecast.py), refitted at the same quarterly origins as the
point forecast and from the same bid-time information: data to D-2, the early
feature matrix the offers see.

The table is an experiment's input rather than the site's, so it lives under
data/processed/benchmarks/ with the other cached runs and is not committed. Like
the point tables it is extended in place - an origin already fitted does not change
when new months arrive - and fingerprinted, so a changed feature set or forest
cannot extend a table built from another.

Usage:
    python scripts/build_quantile_walk_forward.py            # extend the cached table
    python scripts/build_quantile_walk_forward.py --rebuild  # refit every origin
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.analysis.price_forecast import (
    WALK_FORWARD_CADENCE_MONTHS,
    build_feature_matrix,
    load_bess_capacity,
    resolve_feature_cols,
)
from src.analysis.quantile_forecast import FOREST_PARAMS, QUANTILES, walk_forward_quantiles

PROCESSED = ROOT / "data" / "processed"
BENCH = PROCESSED / "benchmarks"
TABLE = BENCH / "forecast_quantiles_early.parquet"
FOLDS = BENCH / "forecast_quantiles_early_folds.json"
FINGERPRINT = BENCH / "forecast_quantiles_early_features.json"
INFORMATION_LAG_DAYS = 2      # bid time: the offers' information set


def load_or_build(*, rebuild: bool = False, verbose: bool = True) -> tuple:
    """The cached quantile table, extended with any origins it is missing. Returns (table, folds)."""
    from scripts.build_forecast_walk_forward import backtest_window

    auctions = pd.read_parquet(PROCESSED / "auctions.parquet")
    market_index = pd.read_parquet(PROCESSED / "market_index.parquet")
    generation = pd.read_parquet(PROCESSED / "generation_daily.parquet")
    capacity = load_bess_capacity(PROCESSED / "bess_fleet_capacity.parquet")
    features = build_feature_matrix(market_index, generation, capacity,
                                    information_lag_days=INFORMATION_LAG_DAYS)
    start, end = backtest_window(auctions, market_index)
    fingerprint = {
        "feature_cols": resolve_feature_cols(features),
        "quantiles": list(QUANTILES),
        "forest": {k: v for k, v in FOREST_PARAMS.items() if k != "n_jobs"},
        "cadence_months": WALK_FORWARD_CADENCE_MONTHS,
        "information_lag_days": INFORMATION_LAG_DAYS,
    }

    cached, folds = pd.DataFrame(), []
    if TABLE.exists() and not rebuild:
        previous = json.loads(FINGERPRINT.read_text()) if FINGERPRINT.exists() else None
        if previous != fingerprint:
            raise RuntimeError(f"{TABLE.name} was built with different settings; re-run with --rebuild")
        cached = pd.read_parquet(TABLE)
        folds = json.loads(FOLDS.read_text()) if FOLDS.exists() else []
    done = sorted(pd.to_datetime(cached["origin"]).unique()) if not cached.empty else []
    if verbose:
        print(f"Window {start.date()} → {end.date()}; {len(done)} origin(s) already cached", flush=True)

    started = time.time()
    fresh, new_folds = walk_forward_quantiles(
        features, start, end, skip_origins=done,
        on_fold=(lambda f: print(f"  {f['origin']} → {f['predicts_until']}: {f['train_rows']:,} rows, "
                                 f"share below q10/q50/q90 {f['share_below']['q10']}/"
                                 f"{f['share_below']['q50']}/{f['share_below']['q90']}",
                                 flush=True)) if verbose else None,
    )
    if fresh.empty:
        return cached, folds

    table = (pd.concat([cached, fresh], ignore_index=True)
             .drop_duplicates(["settlementDate", "settlementPeriod"], keep="last")
             .sort_values(["settlementDate", "settlementPeriod"], ignore_index=True))
    folds = sorted(folds + new_folds, key=lambda f: f["origin"])
    BENCH.mkdir(parents=True, exist_ok=True)
    table.to_parquet(TABLE, index=False)
    FOLDS.write_text(json.dumps(folds, indent=2) + "\n")
    FINGERPRINT.write_text(json.dumps(fingerprint, indent=2) + "\n")
    if verbose:
        print(f"Fitted {len(new_folds)} origin(s) in {(time.time() - started) / 60:.1f} min → "
              f"{TABLE.relative_to(ROOT)} ({len(table):,} rows)", flush=True)
    return table, folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="refit every origin")
    args = parser.parse_args()
    load_or_build(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
