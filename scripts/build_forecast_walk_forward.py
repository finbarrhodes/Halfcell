#!/usr/bin/env python3
"""
Build the walk-forward forecast table
=====================================
Out-of-sample price forecasts for every day of the backtest, produced by refitting
the model at quarterly origins and predicting only the days after each one.

Why this exists: the revenue backtest starts in September 2021, while a single
fixed train/test split trains on everything before March 2025. Predictions for the
42 months before that split would then come from a model fitted on those very days
— in-sample forecasts that flatter both the forecast metrics and the revenue they
drive. Measured on the cache this was worth £18.8k/MW/yr of apparent forecast value
in-sample against £2.3k out-of-sample (2026-09-17).

The table is cached at data/processed/forecast_walk_forward.parquet and extended in
place: each origin's training data is historical, so an origin already computed does
not change when new months arrive, and a refresh fits only the newest origin.

Usage:
    python scripts/build_forecast_walk_forward.py            # extend the cached table
    python scripts/build_forecast_walk_forward.py --rebuild  # refit every origin
    python scripts/build_forecast_walk_forward.py --train-years 3   # rolling window
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.analysis.price_forecast import (
    WALK_FORWARD_CADENCE_MONTHS,
    build_feature_matrix,
    load_bess_capacity,
    walk_forward_predictions,
)

PROCESSED = Path(__file__).parent.parent / "data" / "processed"
TABLE = PROCESSED / "forecast_walk_forward.parquet"
FOLDS = PROCESSED / "forecast_walk_forward_folds.json"


def backtest_window(auctions: pd.DataFrame, market_index: pd.DataFrame) -> tuple:
    """The overlapping range the backtest runs over — the same bounds precompute uses."""
    start = max(auctions["EFA Date"].min(), market_index["settlementDate"].min())
    end = min(auctions["EFA Date"].max(), market_index["settlementDate"].max())
    return pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()


def load_or_build(
    *,
    model_type: str = "rf",
    cadence_months: int = WALK_FORWARD_CADENCE_MONTHS,
    train_years: float | None = None,
    rebuild: bool = False,
    verbose: bool = True,
) -> tuple:
    """
    The cached walk-forward table, extended with any origins it is missing.

    Returns (predictions, folds). Writes both back to data/processed/ when anything
    was fitted.
    """
    auctions = pd.read_parquet(PROCESSED / "auctions.parquet")
    market_index = pd.read_parquet(PROCESSED / "market_index.parquet")
    generation = pd.read_parquet(PROCESSED / "generation_daily.parquet")
    capacity = load_bess_capacity(PROCESSED / "bess_fleet_capacity.parquet")
    features = build_feature_matrix(market_index, generation, capacity)
    start, end = backtest_window(auctions, market_index)

    cached = pd.DataFrame()
    folds: list = []
    if TABLE.exists() and not rebuild:
        cached = pd.read_parquet(TABLE)
        if FOLDS.exists():
            folds = json.loads(FOLDS.read_text())
    done = sorted(pd.to_datetime(cached["origin"]).unique()) if not cached.empty else []
    if verbose:
        print(f"Window {start.date()} → {end.date()}; {len(done)} origin(s) already cached")

    started = time.time()
    fresh, new_folds = walk_forward_predictions(
        features, start, end,
        model_type=model_type, cadence_months=cadence_months, train_years=train_years,
        skip_origins=done,
        on_fold=(lambda f: print(f"  {f['origin']} → {f['predicts_until']}: "
                                 f"{f['train_rows']:,} training rows, "
                                 f"RMSE {f['metrics']['rmse']}, ρ {f['metrics']['spearman']}",
                                 flush=True)) if verbose else None,
    )

    if fresh.empty:
        if verbose:
            print("Nothing to fit; the cached table already covers the window.")
        return cached, folds

    predictions = (
        pd.concat([cached, fresh], ignore_index=True)
        .drop_duplicates(["settlementDate", "settlementPeriod"], keep="last")
        .sort_values(["settlementDate", "settlementPeriod"], ignore_index=True)
    )
    folds = sorted(folds + new_folds, key=lambda f: f["origin"])
    predictions.to_parquet(TABLE, index=False)
    FOLDS.write_text(json.dumps(folds, indent=2) + "\n")
    if verbose:
        mins = (time.time() - started) / 60
        print(f"Fitted {len(new_folds)} origin(s) in {mins:.1f} min → {TABLE.name} "
              f"({len(predictions):,} rows, {len(folds)} folds)")
    return predictions, folds


def pooled_metrics(predictions: pd.DataFrame, market_index: pd.DataFrame) -> dict:
    """Metrics over every walk-forward prediction, as one out-of-sample series."""
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np

    actual = (market_index[market_index["dataProvider"] == "APXMIDP"]
              .assign(settlementDate=lambda d: d["settlementDate"].dt.normalize())
              [["settlementDate", "settlementPeriod", "price"]])
    joined = predictions.merge(actual, on=["settlementDate", "settlementPeriod"], how="inner").dropna()
    y, p = joined["price"].to_numpy(), joined["prediction"].to_numpy()
    spike = y >= np.percentile(y, 90)
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y, p))), 2),
        "mae": round(float(mean_absolute_error(y, p)), 2),
        "spearman": round(float(spearmanr(y, p).statistic), 3),
        "spike_rmse": round(float(np.sqrt(mean_squared_error(y[spike], p[spike]))), 2),
        "n_samples": int(len(joined)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="rf", help="model type to refit at each origin")
    parser.add_argument("--cadence", type=int, default=WALK_FORWARD_CADENCE_MONTHS,
                        help="months between retrainings")
    parser.add_argument("--train-years", type=float, default=None,
                        help="cap the training window (rolling); omit for an expanding window")
    parser.add_argument("--rebuild", action="store_true", help="refit every origin")
    args = parser.parse_args()

    predictions, folds = load_or_build(
        model_type=args.model, cadence_months=args.cadence,
        train_years=args.train_years, rebuild=args.rebuild,
    )
    market_index = pd.read_parquet(PROCESSED / "market_index.parquet")
    metrics = pooled_metrics(predictions, market_index)
    print(f"\nWalk-forward, pooled over {len(folds)} folds: RMSE {metrics['rmse']} £/MWh, "
          f"MAE {metrics['mae']}, Spearman ρ {metrics['spearman']}, "
          f"spike RMSE {metrics['spike_rmse']}, n {metrics['n_samples']:,}")


if __name__ == "__main__":
    main()
