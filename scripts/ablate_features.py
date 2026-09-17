#!/usr/bin/env python3
"""
Feature ablation on walk-forward folds
======================================
Measures what a feature group is worth by building the matrix with and without
it and rerunning the same walk-forward harness.

Scored on spread calibration first. That ordering is deliberate and is the
lesson of the 2026-09-17 model benchmark: RMSE, Spearman and spike-RMSE all
ranked LEAR best while it earned £19k/MW/yr less than reusing yesterday's
prices, because none of them charge a forecast for inventing spread that never
arrives. A feature group that improves average error while worsening spread bias
would lose money, and only one of these columns would show it.

The variants are separate prediction tables under data/processed/benchmarks/, not
extensions of the shipped one: mixing feature sets inside a single series is the
failure the shipped cache now fingerprints against.

Usage:
    python scripts/ablate_features.py                       # baseline vs + wind
    python scripts/ablate_features.py --revenue             # and a dispatch run each
    python scripts/ablate_features.py --models rf,lgb       # more than one model
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from scripts.benchmark_walk_forward import baselines, revenue_for, split_metrics
from scripts.build_forecast_walk_forward import backtest_window
from src.analysis.price_forecast import (
    WALK_FORWARD_CADENCE_MONTHS,
    build_feature_matrix,
    load_bess_capacity,
    resolve_feature_cols,
    walk_forward_predictions,
)

PROCESSED = Path(__file__).parent.parent / "data" / "processed"
BENCH = PROCESSED / "benchmarks"
REPORTS = Path(__file__).parent.parent / "reports"
SELECT_BEFORE = "2025-01-01"


def variants(market_index, generation, capacity, wind) -> dict:
    """The feature matrices to compare, keyed by the name used in the report."""
    return {
        "baseline": lambda: build_feature_matrix(market_index, generation, capacity),
        "wind": lambda: build_feature_matrix(market_index, generation, capacity, wind_forecast=wind),
    }


def predictions_for(name: str, matrix, window, model_type: str, cadence: int, verbose=True):
    """Walk-forward predictions for one variant, cached per variant and model."""
    BENCH.mkdir(parents=True, exist_ok=True)
    table = BENCH / f"ablation_{name}_{model_type}_{cadence}m.parquet"
    folds_file = table.with_suffix(".folds.json")
    if table.exists() and folds_file.exists():
        if verbose:
            print(f"    reused {table.name}", flush=True)
        return pd.read_parquet(table), json.loads(folds_file.read_text())

    predictions, folds = walk_forward_predictions(
        matrix, window[0], window[1], model_type=model_type, cadence_months=cadence,
        on_fold=(lambda f: print(f"    {f['origin']}: RMSE {f['metrics']['rmse']}, "
                                 f"ρ {f['metrics']['spearman']}", flush=True)) if verbose else None,
    )
    predictions.to_parquet(table, index=False)
    folds_file.write_text(json.dumps(folds, indent=2) + "\n")
    return predictions, folds


def write_report(rows: list, select_before: str) -> None:
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "feature_ablation.json").write_text(json.dumps(rows, indent=2) + "\n")

    def cell(row, key, half="confirmation"):
        block = row["metrics"].get(half) or {}
        return block.get(key, "—")

    lines = [
        "# Feature ablation on walk-forward folds",
        "",
        "Spread bias and MAE first: they are the statistics that decide whether a forecast",
        "helps a battery, and the only ones that charge it for spread it invents. Figures are",
        f"the confirmation folds (from {select_before}); the selection folds drove nothing here",
        "but are in the JSON.",
        "",
        "| Model | Features | Spread bias | Spread MAE | RMSE | Spearman | Spike RMSE |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['variant']} | {cell(row, 'spread_bias')} | {cell(row, 'spread_mae')} "
            f"| {cell(row, 'rmse')} | {cell(row, 'spearman')} | {cell(row, 'spike_rmse')} |"
        )
    if any("revenue" in row for row in rows):
        lines += ["", "| Model | Features | £k/MW/yr | Foresight ratio |", "|---|---|---|---|"]
        for row in rows:
            rev = row.get("revenue")
            if rev:
                lines.append(f"| {row['model']} | {row['variant']} | "
                             f"{rev['annualised_per_mw'] / 1e3:.1f} | {rev['foresight_ratio'] * 100:.1f}% |")
    (REPORTS / "feature_ablation.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="rf", help="comma-separated model types")
    parser.add_argument("--cadence", type=int, default=WALK_FORWARD_CADENCE_MONTHS)
    parser.add_argument("--select-before", default=SELECT_BEFORE)
    parser.add_argument("--revenue", action="store_true",
                        help="also run a dispatch backtest per variant (~9 min each)")
    args = parser.parse_args()

    auctions = pd.read_parquet(PROCESSED / "auctions.parquet")
    market_index = pd.read_parquet(PROCESSED / "market_index.parquet")
    generation = pd.read_parquet(PROCESSED / "generation_daily.parquet")
    delivery = pd.read_parquet(PROCESSED / "response_delivery.parquet")
    wind = pd.read_parquet(PROCESSED / "wind_forecast.parquet")
    capacity = load_bess_capacity()
    window = backtest_window(auctions, market_index)

    pf_base = naive_base = None
    if args.revenue:
        pf_base, naive_base = baselines(auctions, market_index, delivery, window, True)
        print(f"Baselines: perfect foresight £{pf_base / 1e3:.1f}k, naive £{naive_base / 1e3:.1f}k per MW/yr")

    rows = []
    built = variants(market_index, generation, capacity, wind)
    for model_type in [m.strip() for m in args.models.split(",") if m.strip()]:
        for name, build in built.items():
            print(f"\n── {model_type} / {name} " + "─" * 30, flush=True)
            matrix = build()
            cols = resolve_feature_cols(matrix)
            print(f"    {len(cols)} feature columns, {len(matrix):,} rows", flush=True)
            started = time.time()
            predictions, folds = predictions_for(name, matrix, window, model_type, args.cadence)
            row = {
                "model": model_type,
                "variant": name,
                "feature_cols": cols,
                "fit_minutes": round((time.time() - started) / 60, 2),
                "metrics": split_metrics(predictions, market_index, args.select_before),
            }
            confirmation = row["metrics"]["confirmation"]
            print(f"    spread bias {confirmation['spread_bias']}, MAE {confirmation['spread_mae']}, "
                  f"RMSE {confirmation['rmse']}, ρ {confirmation['spearman']}", flush=True)

            if args.revenue:
                print("    dispatch backtest…", flush=True)
                row["revenue"] = revenue_for(predictions, auctions, market_index, delivery,
                                             window, pf_base, naive_base)
                print(f"    £{row['revenue']['annualised_per_mw'] / 1e3:.1f}k/MW/yr, "
                      f"foresight {row['revenue']['foresight_ratio'] * 100:.1f}%", flush=True)
            rows.append(row)
            write_report(rows, args.select_before)

    print("\nWrote reports/feature_ablation.md")
    for row in rows:
        conf = row["metrics"]["confirmation"]
        print(f"  {row['model']:<4} {row['variant']:<9} spread bias {conf['spread_bias']:>8}  "
              f"RMSE {conf['rmse']:>6}"
              + (f"  £{row['revenue']['annualised_per_mw'] / 1e3:.1f}k" if "revenue" in row else ""))


if __name__ == "__main__":
    main()
