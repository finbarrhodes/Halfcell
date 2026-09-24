#!/usr/bin/env python3
"""
Model benchmark on walk-forward folds
=====================================
Compares price-forecasting models on the footing the cache now uses: refit at
quarterly origins, each predicting only the days that follow.

Why not the old fixed-split benchmarks: they compared models on one 18-month
window (benchmark_models.py) or on revenue over the full backtest using a model
trained on part of it (benchmark_backtest.py). The second is the leakage removed
from the cache on 2026-09-17, and it was not neutral between models — in-sample
advantage scales with how hard a model fits its own training data, so a 300-tree
Random Forest gained far more from being asked about days it trained on than a
regularised linear model like LEAR. The shipped choice of RF rested on that.

Selection protocol: folds are split at --select-before. Anything at or after that
date is the confirmation set and should not drive the choice of model. Reporting
both halves separately is the point — picking a model on all 20 folds and then
quoting those same folds is a milder version of the mistake this script exists to
avoid.

Usage:
    python scripts/benchmark_walk_forward.py                      # forecast accuracy
    python scripts/benchmark_walk_forward.py --revenue            # and a dispatch run each
    python scripts/benchmark_walk_forward.py --models rf,lear     # a subset
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from scripts.build_forecast_walk_forward import backtest_window, load_or_build, pooled_metrics
from src.analysis.price_forecast import (
    WALK_FORWARD_CADENCE_MONTHS,
    build_feature_matrix,
    load_bess_capacity,
    run_forecast_backtest,
    walk_forward_predictions,
)
from src.analysis.revenue_stack import ALL_SERVICES, REFERENCE_BATTERY

PROCESSED = Path(__file__).parent.parent / "data" / "processed"
CACHE = Path(__file__).parent.parent / "data" / "cache"
REPORTS = Path(__file__).parent.parent / "reports"
DEFAULT_MODELS = "rf,lgb,xgb,lear"
SELECT_BEFORE = "2025-01-01"


BENCH = PROCESSED / "benchmarks"


def predictions_for(model_type: str, features, window, cadence: int, train_years, verbose=True):
    """Walk-forward predictions for one model, cached per configuration."""
    if model_type == "rf" and cadence == WALK_FORWARD_CADENCE_MONTHS and train_years is None:
        if verbose:
            print("  reused the cached shipped table", flush=True)
        return load_or_build(model_type="rf", verbose=False)

    BENCH.mkdir(parents=True, exist_ok=True)
    suffix = f"_{train_years:g}y" if train_years else ""
    table = BENCH / f"forecast_wf_{model_type}_{cadence}m{suffix}.parquet"
    folds_file = table.with_suffix(".folds.json")
    if table.exists() and folds_file.exists():
        if verbose:
            print(f"  reused {table.name}", flush=True)
        return pd.read_parquet(table), json.loads(folds_file.read_text())

    predictions, folds = walk_forward_predictions(
        features, window[0], window[1], model_type=model_type,
        cadence_months=cadence, train_years=train_years,
        on_fold=(lambda f: print(f"  {f['origin']}: RMSE {f['metrics']['rmse']}, "
                                 f"ρ {f['metrics']['spearman']}", flush=True)) if verbose else None,
    )
    predictions.to_parquet(table, index=False)
    folds_file.write_text(json.dumps(folds, indent=2) + "\n")
    return predictions, folds


def split_metrics(predictions: pd.DataFrame, market_index: pd.DataFrame, select_before: str) -> dict:
    """Pooled metrics for the whole run, and for the selection and confirmation halves."""
    origin = pd.to_datetime(predictions["origin"])
    cut = pd.Timestamp(select_before)
    out = {"all": pooled_metrics(predictions, market_index)}
    for name, mask in (("selection", origin < cut), ("confirmation", origin >= cut)):
        subset = predictions[mask]
        out[name] = pooled_metrics(subset, market_index) if not subset.empty else None
    return out


def baselines(auctions, market_index, delivery, window, full_window: bool) -> tuple:
    """Perfect-foresight and naive revenue for this window: cached if it is the full one."""
    if full_window:
        manifest = json.loads((CACHE / "manifest.json").read_text())
        return (manifest["pf_mpc"]["summary"]["annualised_per_mw"],
                manifest["naive_mpc"]["summary"]["annualised_per_mw"])

    from src.analysis.revenue_stack import run_backtest
    print("  baselines for this window…", flush=True)
    pf = run_backtest(auctions, market_index, REFERENCE_BATTERY, ALL_SERVICES,
                      window[0], window[1], delivery=delivery)
    naive = run_forecast_backtest(
        strategy="naive", market_index=market_index, auctions=auctions, battery=REFERENCE_BATTERY,
        services=ALL_SERVICES, start_date=window[0], end_date=window[1], delivery=delivery)
    return (pf["summary"]["annualised_per_mw"], naive["summary"]["annualised_per_mw"])


def revenue_for(predictions, auctions, market_index, delivery, window, pf, naive) -> dict:
    """Annualised revenue and foresight ratio for one model over this window."""
    result = run_forecast_backtest(
        strategy="ml", market_index=market_index, auctions=auctions, battery=REFERENCE_BATTERY,
        services=ALL_SERVICES, start_date=window[0], end_date=window[1], predictions=predictions,
        delivery=delivery,
    )
    per_mw = result["summary"]["annualised_per_mw"]
    return {
        "annualised_per_mw": round(per_mw, 1),
        "foresight_ratio": round((per_mw - naive) / (pf - naive), 4),
        "breach_periods": result["summary"]["soe_breach_periods"],
    }


def write_report(rows: list, select_before: str) -> None:
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "walk_forward_benchmark.json").write_text(json.dumps(rows, indent=2) + "\n")

    def cell(row, half, key):
        block = row["metrics"].get(half) or {}
        return f"{block.get(key, '—')}"

    lines = [
        "# Model benchmark on walk-forward folds",
        "",
        f"Refit every {WALK_FORWARD_CADENCE_MONTHS} months, each origin predicting only the days "
        f"that follow. Folds before {select_before} are the selection set; the rest are held back "
        "as confirmation.",
        "",
        "Spread bias is reported beside the accuracy metrics because it is the only one that "
        "charges a forecast for spread it invents: LEAR won every other column here and earned "
        "£19k/MW/yr less than reusing yesterday's prices.",
        "",
        "| Model | RMSE (sel) | RMSE (conf) | Spearman (sel) | Spearman (conf) | Spike RMSE (conf) "
        "| Spread bias (conf) | Spread MAE (conf) | Fit time |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {cell(row, 'selection', 'rmse')} | {cell(row, 'confirmation', 'rmse')} "
            f"| {cell(row, 'selection', 'spearman')} | {cell(row, 'confirmation', 'spearman')} "
            f"| {cell(row, 'confirmation', 'spike_rmse')} | {cell(row, 'confirmation', 'spread_bias')} "
            f"| {cell(row, 'confirmation', 'spread_mae')} | {row['fit_minutes']:.1f} min |"
        )
    if any("revenue" in row for row in rows):
        lines += ["", "| Model | £k/MW/yr | Foresight ratio | Unavailable periods |", "|---|---|---|---|"]
        for row in rows:
            rev = row.get("revenue")
            if rev:
                lines.append(f"| {row['model']} | {rev['annualised_per_mw'] / 1e3:.1f} "
                             f"| {rev['foresight_ratio'] * 100:.1f}% | {rev['breach_periods']} |")
    (REPORTS / "walk_forward_benchmark.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=DEFAULT_MODELS, help="comma-separated model types")
    parser.add_argument("--cadence", type=int, default=WALK_FORWARD_CADENCE_MONTHS)
    parser.add_argument("--train-years", type=float, default=None)
    parser.add_argument("--select-before", default=SELECT_BEFORE,
                        help="folds from this date are held back as confirmation")
    parser.add_argument("--revenue", action="store_true",
                        help="also run a dispatch backtest per model (~9 min each)")
    parser.add_argument("--revenue-from", default=None,
                        help="restrict the dispatch backtest to this start date; baselines are "
                             "recomputed for the same window rather than read from the cache")
    args = parser.parse_args()

    auctions = pd.read_parquet(PROCESSED / "auctions.parquet")
    market_index = pd.read_parquet(PROCESSED / "market_index.parquet")
    generation = pd.read_parquet(PROCESSED / "generation_daily.parquet")
    delivery = pd.read_parquet(PROCESSED / "response_delivery.parquet")
    features = build_feature_matrix(market_index, generation, load_bess_capacity())
    window = backtest_window(auctions, market_index)
    default_setup = args.cadence == WALK_FORWARD_CADENCE_MONTHS and args.train_years is None

    revenue_window = window if args.revenue_from is None else (pd.Timestamp(args.revenue_from), window[1])
    full_window = args.revenue_from is None
    print(f"Window {window[0].date()} → {window[1].date()}, refit every {args.cadence} months")
    print(f"Selection folds before {args.select_before}, confirmation from it")
    pf_base = naive_base = None
    if args.revenue:
        print(f"Revenue over {pd.Timestamp(revenue_window[0]).date()} → {window[1].date()}")
        pf_base, naive_base = baselines(auctions, market_index, delivery, revenue_window, full_window)
        print(f"  perfect foresight £{pf_base / 1e3:.1f}k, naive £{naive_base / 1e3:.1f}k per MW/yr")
    print("", flush=True)

    rows = []
    for model_type in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"── {model_type} " + "─" * 40, flush=True)
        started = time.time()
        predictions, folds = predictions_for(
            model_type, features, window, args.cadence, args.train_years
        )
        fit_minutes = (time.time() - started) / 60
        row = {
            "model": model_type,
            "fit_minutes": round(fit_minutes, 2),
            "folds": len(folds),
            "metrics": split_metrics(predictions, market_index, args.select_before),
        }
        confirmation = row["metrics"]["confirmation"]
        print(f"  {model_type}: confirmation RMSE {confirmation['rmse']}, "
              f"ρ {confirmation['spearman']} ({fit_minutes:.1f} min)", flush=True)

        if args.revenue:
            print("  dispatch backtest…", flush=True)
            row["revenue"] = revenue_for(predictions, auctions, market_index, delivery,
                                         revenue_window, pf_base, naive_base)
            row["revenue"]["window_from"] = str(pd.Timestamp(revenue_window[0]).date())
            print(f"  £{row['revenue']['annualised_per_mw'] / 1e3:.1f}k/MW/yr, "
                  f"foresight {row['revenue']['foresight_ratio'] * 100:.1f}%", flush=True)

        rows.append(row)
        write_report(rows, args.select_before)   # written as we go, so a long run is never wasted

    print(f"\nWrote reports/walk_forward_benchmark.md and .json")
    for row in rows:
        conf = row["metrics"]["confirmation"]
        print(f"  {row['model']:<5} confirmation RMSE {conf['rmse']:>6}  ρ {conf['spearman']:>6}"
              + (f"  £{row['revenue']['annualised_per_mw'] / 1e3:.1f}k" if "revenue" in row else ""))


if __name__ == "__main__":
    main()
