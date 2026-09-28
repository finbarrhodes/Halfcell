#!/usr/bin/env python3
"""
Error bars on the forecast's value
==================================
Halfcell reports that the model earns about £3k/MW/yr more than reusing an older
day's prices. That is a small number next to a five-year backtest's noise, and
this script is the check on whether it is distinguishable from it at all.

Two questions, two tools (src/analysis/significance.py):

  Is the forecast more accurate?  Diebold-Mariano on paired daily losses, with a
    Newey-West long-run variance so serial correlation is carried rather than
    assumed away. Run on squared error, which is what the metric tables report,
    and on the day's spread error, which is what the offer stage consumes.

  Is it worth more money?  A moving block bootstrap of the daily revenue
    difference. Revenue cannot be resampled day by day - a dispatch decision
    carries state into the next day - so contiguous blocks are drawn instead,
    four weeks by default.

Both forecasts are put on the same footing: the same days, the engine as shipped
(offers priced by the day-ahead plan on bid-time information, dispatch on
forecast vintages, a constant shrink), and the accuracy tests use the vintage
each decision actually sees.

--model tests a candidate in place of the shipped Random Forest. Its two prediction
tables live under data/processed/benchmarks/ and are fitted on first use; the shipped
pair is left alone, because load_or_build caches by file path and fingerprints only
the feature set and the lag, so asking it for another model would hand back the
Random Forest's predictions under the new name.

Usage:
    python scripts/forecast_error_bars.py                 # cached runs if present
    python scripts/forecast_error_bars.py --fresh --block 14
    python scripts/forecast_error_bars.py --model hgb-pinball
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

PROCESSED = ROOT / "data" / "processed"
BENCH = PROCESSED / "benchmarks"
REPORTS = ROOT / "reports"
SELECT_BEFORE = "2025-01-01"


def prediction_tables(model_type: str = "rf") -> tuple:
    """
    The (day-ahead, bid-time) walk-forward tables for a model, fitted if absent.

    The shipped Random Forest reads its published pair. Any other model gets its own
    under benchmarks/, never load_or_build's: that caches by file path and its
    fingerprint covers the feature set and the lag but not the model, so a second
    model asked for through it would silently be handed the first one's predictions.
    """
    from scripts.build_forecast_walk_forward import backtest_window, load_or_build
    from src.analysis.price_forecast import (
        WALK_FORWARD_CADENCE_MONTHS,
        build_feature_matrix,
        load_bess_capacity,
        walk_forward_predictions,
    )

    if model_type == "rf":
        return (load_or_build(verbose=False)[0],
                load_or_build(verbose=False, information_lag_days=2)[0])

    auctions = pd.read_parquet(PROCESSED / "auctions.parquet")
    market_index = pd.read_parquet(PROCESSED / "market_index.parquet")
    generation = pd.read_parquet(PROCESSED / "generation_daily.parquet")
    capacity = load_bess_capacity()
    window = backtest_window(auctions, market_index)

    tables = []
    for lag in (1, 2):
        suffix = "" if lag == 1 else "_lag2"
        path = BENCH / f"forecast_wf_{model_type}_{WALK_FORWARD_CADENCE_MONTHS}m{suffix}.parquet"
        if path.exists():
            print(f"  reused {path.name}", flush=True)
            tables.append(pd.read_parquet(path))
            continue
        print(f"  fitting {path.name}…", flush=True)
        started = time.time()
        predictions, folds = walk_forward_predictions(
            build_feature_matrix(market_index, generation, capacity, information_lag_days=lag),
            window[0], window[1], model_type=model_type,
            cadence_months=WALK_FORWARD_CADENCE_MONTHS,
        )
        BENCH.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(path, index=False)
        path.with_suffix(".folds.json").write_text(json.dumps(folds, indent=2) + "\n")
        print(f"    {len(folds)} origins in {(time.time() - started) / 60:.1f} min", flush=True)
        tables.append(predictions)
    return tuple(tables)


def daily_revenue(strategy: str, fresh: bool = False, model_type: str = "rf") -> pd.DataFrame:
    """Per-day revenue for one strategy, at the settings the site publishes."""
    from scripts.build_forecast_walk_forward import backtest_window, load_or_build
    from scripts.precompute_cache import (
        FORECAST_VINTAGES,
        OFFER_INFORMATION,
        OFFER_VALUATION,
        PRICE_SHRINK,
    )
    from src.analysis.price_forecast import run_forecast_backtest
    from src.analysis.revenue_stack import ALL_SERVICES, REFERENCE_BATTERY

    # naive has no model, so its run is shared across candidates; the shipped model
    # keeps the unsuffixed name it already writes
    tag = strategy if strategy == "naive" or model_type == "rf" else f"{strategy}_{model_type}"
    cached = BENCH / f"daily_revenue_{tag}.parquet"
    if cached.exists() and not fresh:
        print(f"  reused {cached.name}", flush=True)
        return pd.read_parquet(cached)

    started = time.time()
    auctions = pd.read_parquet(PROCESSED / "auctions.parquet")
    market_index = pd.read_parquet(PROCESSED / "market_index.parquet")
    delivery = pd.read_parquet(PROCESSED / "response_delivery.parquet")
    start, end = backtest_window(auctions, market_index)
    predictions = early = None
    if strategy == "ml":
        predictions, early = prediction_tables(model_type)
    result = run_forecast_backtest(
        strategy=strategy, market_index=market_index, auctions=auctions,
        battery=REFERENCE_BATTERY, services=ALL_SERVICES, start_date=start, end_date=end,
        predictions=predictions, early_predictions=early, delivery=delivery,
        offer_valuation=OFFER_VALUATION, price_shrink=PRICE_SHRINK[strategy],
        offer_information=OFFER_INFORMATION, forecast_vintages=FORECAST_VINTAGES,
    )
    BENCH.mkdir(parents=True, exist_ok=True)
    result["daily"].to_parquet(cached, index=False)
    print(f"  {strategy}: £{result['summary']['annualised_per_mw'] / 1e3:.1f}k/MW/yr "
          f"[{(time.time() - started) / 60:.1f} min]", flush=True)
    return result["daily"]


def forecast_losses(market_index: pd.DataFrame, model_type: str = "rf") -> dict:
    """
    Paired daily losses for each forecast, by the vintage each decision sees.

    "dispatch" is the day-ahead pair - the ML table against yesterday's prices -
    and "offer" the bid-time pair, from data to D-2. Squared error is averaged over
    the day so both losses are one number per day, which is the unit the tests
    resample; spread error is the day's predicted range against its realised one.
    """
    from src.analysis.shrink import day_matrix, naive_predictions

    apx = market_index[(market_index["dataProvider"] == "APXMIDP")
                       & (market_index["settlementPeriod"] <= 48)]
    actual = day_matrix(apx, "price")
    day_ahead, bid_time = prediction_tables(model_type)
    tables = {
        ("dispatch", "ml"): day_matrix(day_ahead, "prediction"),
        ("dispatch", "naive"): day_matrix(naive_predictions(market_index, days_back=1), "prediction"),
        ("offer", "ml"): day_matrix(bid_time, "prediction"),
        ("offer", "naive"): day_matrix(naive_predictions(market_index, days_back=2), "prediction"),
    }
    out = {}
    for stage in ("dispatch", "offer"):
        ml, naive = tables[(stage, "ml")], tables[(stage, "naive")]
        days = ml.index.intersection(naive.index).intersection(actual.index)
        a, m, n = actual.loc[days], ml.loc[days], naive.loc[days]
        keep = a.notna().all(axis=1) & m.notna().all(axis=1) & n.notna().all(axis=1)
        a, m, n = a[keep], m[keep], n[keep]
        spread = lambda x: x.max(axis=1) - x.min(axis=1)
        out[stage] = {
            "days": a.index,
            "squared_error": {"ml": ((m - a) ** 2).mean(axis=1), "naive": ((n - a) ** 2).mean(axis=1)},
            "spread_error": {"ml": (spread(m) - spread(a)).abs(), "naive": (spread(n) - spread(a)).abs()},
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--block", type=int, default=28, help="bootstrap block length in days")
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--fresh", action="store_true", help="re-run the dispatch backtests")
    parser.add_argument("--select-before", default=SELECT_BEFORE)
    parser.add_argument("--model", default="rf",
                        help="model spec to test in place of the shipped Random Forest")
    args = parser.parse_args()

    from src.analysis.revenue_stack import REFERENCE_BATTERY
    from src.analysis.significance import bootstrap_interval, diebold_mariano, paired_daily

    market_index = pd.read_parquet(PROCESSED / "market_index.parquet")

    print(f"Model under test: {args.model}")
    print("Accuracy: Diebold-Mariano on paired daily losses")
    accuracy = {}
    for stage, data in forecast_losses(market_index, args.model).items():
        for loss in ("squared_error", "spread_error"):
            test = diebold_mariano(data[loss]["ml"].to_numpy(), data[loss]["naive"].to_numpy())
            accuracy[f"{stage}/{loss}"] = test
            print(f"  {stage:<9} {loss:<14} mean gain {test['mean_difference']:>10.1f}  "
                  f"t {test['t_stat']:>6.2f}  p {test['p_value']:.2e}  n {test['n']}")

    print("\nRevenue: moving block bootstrap of the daily difference")
    daily = {s: daily_revenue(s, args.fresh, args.model) for s in ("ml", "naive")}
    paired = paired_daily(daily["ml"], daily["naive"])
    per_mw_year = 365.25 / REFERENCE_BATTERY.power_mw / 1e3      # £/day -> £k/MW/yr
    revenue = {}
    halves = {"all": paired,
              "selection": paired[paired.index < pd.Timestamp(args.select_before)],
              "confirmation": paired[paired.index >= pd.Timestamp(args.select_before)]}
    for name, frame in halves.items():
        if frame.empty:
            continue
        out = bootstrap_interval(frame["difference"].to_numpy(), block=args.block,
                                 resamples=args.resamples, scale=per_mw_year)
        revenue[name] = out
        print(f"  {name:<13} ML − naive £{out['mean']:>5.2f}k/MW/yr  "
              f"95% [{out['low']:>6.2f}, {out['high']:>5.2f}]  "
              f"P(≤0) {out['share_below_zero']:.3f}  days {len(frame)}")

    REPORTS.mkdir(exist_ok=True)
    # The shipped model keeps the canonical filenames; a candidate writes beside them
    stem = "forecast_error_bars" if args.model == "rf" else f"forecast_error_bars_{args.model}"
    (REPORTS / f"{stem}.json").write_text(json.dumps(
        {"model": args.model, "accuracy": accuracy, "revenue": revenue, "block_days": args.block,
         "resamples": args.resamples, "select_before": args.select_before}, indent=2) + "\n")

    described = ("the shipped Random Forest" if args.model == "rf"
                 else f"the candidate model `{args.model}`")
    fmt = lambda x: f"{x:+.2f}"
    lines = [
        "# Error bars on the forecast's value",
        "",
        f"Paired daily comparisons of {described} against reusing an older day's "
        f"prices, on the engine as published. Blocks of {args.block} days, {args.resamples:,} "
        f"resamples.",
        "",
        "## Is it more accurate? Diebold-Mariano on paired daily losses",
        "",
        "| Stage | Loss | Mean gain | t | p |",
        "|---|---|---|---|---|",
    ]
    for key, test in accuracy.items():
        stage, loss = key.split("/")
        lines.append(f"| {stage} | {loss.replace('_', ' ')} | {test['mean_difference']:.1f} | "
                     f"{test['t_stat']:.2f} | {test['p_value']:.2e} |")
    lines += [
        "",
        "A positive mean gain means the model loses less than naive. The dispatch row uses the "
        "day-ahead vintage each strategy dispatches on; the offer row uses the bid-time vintage "
        "offers see.",
        "",
        "## Is it worth more money? Block bootstrap of the daily revenue difference",
        "",
        "| Window | ML − naive (£k/MW/yr) | 95% interval | P(≤ 0) | Days |",
        "|---|---|---|---|---|",
    ]
    for name, out in revenue.items():
        lines.append(f"| {name} | {out['mean']:.2f} | [{fmt(out['low'])}, {fmt(out['high'])}] | "
                     f"{out['share_below_zero']:.3f} | {len(halves[name])} |")
    (REPORTS / f"{stem}.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote reports/{stem}.md")


if __name__ == "__main__":
    main()
