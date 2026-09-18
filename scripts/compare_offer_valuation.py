#!/usr/bin/env python3
"""
Offer valuation: the per-block formula against the day-ahead trading plan
=========================================================================
Runs the full dispatch backtest for each price signal (perfect foresight, naive,
walk-forward ML) under each way of valuing the trading an offer gives up, and
reports revenue split into FR and trading, for the selection folds (before
--select-before) and the confirmation folds separately.

What to expect if the plan is doing its job: the gain shows up first under
perfect foresight, where the forecast's shape across the day is right by
construction. Under a real forecast the plan can also be misled - it will pay
for headroom to chase a cross-block spread the forecast invents - which is what
price_shrink is for. Choose the shrink on the selection folds only.

":bid" gives naive and ML offers only what existed at the bid deadline
(run_forecast_backtest's offer_information="bid_time"). Without it, offers see a
forecast of D built from all of D-1, ten hours of which had not happened when
the offers closed. ":vint" does the same for dispatch: tomorrow is planned on
the early forecast and the plan ends after it (forecast_vintages). For perfect
foresight "bid" means nothing and "vint" only shortens the horizon.

Each run is cached under data/processed/benchmarks/, keyed by its settings and
by the source files the engine is built from, so a rerun after a code change
recomputes and a rerun without one only reports. --fresh ignores the cache.

Usage:
    python scripts/compare_offer_valuation.py                        # default grid, 4 at a time
    python scripts/compare_offer_valuation.py --runs pf:formula,pf:lp
    python scripts/compare_offer_valuation.py --runs ml:lp:0.5 --jobs 1
    python scripts/compare_offer_valuation.py --runs naive:lp:bid,ml:lp:0.5:bid:vint
"""

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

PROCESSED = ROOT / "data" / "processed"
BENCH = PROCESSED / "benchmarks"
REPORTS = ROOT / "reports"
SELECT_BEFORE = "2025-01-01"
DEFAULT_RUNS = "pf:formula,pf:lp,naive:formula,naive:lp,ml:formula,ml:lp"

# The engine: a change to any of these invalidates cached runs
ENGINE_SOURCES = [
    "src/analysis/revenue_stack.py", "src/analysis/fr_allocation.py", "src/analysis/neso_rules.py",
    "src/analysis/price_forecast.py", "src/optimisation/mpc.py", "src/optimisation/day_ahead.py",
]


def parse_run(spec: str) -> tuple:
    """
    "ml:lp:0.5:bid:vint" -> ("ml", "lp", 0.5, True, True). After strategy and
    valuation come any of: a shrink (default 1), "bid" for bid-time offers, and
    "vint" for forecast vintages in dispatch. Perfect foresight ignores "bid".
    """
    parts = spec.strip().split(":")
    strategy, valuation, rest = parts[0], parts[1], parts[2:]
    flags = {part for part in rest if part in ("bid", "vint")}
    numbers = [part for part in rest if part not in flags]
    if strategy not in ("pf", "naive", "ml") or valuation not in ("formula", "lp") or len(numbers) > 1:
        raise ValueError(f"bad run spec {spec!r}")
    shrink = float(numbers[0]) if numbers else 1.0
    return strategy, valuation, shrink, "bid" in flags and strategy != "pf", "vint" in flags


def engine_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in ENGINE_SOURCES:
        digest.update((ROOT / path).read_bytes())
    return digest.hexdigest()[:12]


def run_name(strategy: str, valuation: str, shrink: float, bid_time: bool = False, vintages: bool = False) -> str:
    return (f"{strategy}_{valuation}" + (f"_shrink{shrink:g}" if valuation == "lp" and shrink != 1.0 else "")
            + ("_bid" if bid_time else "") + ("_vint" if vintages else ""))


def backtest(strategy: str, valuation: str, shrink: float, bid_time: bool, vintages: bool,
             fingerprint: str, fresh: bool) -> tuple:
    """One full-window backtest, cached. Runs in a worker process."""
    from scripts.build_forecast_walk_forward import backtest_window, load_or_build
    from src.analysis.price_forecast import run_forecast_backtest
    from src.analysis.revenue_stack import ALL_SERVICES, REFERENCE_BATTERY, run_backtest

    name = run_name(strategy, valuation, shrink, bid_time, vintages)
    monthly_file = BENCH / f"offer_valuation_{name}_{fingerprint}.parquet"
    summary_file = monthly_file.with_suffix(".json")
    if monthly_file.exists() and summary_file.exists() and not fresh:
        return name, pd.read_parquet(monthly_file), json.loads(summary_file.read_text()), 0.0

    started = time.time()
    auctions = pd.read_parquet(PROCESSED / "auctions.parquet")
    market_index = pd.read_parquet(PROCESSED / "market_index.parquet")
    delivery = pd.read_parquet(PROCESSED / "response_delivery.parquet")
    start, end = backtest_window(auctions, market_index)
    common = dict(services=ALL_SERVICES, start_date=start, end_date=end, delivery=delivery,
                  offer_valuation=valuation, price_shrink=shrink)
    if strategy == "pf":
        result = run_backtest(auctions, market_index, REFERENCE_BATTERY, forecast_vintages=vintages, **common)
    else:
        ml = strategy == "ml"
        predictions = load_or_build(model_type="rf", verbose=False)[0] if ml else None
        early_predictions = (load_or_build(model_type="rf", verbose=False, information_lag_days=2)[0]
                             if ml and (bid_time or vintages) else None)
        result = run_forecast_backtest(strategy=strategy, market_index=market_index, auctions=auctions,
                                       battery=REFERENCE_BATTERY, predictions=predictions,
                                       offer_information="bid_time" if bid_time else "day_ahead",
                                       forecast_vintages=vintages,
                                       early_predictions=early_predictions, **common)

    BENCH.mkdir(parents=True, exist_ok=True)
    monthly = result["monthly"]
    monthly.drop(columns=["month"]).to_parquet(monthly_file, index=False)
    summary = {k: v for k, v in result["summary"].items() if not isinstance(v, dict)}
    summary_file.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return name, pd.read_parquet(monthly_file), summary, (time.time() - started) / 60


def split_revenue(monthly: pd.DataFrame, power_mw: float, select_before: str) -> dict:
    """£/MW/yr, net and split into FR and trading, over all months and each half."""
    fr_cols = [c for c in monthly.columns if c.endswith("_rev")]
    months = pd.to_datetime(monthly["month_dt"])
    out = {}
    for half, mask in (("all", months == months),
                       ("selection", months < pd.Timestamp(select_before)),
                       ("confirmation", months >= pd.Timestamp(select_before))):
        m = monthly[mask.to_numpy()]
        years = len(m) / 12
        if years == 0:
            out[half] = None
            continue
        per = lambda x: round(float(x) / years / power_mw / 1e3, 2)      # £k/MW/yr
        out[half] = {
            "net": per(m["net_revenue"].sum()),
            "fr": per(m[fr_cols].sum().sum()),
            "trading": per(m["imbalance_revenue_gbp"].sum()),
            "costs": per(m["cycling_cost"].sum()),
            "months": len(m),
        }
    return out


def foresight(rows: dict, valuation_key: str, half: str) -> float | None:
    """(ML − naive) / (PF − naive) within one valuation; PF is the same with or without _bid."""
    vint = valuation_key.endswith("_vint")
    pf_key = valuation_key.split("_shrink")[0].split("_bid")[0].removesuffix("_vint") + ("_vint" if vint else "")
    try:
        pf = rows[f"pf_{pf_key}"]["revenue"][half]["net"]
        naive, ml = (rows[f"{s}_{valuation_key}"]["revenue"][half]["net"] for s in ("naive", "ml"))
    except (KeyError, TypeError):
        return None
    return round((ml - naive) / (pf - naive), 4) if pf != naive else None


def write_report(rows: dict, select_before: str, fingerprint: str, report: str = "offer_valuation") -> None:
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"{report}.json").write_text(json.dumps(
        {"engine": fingerprint, "select_before": select_before, "runs": rows}, indent=2) + "\n")

    def cells(row, half):
        r = row["revenue"].get(half)
        return ["—"] * 3 if r is None else [f"{r['net']:.1f}", f"{r['fr']:.1f}", f"{r['trading']:.1f}"]

    lines = [
        "# Offer valuation: formula against the day-ahead plan",
        "",
        f"£k per MW per year, net of cycling costs, split into availability (FR) and trading. "
        f"Selection is before {select_before}; confirmation from it. Engine `{fingerprint}`.",
        "",
        "| Run | Net (sel) | FR (sel) | Trading (sel) | Net (conf) | FR (conf) | Trading (conf) | Breach periods |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, row in rows.items():
        lines.append(f"| {name} | " + " | ".join(cells(row, "selection") + cells(row, "confirmation"))
                     + f" | {row['summary'].get('soe_breach_periods', '—')} |")
    valuations = sorted({name.split("_", 1)[1] for name in rows if not name.startswith("pf_")})
    lines += ["", "| Valuation | Foresight ratio (sel) | Foresight ratio (conf) |", "|---|---|---|"]
    for v in valuations:
        sel, conf = foresight(rows, v, "selection"), foresight(rows, v, "confirmation")
        fmt = lambda x: "—" if x is None else f"{x * 100:.1f}%"
        lines.append(f"| {v} | {fmt(sel)} | {fmt(conf)} |")
    (REPORTS / f"{report}.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default=DEFAULT_RUNS, help="comma-separated strategy:valuation[:shrink]")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--select-before", default=SELECT_BEFORE)
    parser.add_argument("--fresh", action="store_true", help="ignore cached runs")
    parser.add_argument("--report", default="offer_valuation", help="report name under reports/")
    args = parser.parse_args()

    from src.analysis.revenue_stack import REFERENCE_BATTERY

    specs = [parse_run(s) for s in args.runs.split(",") if s.strip()]
    fingerprint = engine_fingerprint()
    print(f"Engine {fingerprint}; {len(specs)} runs, {args.jobs} at a time", flush=True)

    rows = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(backtest, *spec, fingerprint, args.fresh) for spec in specs]
        for future in as_completed(futures):
            name, monthly, summary, minutes = future.result()
            rows[name] = {"summary": summary,
                          "revenue": split_revenue(monthly, REFERENCE_BATTERY.power_mw, args.select_before)}
            conf = rows[name]["revenue"]["confirmation"]
            print(f"  {name:<24} net £{rows[name]['revenue']['all']['net']:.1f}k "
                  f"(conf £{conf['net']:.1f}k: FR {conf['fr']:.1f}, trading {conf['trading']:.1f})"
                  + (f"  [{minutes:.1f} min]" if minutes else "  [cached]"), flush=True)
            ordered = {run_name(*spec): rows[run_name(*spec)] for spec in specs if run_name(*spec) in rows}
            write_report(ordered, args.select_before, fingerprint, args.report)
    print(f"Wrote reports/{args.report}.md")


if __name__ == "__main__":
    main()
