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
    python scripts/compare_offer_valuation.py --runs ml:lp:1:bid:vint:dyn      # per-day weight
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
DYNAMIC_FALLBACK = 0.5        # the shipped constant, used until a weight can be fitted
# Conformal bands calibrate on a trailing year: on the whole history they inherit the
# 2021-22 crisis and over-cover (0.72 against a 0.60 target, median width £72 against £50).
GUARD_WINDOW_DAYS = 365
DEFAULT_RUNS = "pf:formula,pf:lp,naive:formula,naive:lp,ml:formula,ml:lp"

# The engine: a change to any of these invalidates cached runs
ENGINE_SOURCES = [
    "src/analysis/revenue_stack.py", "src/analysis/fr_allocation.py", "src/analysis/neso_rules.py",
    "src/analysis/price_forecast.py", "src/optimisation/mpc.py", "src/optimisation/day_ahead.py",
    # the weights and bands runs plan against are part of the engine too, and so is
    # the code that builds the bands the qr/cqr/spci/ens runs read
    "src/analysis/shrink.py", "src/analysis/intervals.py", "src/analysis/spci.py",
    "src/analysis/quantile_forecast.py", "scripts/interval_benchmark.py",
    "scripts/build_quantile_walk_forward.py",
]
# Bands built by scripts/interval_benchmark.py rather than by the engine's own split
# conformal: "qr" the quantile forest's quantiles, "cqr" those conformalised, "spci"
# SPCI on the residuals ("spci_b" with the β search, "spci_w" refitted weekly), "ens"
# the average of qr, split conformal and spci.
BAND_METHODS = ("qr", "cqr", "spci", "spci_b", "spci_w", "ens")
# The configuration the site publishes, which every variant here is trying to beat
# Recovery credit is part of it from 2026-09-24, so a variant compared against these
# needs "recany" in its spec to run on the same dispatch engine
SHIPPED = {"pf": "pf_lp_recany_vint", "naive": "naive_lp_shrink0.5_recany_bid_vint",
           "ml": "ml_lp_shrink0.5_recany_bid_vint"}


def dispatch_tag(name: str) -> str | None:
    """The recovery-credit or margin tag in a run name: runs compare only within one."""
    return next((tag for tag in ("recany", "rec", "margin") if tag in name.split("_")), None)


def parse_run(spec: str) -> tuple:
    """
    "ml:lp:0.5:bid:vint" -> ("ml", "lp", 0.5, True, True, False). After strategy
    and valuation come numbers and flags. "bid" is bid-time offers, "vint" forecast
    vintages in dispatch, "dyn" a weight fitted per day (Mincer-Zarnowitz slope), and
    "sm5" smooths dispatch's own plan over five half-hours ("osm5" the offer plan's),
    "tilt" a weight that leans on how loud the day looks, and "cp" conformal guard
    bands, per settlement period unless "blk" (EFA block) or "dayg" (one set) is
    given. Any of BAND_METHODS in place of "cp" uses that method's bands instead
    (ML only: they are built on the ML forecast). The first number is the constant
    shrink, the risk factor with "dyn", or the intercept with "tilt"; the second is
    the tilt, or alpha with a band. Perfect foresight ignores everything but "vint",
    "rec" and "recany": those credit recovery through the Reserved Capacity at the
    price (revenue_stack.run_dispatch), a dispatch change that applies to every
    signal. "rec" spends the recovery allowance only on reserve flows, "recany" on
    every trade: the two readings bracket the answer. Both keep a margin inside each
    new block's requirement; "margin" keeps that margin without the credit.
    """
    parts = spec.strip().split(":")
    strategy, valuation, rest = parts[0], parts[1], parts[2:]
    flags = {part for part in rest
             if part in ("bid", "vint", "dyn", "tilt", "cp", "blk", "dayg", "rec", "recany", "margin") + BAND_METHODS}
    # "sm5" smooths dispatch's plan over five half-hours, "osm5" the offer plan's
    smoothing = {part[:-len(part.lstrip("abcdefghijklmnopqrstuvwxyz"))] or part: part
                 for part in rest if part.startswith(("sm", "osm"))}
    dispatch_smooth = int(smoothing["sm"][2:]) if "sm" in smoothing else 0
    offer_smooth = int(smoothing["osm"][3:]) if "osm" in smoothing else 0
    numbers = [part for part in rest if part not in flags and not part.startswith(("sm", "osm"))]
    if strategy not in ("pf", "naive", "ml") or valuation not in ("formula", "lp") or len(numbers) > 2:
        raise ValueError(f"bad run spec {spec!r}")
    shrink = float(numbers[0]) if numbers else 1.0
    second = float(numbers[1]) if len(numbers) > 1 else 0.0
    mode = "tilt" if "tilt" in flags else "slope" if "dyn" in flags else None
    guard = None
    methods = [m for m in BAND_METHODS if m in flags]
    if len(methods) > 1 or (methods and "cp" in flags) or (methods and strategy == "naive"):
        raise ValueError(f"bad run spec {spec!r}: one band at a time, and {methods} bands are ML-only")
    if "cp" in flags and strategy != "pf":
        guard = (second or 0.2, "block" if "blk" in flags else "day" if "dayg" in flags else "period")
    elif methods and strategy != "pf":
        guard = (second or 0.2, methods[0])
    recovery = "any" if "recany" in flags else "reserve" if "rec" in flags else None
    if recovery and "margin" in flags:
        raise ValueError(f"bad run spec {spec!r}: credited recovery already keeps the margin")
    return (strategy, valuation, shrink, "bid" in flags and strategy != "pf", "vint" in flags,
            mode if strategy != "pf" else None, second, guard, dispatch_smooth, offer_smooth, recovery,
            "margin" in flags)


def engine_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in ENGINE_SOURCES:
        digest.update((ROOT / path).read_bytes())
    return digest.hexdigest()[:12]


def run_name(strategy: str, valuation: str, shrink: float, bid_time: bool = False,
             vintages: bool = False, dynamic=None, second: float = 0.0, guard=None,
             dispatch_smooth: int = 0, offer_smooth: int = 0, recovery: str | None = None,
             margin: bool = False) -> str:
    weight = (f"_a{shrink:g}b{second:+g}" if dynamic == "tilt" else
              f"_dyn{shrink:g}" if dynamic else
              f"_shrink{shrink:g}" if valuation == "lp" and shrink != 1.0 else "")
    band = ("" if not guard else f"_{guard[1]}{guard[0]:g}" if guard[1] in BAND_METHODS
            else f"_cp{guard[0]:g}{guard[1][:3]}")
    smooth = (f"_sm{dispatch_smooth}" if dispatch_smooth else "") + (f"_osm{offer_smooth}" if offer_smooth else "")
    credit = {"reserve": "_rec", "any": "_recany"}.get(recovery, "_margin" if margin else "")
    return (f"{strategy}_{valuation}" + weight + band + smooth + credit
            + ("_bid" if bid_time else "") + ("_vint" if vintages else ""))


def backtest(strategy: str, valuation: str, shrink: float, bid_time: bool, vintages: bool,
             dynamic, second: float, guard, dispatch_smooth: int, offer_smooth: int,
             recovery: str | None, margin: bool, fingerprint: str, fresh: bool) -> tuple:
    """One full-window backtest, cached. Runs in a worker process."""
    from scripts.build_forecast_walk_forward import backtest_window, load_or_build
    from src.analysis.price_forecast import run_forecast_backtest
    from src.analysis.revenue_stack import ALL_SERVICES, REFERENCE_BATTERY, run_backtest

    name = run_name(strategy, valuation, shrink, bid_time, vintages, dynamic, second, guard,
                    dispatch_smooth, offer_smooth, recovery, margin)
    monthly_file = BENCH / f"offer_valuation_{name}_{fingerprint}.parquet"
    summary_file = monthly_file.with_suffix(".json")
    if monthly_file.exists() and summary_file.exists() and not fresh:
        return name, pd.read_parquet(monthly_file), json.loads(summary_file.read_text()), 0.0

    started = time.time()
    auctions = pd.read_parquet(PROCESSED / "auctions.parquet")
    market_index = pd.read_parquet(PROCESSED / "market_index.parquet")
    delivery = pd.read_parquet(PROCESSED / "response_delivery.parquet")
    start, end = backtest_window(auctions, market_index)
    # With a per-day weight the number in the spec is the risk factor on each fitted
    # weight, and the constant becomes only the fallback until there is history to fit.
    common = dict(services=ALL_SERVICES, start_date=start, end_date=end, delivery=delivery,
                  offer_valuation=valuation,
                  price_shrink=DYNAMIC_FALLBACK if dynamic == "slope" else shrink)
    if strategy == "pf":
        # Smoothing answers forecast error, which perfect foresight does not have - it
        # keeps actual prices, as it keeps a shrink of 1
        result = run_backtest(auctions, market_index, REFERENCE_BATTERY, forecast_vintages=vintages,
                              credit_recovery=recovery, block_start_margin=margin, **common)
    else:
        ml = strategy == "ml"
        prebuilt = guard is not None and guard[1] in BAND_METHODS
        if prebuilt:
            from scripts.interval_benchmark import bands
        predictions = load_or_build(model_type="rf", verbose=False)[0] if ml else None
        early_predictions = (load_or_build(model_type="rf", verbose=False, information_lag_days=2)[0]
                             if ml and (bid_time or vintages) else None)
        result = run_forecast_backtest(strategy=strategy, market_index=market_index, auctions=auctions,
                                       battery=REFERENCE_BATTERY, predictions=predictions,
                                       offer_information="bid_time" if bid_time else "day_ahead",
                                       forecast_vintages=vintages,
                                       early_predictions=early_predictions,
                                       dynamic_shrink=dynamic or False, shrink_risk_factor=shrink,
                                       shrink_tilt=second,
                                       guard_alpha=guard[0] if guard and not prebuilt else None,
                                       guard_group=guard[1] if guard and not prebuilt else "period",
                                       guard_window_days=GUARD_WINDOW_DAYS,
                                       guard_bands=bands(guard[1], guard[0]) if prebuilt else None,
                                       plan_smoothing=offer_smooth,
                                       dispatch_smoothing=dispatch_smooth, credit_recovery=recovery,
                                       block_start_margin=margin, **common)

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
        # Months, not days: the first and last are partial, so this runs about 1.6%
        # low against revenue_stack's day-based divisor. Every run in a report shares
        # the window, so the bias scales all of them alike and cancels exactly in the
        # comparisons and foresight ratios below. Left as is to keep the committed
        # reports comparable without re-running every backtest behind them.
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
    # Perfect foresight has one run per valuation and horizon: everything the weight
    # names add (_shrink, _dyn, _a..b.., _bid) belongs to a forecast, not to it.
    # Recovery credit and the block-start margin are dispatch changes it shares, so its
    # ceiling carries the same tag.
    parts = valuation_key.split("_")
    credit = dispatch_tag(valuation_key)
    pf_key = (parts[0] + (f"_{credit}" if credit else "")
              + ("_vint" if valuation_key.endswith("_vint") else ""))
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
    fmt = lambda x: "—" if x is None else f"{x * 100:.1f}%"
    lines += ["", "| Valuation | Foresight ratio (sel) | Foresight ratio (conf) |", "|---|---|---|"]
    for v in valuations:
        sel, conf = foresight(rows, v, "selection"), foresight(rows, v, "confirmation")
        lines.append(f"| {v} | {fmt(sel)} | {fmt(conf)} |")
    if SHIPPED["pf"] in rows and SHIPPED["naive"] in rows:
        # Bands built on the ML forecast have no naive twin, so they are measured against
        # the published naive and perfect-foresight runs instead
        lines += ["", "Against the published signals: net £k/MW/yr less the shipped ML run, and the "
                      "foresight ratio with the shipped naive run as the floor.", "",
                  "| Run | Δ net vs shipped (sel) | Δ net vs shipped (conf) | Foresight (sel) | Foresight (conf) |",
                  "|---|---|---|---|---|"]
        for name, row in rows.items():
            # Recovery credit changes dispatch for every signal: only runs on the shipped
            # dispatch engine are measured against the shipped naive and ceiling
            if not name.startswith("ml_") or dispatch_tag(name) != dispatch_tag(SHIPPED["ml"]):
                continue
            cells = []
            for half in ("selection", "confirmation"):
                shipped = rows.get(SHIPPED["ml"], {}).get("revenue", {}).get(half)
                cells.append("—" if shipped is None else f"{row['revenue'][half]['net'] - shipped['net']:+.2f}")
            for half in ("selection", "confirmation"):
                pf, naive = (rows[SHIPPED[k]]["revenue"][half]["net"] for k in ("pf", "naive"))
                ratio = (row["revenue"][half]["net"] - naive) / (pf - naive) if pf != naive else None
                cells.append(fmt(ratio))
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
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
