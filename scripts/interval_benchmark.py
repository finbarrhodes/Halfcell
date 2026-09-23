#!/usr/bin/env python3
"""
Interval benchmark: every guard band scored as a forecast
=========================================================
Builds each way of banding the bid-time forecast and scores them on the same days,
in the measures O'Connor et al. (2025) report - coverage, width, pinball loss and
the Winkler interval score - before any of them is asked to earn money
(scripts/compare_offer_valuation.py does that, reading its bands from here).

  scp       split conformal per settlement period on a trailing year: the bands
            the engine already had (src/analysis/intervals.py)
  qr        the quantile forest's own quantiles (src/analysis/quantile_forecast.py)
  cqr       the same, conformalised per settlement period on a trailing year
  spci      SPCI on the point forecast's residuals (src/analysis/spci.py), refitted
            monthly; spci_b adds the reference's β search, spci_w refits weekly
  ens       the average of qr, scp and spci - the paper's Q-Ens, with split
            conformal standing in for EnbPI

alpha is per tail, so each band is a 1 − 2α interval: 0.1 and 0.3 are the paper's
80% and 40% pairs, 0.2 the one the engine's conformal runs used.

Bands are cached under data/processed/benchmarks/, keyed by their settings, the
source files that build them, and the tables they are built from.

What it found (2026-09-23)
--------------------------
Scored as forecasts, the quantile forest's bands are the best here: a Winkler score
of 110.6 at α = 0.2 against 121.7 for split conformal, and ahead at 0.1 and 0.3 as
well. CQR brings their coverage to target (0.597 for 0.60) for about a point of
Winkler. SPCI only edges split conformal (119.2), and wins just one year, 2022,
when the forecast's errors ran the same way for months. Refitting it weekly
instead of monthly moves that to 118.5 (reports/interval_benchmark_spci_refit.md),
so the paper's daily refit would not close the gap.

None of it earns money. Every band loses to the shipped constant shrink on the
selection folds, by £2.2k-5.1k/MW/yr (reports/offer_valuation_quantile.md), and the
best-scored bands lose the most. The quantile forest is right that the loud days are
uncertain: its band is twice as wide on the fifth of days with the biggest realised
spreads (rank correlation with the spread +0.53). That is the trouble, because those
are the days whose spread is real. So the better the band, the more of the best
trading it declines - the same thing that sank the per-day shrink slope
(src/analysis/shrink.py).

Usage:
    python scripts/interval_benchmark.py
    python scripts/interval_benchmark.py --alphas 0.2 --methods scp,qr,cqr
"""

import argparse
import hashlib
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
ALPHAS = (0.1, 0.2, 0.3)
METHODS = ("scp", "qr", "cqr", "spci", "spci_b", "spci_w", "ens")
WINDOW_DAYS = 365             # trailing calibration year, as compare_offer_valuation's cp runs
SPCI_REFIT = {"spci": 30, "spci_b": 30, "spci_w": 7}
# The code a band depends on: a change to any of these rebuilds the cached bands
SOURCES = ["src/analysis/intervals.py", "src/analysis/spci.py", "src/analysis/quantile_forecast.py",
           "scripts/interval_benchmark.py"]


def _inputs() -> tuple:
    """(early point forecasts, quantile table, market index), as the offers see them."""
    from scripts.build_forecast_walk_forward import load_or_build
    from scripts.build_quantile_walk_forward import load_or_build as load_quantiles

    early = load_or_build(model_type="rf", verbose=False, information_lag_days=2)[0]
    quantiles = load_quantiles(verbose=False)[0]
    market_index = pd.read_parquet(PROCESSED / "market_index.parquet")
    return early, quantiles, market_index


def _fingerprint(method: str, alpha: float, *tables: pd.DataFrame) -> str:
    """
    The cache key for one band: its settings, the code, and the *contents* of the tables
    it is built from. Contents rather than shape, because a rebuilt forecast or a
    revised settlement price changes the band without changing a table's length.
    """
    digest = hashlib.sha256(f"{method}:{alpha:g}".encode())
    for path in SOURCES:
        digest.update((ROOT / path).read_bytes())
    for table in tables:
        digest.update(pd.util.hash_pandas_object(table, index=False).to_numpy().tobytes())
    return digest.hexdigest()[:12]


def _band_file(method: str, alpha: float, early, quantiles, market_index) -> Path:
    apx = market_index[market_index["dataProvider"] == "APXMIDP"][["settlementDate", "settlementPeriod", "price"]]
    return BENCH / f"bands_{method}_{alpha:g}_{_fingerprint(method, alpha, early, quantiles, apx)}.parquet"


def _to_frame(low: dict, high: dict) -> pd.DataFrame:
    rows = [pd.DataFrame({"settlementDate": day, "settlementPeriod": low[day].index.astype(int),
                          "low": low[day].to_numpy(dtype=float), "high": high[day].to_numpy(dtype=float)})
            for day in sorted(low)]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["settlementDate", "settlementPeriod", "low", "high"])


def _from_frame(frame: pd.DataFrame) -> tuple:
    low, high = {}, {}
    for day, rows in frame.groupby("settlementDate"):
        index = rows["settlementPeriod"].astype(int).to_numpy()
        low[pd.Timestamp(day)] = pd.Series(rows["low"].to_numpy(), index=index)
        high[pd.Timestamp(day)] = pd.Series(rows["high"].to_numpy(), index=index)
    return low, high


def _spci_tables(early, market_index, alphas, refit_days, verbose=True) -> dict:
    """SPCI's residual quantiles at every level the requested bands read, from one set of fits."""
    from src.analysis.spci import levels, walk_forward_spci_quantiles

    needed = set()
    for alpha in alphas:
        for beta in (False, True):
            lows, highs = levels(alpha, beta)
            needed |= {float(q) for q in (*lows, *highs)}
    started = time.time()
    tables, _ = walk_forward_spci_quantiles(early, market_index, sorted(needed), refit_days=refit_days)
    if verbose:
        print(f"  SPCI, refit every {refit_days} days: {(time.time() - started) / 60:.1f} min", flush=True)
    return tables


def build(method: str, alpha: float, early, quantiles, market_index, spci_tables=None) -> tuple:
    """(low_by_date, high_by_date) for one method at one alpha, uncached."""
    from src.analysis.intervals import (
        combine_bands,
        quantile_bands,
        walk_forward_bands,
        walk_forward_cqr_bands,
    )
    from src.analysis.spci import spci_bands

    if method == "scp":
        return walk_forward_bands(early, market_index, alpha=alpha, group="period",
                                  window_days=WINDOW_DAYS)[:2]
    if method == "qr":
        return quantile_bands(early, quantiles, alpha)[:2]
    if method == "cqr":
        return walk_forward_cqr_bands(early, quantiles, market_index, alpha=alpha, group="period",
                                      window_days=WINDOW_DAYS)[:2]
    if method in SPCI_REFIT:
        tables = (spci_tables or {}).get(SPCI_REFIT[method]) or _spci_tables(
            early, market_index, [alpha], SPCI_REFIT[method])
        return spci_bands(tables, alpha, optimise_beta=method == "spci_b")
    if method == "ens":
        members = [bands(m, alpha, (early, quantiles, market_index), spci_tables)
                   for m in ("qr", "scp", "spci")]
        return combine_bands(*[(low, high, []) for low, high in members])[:2]
    raise ValueError(f"unknown band method {method!r}; use one of {METHODS}")


def bands(method: str, alpha: float, inputs: tuple | None = None, spci_tables=None) -> tuple:
    """(low_by_date, high_by_date) for one method at one alpha, from the cache when it can."""
    early, quantiles, market_index = inputs or _inputs()
    cached = _band_file(method, alpha, early, quantiles, market_index)
    if cached.exists():
        return _from_frame(pd.read_parquet(cached))
    low, high = build(method, alpha, early, quantiles, market_index, spci_tables)
    BENCH.mkdir(parents=True, exist_ok=True)
    _to_frame(low, high).to_parquet(cached, index=False)
    return low, high


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alphas", default=",".join(f"{a:g}" for a in ALPHAS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--report", default="interval_benchmark")
    args = parser.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    methods = [m for m in args.methods.split(",") if m]

    from src.analysis.intervals import band_frame, score_bands

    inputs = _inputs()
    early, quantiles, market_index = inputs
    uncached = {m for a in alphas for m in methods if m in SPCI_REFIT or m == "ens"
                if not _band_file(m, a, early, quantiles, market_index).exists()}
    # The ensemble's SPCI member is the monthly one
    refits = ({SPCI_REFIT[m] for m in uncached if m in SPCI_REFIT}
              | ({SPCI_REFIT["spci"]} if "ens" in uncached else set()))
    spci_tables = {refit: _spci_tables(early, market_index, alphas, refit) for refit in sorted(refits)}

    results = {}
    for alpha in alphas:
        built = {m: bands(m, alpha, inputs, spci_tables) for m in methods}
        common = sorted(set.intersection(*(set(low) for low, _ in built.values())))
        if not common:
            raise SystemExit(f"alpha {alpha:g}: no day is banded by every method; nothing to compare")
        print(f"alpha {alpha:g}: {len(common)} days banded by every method, "
              f"{common[0].date()} to {common[-1].date()}", flush=True)
        results[f"{alpha:g}"] = {"days": len(common), "first": common[0].date().isoformat(),
                                 "last": common[-1].date().isoformat(), "methods": {}}
        for method, (low, high) in built.items():
            frame = band_frame(early, market_index, {d: low[d] for d in common}, {d: high[d] for d in common})
            results[f"{alpha:g}"]["methods"][method] = {
                by or "all": score_bands(frame, alpha, by=by).to_dict(orient="records")
                for by in (None, "year", "block")}
            overall = results[f"{alpha:g}"]["methods"][method]["all"][0]
            print(f"  {method:<7} coverage {overall['coverage']:.3f} (target {1 - 2 * alpha:.2f})  "
                  f"width {overall['mean_width']:6.1f}  pinball {overall['pinball']:6.2f}  "
                  f"winkler {overall['winkler']:6.1f}", flush=True)
    write_report(results, args.report)


def write_report(results: dict, report: str) -> None:
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"{report}.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = [
        "# Interval benchmark: guard bands scored as forecasts",
        "",
        "Bid-time Random Forest forecast, APXMIDP, walk-forward. Every method is scored on the "
        "days all of them band. Alpha is per tail: the target coverage is 1 − 2α. Pinball is the "
        "mean over the two bounds and Winkler the interval score; for both, lower is better. "
        "Built by `scripts/interval_benchmark.py`.",
    ]
    for alpha, block in results.items():
        target = 1 - 2 * float(alpha)
        lines += ["", f"## α = {alpha} ({target:.0%} interval), {block['days']} days, "
                      f"{block['first']} to {block['last']}", "",
                  "| Method | Coverage | Below | Above | Mean width | Pinball | Winkler |",
                  "|---|---|---|---|---|---|---|"]
        for method, scores in block["methods"].items():
            s = scores["all"][0]
            lines.append(f"| {method} | {s['coverage']:.3f} | {s['below']:.3f} | {s['above']:.3f} | "
                         f"{s['mean_width']:.1f} | {s['pinball']:.2f} | {s['winkler']:.1f} |")
        years = sorted({row["year"] for scores in block["methods"].values() for row in scores["year"]})
        lines += ["", "Winkler by year:", "", "| Method | " + " | ".join(str(y) for y in years) + " |",
                  "|---|" + "---|" * len(years)]
        for method, scores in block["methods"].items():
            by_year = {row["year"]: row["winkler"] for row in scores["year"]}
            lines.append(f"| {method} | " + " | ".join(f"{by_year.get(y, float('nan')):.1f}" for y in years) + " |")
    (REPORTS / f"{report}.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote reports/{report}.md")


if __name__ == "__main__":
    main()
