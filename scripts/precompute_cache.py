#!/usr/bin/env python3
"""
Pre-compute Backtest Cache
===========================
Runs every strategy and scenario for the fixed 50 MW / 2h reference battery and
writes the results to data/cache/ for the site to read.

Usage:
    python scripts/precompute_cache.py

Re-run after any data update or methodology change, then commit the updated
cache files.

Written, per strategy (pf_mpc, naive_mpc, ml_mpc):
  <key>.parquet            full stack: FR availability + arbitrage
  soc_<key>.parquet        state of energy for every settlement period
  <key>_arb_only.parquet   arbitrage only, no FR commitments

Shared by all strategies, since FR-only uses no price forecast:
  fr_only.parquet               a site with no interest in arbitrage: FR availability,
                                less the trading it needs to make good delivered energy

Every run is capped at a fifth of each auction's cleared volume, calls contracts on
as GB frequency actually moved (data/processed/response_delivery.parquet), and
prices that delivery into offers.

The ML strategy's forecasts come from data/processed/forecast_walk_forward.parquet,
where the model is refit at quarterly origins and predicts only the days after each
one, so no day is forecast by a model that trained on it. Missing origins are fitted
and cached on the way through; see scripts/build_forecast_walk_forward.py.

Every forecast is used only once the data behind it exists. Offers for day D close
at 14:00 on D-1, so they see the early forecast of D, built from data to D-2
(forecast_walk_forward_early.parquet for ML, D-2's prices for naive); dispatch
plans today on the day-ahead forecast and tomorrow on the early one, and ends its
plan there. Offers are priced by the day-ahead plan (src/optimisation/day_ahead.py)
on a forecast shrunk towards its daily mean by PRICE_SHRINK, chosen per signal on
the folds before 2025 (reports/offer_valuation_vintages.md). Perfect foresight
uses the same engine and horizon with actual prices.

Strategies:
  1. Perfect Foresight + MPC  — revenue ceiling
  2. Naive (D-1 prices) + MPC — zero-skill floor
  3. ML (Random Forest) + MPC — realistic best case; main result

Runtime is roughly seven dispatch runs at ~6-8 minutes each with the compile-once
MPC solver, plus model training. It lengthens with every refresh, because
DEFAULT_TEST_START is fixed and new data accrues to the test period, so re-check
it against the refresh workflow's timeout-minutes from time to time.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from the repo root or the scripts/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from scripts.build_forecast_walk_forward import fold_metrics, load_or_build, pooled_metrics
from src.analysis.price_forecast import (
    DEFAULT_TEST_START,
    WALK_FORWARD_CADENCE_MONTHS,
    build_feature_matrix,
    get_feature_importances,
    load_bess_capacity,
    run_forecast_backtest,
    train_forecast_model,
)
from src.analysis.revenue_stack import (
    ALL_SERVICES,
    AUCTION_SHARE_CAP,
    REFERENCE_BATTERY,
    run_backtest,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROCESSED = Path(__file__).parent.parent / "data" / "processed"
CACHE     = Path(__file__).parent.parent / "data" / "cache"

# ---------------------------------------------------------------------------
# Fixed battery configuration (documented in methodology expander in the app)
# ---------------------------------------------------------------------------

BATTERY = REFERENCE_BATTERY
INITIAL_SOC    = 0.5   # Neutral midpoint; SoC tracked continuously thereafter
DISPATCH_METHOD = "mpc"   # recorded in the manifest; MPC is the only dispatch path
PRE_EAC_RULE    = "d1"    # pre-EAC service chosen on D-1 clearing prices
HORIZON         = 96    # 48h rolling LP horizon
SERVICES        = ALL_SERVICES
OFFER_VALUATION = "lp"        # offers priced by the day-ahead trading plan
OFFER_INFORMATION = "bid_time"  # offers see only what existed at 14:00 on D-1
FORECAST_VINTAGES = True      # dispatch plans tomorrow on the early forecast, and stops there
# Weight on the plan's forecast deviations from its daily mean, per signal. Chosen on
# the pre-2025 folds only (0.25 / 0.5 / 0.75 / 1 tried; reports/offer_valuation_vintages.md):
# 0.5 for both, narrowly for ML, whose 0.5 and 0.75 are within £0.3k on either half.
# Perfect foresight has nothing to hedge against, so it plans on actual prices as they are.
PRICE_SHRINK = {"pf": 1.0, "naive": 0.5, "ml": 0.5}
ML_MODEL_TYPE   = "rf"  # Random Forest selected at precompute time (see methodology expander)
N_IMPORTANCES   = 20    # Top-N feature importances stored in the manifest for display


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent.parent,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _data_mtimes() -> dict:
    return {
        f.name: f.stat().st_mtime
        for f in sorted(PROCESSED.glob("*.parquet"))
    }


def _print_section(n: int, total: int, label: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  [{n}/{total}] {label}")
    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _summary_line(label: str, summary: dict) -> None:
    print(f"  {label:<20}: £{summary.get('annualised_per_mw', 0):>10,.0f} / MW / yr"
          f"   delivered {summary.get('total_delivery_mwh', 0):>9,.0f} MWh"
          f"   state-of-energy breaches: {summary.get('soe_breach_periods', 0):,} periods")


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Load source data
    # ------------------------------------------------------------------
    print("Loading processed data…")
    auctions  = pd.read_parquet(PROCESSED / "auctions.parquet")
    mkt_index = pd.read_parquet(PROCESSED / "market_index.parquet")
    gen_daily = pd.read_parquet(PROCESSED / "generation_daily.parquet")
    delivery  = pd.read_parquet(PROCESSED / "response_delivery.parquet")

    # Full overlapping date range
    start_date = max(
        auctions["EFA Date"].min(), mkt_index["settlementDate"].min()
    ).date()
    end_date = min(
        auctions["EFA Date"].max(), mkt_index["settlementDate"].max()
    ).date()
    print(f"Date range: {start_date} → {end_date}")

    git_sha     = _git_sha()
    data_mtimes = _data_mtimes()
    manifest    = {}

    base_params = dict(
        power_mw             = BATTERY.power_mw,
        duration_h           = BATTERY.duration_h,
        efficiency_rt        = BATTERY.efficiency_rt,
        cycling_cost_per_mwh = BATTERY.cycling_cost_per_mwh,
        availability_factor  = BATTERY.availability_factor,
        initial_soc          = INITIAL_SOC,
        dispatch_method      = DISPATCH_METHOD,
        horizon              = HORIZON,
        pre_eac_rule         = PRE_EAC_RULE,
        auction_share_cap    = AUCTION_SHARE_CAP,
        delivery_modelled    = True,
        offer_valuation      = OFFER_VALUATION,
        offer_information    = OFFER_INFORMATION,
        forecast_vintages    = FORECAST_VINTAGES,
        start_date           = str(start_date),
        end_date             = str(end_date),
    )

    def entry(result: dict, scenarios: dict, params: dict = base_params, **extra) -> dict:
        return dict(
            computed_at = datetime.now(timezone.utc).isoformat(),
            git_sha     = git_sha,
            data_mtimes = data_mtimes,
            params      = params,
            summary     = result["summary"],
            scenarios   = scenarios,
            **extra,
        )

    # ------------------------------------------------------------------
    # 1. FR availability only — no price forecast, so shared by all three
    # ------------------------------------------------------------------
    _print_section(1, 4, "FR availability only (scenario shared by all strategies)")
    fr_only = run_backtest(auctions, mkt_index, BATTERY, SERVICES, start_date, end_date,
                           include_arbitrage=False, pre_eac_rule=PRE_EAC_RULE, delivery=delivery,
                           forecast_vintages=FORECAST_VINTAGES)
    fr_only["monthly"].to_parquet(CACHE / "fr_only.parquet", index=False)
    _summary_line("FR only", fr_only["summary"])
    fr_scenarios = {"fr_only": fr_only["summary"]}

    def run_pair(key: str, run) -> tuple[dict, dict]:
        """Full stack and arbitrage-only for one strategy; writes both to the cache."""
        full = run(SERVICES)
        full["monthly"].to_parquet(CACHE / f"{key}.parquet", index=False)
        if full.get("soc_trajectory") is not None:
            full["soc_trajectory"].to_parquet(CACHE / f"soc_{key}.parquet", index=False)
        arb = run([])
        arb["monthly"].to_parquet(CACHE / f"{key}_arb_only.parquet", index=False)
        _summary_line("Full stack", full["summary"])
        _summary_line("Arbitrage only", arb["summary"])
        return full, {"arb_only": arb["summary"], **fr_scenarios}

    # ------------------------------------------------------------------
    # 2. Perfect Foresight + MPC
    # ------------------------------------------------------------------
    _print_section(2, 4, "Perfect Foresight + MPC")
    pf, pf_scenarios = run_pair("pf_mpc", lambda svc: run_backtest(
        auctions, mkt_index, BATTERY, svc, start_date, end_date,
        initial_soc_frac=INITIAL_SOC, horizon=HORIZON, pre_eac_rule=PRE_EAC_RULE, delivery=delivery,
        offer_valuation=OFFER_VALUATION, price_shrink=PRICE_SHRINK["pf"],
        forecast_vintages=FORECAST_VINTAGES,
    ))
    manifest["pf_mpc"] = entry(pf, pf_scenarios, params={**base_params, "price_shrink": PRICE_SHRINK["pf"]})

    # ------------------------------------------------------------------
    # 3. Naive (D-1 prices) + MPC
    # ------------------------------------------------------------------
    _print_section(3, 4, "Naive (D-1 prices) + MPC")
    naive, naive_scenarios = run_pair("naive_mpc", lambda svc: run_forecast_backtest(
        strategy="naive", market_index=mkt_index, auctions=auctions, battery=BATTERY,
        services=svc, start_date=start_date, end_date=end_date,
        initial_soc_frac=INITIAL_SOC, horizon=HORIZON, pre_eac_rule=PRE_EAC_RULE, delivery=delivery,
        offer_valuation=OFFER_VALUATION, price_shrink=PRICE_SHRINK["naive"],
        offer_information=OFFER_INFORMATION, forecast_vintages=FORECAST_VINTAGES,
    ))
    manifest["naive_mpc"] = entry(naive, naive_scenarios,
                                  params={**base_params, "price_shrink": PRICE_SHRINK["naive"]})

    # ------------------------------------------------------------------
    # 4. ML (Random Forest) + MPC
    # ------------------------------------------------------------------
    _print_section(4, 4, "ML (Random Forest) + MPC")

    print(f"  Walk-forward forecasts, refit every {WALK_FORWARD_CADENCE_MONTHS} months "
          f"(cached in data/processed)…")
    predictions, folds = load_or_build(model_type=ML_MODEL_TYPE)
    walk_forward = pooled_metrics(predictions, mkt_index)
    print(f"  Out-of-sample over {len(folds)} folds: RMSE {walk_forward['rmse']:.2f} £/MWh  |  "
          f"Spearman ρ: {walk_forward['spearman']:.3f}")
    print("  Early forecasts, from data to D-2, for offers and tomorrow's dispatch…")
    early_predictions, early_folds = load_or_build(model_type=ML_MODEL_TYPE, information_lag_days=2)
    early_metrics = pooled_metrics(early_predictions, mkt_index)
    print(f"  Early, over {len(early_folds)} folds: RMSE {early_metrics['rmse']:.2f} £/MWh  |  "
          f"Spearman ρ: {early_metrics['spearman']:.3f}")

    ml, ml_scenarios = run_pair("ml_mpc", lambda svc: run_forecast_backtest(
        strategy="ml", market_index=mkt_index, auctions=auctions, battery=BATTERY,
        services=svc, start_date=start_date, end_date=end_date, predictions=predictions,
        initial_soc_frac=INITIAL_SOC, horizon=HORIZON, pre_eac_rule=PRE_EAC_RULE, delivery=delivery,
        offer_valuation=OFFER_VALUATION, price_shrink=PRICE_SHRINK["ml"],
        offer_information=OFFER_INFORMATION, forecast_vintages=FORECAST_VINTAGES,
        early_predictions=early_predictions,
    ))

    # Feature importances describe the data, not any one forecast, so they come from a
    # single fit over all history — never used to predict anything. The same fit yields
    # the fixed-split metrics the methodology page cites when explaining why walk-forward
    # replaced them. Stored so the site never has to train a model: the chart needs ~20
    # numbers, not a Random Forest.
    print("  One fit over all history, for feature importances…")
    feature_df = build_feature_matrix(mkt_index, gen_daily, load_bess_capacity())
    model, feature_cols, fixed_train, fixed_test = train_forecast_model(
        feature_df, model_type=ML_MODEL_TYPE, test_start=DEFAULT_TEST_START
    )
    importances = get_feature_importances(model, feature_cols).head(N_IMPORTANCES)
    manifest["ml_mpc"] = entry(
        ml, ml_scenarios,
        params={**base_params, "price_shrink": PRICE_SHRINK["ml"], "ml_model_type": ML_MODEL_TYPE,
                "forecast_validation": "walk-forward",
                "walk_forward_cadence_months": WALK_FORWARD_CADENCE_MONTHS,
                "test_start": str(DEFAULT_TEST_START)},
        model_metrics={
            "walk_forward": walk_forward,
            "early": early_metrics,
            "folds": fold_metrics(predictions, mkt_index, folds),
            "fixed_split": {"train": fixed_train, "test": fixed_test},
        },
        feature_importances=[
            {"feature": str(k), "importance": float(v)} for k, v in importances.items()
        ],
    )

    # ------------------------------------------------------------------
    # Write manifest last: check_cache_consistency.py relies on that order
    # ------------------------------------------------------------------
    manifest_path = CACHE / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    minutes = (datetime.now(timezone.utc) - started).total_seconds() / 60
    print(f"\n{'─' * 60}")
    print(f"✓  Cache written to data/cache/ in {minutes:.1f} min")

if __name__ == "__main__":
    main()
