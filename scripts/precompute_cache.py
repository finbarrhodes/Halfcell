#!/usr/bin/env python3
"""
Pre-compute Backtest Cache
===========================
Runs all six MPC strategy backtests with a fixed 50 MW / 2h battery
configuration and writes results to data/cache/ as Parquet files.

The Streamlit app is a pure viewer that reads from this cache — no live
computation at app startup.

Usage:
    python scripts/precompute_cache.py

Re-run after any data update or methodology change, then commit the updated
cache files.  Runtime: ~20–40 minutes total (six MPC backtests; DNN is slowest).

Strategies computed:
  1. Perfect Foresight + MPC  — revenue ceiling
  2. Naive (D-1 prices) + MPC — zero-skill floor
  3. ML (Random Forest) + MPC
  4. ML (LightGBM) + MPC
  5. ML (LEAR) + MPC
  6. ML (DNN, Lago 2021) + MPC
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from the repo root or the scripts/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.analysis.price_forecast import (
    DEFAULT_TEST_START,
    build_feature_matrix,
    load_bess_capacity,
    run_forecast_backtest,
    train_forecast_model,
)
from src.analysis.revenue_stack import ALL_SERVICES, REFERENCE_BATTERY, run_backtest

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
DISPATCH_METHOD = "mpc"
HORIZON         = 96    # 48h rolling LP horizon
SERVICES        = ALL_SERVICES
ML_MODEL_TYPE   = "rf"  # Current production model — update after revenue comparison

# Set to a subset of ["rf", "lgb", "lear", "dnn"] to skip individual models.
# Useful if e.g. PyTorch is unavailable (comment out "dnn") or a model fails.
MODELS_TO_RUN = ["rf", "lgb", "lear", "dnn"]


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

def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load source data
    # ------------------------------------------------------------------
    print("Loading processed data…")
    auctions  = pd.read_parquet(PROCESSED / "auctions.parquet")
    mkt_index = pd.read_parquet(PROCESSED / "market_index.parquet")
    gen_daily = pd.read_parquet(PROCESSED / "generation_daily.parquet")

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
        start_date           = str(start_date),
        end_date             = str(end_date),
    )

    # ------------------------------------------------------------------
    # 1. Perfect Foresight + MPC
    # ------------------------------------------------------------------
    _print_section(1, 6, "Perfect Foresight + MPC")

    pf = run_backtest(
        auctions, mkt_index, BATTERY, SERVICES, start_date, end_date,
        initial_soc_frac=INITIAL_SOC,
        dispatch_method=DISPATCH_METHOD,
        horizon=HORIZON,
    )
    pf["monthly"].to_parquet(CACHE / "pf_mpc.parquet", index=False)
    if pf.get("soc_trajectory") is not None:
        pf["soc_trajectory"].to_parquet(CACHE / "soc_pf_mpc.parquet", index=False)

    manifest["pf_mpc"] = dict(
        computed_at = datetime.now(timezone.utc).isoformat(),
        git_sha     = git_sha,
        data_mtimes = data_mtimes,
        params      = base_params,
        summary     = pf["summary"],
    )
    print(f"  Total net revenue : £{pf['summary']['total_net']:>12,.0f}")
    print(f"  Ann. per MW       : £{pf['summary']['annualised_per_mw']:>10,.0f} / MW / yr")

    # ------------------------------------------------------------------
    # 2. Naive (D-1 prices) + MPC
    # ------------------------------------------------------------------
    _print_section(2, 6, "Naive (D-1 prices) + MPC")

    naive = run_forecast_backtest(
        strategy        = "naive",
        market_index    = mkt_index,
        auctions        = auctions,
        battery         = BATTERY,
        services        = SERVICES,
        start_date      = start_date,
        end_date        = end_date,
        initial_soc_frac = INITIAL_SOC,
        dispatch_method = DISPATCH_METHOD,
        horizon         = HORIZON,
    )
    naive["monthly"].to_parquet(CACHE / "naive_mpc.parquet", index=False)
    if naive.get("soc_trajectory") is not None:
        naive["soc_trajectory"].to_parquet(CACHE / "soc_naive_mpc.parquet", index=False)

    manifest["naive_mpc"] = dict(
        computed_at = datetime.now(timezone.utc).isoformat(),
        git_sha     = git_sha,
        data_mtimes = data_mtimes,
        params      = base_params,
        summary     = naive["summary"],
    )
    print(f"  Total net revenue : £{naive['summary']['total_net']:>12,.0f}")
    print(f"  Ann. per MW       : £{naive['summary']['annualised_per_mw']:>10,.0f} / MW / yr")

    # ------------------------------------------------------------------
    # Shared ML setup: build feature matrix once for all ML models
    # ------------------------------------------------------------------
    print(f"\n{'─' * 60}")
    print("  Building shared feature matrix for ML models…")
    print(f"{'─' * 60}")
    feature_df = build_feature_matrix(mkt_index, gen_daily, load_bess_capacity())
    print(f"  Feature matrix: {len(feature_df):,} rows, test split: {DEFAULT_TEST_START}")

    ml_base_params = {**base_params, "test_start": str(DEFAULT_TEST_START)}

    # ------------------------------------------------------------------
    # 3. ML (Random Forest) + MPC
    # ------------------------------------------------------------------
    if "rf" in MODELS_TO_RUN:
        _print_section(3, 6, "ML (Random Forest) + MPC")

        print("  Training RF model…")
        model, feature_cols, train_metrics, test_metrics = train_forecast_model(
            feature_df, model_type="rf", test_start=DEFAULT_TEST_START
        )
        print(f"  Test RMSE: {test_metrics['rmse']:.2f} £/MWh  |  "
              f"Spearman ρ: {test_metrics['spearman']:.3f}")

        print("  Running RF + MPC backtest…")
        ml = run_forecast_backtest(
            strategy        = "ml",
            market_index    = mkt_index,
            auctions        = auctions,
            battery         = BATTERY,
            services        = SERVICES,
            start_date      = start_date,
            end_date        = end_date,
            model           = model,
            feature_df      = feature_df,
            feature_cols    = feature_cols,
            initial_soc_frac = INITIAL_SOC,
            dispatch_method = DISPATCH_METHOD,
            horizon         = HORIZON,
        )
        ml["monthly"].to_parquet(CACHE / "ml_mpc.parquet", index=False)
        if ml.get("soc_trajectory") is not None:
            ml["soc_trajectory"].to_parquet(CACHE / "soc_ml_mpc.parquet", index=False)

        manifest["ml_mpc"] = dict(
            computed_at   = datetime.now(timezone.utc).isoformat(),
            git_sha       = git_sha,
            data_mtimes   = data_mtimes,
            params        = {**ml_base_params, "ml_model_type": "rf"},
            summary       = ml["summary"],
            model_metrics = {"train": train_metrics, "test": test_metrics},
        )
        print(f"  Total net revenue : £{ml['summary']['total_net']:>12,.0f}")
        print(f"  Ann. per MW       : £{ml['summary']['annualised_per_mw']:>10,.0f} / MW / yr")
    else:
        print("\n  [3/6] RF skipped (not in MODELS_TO_RUN)")

    # ------------------------------------------------------------------
    # 4. ML (LightGBM) + MPC
    # ------------------------------------------------------------------
    if "lgb" in MODELS_TO_RUN:
        _print_section(4, 6, "ML (LightGBM) + MPC")

        print("  Training LGB model…")
        lgb_model, lgb_feat_cols, lgb_train_m, lgb_test_m = train_forecast_model(
            feature_df, model_type="lgb", test_start=DEFAULT_TEST_START
        )
        print(f"  Test RMSE: {lgb_test_m['rmse']:.2f} £/MWh  |  "
              f"Spearman ρ: {lgb_test_m['spearman']:.3f}")

        print("  Running LGB + MPC backtest…")
        lgb_result = run_forecast_backtest(
            strategy        = "ml",
            market_index    = mkt_index,
            auctions        = auctions,
            battery         = BATTERY,
            services        = SERVICES,
            start_date      = start_date,
            end_date        = end_date,
            model           = lgb_model,
            feature_df      = feature_df,
            feature_cols    = lgb_feat_cols,
            initial_soc_frac = INITIAL_SOC,
            dispatch_method = DISPATCH_METHOD,
            horizon         = HORIZON,
        )
        lgb_result["monthly"].to_parquet(CACHE / "lgb_mpc.parquet", index=False)
        if lgb_result.get("soc_trajectory") is not None:
            lgb_result["soc_trajectory"].to_parquet(CACHE / "soc_lgb_mpc.parquet", index=False)

        manifest["lgb_mpc"] = dict(
            computed_at   = datetime.now(timezone.utc).isoformat(),
            git_sha       = git_sha,
            data_mtimes   = data_mtimes,
            params        = {**ml_base_params, "ml_model_type": "lgb"},
            summary       = lgb_result["summary"],
            model_metrics = {"train": lgb_train_m, "test": lgb_test_m},
        )
        print(f"  Total net revenue : £{lgb_result['summary']['total_net']:>12,.0f}")
        print(f"  Ann. per MW       : £{lgb_result['summary']['annualised_per_mw']:>10,.0f} / MW / yr")
    else:
        print("\n  [4/6] LGB skipped (not in MODELS_TO_RUN)")

    # ------------------------------------------------------------------
    # 5. ML (LEAR) + MPC
    # ------------------------------------------------------------------
    if "lear" in MODELS_TO_RUN:
        _print_section(5, 6, "ML (LEAR) + MPC")

        print("  Training LEAR model (48 per-period Lasso regressors)…")
        lear_model, lear_feat_cols, lear_train_m, lear_test_m = train_forecast_model(
            feature_df, model_type="lear", test_start=DEFAULT_TEST_START
        )
        print(f"  Test RMSE: {lear_test_m['rmse']:.2f} £/MWh  |  "
              f"Spearman ρ: {lear_test_m['spearman']:.3f}")

        print("  Running LEAR + MPC backtest…")
        lear_result = run_forecast_backtest(
            strategy        = "ml",
            market_index    = mkt_index,
            auctions        = auctions,
            battery         = BATTERY,
            services        = SERVICES,
            start_date      = start_date,
            end_date        = end_date,
            model           = lear_model,
            feature_df      = feature_df,
            feature_cols    = lear_feat_cols,
            initial_soc_frac = INITIAL_SOC,
            dispatch_method = DISPATCH_METHOD,
            horizon         = HORIZON,
        )
        lear_result["monthly"].to_parquet(CACHE / "lear_mpc.parquet", index=False)
        if lear_result.get("soc_trajectory") is not None:
            lear_result["soc_trajectory"].to_parquet(CACHE / "soc_lear_mpc.parquet", index=False)

        manifest["lear_mpc"] = dict(
            computed_at   = datetime.now(timezone.utc).isoformat(),
            git_sha       = git_sha,
            data_mtimes   = data_mtimes,
            params        = {**ml_base_params, "ml_model_type": "lear"},
            summary       = lear_result["summary"],
            model_metrics = {"train": lear_train_m, "test": lear_test_m},
        )
        print(f"  Total net revenue : £{lear_result['summary']['total_net']:>12,.0f}")
        print(f"  Ann. per MW       : £{lear_result['summary']['annualised_per_mw']:>10,.0f} / MW / yr")
    else:
        print("\n  [5/6] LEAR skipped (not in MODELS_TO_RUN)")

    # ------------------------------------------------------------------
    # 6. ML (DNN, Lago 2021) + MPC
    # ------------------------------------------------------------------
    if "dnn" in MODELS_TO_RUN:
        _print_section(6, 6, "ML (DNN, Lago 2021) + MPC")

        print("  Training DNN model (requires PyTorch — slowest step)…")
        dnn_model, dnn_feat_cols, dnn_train_m, dnn_test_m = train_forecast_model(
            feature_df, model_type="dnn", test_start=DEFAULT_TEST_START
        )
        print(f"  Test RMSE: {dnn_test_m['rmse']:.2f} £/MWh  |  "
              f"Spearman ρ: {dnn_test_m['spearman']:.3f}")

        print("  Running DNN + MPC backtest…")
        dnn_result = run_forecast_backtest(
            strategy        = "ml",
            market_index    = mkt_index,
            auctions        = auctions,
            battery         = BATTERY,
            services        = SERVICES,
            start_date      = start_date,
            end_date        = end_date,
            model           = dnn_model,
            feature_df      = feature_df,
            feature_cols    = dnn_feat_cols,
            initial_soc_frac = INITIAL_SOC,
            dispatch_method = DISPATCH_METHOD,
            horizon         = HORIZON,
        )
        dnn_result["monthly"].to_parquet(CACHE / "dnn_mpc.parquet", index=False)
        if dnn_result.get("soc_trajectory") is not None:
            dnn_result["soc_trajectory"].to_parquet(CACHE / "soc_dnn_mpc.parquet", index=False)

        manifest["dnn_mpc"] = dict(
            computed_at   = datetime.now(timezone.utc).isoformat(),
            git_sha       = git_sha,
            data_mtimes   = data_mtimes,
            params        = {**ml_base_params, "ml_model_type": "dnn"},
            summary       = dnn_result["summary"],
            model_metrics = {"train": dnn_train_m, "test": dnn_test_m},
        )
        print(f"  Total net revenue : £{dnn_result['summary']['total_net']:>12,.0f}")
        print(f"  Ann. per MW       : £{dnn_result['summary']['annualised_per_mw']:>10,.0f} / MW / yr")
    else:
        print("\n  [6/6] DNN skipped (not in MODELS_TO_RUN)")

    # ------------------------------------------------------------------
    # Write manifest
    # ------------------------------------------------------------------
    print(f"\n{'─' * 60}")
    manifest_path = CACHE / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print("\n✓  Cache written to data/cache/")
    print("   pf_mpc.parquet        naive_mpc.parquet")
    print("   ml_mpc.parquet        lgb_mpc.parquet")
    print("   lear_mpc.parquet      dnn_mpc.parquet")
    print("   soc_*.parquet         manifest.json")
    print("\nCommit data/cache/ to the repository to make results available on")
    print("Streamlit Cloud without any compute step at deploy time.")


if __name__ == "__main__":
    main()
