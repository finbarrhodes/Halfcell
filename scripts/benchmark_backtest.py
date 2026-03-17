#!/usr/bin/env python3
"""
Revenue Backtest Benchmark: RF vs LightGBM vs LEAR vs DNN
==========================================================
Trains all four ML models and runs full MPC revenue backtests for each,
reporting a side-by-side revenue and revenue-gap comparison.

Uses the same battery config, dispatch method, and date range as
precompute_cache.py — results are directly comparable to the production cache.

Usage:
    python scripts/benchmark_backtest.py

Metrics reported:
  Net revenue     — total arbitrage + FR net of cycling cost over the period
  Ann. per MW     — annualised net revenue per MW of power capacity
  Revenue gap     — (ML - Naive) / (PF - Naive): fraction of capturable
                    improvement over naive that each model delivers
  Arb. gap        — same ratio computed on arbitrage revenue only,
                    isolating the forecast-sensitive slice
  Train time      — model training wall-clock seconds
  Backtest time   — MPC dispatch wall-clock seconds
"""

import sys
import time
from pathlib import Path

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
# Configuration — mirrors precompute_cache.py exactly
# ---------------------------------------------------------------------------

PROCESSED = Path(__file__).parent.parent / "data" / "processed"

BATTERY = REFERENCE_BATTERY
INITIAL_SOC     = 0.5
DISPATCH_METHOD = "greedy"   # faster than MPC; relative model ranking is consistent
HORIZON         = 96         # unused in greedy mode; retained for MPC compatibility
SERVICES        = ALL_SERVICES

ML_MODELS = ["rf", "lgb", "lear", "dnn"]
ML_LABELS = {
    "rf":   "Random Forest",
    "lgb":  "LightGBM",
    "lear": "LEAR",
    "dnn":  "DNN (Lago 2021)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(n: int) -> str:
    return "─" * n


def _arb_net(result: dict) -> float:
    """Extract total arbitrage net (imbalance revenue minus cycling cost)."""
    df = result["monthly"]
    arb = 0.0
    if "imbalance_revenue_gbp" in df.columns:
        arb += df["imbalance_revenue_gbp"].sum()
    if "cycling_cost_gbp" in df.columns:
        arb -= df["cycling_cost_gbp"].sum()
    return arb


def _rev_gap(ml_net: float, naive_net: float, pf_net: float) -> float | None:
    denom = pf_net - naive_net
    if abs(denom) < 1.0:
        return None
    return (ml_net - naive_net) / denom


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading processed data…")
    auctions  = pd.read_parquet(PROCESSED / "auctions.parquet")
    mkt_index = pd.read_parquet(PROCESSED / "market_index.parquet")
    gen_daily = pd.read_parquet(PROCESSED / "generation_daily.parquet")

    start_date = max(
        auctions["EFA Date"].min(), mkt_index["settlementDate"].min()
    ).date()
    end_date = min(
        auctions["EFA Date"].max(), mkt_index["settlementDate"].max()
    ).date()
    print(f"Date range : {start_date} → {end_date}")
    print(f"Test split : {DEFAULT_TEST_START} (model trained on pre-split data)\n")

    # ------------------------------------------------------------------
    # Build feature matrix once (shared by all ML models)
    # ------------------------------------------------------------------
    print("Building feature matrix…")
    feature_df = build_feature_matrix(mkt_index, gen_daily, load_bess_capacity())

    # ------------------------------------------------------------------
    # Perfect Foresight (revenue ceiling)
    # ------------------------------------------------------------------
    print("Running Perfect Foresight + MPC…", end=" ", flush=True)
    t0 = time.time()
    pf = run_backtest(
        auctions, mkt_index, BATTERY, SERVICES, start_date, end_date,
        initial_soc_frac=INITIAL_SOC,
        dispatch_method=DISPATCH_METHOD,
        horizon=HORIZON,
    )
    pf_time = time.time() - t0
    pf_net     = pf["summary"]["total_net"]
    pf_ann     = pf["summary"]["annualised_per_mw"]
    pf_arb_net = _arb_net(pf)
    print(f"done ({pf_time:.0f}s)  —  net £{pf_net:,.0f}")

    # ------------------------------------------------------------------
    # Naive baseline (zero-skill floor)
    # ------------------------------------------------------------------
    print("Running Naive (D-1) + MPC…", end=" ", flush=True)
    t0 = time.time()
    naive = run_forecast_backtest(
        strategy="naive",
        market_index=mkt_index,
        auctions=auctions,
        battery=BATTERY,
        services=SERVICES,
        start_date=start_date,
        end_date=end_date,
        initial_soc_frac=INITIAL_SOC,
        dispatch_method=DISPATCH_METHOD,
        horizon=HORIZON,
    )
    naive_time = time.time() - t0
    naive_net     = naive["summary"]["total_net"]
    naive_ann     = naive["summary"]["annualised_per_mw"]
    naive_arb_net = _arb_net(naive)
    print(f"done ({naive_time:.0f}s)  —  net £{naive_net:,.0f}")

    # ------------------------------------------------------------------
    # ML models
    # ------------------------------------------------------------------
    results = {}
    for model_type in ML_MODELS:
        label = ML_LABELS[model_type]
        print(f"\nTraining {label}…", end=" ", flush=True)
        t_train = time.time()
        model, feature_cols, _, test_metrics = train_forecast_model(
            feature_df, model_type=model_type, test_start=DEFAULT_TEST_START
        )
        train_time = time.time() - t_train
        print(f"done ({train_time:.0f}s)  RMSE={test_metrics['rmse']:.2f}  ρ={test_metrics['spearman']:.3f}")

        print(f"  Backtest {label} + MPC…", end=" ", flush=True)
        t_bt = time.time()
        ml = run_forecast_backtest(
            strategy="ml",
            market_index=mkt_index,
            auctions=auctions,
            battery=BATTERY,
            services=SERVICES,
            start_date=start_date,
            end_date=end_date,
            model=model,
            feature_df=feature_df,
            feature_cols=feature_cols,
            initial_soc_frac=INITIAL_SOC,
            dispatch_method=DISPATCH_METHOD,
            horizon=HORIZON,
        )
        bt_time = time.time() - t_bt
        ml_net     = ml["summary"]["total_net"]
        ml_ann     = ml["summary"]["annualised_per_mw"]
        ml_arb_net = _arb_net(ml)
        print(f"done ({bt_time:.0f}s)  —  net £{ml_net:,.0f}")

        results[model_type] = dict(
            net       = ml_net,
            ann       = ml_ann,
            arb_net   = ml_arb_net,
            rev_gap   = _rev_gap(ml_net, naive_net, pf_net),
            arb_gap   = _rev_gap(ml_arb_net, naive_arb_net, pf_arb_net),
            train_s   = train_time,
            bt_s      = bt_time,
            rmse      = test_metrics["rmse"],
            spearman  = test_metrics["spearman"],
            spike_rmse= test_metrics.get("spike_rmse"),
        )

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    col_w   = 20
    n_ml    = len(ML_MODELS)
    n_ref   = 2   # PF + Naive
    n_cols  = n_ml + n_ref
    row_w   = 32 + col_w * n_cols
    all_labels = ["Perfect Foresight", "Naive (D-1)"] + [ML_LABELS[m] for m in ML_MODELS]

    print(f"\n\n{_sep(row_w)}")
    print(f"  {'':30}" + "".join(f"{lbl:>{col_w}}" for lbl in all_labels))
    print(_sep(row_w))

    def _row(label: str, vals: list[str], best_idx: int | None = None) -> None:
        row = f"  {label:<30}"
        for i, v in enumerate(vals):
            marker = " ✓" if i == best_idx else "  "
            row += f"{v + marker:>{col_w}}"
        print(row)

    def _best_ml(vals_ml: list, higher_better: bool) -> int | None:
        """Return 0-based index into vals_ml of the best ML model (offset +2 for display)."""
        valid = [(i, v) for i, v in enumerate(vals_ml) if v is not None]
        if not valid:
            return None
        best_i = (
            max(valid, key=lambda x: x[1])[0] if higher_better
            else min(valid, key=lambda x: x[1])[0]
        )
        return best_i + 2  # +2 for PF and Naive columns

    # Net revenue
    pf_net_s    = f"£{pf_net:,.0f}"
    naive_net_s = f"£{naive_net:,.0f}"
    ml_nets     = [results[m]["net"] for m in ML_MODELS]
    best        = _best_ml(ml_nets, higher_better=True)
    _row("Net revenue (£)", [pf_net_s, naive_net_s] + [f"£{v:,.0f}" for v in ml_nets], best)

    # Annualised per MW
    ml_anns = [results[m]["ann"] for m in ML_MODELS]
    best    = _best_ml(ml_anns, higher_better=True)
    _row("Ann. per MW (£/MW/yr)",
         [f"£{pf_ann:,.0f}", f"£{naive_ann:,.0f}"] + [f"£{v:,.0f}" for v in ml_anns], best)

    # Revenue gap
    ml_gaps = [results[m]["rev_gap"] for m in ML_MODELS]
    best    = _best_ml(ml_gaps, higher_better=True)
    _row("Revenue gap",
         ["1.000", "0.000"] + [f"{v:.3f}" if v is not None else "n/a" for v in ml_gaps], best)

    # Arb-only gap
    ml_arb_gaps = [results[m]["arb_gap"] for m in ML_MODELS]
    best        = _best_ml(ml_arb_gaps, higher_better=True)
    _row("Arb-only gap",
         ["1.000", "0.000"] + [f"{v:.3f}" if v is not None else "n/a" for v in ml_arb_gaps], best)

    print(_sep(row_w))

    # Forecast metrics
    ml_rmses    = [results[m]["rmse"]     for m in ML_MODELS]
    ml_rhos     = [results[m]["spearman"] for m in ML_MODELS]
    ml_spikes   = [results[m]["spike_rmse"] for m in ML_MODELS]
    _row("RMSE (£/MWh)",
         ["—", "—"] + [f"{v:.2f}" for v in ml_rmses],
         _best_ml(ml_rmses, higher_better=False))
    _row("Spearman ρ",
         ["—", "—"] + [f"{v:.4f}" for v in ml_rhos],
         _best_ml(ml_rhos, higher_better=True))
    _row("Spike RMSE",
         ["—", "—"] + [f"{v:.2f}" if v is not None else "n/a" for v in ml_spikes],
         _best_ml(ml_spikes, higher_better=False))

    print(_sep(row_w))

    # Timing
    print(f"  {'Train time (s)':<30}"
          + f"{'—':>{col_w}}" + f"{'—':>{col_w}}"
          + "".join(f"{results[m]['train_s']:>{col_w - 2}.0f}s " for m in ML_MODELS))
    print(f"  {'Backtest time (s)':<30}"
          + f"{pf_time:>{col_w - 2}.0f}s "
          + f"{naive_time:>{col_w - 2}.0f}s "
          + "".join(f"{results[m]['bt_s']:>{col_w - 2}.0f}s " for m in ML_MODELS))
    print(_sep(row_w))
    print()


if __name__ == "__main__":
    main()
