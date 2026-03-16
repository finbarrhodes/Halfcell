#!/usr/bin/env python3
"""
Model Benchmark: RF vs LightGBM
=================================
Trains RF and LightGBM on the current dataset and prints a side-by-side
metric comparison. No backtest — just forecasting accuracy.

Usage:
    python scripts/benchmark_models.py

Metrics reported (test set only, strict temporal split at 2025-03-01):
  RMSE        — overall point-forecast error
  MAE         — mean absolute error
  Spearman ρ  — ordinal ranking accuracy (governs dispatch quality)
  Spike RMSE  — RMSE on top-decile price periods (most important for arbitrage)
  Train time  — wall-clock seconds
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
    train_forecast_model,
)

PROCESSED = Path(__file__).parent.parent / "data" / "processed"
MODELS = ["rf", "lgb", "lear"]
MODEL_LABELS = {"rf": "Random Forest", "lgb": "LightGBM", "lear": "LEAR"}


def main() -> None:
    print("Loading processed data…")
    mkt_index = pd.read_parquet(PROCESSED / "market_index.parquet")
    gen_daily  = pd.read_parquet(PROCESSED / "generation_daily.parquet")

    print("Building feature matrix…")
    feature_df = build_feature_matrix(mkt_index, gen_daily, load_bess_capacity())

    train_n = (feature_df["settlementDate"] < pd.Timestamp(DEFAULT_TEST_START)).sum()
    test_n  = (feature_df["settlementDate"] >= pd.Timestamp(DEFAULT_TEST_START)).sum()
    print(f"  Train periods : {train_n:,}  (up to {DEFAULT_TEST_START})")
    print(f"  Test periods  : {test_n:,}   (from {DEFAULT_TEST_START})\n")

    results = {}
    for model_type in MODELS:
        label = MODEL_LABELS[model_type]
        print(f"Training {label}…", end=" ", flush=True)
        t0 = time.time()
        _, _, train_m, test_m = train_forecast_model(
            feature_df, model_type=model_type, test_start=DEFAULT_TEST_START
        )
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")
        results[model_type] = {"train": train_m, "test": test_m, "elapsed_s": elapsed}

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    metrics = ["rmse", "mae", "spearman", "spike_rmse"]
    metric_labels = {
        "rmse":       "RMSE (£/MWh)",
        "mae":        "MAE  (£/MWh)",
        "spearman":   "Spearman ρ",
        "spike_rmse": "Spike RMSE   (top-decile)",
    }
    col_w = 18

    sep_len = 30 + col_w * len(MODELS)
    print(f"\n{'─' * sep_len}")
    print(f"  {'Metric':<28}" + "".join(f"{MODEL_LABELS[m]:>{col_w}}" for m in MODELS))
    print(f"{'─' * sep_len}")

    for metric in metrics:
        label = metric_labels[metric]
        row = f"  {label:<28}"
        vals = []
        for model_type in MODELS:
            v = results[model_type]["test"].get(metric)
            vals.append(v)
        # Spearman: higher is better; all others: lower is better
        higher_better = metric == "spearman"
        valid = [(i, v) for i, v in enumerate(vals) if v is not None]
        best_idx = (
            max(valid, key=lambda x: x[1])[0] if higher_better
            else min(valid, key=lambda x: x[1])[0]
        ) if valid else None

        for i, (model_type, v) in enumerate(zip(MODELS, vals)):
            if v is None:
                cell = "n/a"
            elif metric == "spearman":
                cell = f"{v:.4f}"
            else:
                cell = f"{v:.2f}"
            marker = " ✓" if i == best_idx else "  "
            row += f"{cell + marker:>{col_w}}"
        print(row)

    print(f"{'─' * sep_len}")
    row = f"  {'Train time (s)':<28}"
    row += "".join(f"{results[m]['elapsed_s']:>{col_w - 2}.1f}s " for m in MODELS)
    print(row)
    print(f"{'─' * sep_len}\n")

    # Train-set RMSE (overfit indicator)
    print("  Train-set RMSE (overfit indicator):")
    for model_type in MODELS:
        label = MODEL_LABELS[model_type]
        tr = results[model_type]["train"]["rmse"]
        te = results[model_type]["test"]["rmse"]
        ratio = te / tr if tr else float("inf")
        print(f"    {label:<18}  train={tr:.2f}  test={te:.2f}  ratio={ratio:.2f}")
    print()


if __name__ == "__main__":
    main()
