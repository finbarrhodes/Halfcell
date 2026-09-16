"""
Price Forecast & Forecast-Driven Dispatch
==========================================
Orchestration layer for price forecasting and MPC backtest execution.

Implements three dispatch strategies for the BESS revenue backtester:

  1. Perfect Foresight — actual day-D prices fed to the optimizer (revenue ceiling).
     Already handled by revenue_stack.py; not repeated here.

  2. Naive baseline — uses actual day D-1 prices as the forecast for day D.
     No ML required; sets the "zero skill" floor.

  3. ML model — trains a Random Forest, XGBoost, LightGBM, or LEAR regressor on
     features available at end of day D-1 (lagged prices, generation mix, cyclical
     temporal encodings) with a strict temporal train/test split.

All three strategies run the same engine, revenue_stack.run_strategy. The forecast
values each EFA block's arbitrage when the FR allocation decides what to offer, and
drives the MPC dispatch; revenue is realised against actual day-D prices.

Model definitions   → src/analysis/forecasting_models.py
Feature engineering → src/analysis/features.py
MPC LP solver       → src/optimisation/mpc.py

New module kept separate from revenue_stack.py so the perfect-foresight backtester
remains a clean, standalone baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.forecasting_models import (
    _LogTransformModel,
    _LEARModel,
    _DNNModel,
    _build_model,
)
from src.analysis.features import (
    FEATURE_COLS,
    _build_lear_extra_features,
)

# Re-export data-loading and feature-building functions so existing callers
# that import them from this module continue to work unchanged.
from src.analysis.features import load_bess_capacity, build_feature_matrix  # noqa: F401


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default temporal split: everything before this date is training data.
# ~72 months train (Jan 2019–Feb 2025), ~12 months test (Mar 2025–Feb 2026).
DEFAULT_TEST_START = "2025-03-01"


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_forecast_model(
    feature_df: pd.DataFrame,
    model_type: str = "xgb",
    test_start: str = DEFAULT_TEST_START,
) -> tuple:
    """
    Train a price forecasting model with a strict temporal train/test split.

    Parameters
    ----------
    feature_df  : DataFrame from build_feature_matrix()
    model_type  : "rf", "xgb", "lgb", "lear", or "dnn"
    test_start  : ISO date string — all rows on or after this date form the test set

    Returns
    -------
    (model, feature_cols, train_metrics, test_metrics) where:
      model         : fitted _LogTransformModel wrapping the base estimator
      feature_cols  : list of column names used as features
      train_metrics : dict {rmse, mae, spearman, n_samples[, spike_rmse]}
      test_metrics  : dict {rmse, mae, spearman, n_samples[, spike_rmse]}
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from scipy.stats import spearmanr

    # Resolve the full feature column list (base + any per-fuel gen columns present)
    gen_fuel_cols = sorted([
        c for c in feature_df.columns
        if c.startswith("gen_") and c not in ("gen_total", "gen_renewable_frac", "gen_fossil_frac")
    ])
    feature_cols = FEATURE_COLS + [c for c in gen_fuel_cols if c not in FEATURE_COLS]
    feature_cols = [c for c in feature_cols if c in feature_df.columns]

    train = feature_df[feature_df["settlementDate"] < pd.Timestamp(test_start)]
    test  = feature_df[feature_df["settlementDate"] >= pd.Timestamp(test_start)]

    X_train = train[feature_cols].fillna(0)
    y_train = train["apx_price"]
    X_test  = test[feature_cols].fillna(0)
    y_test  = test["apx_price"]

    # LEAR / DNN: inject wide price features (all 48 SPs × D-1/D-2/D-7) and
    # DoW dummies, plus settlementPeriod. Built from the full feature_df with no
    # look-ahead. Tree-model paths are unaffected.
    # LEAR uses settlementPeriod as an internal routing key (popped in _LEARModel).
    # DNN uses it as a plain numeric input feature.
    _lear_extra: pd.DataFrame | None = None
    if model_type in ("lear", "dnn"):
        _lear_extra  = _build_lear_extra_features(feature_df)
        _extra_cols  = [c for c in _lear_extra.columns
                        if c not in ("settlementDate", "settlementPeriod")]

        _train_extra = (
            train[["settlementDate", "settlementPeriod"]]
            .merge(_lear_extra, on=["settlementDate", "settlementPeriod"], how="left")
        )
        _test_extra = (
            test[["settlementDate", "settlementPeriod"]]
            .merge(_lear_extra, on=["settlementDate", "settlementPeriod"], how="left")
        )
        X_train = pd.concat(
            [X_train.reset_index(drop=True),
             _train_extra[_extra_cols].reset_index(drop=True)],
            axis=1,
        ).fillna(0)
        X_train["settlementPeriod"] = train["settlementPeriod"].values
        X_test = pd.concat(
            [X_test.reset_index(drop=True),
             _test_extra[_extra_cols].reset_index(drop=True)],
            axis=1,
        ).fillna(0)
        X_test["settlementPeriod"] = test["settlementPeriod"].values

    # Wrap with signed-log transform: fit/predict both operate in price space
    model = _LogTransformModel(_build_model(model_type))
    model.fit(X_train, y_train)

    # LEAR / DNN: cache extra features on the model so predict_day_prices can
    # retrieve the wide lag columns without rebuilding them on every call
    if model_type in ("lear", "dnn") and _lear_extra is not None:
        model._model._lear_extra_df = _lear_extra

    def _metrics(X, y):
        pred = model.predict(X)
        y_arr = np.asarray(y)
        rmse = float(np.sqrt(mean_squared_error(y_arr, pred)))
        mae  = float(mean_absolute_error(y_arr, pred))

        # Spearman rank correlation of the 48-period daily rankings
        # (ordinal accuracy — directly governs greedy dispatch quality)
        sp_result  = spearmanr(y_arr, pred)
        sp_corr    = float(
            sp_result.statistic if hasattr(sp_result, "statistic") else sp_result.correlation
        )

        # Spike RMSE: RMSE restricted to top-decile actual price periods
        # (the periods that matter most for BESS arbitrage revenue)
        threshold  = float(np.percentile(y_arr, 90))
        spike_mask = y_arr >= threshold
        spike_rmse = (
            float(np.sqrt(mean_squared_error(y_arr[spike_mask], pred[spike_mask])))
            if spike_mask.sum() >= 10 else None
        )

        m = {
            "rmse":     round(rmse, 2),
            "mae":      round(mae, 2),
            "spearman": round(sp_corr, 3),
            "n_samples": len(y_arr),
        }
        if spike_rmse is not None:
            m["spike_rmse"] = round(spike_rmse, 2)
        return m

    train_metrics = _metrics(X_train, y_train)
    test_metrics  = _metrics(X_test,  y_test)

    return model, feature_cols, train_metrics, test_metrics


# ---------------------------------------------------------------------------
# Feature importance helper
# ---------------------------------------------------------------------------

def get_feature_importances(model, feature_cols: list) -> pd.Series:
    """Return feature importances as a named Series, sorted descending.

    For LEAR, uses the model's own stored feature names (which include the
    wide lag and DoW columns) rather than the base feature_cols list.
    """
    fi = model.feature_importances_
    names = (
        model._model._feat_names
        if isinstance(model._model, _LEARModel) and model._model._feat_names is not None
        else feature_cols
    )
    return pd.Series(fi, index=names[: len(fi)]).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Revenue gap metric
# ---------------------------------------------------------------------------

def compute_revenue_gap(
    ml_net: float,
    naive_net: float,
    perfect_net: float,
) -> float | None:
    """
    Revenue gap: fraction of the theoretically capturable improvement over naive
    that the ML forecast actually delivers.

        gap = (ml_net - naive_net) / (perfect_net - naive_net)

    A value of 1.0 means ML matches perfect foresight; 0.0 means ML is no better
    than naive; negative means ML underperforms naive.

    Returns None if the denominator is near-zero (no arbitrage headroom).
    """
    denom = perfect_net - naive_net
    if abs(denom) < 1.0:
        return None
    return (ml_net - naive_net) / denom


# ---------------------------------------------------------------------------
# Per-day prediction helpers
# ---------------------------------------------------------------------------

def predict_day_prices(
    model,
    feature_cols: list,
    feature_df: pd.DataFrame,
    target_date: pd.Timestamp,
) -> pd.Series:
    """
    Predict APXMIDP prices for all 48 settlement periods of target_date.

    Returns a Series indexed by settlementPeriod (1–48).
    Returns an empty Series if features for that date are unavailable.
    """
    day_df = (
        feature_df[feature_df["settlementDate"] == pd.Timestamp(target_date)]
        .sort_values("settlementPeriod")
    )
    if day_df.empty:
        return pd.Series(dtype=float)

    X = day_df[feature_cols].fillna(0).reset_index(drop=True)

    # LEAR / DNN: append wide lag features and settlementPeriod.
    # LEAR pops settlementPeriod internally as a routing key; DNN keeps it as a feature.
    if isinstance(model._model, (_LEARModel, _DNNModel)):
        lear_extra = model._model._lear_extra_df
        day_extra  = (
            lear_extra[lear_extra["settlementDate"] == pd.Timestamp(target_date)]
            .sort_values("settlementPeriod")
            .reset_index(drop=True)
        )
        if day_extra.empty:
            return pd.Series(dtype=float)
        extra_cols = [c for c in day_extra.columns
                      if c not in ("settlementDate", "settlementPeriod")]
        extra_part = day_extra[extra_cols].reset_index(drop=True)
        sp_col     = pd.DataFrame({"settlementPeriod": day_df["settlementPeriod"].values})
        X = pd.concat([X, extra_part, sp_col], axis=1).fillna(0)

    preds = model.predict(X)
    return pd.Series(preds, index=day_df["settlementPeriod"].values)


def naive_day_prices(
    market_index: pd.DataFrame,
    target_date: pd.Timestamp,
) -> pd.Series:
    """
    Naive forecast: return yesterday's APXMIDP prices as the forecast for target_date.
    Returns a Series indexed by settlementPeriod (1–48).
    Returns an empty Series if yesterday's data is unavailable.
    """
    yesterday = pd.Timestamp(target_date) - pd.Timedelta(days=1)
    apx = market_index[market_index["dataProvider"] == "APXMIDP"]
    prev = apx[apx["settlementDate"].dt.normalize() == yesterday]
    if prev.empty:
        return pd.Series(dtype=float)
    return prev.set_index("settlementPeriod")["price"]


# ---------------------------------------------------------------------------
# Forecast-driven dispatch backtester
# ---------------------------------------------------------------------------

def run_forecast_backtest(
    strategy: str,
    market_index: pd.DataFrame,
    auctions: pd.DataFrame,
    battery,                  # BatterySpec from revenue_stack
    services: list,
    start_date,
    end_date,
    model=None,
    feature_df: pd.DataFrame = None,
    feature_cols: list = None,
    initial_soc_frac: float = 0.5,
    horizon: int = 96,
    *,
    include_arbitrage: bool = True,
    pre_eac_rule: str = "d1",
    delivery: pd.DataFrame | None = None,
) -> dict:
    """
    Forecast-driven revenue backtest for the 'naive' or 'ml' strategy.

    The forecast for day D, built only from information available by the end of
    D-1, does two jobs: it values each EFA block's arbitrage when the FR
    allocation decides how much capacity to offer, and it drives the MPC
    dispatch. Revenue is realised against actual prices. Everything after the
    forecast is shared with the perfect-foresight run (revenue_stack.run_strategy).

    Parameters
    ----------
    strategy         : "naive" or "ml"
    market_index     : DataFrame from load_market_index()
    auctions         : DataFrame from load_auctions()
    battery          : BatterySpec instance
    services         : products the battery may offer ([] for arbitrage only)
    start_date       : inclusive backtest start
    end_date         : inclusive backtest end
    model            : fitted model object (required for strategy="ml")
    feature_df       : feature matrix from build_feature_matrix() (required for strategy="ml")
    feature_cols     : feature column list from train_forecast_model() (required for strategy="ml")
    initial_soc_frac : starting state of energy as a fraction of energy_mwh (default 0.5)
    horizon          : MPC planning horizon in settlement periods (default 96 = 48h)
    include_arbitrage: False runs the FR-only scenario; no forecast is needed
    pre_eac_rule     : "d1" — see revenue_stack.compute_fr_schedule
    delivery         : response delivery table (response_delivery.parquet), or None to leave
                       response undelivered

    Returns
    -------
    dict with the same keys as revenue_stack.run_backtest()
    """
    from src.analysis.revenue_stack import _apx_by_date, _in_range, run_strategy

    if strategy not in ("naive", "ml"):
        raise ValueError(f"Unknown strategy '{strategy}'")

    forecast_prices_by_date: dict = {}
    if include_arbitrage:
        apx_by_date = _apx_by_date(market_index)
        for date in sorted(d for d in apx_by_date if _in_range(d, start_date, end_date)):
            if strategy == "naive":
                fp = naive_day_prices(market_index, date)
            else:
                fp = predict_day_prices(model, feature_cols, feature_df, date)
            if not fp.empty:
                forecast_prices_by_date[date] = fp

    return run_strategy(
        auctions, market_index, battery, forecast_prices_by_date, services, start_date, end_date,
        initial_soc_frac=initial_soc_frac, horizon=horizon,
        include_arbitrage=include_arbitrage, pre_eac_rule=pre_eac_rule, delivery=delivery,
    )
