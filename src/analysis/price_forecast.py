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

def resolve_feature_cols(feature_df: pd.DataFrame) -> list:
    """
    The feature columns a model will actually be given: the declared list plus any
    per-fuel generation columns present, filtered to what the matrix holds.

    That filter is what makes a feature group optional — build_feature_matrix
    either produced the group or it did not, and nothing else has to change. It
    also means the declared list is not the truth about what a model saw, which
    is why the walk-forward cache fingerprints the resolved list.
    """
    gen_fuel_cols = sorted(
        c for c in feature_df.columns
        if c.startswith("gen_") and c not in ("gen_total", "gen_renewable_frac", "gen_fossil_frac")
    )
    declared = FEATURE_COLS + [c for c in gen_fuel_cols if c not in FEATURE_COLS]
    return [c for c in declared if c in feature_df.columns]


def spread_calibration(dates, actual, predicted) -> dict:
    """
    How well a forecast predicts each day's price *spread*, which is what arbitrage trades on.

    RMSE and rank correlation both miss the failure that costs the most money: a forecast
    that invents a spread which never materialises. That costs a bad trade and then inflates
    the shadow arbitrage value until the allocation stage declines frequency response
    contracts worth having, while a forecast that merely misses a real spread only forgoes
    upside. The error is asymmetric, so it needs a signed statistic.

    Measured on 2026-09-17: Random Forest -31.5 £/MWh, LightGBM -19.6, LEAR +271.8. LEAR won
    every accuracy metric and earned £19k/MW/yr less than reusing yesterday's prices.

    Returns spread_bias (mean signed error, negative means conservative) and spread_mae.
    """
    frame = pd.DataFrame({
        "date": pd.to_datetime(pd.Series(list(dates)).values),
        "actual": np.asarray(actual, dtype=float),
        "predicted": np.asarray(predicted, dtype=float),
    })
    daily = frame.groupby("date").agg(
        actual_spread=("actual", lambda v: v.max() - v.min()),
        predicted_spread=("predicted", lambda v: v.max() - v.min()),
    )
    error = daily["predicted_spread"] - daily["actual_spread"]
    if error.empty:
        return {}
    return {"spread_bias": round(float(error.mean()), 2), "spread_mae": round(float(error.abs().mean()), 2)}


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

    feature_cols = resolve_feature_cols(feature_df)

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

    def _metrics(X, y, dates):
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
        m.update(spread_calibration(dates, y_arr, pred))
        return m

    train_metrics = _metrics(X_train, y_train, train["settlementDate"])
    test_metrics  = _metrics(X_test,  y_test,  test["settlementDate"])

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
    days_back: int = 1,
) -> pd.Series:
    """
    Naive forecast: the APXMIDP prices of the last complete day, `days_back`
    before target_date - yesterday's by default, the day before at an offer's
    bid deadline (see build_feature_matrix's information_lag_days).
    Returns a Series indexed by settlementPeriod (1–48).
    Returns an empty Series if that day's data is unavailable.
    """
    yesterday = pd.Timestamp(target_date) - pd.Timedelta(days=days_back)
    apx = market_index[market_index["dataProvider"] == "APXMIDP"]
    prev = apx[apx["settlementDate"].dt.normalize() == yesterday]
    if prev.empty:
        return pd.Series(dtype=float)
    return prev.set_index("settlementPeriod")["price"]


# ---------------------------------------------------------------------------
# Walk-forward retraining
# ---------------------------------------------------------------------------

# Months between retrainings. Quarterly is a compromise: an operator would retrain
# at least this often, and it costs ~20 fits over the backtest (14-31 s each) where
# monthly would cost 60. The prediction table is cached, so a refresh only fits the
# origins it does not already have.
WALK_FORWARD_CADENCE_MONTHS = 3


def walk_forward_origins(start_date, end_date, cadence_months: int = WALK_FORWARD_CADENCE_MONTHS) -> list:
    """Retraining dates covering [start_date, end_date], each the first of a month."""
    start, end = pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize()
    origins, origin = [], start.replace(day=1)
    while origin <= end:
        origins.append(origin)
        origin = origin + pd.DateOffset(months=cadence_months)
    return origins


def walk_forward_predictions(
    feature_df: pd.DataFrame,
    start_date,
    end_date,
    *,
    model_type: str = "rf",
    cadence_months: int = WALK_FORWARD_CADENCE_MONTHS,
    train_years: float | None = None,
    skip_origins=None,
    on_fold=None,
) -> tuple:
    """
    Out-of-sample price forecasts for every day in [start_date, end_date].

    A single fixed split cannot cover a revenue backtest that starts before the
    split date: predictions for those days would come from a model fitted on them,
    which flatters both the forecast and the revenue it drives. Here the model is
    refitted at each origin on data strictly before it and used only for the days
    up to the next origin, so no day is ever predicted by a model that saw it.

    The feature matrix starts well before the backtest does, which is what makes
    this possible: the first origin already has years of history to train on.

    Parameters
    ----------
    feature_df   : DataFrame from build_feature_matrix()
    start_date, end_date : inclusive bounds of the window to predict
    model_type   : any type train_forecast_model() accepts
    cadence_months : months between retrainings
    train_years  : cap the training window at this many years (a rolling window).
                   None trains on all prior data (expanding), which is the default
                   the fixed-split model used.
    skip_origins : origins to leave out, so a cached table can be extended by fitting
                   only the origins it does not already hold
    on_fold      : optional callback, passed each fold's metadata as it completes

    Returns
    -------
    (predictions, folds) where predictions is a DataFrame of
    [settlementDate, settlementPeriod, prediction, origin] and folds is one dict
    per origin: its training span and row count, and the metrics for the days it
    predicted.
    """
    origins = walk_forward_origins(start_date, end_date, cadence_months)
    first, last = pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize()
    dates = feature_df["settlementDate"].drop_duplicates().sort_values()

    skip = {pd.Timestamp(o).normalize() for o in (skip_origins or ())}
    frames, folds = [], []
    for i, origin in enumerate(origins):
        fold_end = origins[i + 1] if i + 1 < len(origins) else last + pd.Timedelta(days=1)
        if origin in skip:
            continue
        window = feature_df[feature_df["settlementDate"] < fold_end]
        if train_years is not None:
            window = window[window["settlementDate"] >= origin - pd.DateOffset(months=round(train_years * 12))]
        train_rows = int((window["settlementDate"] < origin).sum())
        if train_rows == 0 or (window["settlementDate"] >= origin).sum() == 0:
            continue

        # test_start=origin makes this fit on the history and score exactly this fold
        model, cols, _train_metrics, fold_metrics = train_forecast_model(
            window, model_type=model_type, test_start=origin
        )
        for day in dates[(dates >= max(origin, first)) & (dates < fold_end) & (dates <= last)]:
            forecast = predict_day_prices(model, cols, feature_df, day)
            if forecast.empty:
                continue
            frames.append(pd.DataFrame({
                "settlementDate": day,
                "settlementPeriod": forecast.index.astype(int),
                "prediction": forecast.to_numpy(dtype=float),
                "origin": origin,
            }))
        folds.append({
            "origin": origin.date().isoformat(),
            "predicts_until": (fold_end - pd.Timedelta(days=1)).date().isoformat(),
            "train_start": window["settlementDate"].min().date().isoformat(),
            "train_rows": train_rows,
            "metrics": fold_metrics,
        })
        if on_fold is not None:
            on_fold(folds[-1])

    columns = ["settlementDate", "settlementPeriod", "prediction", "origin"]
    predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    return predictions, folds


def forecast_series_by_date(predictions: pd.DataFrame, start_date=None, end_date=None) -> dict:
    """Prediction table as {date: Series indexed by settlementPeriod}, as dispatch wants it."""
    from src.analysis.revenue_stack import _in_range

    table = predictions.assign(settlementDate=pd.to_datetime(predictions["settlementDate"]).dt.normalize())
    return {
        date: group.set_index("settlementPeriod")["prediction"].sort_index()
        for date, group in table.groupby("settlementDate")
        if _in_range(date, start_date, end_date)
    }


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
    predictions: pd.DataFrame | None = None,
    initial_soc_frac: float = 0.5,
    horizon: int = 96,
    *,
    include_arbitrage: bool = True,
    pre_eac_rule: str = "d1",
    delivery: pd.DataFrame | None = None,
    offer_valuation: str = "formula",
    price_shrink: float = 1.0,
    offer_information: str = "day_ahead",
    forecast_vintages: bool = False,
    early_predictions: pd.DataFrame | None = None,
    dynamic_shrink: str | bool = False,
    shrink_risk_factor: float = 1.0,
    shrink_tilt: float = 0.0,
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
    predictions      : walk-forward prediction table from walk_forward_predictions(). Given
                       for strategy="ml", it replaces model/feature_df/feature_cols, and every
                       day is forecast by a model that never saw it
    initial_soc_frac : starting state of energy as a fraction of energy_mwh (default 0.5)
    horizon          : MPC planning horizon in settlement periods (default 96 = 48h)
    include_arbitrage: False runs the FR-only scenario; no forecast is needed
    pre_eac_rule     : "d1" — see revenue_stack.compute_fr_schedule
    delivery         : response delivery table (response_delivery.parquet), or None to leave
                       response undelivered
    offer_valuation, price_shrink : how offers price trading; see revenue_stack.run_strategy
    offer_information: "day_ahead" gives offers the same forecast of D that dispatch uses,
                       which needs all of D-1 - ten hours past the 14:00 bid deadline.
                       "bid_time" gives them the early forecast: only what existed at the
                       deadline
    forecast_vintages: dispatch plans tomorrow on the early forecast and stops after it,
                       rather than reading forecasts for tomorrow and the day after that
                       need data still to come; see revenue_stack.run_dispatch
    early_predictions: for ml with either of the above, a walk-forward table built with
                       information_lag_days=2. For naive the early forecast is D-2's prices
    dynamic_shrink  : how to set the weight per day instead of holding price_shrink
                       constant, walk-forward from the same early forecasts the offers see
                       (src/analysis/shrink.py). "slope" fits the Mincer-Zarnowitz scaling;
                       "tilt" leans the weight on how loud the day looks, by shrink_tilt
                       per unit of amplitude percentile, around price_shrink. price_shrink
                       is also the fallback until there is enough history
    shrink_risk_factor: multiplies every fitted weight, for the caution an asymmetric
                       decision cost calls for. Tune it on selection folds only

    Returns
    -------
    dict with the same keys as revenue_stack.run_backtest()
    """
    from src.analysis.revenue_stack import _apx_by_date, _in_range, run_strategy

    if strategy not in ("naive", "ml"):
        raise ValueError(f"Unknown strategy '{strategy}'")

    forecast_prices_by_date: dict = {}
    if include_arbitrage and strategy == "ml" and predictions is not None:
        forecast_prices_by_date = forecast_series_by_date(predictions, start_date, end_date)
    elif include_arbitrage:
        apx_by_date = _apx_by_date(market_index)
        for date in sorted(d for d in apx_by_date if _in_range(d, start_date, end_date)):
            if strategy == "naive":
                fp = naive_day_prices(market_index, date)
            else:
                fp = predict_day_prices(model, feature_cols, feature_df, date)
            if not fp.empty:
                forecast_prices_by_date[date] = fp

    if offer_information not in ("day_ahead", "bid_time"):
        raise ValueError(f"Unknown offer_information '{offer_information}'")
    early_forecast = None
    if include_arbitrage and (offer_information == "bid_time" or forecast_vintages):
        if strategy == "ml":
            if early_predictions is None:
                raise ValueError("bid-time offers or forecast vintages with strategy='ml' need early_predictions")
            early_forecast = forecast_series_by_date(early_predictions, start_date, end_date)
        else:
            early_forecast = {}
            apx_by_date = _apx_by_date(market_index)
            for date in sorted(d for d in apx_by_date if _in_range(d, start_date, end_date)):
                fp = naive_day_prices(market_index, date, days_back=2)
                if not fp.empty:
                    early_forecast[date] = fp

    shrink_by_date = None
    if include_arbitrage and dynamic_shrink:
        from src.analysis.shrink import naive_predictions, tilted_weights, walk_forward_slopes

        # Calibrate whichever forecast the offers actually see, so the weight measures
        # the belief owed to that forecast rather than to a sharper one
        mode = "slope" if dynamic_shrink is True else str(dynamic_shrink)
        if mode not in ("slope", "tilt"):
            raise ValueError(f"dynamic_shrink must be 'slope' or 'tilt', got {dynamic_shrink!r}")
        bid_time = offer_information == "bid_time"
        if strategy == "ml":
            source = early_predictions if bid_time else predictions
            if source is None:
                raise ValueError("dynamic_shrink with strategy='ml' needs a prediction table")
        else:
            source = naive_predictions(market_index, days_back=2 if bid_time else 1)
        if mode == "slope":
            shrink_by_date = walk_forward_slopes(
                source, market_index, risk_factor=shrink_risk_factor, fallback=price_shrink)[0]
        else:
            shrink_by_date = tilted_weights(source, market_index, intercept=price_shrink,
                                            tilt=shrink_tilt)
        shrink_by_date = {d: w for d, w in shrink_by_date.items() if _in_range(d, start_date, end_date)}

    return run_strategy(
        auctions, market_index, battery, forecast_prices_by_date, services, start_date, end_date,
        initial_soc_frac=initial_soc_frac, horizon=horizon,
        include_arbitrage=include_arbitrage, pre_eac_rule=pre_eac_rule, delivery=delivery,
        offer_valuation=offer_valuation, price_shrink=price_shrink,
        early_forecast_prices_by_date=early_forecast,
        offers_at_bid_time=offer_information == "bid_time",
        forecast_vintages=forecast_vintages,
        price_shrink_by_date=shrink_by_date,
    )
