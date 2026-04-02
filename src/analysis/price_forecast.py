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

All three strategies share the same dispatch logic: given a price forecast for day D,
pick the N cheapest periods to charge and N most expensive to discharge; then realise
revenue against the actual day-D prices.

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
    _AsinhTransformModel,
    _LEARCalibratedEnsemble,
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
# LEAR calibration window definitions (Lago et al., 2021)
# ---------------------------------------------------------------------------

# (label, weeks) pairs; None = full training history.
# Short windows adapt to recent regime shifts (BESS fleet growth, price
# suppression); long windows provide stable baselines. Simple averaging of
# all window predictions consistently outperforms any single window.
LEAR_CALIBRATION_WINDOWS: list[tuple[str | None, int | None]] = [
    ("8w",  8),
    ("26w", 26),
    ("52w", 52),
    ("2yr", 104),
    ("3yr", 156),
    (None,  None),   # full history
]

# Skip any window whose training slice falls below this row count.
# _LEARModel silently omits settlement periods with < 30 rows; below this
# total threshold enough periods are skipped to degrade the ensemble average.
_LEAR_MIN_TRAIN_ROWS: int = 48 * 30  # 1 440


# ---------------------------------------------------------------------------
# Shared metrics helper
# ---------------------------------------------------------------------------

def _compute_metrics(model, X, y) -> dict:
    """Compute RMSE, MAE, Spearman ρ, and Spike-RMSE for model predictions."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from scipy.stats import spearmanr

    pred  = model.predict(X)
    y_arr = np.asarray(y)
    rmse  = float(np.sqrt(mean_squared_error(y_arr, pred)))
    mae   = float(mean_absolute_error(y_arr, pred))

    sp_result = spearmanr(y_arr, pred)
    sp_corr   = float(
        sp_result.statistic if hasattr(sp_result, "statistic") else sp_result.correlation
    )

    threshold  = float(np.percentile(y_arr, 90))
    spike_mask = y_arr >= threshold
    spike_rmse = (
        float(np.sqrt(mean_squared_error(y_arr[spike_mask], pred[spike_mask])))
        if spike_mask.sum() >= 10 else None
    )

    m = {
        "rmse":      round(rmse, 2),
        "mae":       round(mae, 2),
        "spearman":  round(sp_corr, 3),
        "n_samples": len(y_arr),
    }
    if spike_rmse is not None:
        m["spike_rmse"] = round(spike_rmse, 2)
    return m


# ---------------------------------------------------------------------------
# LEAR calibration-window training
# ---------------------------------------------------------------------------

def _train_lear_calibrated(
    feature_df: pd.DataFrame,
    feature_cols: list,
    test_start: str,
    _lear_extra: pd.DataFrame,
    _extra_cols: list,
) -> tuple:
    """
    Train a LEAR calibration-window ensemble following Lago et al. (2021).

    Trains one _AsinhTransformModel(_LEARModel()) per window in
    LEAR_CALIBRATION_WINDOWS, skipping any window below _LEAR_MIN_TRAIN_ROWS.
    Returns a _LEARCalibratedEnsemble whose predictions are the simple average
    across all valid windows.

    Returns
    -------
    (ensemble, train_metrics, test_metrics)
    """
    test_ts = pd.Timestamp(test_start)
    test    = feature_df[feature_df["settlementDate"] >= test_ts]

    # --- Option A: drop same-period lag columns that are strictly redundant for LEAR.
    # apx_lag_1d/2d/7d/14d each equal the row's own wide-lag column (lear_lag{n}d_d_sp_{sp})
    # so they add zero information but worsen the Gram matrix condition number.
    _LEAR_REDUNDANT_LAGS = {"apx_lag_1d", "apx_lag_2d", "apx_lag_7d", "apx_lag_14d"}
    feature_cols = [c for c in feature_cols if c not in _LEAR_REDUNDANT_LAGS]

    # Build X_test once — identical for all windows
    _test_extra = (
        test[["settlementDate", "settlementPeriod"]]
        .merge(_lear_extra, on=["settlementDate", "settlementPeriod"], how="left")
    )
    X_test = pd.concat(
        [test[feature_cols].fillna(0).reset_index(drop=True),
         _test_extra[_extra_cols].reset_index(drop=True)],
        axis=1,
    ).fillna(0)
    X_test["settlementPeriod"] = test["settlementPeriod"].values
    y_test = test["apx_price"]

    constituents: list = []
    X_train_full: pd.DataFrame | None = None
    y_train_full: pd.Series   | None = None

    for _label, weeks in LEAR_CALIBRATION_WINDOWS:
        window_start = (
            test_ts - pd.Timedelta(weeks=weeks)
            if weeks is not None else pd.Timestamp("1900-01-01")
        )
        train = feature_df[
            (feature_df["settlementDate"] >= window_start) &
            (feature_df["settlementDate"] < test_ts)
        ]
        if len(train) < _LEAR_MIN_TRAIN_ROWS:
            continue

        _train_extra = (
            train[["settlementDate", "settlementPeriod"]]
            .merge(_lear_extra, on=["settlementDate", "settlementPeriod"], how="left")
        )
        X_win = pd.concat(
            [train[feature_cols].fillna(0).reset_index(drop=True),
             _train_extra[_extra_cols].reset_index(drop=True)],
            axis=1,
        ).fillna(0)
        X_win["settlementPeriod"] = train["settlementPeriod"].values
        y_win = train["apx_price"]

        m = _AsinhTransformModel(_build_model("lear"))
        m.fit(X_win, y_win)
        # Skip windows where too few SP models were fitted — this happens when
        # n_samples_per_SP <= n_features (short windows with wide lag matrix).
        # A constituent with fewer than 24 fitted models would predict zero for
        # the missing periods and degrade the ensemble average.
        if len(m._model._models) < 24:
            continue
        m._model._lear_extra_df = _lear_extra   # shared reference, no copies
        constituents.append(m)

        if weeks is None:   # full-history window — use for train metrics
            X_train_full = X_win
            y_train_full = y_win

    if not constituents:
        raise RuntimeError(
            "No valid LEAR calibration windows — all windows below "
            f"the minimum row threshold of {_LEAR_MIN_TRAIN_ROWS}."
        )

    ensemble     = _LEARCalibratedEnsemble(constituents)
    train_metrics = (
        _compute_metrics(ensemble, X_train_full, y_train_full)
        if X_train_full is not None else {}
    )
    test_metrics = _compute_metrics(ensemble, X_test, y_test)

    return ensemble, train_metrics, test_metrics


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
      model         : fitted _AsinhTransformModel wrapping the base estimator
      feature_cols  : list of column names used as features
      train_metrics : dict {rmse, mae, spearman, n_samples[, spike_rmse]}
      test_metrics  : dict {rmse, mae, spearman, n_samples[, spike_rmse]}
    """
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
        _lear_extra = _build_lear_extra_features(feature_df)
        _extra_cols = [c for c in _lear_extra.columns
                       if c not in ("settlementDate", "settlementPeriod")]

        # LEAR: delegate entirely to the calibration-window ensemble helper.
        # All window slicing, X assembly, fitting, and metrics are handled there.
        if model_type == "lear":
            ensemble, train_metrics, test_metrics = _train_lear_calibrated(
                feature_df, feature_cols, test_start, _lear_extra, _extra_cols
            )
            return ensemble, feature_cols, train_metrics, test_metrics

        # DNN: augment X_train / X_test with the wide lag features
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

    # Wrap with arcsinh transform: fit/predict both operate in price space
    model = _AsinhTransformModel(_build_model(model_type))
    model.fit(X_train, y_train)

    # DNN: cache extra features on the model so predict_day_prices can
    # retrieve the wide lag columns without rebuilding them on every call
    if model_type == "dnn" and _lear_extra is not None:
        model._model._lear_extra_df = _lear_extra

    train_metrics = _compute_metrics(model, X_train, y_train)
    test_metrics  = _compute_metrics(model, X_test,  y_test)

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

def _dispatch_day(
    forecast_prices: pd.Series,
    actual_prices: pd.Series,
    n_periods: int,
    energy_out: float,
    energy_in: float,
    cycling_cost_per_mwh: float,
) -> dict | None:
    """
    Given a price forecast and actual prices for a single day:
    - Use forecast to identify which periods to charge / discharge
    - Realise revenue against actual prices

    Returns a dict {imbalance_revenue_gbp, cycling_cost_gbp, mwh_cycled}
    or None if the trade is not executed (insufficient data or not profitable).
    """
    if len(forecast_prices) < n_periods * 2 or len(actual_prices) < n_periods * 2:
        return None

    # Use FORECAST to rank periods
    charge_periods    = forecast_prices.nsmallest(n_periods).index
    discharge_periods = forecast_prices.nlargest(n_periods).index

    # Realise revenue against ACTUAL prices
    avg_charge    = actual_prices.reindex(charge_periods).dropna().mean()
    avg_discharge = actual_prices.reindex(discharge_periods).dropna().mean()

    if pd.isna(avg_charge) or pd.isna(avg_discharge):
        return None

    gross_profit = avg_discharge * energy_out - avg_charge * energy_in
    cycling_wear = cycling_cost_per_mwh * energy_out

    # Only execute if forecast-implied schedule is realised-profitable
    if gross_profit <= cycling_wear:
        return None

    return {
        "imbalance_revenue_gbp": gross_profit,
        "cycling_cost_gbp":      cycling_wear,
        "mwh_cycled":            energy_out,
    }


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
    dispatch_method: str = "greedy",
    horizon: int = 96,
) -> dict:
    """
    Run a forecast-driven revenue backtest for either the 'naive' or 'ml' strategy.

    Uses a two-pass approach:
      1. Collect forecast prices for all days in the period.
      2. Run compute_daily_fr_schedule() to determine per-EFA-block FR/arb allocation:
         for each block, compare the confirmed FR clearing price against a shadow arb
         estimate from the forecast prices for that block's 8 settlement periods.
      3. Dispatch within the allocated arb_mw for each EFA block, realising revenue
         against actual prices. SoC is tracked across all 6 blocks per day and
         carried into the next day.

    Parameters
    ----------
    strategy         : "naive" or "ml"
    market_index     : DataFrame from load_market_index()
    auctions         : DataFrame from load_auctions()
    battery          : BatterySpec instance
    services         : list of service codes to include
    start_date       : inclusive backtest start
    end_date         : inclusive backtest end
    model            : fitted model object (required for strategy="ml")
    feature_df       : feature matrix from build_feature_matrix() (required for strategy="ml")
    feature_cols     : feature column list from train_forecast_model() (required for strategy="ml")
    initial_soc_frac : float — starting SoC as fraction of energy_mwh (default 0.5)
    dispatch_method  : "greedy" (default) or "mpc". When "mpc", re-solves LP every
                       30 minutes over a rolling horizon using forecast prices for
                       planning and actual prices for revenue settlement.
    horizon          : MPC planning horizon in settlement periods (default 96 = 48h).

    Returns
    -------
    dict with keys:
      'monthly'        : wide-format DataFrame, same schema as revenue_stack.run_backtest()
      'summary'        : dict of aggregate stats
      'soc_trajectory' : DataFrame [date, sp, soc_frac] when dispatch_method="mpc", else None
    """
    from src.analysis.revenue_stack import (
        calc_ancillary_revenue,
        compute_daily_fr_schedule,
        EFA_PERIODS,
        FR_SOC_LOWER,
        FR_SOC_UPPER,
        _efa_prices,
        _build_arb_mw_by_period,
        _run_mpc_dispatch,
        _build_result,
        ALL_SERVICES,
    )

    if services is None:
        services = ALL_SERVICES

    # Build full apx_by_date lookup (unfiltered) so EFA 1 D-1 lookups work on first day
    apx_all = market_index[market_index["dataProvider"] == "APXMIDP"].copy()
    apx_all["settlementDate"] = pd.to_datetime(apx_all["settlementDate"]).dt.normalize()
    apx_all = apx_all[apx_all["settlementPeriod"] <= 48]
    apx_by_date = {
        date: grp.set_index("settlementPeriod")["price"]
        for date, grp in apx_all.groupby("settlementDate")
    }

    sd = pd.Timestamp(start_date) if start_date else apx_all["settlementDate"].min()
    ed = pd.Timestamp(end_date)   if end_date   else apx_all["settlementDate"].max()
    sorted_dates = sorted(
        d for d in apx_by_date if sd <= d <= ed
    )

    # --- First pass: collect forecast prices for all days in range ---
    forecast_prices_by_date: dict = {}
    for date in sorted_dates:
        if strategy == "naive":
            fp = naive_day_prices(market_index, date)
        elif strategy == "ml":
            fp = predict_day_prices(model, feature_cols, feature_df, date)
        else:
            raise ValueError(f"Unknown strategy '{strategy}'")
        if not fp.empty:
            forecast_prices_by_date[date] = fp

    # --- Per-EFA-block capacity allocation ---
    # For each block: confirmed FR clearing price vs forecast-based shadow arb estimate.
    fr_schedule = compute_daily_fr_schedule(
        auctions, forecast_prices_by_date, battery, services, start_date, end_date
    )
    # arb_sched_map: {(date, efa): arb_mw}
    arb_sched_map = {
        (pd.Timestamp(d).normalize(), int(e)): battery.power_mw - float(v)
        for (d, e), v in fr_schedule.items()
    }
    avg_fr_mw  = float(fr_schedule.mean()) if len(fr_schedule) > 0 else battery.power_mw
    avg_arb_mw = battery.power_mw - avg_fr_mw

    # --- Ancillary revenue (scaled by per-EFA-block fr_mw) ---
    anc = calc_ancillary_revenue(
        auctions, battery, services, start_date, end_date, fr_schedule=fr_schedule,
    )
    if not anc.empty:
        anc_wide = anc.pivot_table(
            index="month", columns="service", values="revenue_gbp", fill_value=0
        )
        anc_wide.columns = [f"{c}_rev" for c in anc_wide.columns]
    else:
        anc_wide = pd.DataFrame()

    # --- Second pass: dispatch with SoC tracking ---
    actual_by_period = {
        (d, int(sp)): float(price)
        for d, series in apx_by_date.items()
        for sp, price in series.items()
    }
    all_periods = [(date, sp) for date in sorted_dates for sp in range(1, 49)]

    soc_traj = []

    if dispatch_method == "mpc":
        # Rolling MPC: re-solves LP every 30 min using forecast prices for planning,
        # actual prices for settlement. Forecast EFA 1 D-1 periods are absent from
        # forecast_by_period — LP treats them as zero-revenue (conservative).
        arb_mw_by_period = _build_arb_mw_by_period(arb_sched_map)
        forecast_by_period = {
            (date, int(sp)): float(price)
            for date, fp_series in forecast_prices_by_date.items()
            for sp, price in fp_series.items()
        }
        arb_rows, soc_traj = _run_mpc_dispatch(
            all_periods=all_periods,
            actual_by_period=actual_by_period,
            forecast_by_period=forecast_by_period,
            arb_mw_by_period=arb_mw_by_period,
            battery=battery,
            initial_soc_frac=initial_soc_frac,
            horizon=horizon,
        )

    else:
        # Greedy EFA-block dispatch (original)
        n_periods = max(1, int(battery.duration_h * 2))
        soc       = initial_soc_frac * battery.energy_mwh
        soc_min   = FR_SOC_LOWER * battery.energy_mwh
        soc_max   = FR_SOC_UPPER * battery.energy_mwh
        arb_rows  = []

        for date in sorted_dates:
            fp_day = forecast_prices_by_date.get(date)
            if fp_day is None:
                continue
            fp_by_date = {date: fp_day}

            for efa in range(1, 7):
                actual_efa   = _efa_prices(apx_by_date, date, efa)
                forecast_efa = _efa_prices(fp_by_date, date, efa)

                arb_mw_d = arb_sched_map.get((date, efa), 0.0)
                if arb_mw_d <= 0:
                    continue

                nominal_energy_out = arb_mw_d * battery.duration_h
                energy_out = min(nominal_energy_out, max(0.0, soc - soc_min))
                energy_in  = min(
                    energy_out / battery.efficiency_rt,
                    max(0.0, soc_max - soc),
                )
                if energy_out <= 0 or energy_in <= 0:
                    continue

                result_d = _dispatch_day(
                    forecast_efa, actual_efa,
                    n_periods, energy_out, energy_in,
                    battery.cycling_cost_per_mwh,
                )
                if result_d:
                    soc = soc - result_d["mwh_cycled"] + result_d["mwh_cycled"] / battery.efficiency_rt
                    soc = float(np.clip(soc, 0, battery.energy_mwh))
                    result_d["date"] = date
                    arb_rows.append(result_d)

    if arb_rows:
        daily_arb = pd.DataFrame(arb_rows)
        daily_arb["month"] = pd.to_datetime(daily_arb["date"]).dt.to_period("M")
        imb_wide = (
            daily_arb.groupby("month")
            .agg(
                imbalance_revenue_gbp=("imbalance_revenue_gbp", "sum"),
                cycling_cost_gbp=("cycling_cost_gbp", "sum"),
                mwh_cycled=("mwh_cycled", "sum"),
            )
            .reset_index()
            .set_index("month")
        )
    else:
        imb_wide = pd.DataFrame()

    # --- Merge streams, apply availability factor, compute summary ---
    return _build_result(anc_wide, imb_wide, battery, avg_fr_mw, avg_arb_mw, soc_traj)
