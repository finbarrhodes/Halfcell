"""
Price Forecast & Forecast-Driven Dispatch
==========================================
Implements three dispatch strategies for the BESS revenue backtester:

  1. Perfect Foresight — actual day-D prices fed to the optimizer (revenue ceiling).
     Already handled by revenue_stack.py; not repeated here.

  2. Naive baseline — uses actual day D-1 prices as the forecast for day D.
     No ML required; sets the "zero skill" floor.

  3. ML model — trains a Random Forest or XGBoost regressor on features available
     at end of day D-1 (lagged prices, generation mix, cyclical temporal encodings)
     with a strict temporal train/test split.

All three strategies share the same dispatch logic: given a price forecast for day D,
pick the N cheapest periods to charge and N most expensive to discharge; then realise
revenue against the actual day-D prices.

New module kept separate from revenue_stack.py so the perfect-foresight backtester
remains a clean, standalone baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Log-transform model wrapper
# ---------------------------------------------------------------------------

class _LogTransformModel:
    """
    Wraps a sklearn/xgboost/lightgbm estimator with a signed-log1p target
    transform so that fit() and predict() both operate in price space.

    Signed-log1p handles negative prices gracefully:
        transform  : sign(y) * log1p(|y|)
        inverse    : sign(p) * expm1(|p|)
    """

    def __init__(self, base_model):
        self._model = base_model
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X, y):
        y_log = np.sign(y) * np.log1p(np.abs(y))
        self._model.fit(X, y_log)
        self.feature_importances_ = self._model.feature_importances_
        return self

    def predict(self, X):
        pred_log = self._model.predict(X)
        return np.sign(pred_log) * np.expm1(np.abs(pred_log))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default temporal split: everything before this date is training data.
# ~72 months train (Jan 2019–Feb 2025), ~12 months test (Mar 2025–Feb 2026).
DEFAULT_TEST_START = "2025-03-01"


# ---------------------------------------------------------------------------
# BESS fleet capacity loader
# ---------------------------------------------------------------------------

def load_bess_capacity(path: str | None = None) -> pd.DataFrame:
    """
    Load the monthly GB BESS fleet capacity series from
    data/processed/bess_fleet_capacity.parquet.

    Returns a DataFrame with columns:
        month_start (datetime64[ns]) — first day of each month
        bess_fleet_mw (float)       — cumulative operational capacity (MW)

    Sorted ascending. If the parquet file does not exist (e.g. REPD data has
    not yet been collected), returns an empty DataFrame so callers degrade
    gracefully.
    """
    from pathlib import Path as _Path
    if path is None:
        _candidate = _Path(__file__).parent.parent.parent / "data" / "processed" / "bess_fleet_capacity.parquet"
    else:
        _candidate = _Path(path)

    if not _candidate.exists():
        return pd.DataFrame(columns=["month_start", "bess_fleet_mw"])

    df = pd.read_parquet(_candidate)
    # The parquet stores month as a first-of-month timestamp in the "month" column.
    df = df.rename(columns={"month": "month_start"})
    df["month_start"] = pd.to_datetime(df["month_start"]).dt.normalize()
    return df[["month_start", "bess_fleet_mw"]].sort_values("month_start").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_feature_matrix(
    market_index: pd.DataFrame,
    generation_daily: pd.DataFrame,
    bess_capacity: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Construct a feature matrix for price forecasting.

    Each row represents one (settlementDate, settlementPeriod) observation.
    Target column ``apx_price`` is the APXMIDP price for that period.
    All feature columns use only information available at the end of day D-1,
    ensuring no look-ahead bias when training or backtesting.

    Parameters
    ----------
    market_index : DataFrame
        From data/processed/market_index.parquet. Must contain columns:
        settlementDate, settlementPeriod, dataProvider, price.
    generation_daily : DataFrame
        From data/processed/generation_daily.parquet. Must contain columns:
        settlementDate, fuelGroup, generation.
    bess_capacity : DataFrame, optional
        From load_bess_capacity(). Must contain columns:
        month_start (datetime64, first of month), bess_fleet_mw (float).
        When provided, adds ``bess_fleet_mw`` and ``bess_spread_suppression``
        feature columns. When None, those columns are simply absent (the
        dynamic column filter in train_forecast_model handles this gracefully).

    Returns
    -------
    DataFrame with columns:
        settlementDate, settlementPeriod, apx_price  (target),
        + feature columns described below.
    Rows with NaN features (due to lag/rolling windows at the start of the
    series) are dropped.
    """
    # --- Base: APXMIDP prices only ---
    apx = (
        market_index[market_index["dataProvider"] == "APXMIDP"]
        [["settlementDate", "settlementPeriod", "price"]]
        .copy()
        .rename(columns={"price": "apx_price"})
    )
    apx["settlementDate"] = pd.to_datetime(apx["settlementDate"]).dt.normalize()
    apx = apx.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)

    # --- Same-period lagged prices ---
    # Shift by D days within each settlement period group so that day D's feature
    # is the price at that exact same period D days ago.
    apx = apx.sort_values(["settlementPeriod", "settlementDate"])
    for lag_days in [1, 2, 7, 14]:
        apx[f"apx_lag_{lag_days}d"] = (
            apx.groupby("settlementPeriod")["apx_price"]
            .shift(lag_days)
        )

    # --- Previous-day aggregate statistics ---
    # For day D, we want stats computed from all 48 periods of day D-1.
    daily_stats = (
        apx.groupby("settlementDate")["apx_price"]
        .agg(daily_mean="mean", daily_std="std", daily_max="max", daily_min="min")
        .reset_index()
        .sort_values("settlementDate")
    )
    # 7-day rolling mean of daily averages (captures medium-term price level)
    daily_stats["rolling_7d_mean"] = (
        daily_stats["daily_mean"].rolling(7, min_periods=7).mean()
    )
    # Shift by 1 day so day D gets D-1's stats
    daily_stats_lag = daily_stats.copy()
    daily_stats_lag["settlementDate"] = daily_stats_lag["settlementDate"] + pd.Timedelta(days=1)
    daily_stats_lag = daily_stats_lag.rename(columns={
        "daily_mean": "prev_day_mean",
        "daily_std":  "prev_day_std",
        "daily_max":  "prev_day_max",
        "daily_min":  "prev_day_min",
        # rolling_7d_mean column name unchanged
    })
    apx = apx.merge(daily_stats_lag, on="settlementDate", how="left")

    # --- Generation mix features (daily, from day D-1) ---
    gen = generation_daily.copy()
    gen["settlementDate"] = pd.to_datetime(gen["settlementDate"]).dt.normalize()

    # Pivot fuel groups into columns
    gen_wide = gen.pivot_table(
        index="settlementDate",
        columns="fuelGroup",
        values="generation",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    gen_wide.columns.name = None
    # Normalise column names
    gen_wide.columns = [
        "settlementDate" if c == "settlementDate"
        else f"gen_{c.lower().replace(' ', '_')}"
        for c in gen_wide.columns
    ]

    # Derived generation features
    renewable_cols = [c for c in gen_wide.columns if any(
        fuel in c for fuel in ["wind", "hydro", "biomass", "solar"]
    )]
    fossil_cols = [c for c in gen_wide.columns if any(
        fuel in c for fuel in ["gas", "coal", "oil"]
    )]
    all_gen_cols = [c for c in gen_wide.columns if c.startswith("gen_")]
    gen_wide["gen_total"] = gen_wide[all_gen_cols].sum(axis=1)
    gen_wide["gen_renewable_frac"] = (
        gen_wide[[c for c in renewable_cols if c in gen_wide.columns]].sum(axis=1)
        / gen_wide["gen_total"].replace(0, np.nan)
    )
    gen_wide["gen_fossil_frac"] = (
        gen_wide[[c for c in fossil_cols if c in gen_wide.columns]].sum(axis=1)
        / gen_wide["gen_total"].replace(0, np.nan)
    )

    # Shift generation by 1 day: day D gets D-1 generation (available at end-of-D-1)
    gen_wide_lag = gen_wide.copy()
    gen_wide_lag["settlementDate"] = gen_wide_lag["settlementDate"] + pd.Timedelta(days=1)

    apx = apx.merge(gen_wide_lag, on="settlementDate", how="left")

    # --- GB BESS fleet capacity features (monthly, from REPD, D-1 lagged) ---
    # Each row gets the fleet capacity for the month of (settlementDate - 1 day),
    # which is strictly prior to the settlement date — no look-ahead.
    if bess_capacity is not None and not bess_capacity.empty:
        bess = bess_capacity.copy()
        bess["month_start"] = pd.to_datetime(bess["month_start"]).dt.normalize()

        apx["_feature_month"] = (
            (apx["settlementDate"] - pd.Timedelta(days=1))
            .dt.to_period("M")
            .dt.to_timestamp()
        )
        apx = apx.merge(
            bess.rename(columns={"month_start": "_feature_month"}),
            on="_feature_month",
            how="left",
        ).drop(columns=["_feature_month"])

        # Periods before the first REPD entry (pre-2019 or data gap) → 0 MW
        apx["bess_fleet_mw"] = apx["bess_fleet_mw"].fillna(0.0)

        # Analytical suppression feature: BESS penetration as a fraction of total
        # system generation. Encodes the economic mechanism — as fleet grows
        # relative to system size, the merit order flattens and spreads compress.
        apx["bess_spread_suppression"] = (
            apx["bess_fleet_mw"]
            / apx["gen_total"].replace(0, np.nan)
        ).fillna(0.0)

    # --- Cyclical temporal features ---
    sp = apx["settlementPeriod"]
    apx["sp_sin"] = np.sin(2 * np.pi * sp / 48)
    apx["sp_cos"] = np.cos(2 * np.pi * sp / 48)

    dow = apx["settlementDate"].dt.dayofweek  # 0=Mon, 6=Sun
    apx["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    apx["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    doy = apx["settlementDate"].dt.dayofyear
    year_len = apx["settlementDate"].dt.is_leap_year.map({True: 366, False: 365})
    apx["doy_sin"] = np.sin(2 * np.pi * doy / year_len)
    apx["doy_cos"] = np.cos(2 * np.pi * doy / year_len)

    apx["is_weekend"] = (dow >= 5).astype(int)

    # --- UK bank holiday flag ---
    # Price profiles on bank holidays closely resemble Sundays.
    try:
        import holidays as _holidays
        _uk_hols = _holidays.country_holidays("GB", subdiv="ENG")
        apx["is_bank_holiday"] = apx["settlementDate"].apply(
            lambda d: int(d in _uk_hols)
        )
    except ImportError:
        apx["is_bank_holiday"] = 0

    # --- Drop rows where any lag feature is missing ---
    lag_cols = [c for c in apx.columns if "lag" in c or "prev_day" in c or c == "rolling_7d_mean"]
    apx = apx.dropna(subset=lag_cols).reset_index(drop=True)

    return apx


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Same-period lags
    "apx_lag_1d", "apx_lag_2d", "apx_lag_7d", "apx_lag_14d",
    # Previous-day aggregate + medium-term rolling level
    "prev_day_mean", "prev_day_std", "prev_day_max", "prev_day_min",
    "rolling_7d_mean",
    # Generation mix
    "gen_total", "gen_renewable_frac", "gen_fossil_frac",
    # Temporal
    "sp_sin", "sp_cos",
    "dow_sin", "dow_cos",
    "doy_sin", "doy_cos",
    "is_weekend", "is_bank_holiday",
    # GB BESS fleet capacity (structural market variable, from REPD)
    "bess_fleet_mw", "bess_spread_suppression",
]

# Include any individual fuel-group columns that were built (gas, wind, etc.)
# These are appended dynamically in train_forecast_model.


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
    model_type  : "rf" (Random Forest) or "xgb" (XGBoost)
    test_start  : ISO date string — all rows on or after this date form the test set

    Returns
    -------
    (model, feature_cols, train_metrics, test_metrics) where:
      model         : fitted sklearn / xgboost estimator
      feature_cols  : list of column names used as features
      train_metrics : dict {rmse, mae, n_samples}
      test_metrics  : dict {rmse, mae, n_samples}
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

    # Wrap with signed-log transform: fit/predict both operate in price space
    model = _LogTransformModel(_build_model(model_type))
    model.fit(X_train, y_train)

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


def _build_model(model_type: str):
    if model_type == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=300,
            max_features=0.5,       # outperforms "sqrt" with correlated lag features
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        )
    elif model_type == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,     # reduces overfitting on ~20-month training set
            reg_alpha=0.05,         # L1 regularisation
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        )
    elif model_type == "lgb":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'rf', 'xgb', or 'lgb'.")


def get_feature_importances(model, feature_cols: list) -> pd.Series:
    """Return feature importances as a named Series, sorted descending."""
    return (
        pd.Series(model.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
    )


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
    day_df = feature_df[feature_df["settlementDate"] == pd.Timestamp(target_date)]
    if day_df.empty:
        return pd.Series(dtype=float)

    X = day_df[feature_cols].fillna(0)
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
    import numpy as np

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
    # Shared setup: actual prices flat lookup and period list
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
