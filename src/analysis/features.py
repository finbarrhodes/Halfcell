"""
Electricity Price Forecasting — Feature Engineering
=====================================================
Loads input data and constructs the feature matrix used by all price
forecasting models.

Public API
----------
load_bess_capacity()        : Load monthly GB BESS fleet capacity from REPD.
build_feature_matrix()      : Construct the (settlementDate, settlementPeriod)
                              feature matrix with target column ``apx_price``.
_build_lear_extra_features(): Build the wide-form lag and DoW features
                              required by the LEAR / DNN models.
FEATURE_COLS                : Ordered list of base feature column names used
                              by tree-based models (RF, LightGBM, XGBoost).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
# Feature matrix builder
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
# LEAR-specific feature builder
# ---------------------------------------------------------------------------

def _build_lear_extra_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the wide-form lag price features and day-of-week dummies required
    by the LEAR model, following the formulation in Lago et al. (2021).

    For each lag day (D-1, D-2, D-7), all 48 settlement-period prices from
    that day are pivoted into individual columns, giving the LEAR model for
    each period the full shape of prior days' price curves (not just summary
    statistics). This produces 144 wide price columns (48 SPs × 3 lags).

    Seven binary day-of-week dummies (dow_0..dow_6) are also included.
    Linear models require explicit dummies to learn per-day level shifts;
    sin/cos encodings (used by the tree-based models) are insufficient for
    a linear functional form.

    Returns a DataFrame keyed on (settlementDate, settlementPeriod) with
    the same row order as feature_df. NaN values (early series without full
    lag history) are left for the caller to fillna(0) when constructing X.
    """
    apx = (
        feature_df[["settlementDate", "settlementPeriod", "apx_price"]]
        .drop_duplicates(subset=["settlementDate", "settlementPeriod"])
        .copy()
    )

    # Daily × SP wide price matrix: one row per date, 48 SP price columns
    price_wide = apx.pivot_table(
        index="settlementDate",
        columns="settlementPeriod",
        values="apx_price",
    )
    price_wide.columns = [f"d_sp_{int(c)}" for c in price_wide.columns]
    price_wide = price_wide.reset_index()

    # Shift by 1, 2, 7 days so each row's features are strictly prior to settlementDate
    base = feature_df[["settlementDate", "settlementPeriod"]].copy()
    for lag_days in [1, 2, 7]:
        lag_df = price_wide.copy()
        lag_df["settlementDate"] = lag_df["settlementDate"] + pd.Timedelta(days=lag_days)
        lag_df = lag_df.rename(columns={
            c: f"lear_lag{lag_days}d_{c}"
            for c in lag_df.columns
            if c.startswith("d_sp_")
        })
        base = base.merge(lag_df, on="settlementDate", how="left")

    # Day-of-week binary dummies (0 = Monday, 6 = Sunday)
    dow = base["settlementDate"].dt.dayofweek
    for i in range(7):
        base[f"dow_{i}"] = (dow == i).astype(int)

    return base


# ---------------------------------------------------------------------------
# Base feature column list (tree-based models)
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
