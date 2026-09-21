"""
src/analysis/intervals.py
=========================
Conformal prediction intervals for the day-ahead price forecast, and the guard
bands the offer plan trades against.

Why intervals rather than another weight
----------------------------------------
The plan used to pull the forecast's shape towards the day's mean by a constant,
and two attempts to set that constant per day both lost money (2026-09-21): the
weight was calibrated for squared error, and it discounted the loudest days,
which are the days whose spread is real. Revenue turned out to be sensitive to
how much trading value is discounted overall and almost flat in which *days* are
discounted.

What was never tested is which *hours*, and that is where forecast error
genuinely varies. So instead of scaling the shape, this module measures how wrong
the forecast tends to be at each time of day and hands the plan two price paths:
one to sell against and one to buy against. The plan then only commits capacity
to trades that survive its own uncertainty, which is the "guard band" idea
Matsumoto & Sasanuma use for battery arbitrage, and the deterministic equivalent
of a robust objective.

The method
----------
Split conformal prediction (Papadopoulos et al. 2002; Lei et al. 2018): take the
residuals a forecast has already made, and use their empirical quantiles as the
interval around the next one. The finite-sample guarantee comes from the order
statistic, `ceil((n+1)(1-alpha))`, rather than from any assumption about the error
distribution. Bands are one-sided per direction, because selling and buying want
opposite ends and an asymmetric error should not be forced symmetric.

Residuals are grouped before the quantile is taken - Mondrian conformal
prediction, where coverage is conditional on the group rather than only marginal:

  "period" : one calibration set per settlement period, 48 of them. Follows the
             evening peak most closely; each has a few hundred residuals by the
             second year of the backtest.
  "block"  : one per EFA block, pooling eight half-hours. Smoother, eight times
             the data per bin, and it matches the unit the offers are actually
             decided in.
  "day"    : one calibration set for everything, the marginal case.

Everything is walk-forward, on the same quarterly origins as the forecast: a
day's band comes only from residuals of days before its origin, and from the
early (bid-time) forecasts the offers themselves see. `window_days` restricts
calibration to a trailing window, the cheap answer to distribution shift that
EnbPI and adaptive conformal inference handle more carefully; GB's 2021-22 regime
change is the reason it exists here.

What is not done yet: conformalised quantile regression (Romano et al. 2019),
which would make widths depend on the day's features and not only on the hour,
and sequential methods (EnbPI, SPCI) that re-estimate the residual quantiles as
each day lands. Both are the natural next step if the bands pay.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.revenue_stack import _SP_TO_BLOCK

GROUPS = ("period", "block", "day")
# Below this many residuals in a group, fall back to the pooled set: a quantile
# taken from a handful of points is not a guard band, it is an accident.
MIN_CALIBRATION = 60


def group_key(settlement_period: pd.Series, group: str) -> pd.Series:
    """Which calibration set each settlement period belongs to."""
    if group == "period":
        return settlement_period
    if group == "block":
        return settlement_period.map(lambda sp: _SP_TO_BLOCK[int(sp)][1])
    if group == "day":
        return pd.Series(0, index=settlement_period.index)
    raise ValueError(f"group must be one of {GROUPS}, got {group!r}")


def residual_frame(predictions: pd.DataFrame, market_index: pd.DataFrame) -> pd.DataFrame:
    """(settlementDate, settlementPeriod, residual) where residual = actual - forecast."""
    apx = market_index[(market_index["dataProvider"] == "APXMIDP")
                       & (market_index["settlementPeriod"] <= 48)]
    actual = apx.assign(settlementDate=pd.to_datetime(apx["settlementDate"]).dt.normalize())
    forecast = predictions.assign(
        settlementDate=pd.to_datetime(predictions["settlementDate"]).dt.normalize())
    joined = forecast.merge(actual[["settlementDate", "settlementPeriod", "price"]],
                            on=["settlementDate", "settlementPeriod"], how="inner")
    return joined.assign(residual=joined["price"] - joined["prediction"])[
        ["settlementDate", "settlementPeriod", "residual"]].dropna()


def conformal_offsets(residuals: np.ndarray, alpha: float) -> tuple[float, float]:
    """
    The one-sided conformal offsets at level `alpha`, as (low, high).

    The order statistic rather than the plain quantile: with n residuals, the
    `ceil((n+1)(1-alpha))`-th smallest is the smallest value whose coverage holds
    in finite samples. low is at most 0 and high at least 0, so a band never
    turns a sale into a better price than the forecast promised.
    """
    n = len(residuals)
    if not n:
        return 0.0, 0.0
    ordered = np.sort(residuals)
    upper_rank = min(n, int(np.ceil((n + 1) * (1 - alpha)))) - 1
    lower_rank = max(0, int(np.floor((n + 1) * alpha)) - 1)
    return float(min(0.0, ordered[lower_rank])), float(max(0.0, ordered[upper_rank]))


def fit_offsets(residuals: pd.DataFrame, alpha: float, group: str) -> dict:
    """{group key: (low, high)} from a calibration set, with a pooled fallback."""
    pooled = conformal_offsets(residuals["residual"].to_numpy(), alpha)
    keys = group_key(residuals["settlementPeriod"], group)
    offsets = {}
    for key, rows in residuals.groupby(keys.to_numpy())["residual"]:
        offsets[key] = conformal_offsets(rows.to_numpy(), alpha) if len(rows) >= MIN_CALIBRATION else pooled
    offsets["pooled"] = pooled
    return offsets


def walk_forward_bands(
    predictions: pd.DataFrame,
    market_index: pd.DataFrame,
    *,
    alpha: float = 0.2,
    group: str = "period",
    cadence_months: int = 3,
    window_days: int | None = None,
    min_days: int = 120,
) -> tuple[dict, dict, list]:
    """
    Guard bands for every day, calibrated only on residuals from days before it.

    Returns
    -------
    (low_by_date, high_by_date, folds)
        The first two are {date: Series indexed by settlement period} of offsets to
        add to the forecast: `low` (at most zero) for the price a sale is planned
        against, `high` (at least zero) for a purchase. The folds record each
        origin's calibration size and the width it produced, so a run can report
        what it was being careful about.
    """
    if group not in GROUPS:
        raise ValueError(f"group must be one of {GROUPS}, got {group!r}")
    if not 0 < alpha < 0.5:
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")

    residuals = residual_frame(predictions, market_index)
    if residuals.empty:
        return {}, {}, []
    days = pd.DatetimeIndex(sorted(residuals["settlementDate"].unique()))
    periods = pd.Index(sorted(residuals["settlementPeriod"].unique()))
    origins = pd.date_range(days.min().normalize().replace(day=1), days.max(),
                            freq=f"{cadence_months}MS")

    low_by_date, high_by_date, folds = {}, {}, []
    for k, origin in enumerate(origins):
        upto = origins[k + 1] if k + 1 < len(origins) else days.max() + pd.Timedelta(days=1)
        served = days[(days >= origin) & (days < upto)]
        if not len(served):
            continue
        history = residuals[residuals["settlementDate"] < origin]
        if window_days is not None:
            history = history[history["settlementDate"] >= origin - pd.Timedelta(days=window_days)]
        calibration_days = history["settlementDate"].nunique()
        if calibration_days < min_days:
            folds.append({"origin": origin.date().isoformat(), "calibration_days": calibration_days,
                          "banded": False, "days": len(served)})
            continue

        offsets = fit_offsets(history, alpha, group)
        keys = group_key(pd.Series(periods, index=periods), group)
        low = pd.Series([offsets.get(keys[sp], offsets["pooled"])[0] for sp in periods], index=periods)
        high = pd.Series([offsets.get(keys[sp], offsets["pooled"])[1] for sp in periods], index=periods)
        for day in served:
            low_by_date[day], high_by_date[day] = low, high
        folds.append({"origin": origin.date().isoformat(), "calibration_days": calibration_days,
                      "banded": True, "days": len(served),
                      "median_width": round(float((high - low).median()), 1),
                      "width_range": [round(float((high - low).min()), 1),
                                      round(float((high - low).max()), 1)]})
    return low_by_date, high_by_date, folds


def coverage(predictions: pd.DataFrame, market_index: pd.DataFrame, low_by_date: dict,
             high_by_date: dict, group: str = "period") -> pd.DataFrame:
    """
    How often the interval contained the price, and how wide it had to be.

    Reported per group, because marginal coverage can hold while a particular hour
    is badly served - the reason for calibrating per group in the first place.
    """
    residuals = residual_frame(predictions, market_index)
    banded = residuals[residuals["settlementDate"].isin(low_by_date)].copy()
    if banded.empty:
        return pd.DataFrame(columns=["group", "n", "coverage", "median_width"])
    banded["low"] = [low_by_date[d][sp] for d, sp in zip(banded["settlementDate"], banded["settlementPeriod"])]
    banded["high"] = [high_by_date[d][sp] for d, sp in zip(banded["settlementDate"], banded["settlementPeriod"])]
    banded["covered"] = (banded["residual"] >= banded["low"]) & (banded["residual"] <= banded["high"])
    banded["group"] = group_key(banded["settlementPeriod"], group).to_numpy()
    out = banded.groupby("group").agg(n=("covered", "size"), coverage=("covered", "mean"))
    out["median_width"] = (banded["high"] - banded["low"]).groupby(banded["group"]).median()
    return out.reset_index().round({"coverage": 3, "median_width": 1})
