"""
src/analysis/shrink.py
======================
How far to believe a day's forecast shape, estimated from how far past forecasts
turned out to be worth believing.

The offer plan (src/optimisation/day_ahead.py) trades on the forecast's *shape* -
each period's deviation from that day's mean price - and pulls it towards the mean
by a weight before planning. A constant 0.5 ships today. This module estimates the
weight per day instead.

**The statistic.** Write a day's forecast shape as f and what happened as s. The
scaling that minimises squared error is the slope of a regression of s on f through
the origin, `Σ f·s / Σ f²` - the Mincer-Zarnowitz slope, and the same ratio a Kalman
gain or a James-Stein shrinkage factor takes. Pooled over the backtest it is 0.69
for the bid-time Random Forest: a third of the shape the forecast draws does not
arrive, so a plan that believes all of it over-values keeping capacity free.

**Why it varies.** The slope is not the same on every day. Bucketed by how wide the
forecast says the day will be, it runs from 1.21 on the calmest fifth - where the
forecast *understates* the shape - to 0.56 on the widest, which is regression to the
mean: the louder the forecast shouts, the less of it materialises. That, and not the
size of the error, is what a per-day weight can exploit; the spread realised two days
earlier predicts error size well (Spearman +0.59) and this slope hardly at all.
Buckets rather than a fitted curve because the relationship is monotone but not
linear, and a ratio of sums per bucket is hard to overfit and easy to read.

**Two separate things.** The slope is what the data says. Whether to believe even
that much is a decision question: an over-valued spread costs a response contract as
well as a trade, so callers multiply the slope by a risk factor (see
`walk_forward_slopes`'s `risk_factor`), tuned on selection folds exactly as the
constant was. Today's shipped 0.5 is approximately 0.72 x 0.69.

Everything is fitted walk-forward: a day's weight comes only from days before its
origin, and from the same early (bid-time) forecasts the offers themselves see.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Below this many training days a bucketed slope is noise; use the pooled slope, and
# below FALLBACK_MIN_DAYS use the caller's constant instead.
BUCKET_MIN_DAYS = 365
FALLBACK_MIN_DAYS = 120
N_BUCKETS = 5
# A weight above 1 expands the forecast's shape. The data asks for it on calm days;
# the cap keeps a quiet day with a tiny denominator from asking for much more.
MAX_WEIGHT = 1.25


def day_matrix(frame: pd.DataFrame, value: str, date: str = "settlementDate") -> pd.DataFrame:
    """Long (date, settlementPeriod, value) to a day x period matrix."""
    table = frame.assign(**{date: pd.to_datetime(frame[date]).dt.normalize()})
    return table.pivot_table(index=date, columns="settlementPeriod", values=value)


def shapes(forecast: pd.DataFrame, actual: pd.DataFrame) -> tuple:
    """
    Each day's deviations from its own mean, for the days both cover.

    The plan already trades shape rather than level: it cannot move the day's average
    price, only exploit the differences within it.
    """
    days = forecast.index.intersection(actual.index)
    f, a = forecast.loc[days], actual.loc[days]
    return f.sub(f.mean(axis=1), axis=0), a.sub(a.mean(axis=1), axis=0)


def slope(forecast_shape: pd.DataFrame, actual_shape: pd.DataFrame) -> float:
    """
    The scaling of the forecast's shape that best matches what happened,
    `Σ f·s / Σ f²`. 1.0 when a forecast needs no rescaling, below it when the
    forecast draws more shape than arrives.
    """
    f, a = forecast_shape.to_numpy().ravel(), actual_shape.to_numpy().ravel()
    ok = np.isfinite(f) & np.isfinite(a)
    denominator = float(f[ok] @ f[ok])
    return float(f[ok] @ a[ok]) / denominator if denominator > 0 else 1.0


def amplitude(forecast: pd.DataFrame) -> pd.Series:
    """How wide the forecast says each day will be: its spread, in £/MWh."""
    return forecast.max(axis=1) - forecast.min(axis=1)


class SlopeBuckets:
    """Slopes by how wide the forecast says the day will be, plus a pooled fallback."""

    def __init__(self, edges: np.ndarray, slopes: np.ndarray, pooled: float, n_days: int):
        self.edges, self.slopes, self.pooled, self.n_days = edges, slopes, pooled, n_days

    def weight(self, day_amplitude: float) -> float:
        if not self.slopes.size or not np.isfinite(day_amplitude):
            return self.pooled
        return float(self.slopes[int(np.searchsorted(self.edges, day_amplitude))])

    def as_dict(self) -> dict:
        return {"edges": [round(e, 1) for e in self.edges],
                "slopes": [round(s, 3) for s in self.slopes],
                "pooled": round(self.pooled, 3), "train_days": self.n_days}


def fit(forecast: pd.DataFrame, actual: pd.DataFrame, n_buckets: int = N_BUCKETS) -> SlopeBuckets:
    """
    Fit slopes on the days given, bucketed by forecast amplitude.

    With too few days to bucket, the pooled slope is used for every amplitude, so a
    caller never has to special-case a short history.
    """
    f, a = shapes(forecast, actual)
    pooled = slope(f, a)
    if len(f) < BUCKET_MIN_DAYS or n_buckets < 2:
        return SlopeBuckets(np.array([]), np.array([]), pooled, len(f))

    width = amplitude(forecast.loc[f.index])
    edges = np.quantile(width.dropna(), np.linspace(0, 1, n_buckets + 1)[1:-1])
    membership = np.searchsorted(edges, width.to_numpy())
    slopes = np.array([slope(f[membership == k], a[membership == k]) if (membership == k).sum() >= 30
                       else pooled for k in range(n_buckets)])
    return SlopeBuckets(edges, slopes, pooled, len(f))


def walk_forward_slopes(
    predictions: pd.DataFrame,
    market_index: pd.DataFrame,
    *,
    cadence_months: int = 3,
    risk_factor: float = 1.0,
    fallback: float = 0.5,
    value: str = "prediction",
) -> tuple[dict, list]:
    """
    A weight for every day, fitted only on days that came before it.

    The origins match the forecast's own refit cadence: at each one, slopes are fitted
    on every day before it and applied until the next. Days before there is enough
    history take `fallback`, the constant the caller would otherwise have used.

    Parameters
    ----------
    predictions : long (settlementDate, settlementPeriod, `value`) forecasts - the
        early table for offers, or a naive series in the same shape.
    market_index : APXMIDP prices, as the rest of the engine reads them.
    risk_factor : multiplies every fitted slope. 1.0 believes the statistics; below 1
        adds the caution an asymmetric decision cost calls for.
    fallback : the weight before there is enough history to fit one.

    Returns
    -------
    ({date: weight}, [fold record per origin]) - the folds carry the fitted buckets,
    so a run can report what it believed and why.
    """
    forecast = day_matrix(predictions, value)
    apx = market_index[(market_index["dataProvider"] == "APXMIDP")
                       & (market_index["settlementPeriod"] <= 48)]
    actual = day_matrix(apx, "price")
    days = forecast.index.intersection(actual.index).sort_values()
    if not len(days):
        return {}, []

    origins = pd.date_range(days.min().normalize().replace(day=1), days.max(),
                            freq=f"{cadence_months}MS")
    weights, folds = {}, []
    for k, origin in enumerate(origins):
        upto = origins[k + 1] if k + 1 < len(origins) else days.max() + pd.Timedelta(days=1)
        served = days[(days >= origin) & (days < upto)]
        if not len(served):
            continue
        history = days[days < origin]
        if len(history) < FALLBACK_MIN_DAYS:
            weights.update({day: fallback for day in served})
            folds.append({"origin": origin.date().isoformat(), "train_days": len(history),
                          "buckets": None, "weight": fallback, "days": len(served)})
            continue

        buckets = fit(forecast.loc[history], actual.loc[history])
        width = amplitude(forecast.loc[served])
        served_weights = {day: float(np.clip(risk_factor * buckets.weight(width.get(day, np.nan)),
                                             0.0, MAX_WEIGHT))
                          for day in served}
        weights.update(served_weights)
        folds.append({"origin": origin.date().isoformat(), "train_days": len(history),
                      "buckets": buckets.as_dict(), "days": len(served),
                      "weight_mean": round(float(np.mean(list(served_weights.values()))), 3),
                      "weight_range": [round(min(served_weights.values()), 3),
                                       round(max(served_weights.values()), 3)]})
    return weights, folds


def naive_predictions(market_index: pd.DataFrame, days_back: int = 2) -> pd.DataFrame:
    """The naive forecast as a prediction table, so it can be calibrated the same way."""
    apx = market_index[(market_index["dataProvider"] == "APXMIDP")
                       & (market_index["settlementPeriod"] <= 48)]
    table = apx.assign(settlementDate=pd.to_datetime(apx["settlementDate"]).dt.normalize()
                       + pd.Timedelta(days=days_back))
    return table[["settlementDate", "settlementPeriod", "price"]].rename(columns={"price": "prediction"})


def tilted_weights(
    predictions: pd.DataFrame,
    market_index: pd.DataFrame,
    *,
    intercept: float = 0.5,
    tilt: float = 0.0,
    cadence_months: int = 3,
    fallback: float | None = None,
) -> dict:
    """
    A weight that moves with how loud the day's forecast is, by construction rather
    than by calibration: `intercept + tilt x (percentile - 0.5)`.

    The fitted slope (walk_forward_slopes) believes loud days *least*, because that is
    what squared error asks for, and measured on revenue it lost to a constant at every
    risk factor (2026-09-21). The days it disbelieves are the ones whose spread turns
    out to be real, so this family lets the weight lean either way and asks the
    backtest which. tilt=0 reproduces the constant exactly; a positive tilt believes
    the loudest days more.

    The percentile comes from the amplitudes seen before the day's origin only, so a
    day is ranked against history rather than against the whole backtest.
    """
    forecast = day_matrix(predictions, "prediction")
    apx = market_index[(market_index["dataProvider"] == "APXMIDP")
                       & (market_index["settlementPeriod"] <= 48)]
    days = forecast.index.intersection(day_matrix(apx, "price").index).sort_values()
    if not len(days):
        return {}

    width = amplitude(forecast.loc[days])
    origins = pd.date_range(days.min().normalize().replace(day=1), days.max(),
                            freq=f"{cadence_months}MS")
    flat = float(intercept if fallback is None else fallback)
    weights = {}
    for k, origin in enumerate(origins):
        upto = origins[k + 1] if k + 1 < len(origins) else days.max() + pd.Timedelta(days=1)
        served = days[(days >= origin) & (days < upto)]
        history = width[width.index < origin].dropna()
        if not len(served):
            continue
        if len(history) < FALLBACK_MIN_DAYS:
            weights.update({day: flat for day in served})
            continue
        ranked = np.searchsorted(np.sort(history.to_numpy()), width.loc[served].to_numpy()) / len(history)
        weights.update({day: float(np.clip(intercept + tilt * (u - 0.5), 0.0, MAX_WEIGHT))
                        for day, u in zip(served, ranked)})
    return weights
