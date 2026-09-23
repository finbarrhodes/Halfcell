"""
src/analysis/quantile_forecast.py
=================================
Quantile forecasts from the forest the point forecast already uses.

Why this exists
---------------
The conformal guard bands (src/analysis/intervals.py) know how wrong the forecast
has been at each time of day and nothing about the day itself: a calm Sunday and
the morning after a gas-price shock get the same band at 18:00. A quantile
regression forest (Meinshausen 2006) keeps every training target in each leaf
rather than only their mean, so the trees that produce the point forecast also
give a conditional distribution for the day in front of them - wide when the
features resemble the days that went wrong, narrow when they do not.

O'Connor et al. (2025, "Conformal Prediction for Electricity Price Forecasting in
the Day-Ahead and Real-Time Balancing Market") benchmark this as their "QR" arm
against EnbPI and SPCI on Irish prices. On the forecasts they publish, the random
forest's own quantiles are the best of the forest-based methods: an 80% interval
covering 86% of day-ahead prices with a Winkler score of 47, where their
conformal intervals cover 71-75% with scores of 55-56 - and those were built to a
90% target (see intervals.py, "Reading the literature").

The transform
-------------
The point model fits a signed-log1p target (forecasting_models._LogTransformModel).
That bends a mean but not a quantile: the transform is monotone, so the
q-quantile of the transformed price maps back to the q-quantile of the price
exactly. The forest is fitted on the same transformed target and its quantiles
mapped back, which keeps the heavy upper tail the transform exists to tame.

Its trees are grown with the point model's settings, so the two describe the same
partition of the feature space. They are not the same trees - the quantile forest
draws its own bootstrap - and its mean differs from the shipped point forecast by
a few £/MWh at most. Bands built from it are expressed relative to the shipped
forecast, since that is the price the plan actually trades against.

Walk-forward
------------
Same origins, training windows, features and information lag as
price_forecast.walk_forward_predictions: each day's quantiles come from a forest
that never saw it, built from the feature matrix the caller passes - the early one
(information_lag_days=2) for what existed at the bid deadline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.price_forecast import (
    WALK_FORWARD_CADENCE_MONTHS,
    resolve_feature_cols,
    walk_forward_origins,
)

# Enough to report the paper's 80% and 40% intervals (0.1/0.9, 0.3/0.7), the engine's
# 60% one (0.2/0.8), and the median.
QUANTILES = (0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9)

# The point Random Forest's settings (forecasting_models._build_model("rf")), so the
# quantile forest partitions the features the same way. max_samples_leaf=None keeps
# every training target in each leaf: quantile-forest's default of one sampled target
# per leaf is a faster approximation that coarsens the tails.
FOREST_PARAMS = dict(n_estimators=300, max_features=0.5, min_samples_leaf=5,
                     max_samples_leaf=None, n_jobs=-1, random_state=42)


def column(q: float) -> str:
    """The table column holding quantile q: 0.1 -> "q10"."""
    return f"q{int(round(q * 100)):02d}"


def _signed_log(y: np.ndarray) -> np.ndarray:
    return np.sign(y) * np.log1p(np.abs(y))


def _signed_exp(p: np.ndarray) -> np.ndarray:
    return np.sign(p) * np.expm1(np.abs(p))


class QuantileForest:
    """A quantile regression forest on the signed-log price, answering in price space."""

    def __init__(self, **params):
        from quantile_forest import RandomForestQuantileRegressor

        self._forest = RandomForestQuantileRegressor(**{**FOREST_PARAMS, **params})

    def fit(self, X, y):
        self._forest.fit(X, _signed_log(np.asarray(y, dtype=float)))
        return self

    def predict_quantiles(self, X, quantiles=QUANTILES) -> np.ndarray:
        """(rows, len(quantiles)) in £/MWh, non-decreasing along each row."""
        q = self._forest.predict(X, quantiles=list(quantiles))
        # A forest's quantiles are order statistics of one weighted sample, so they
        # cannot cross; sorting only guards against ties broken by rounding.
        return np.sort(_signed_exp(np.asarray(q, dtype=float).reshape(len(X), -1)), axis=1)


def pinball(actual: np.ndarray, forecast: np.ndarray, q: float) -> float:
    """
    Mean pinball (quantile) loss of a q-quantile forecast: q·u for u = actual − forecast
    above zero, (q − 1)·u below. Proper for the q-quantile, so lower is better.
    """
    u = np.asarray(actual, dtype=float) - np.asarray(forecast, dtype=float)
    ok = np.isfinite(u)
    return float(np.mean(np.maximum(q * u[ok], (q - 1) * u[ok]))) if ok.any() else float("nan")


def walk_forward_quantiles(
    feature_df: pd.DataFrame,
    start_date,
    end_date,
    *,
    quantiles=QUANTILES,
    cadence_months: int = WALK_FORWARD_CADENCE_MONTHS,
    train_years: float | None = None,
    skip_origins=None,
    on_fold=None,
    forest_params: dict | None = None,
) -> tuple:
    """
    Out-of-sample price quantiles for every day in [start_date, end_date].

    The quantile counterpart of price_forecast.walk_forward_predictions, on the same
    origins and windows: refit at each origin on the history before it, and used only
    for the days up to the next.

    Returns
    -------
    (table, folds) where table has [settlementDate, settlementPeriod, q.., origin] with
    one "qNN" column per quantile, and folds has one dict per origin: its training span
    and, for the days it forecast, how often the price fell below each quantile and the
    mean pinball loss.
    """
    origins = walk_forward_origins(start_date, end_date, cadence_months)
    first, last = pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize()
    feature_cols = resolve_feature_cols(feature_df)
    names = [column(q) for q in quantiles]
    skip = {pd.Timestamp(o).normalize() for o in (skip_origins or ())}

    frames, folds = [], []
    for i, origin in enumerate(origins):
        fold_end = origins[i + 1] if i + 1 < len(origins) else last + pd.Timedelta(days=1)
        if origin in skip:
            continue
        window = feature_df[feature_df["settlementDate"] < fold_end]
        if train_years is not None:
            window = window[window["settlementDate"] >= origin - pd.DateOffset(months=round(train_years * 12))]
        train = window[window["settlementDate"] < origin].dropna(subset=["apx_price"])
        served = window[(window["settlementDate"] >= max(origin, first))
                        & (window["settlementDate"] <= last)]
        if train.empty or served.empty:
            continue

        forest = QuantileForest(**(forest_params or {})).fit(train[feature_cols].fillna(0), train["apx_price"])
        values = forest.predict_quantiles(served[feature_cols].fillna(0), quantiles)
        frame = pd.DataFrame(values, columns=names, index=served.index)
        frame.insert(0, "settlementPeriod", served["settlementPeriod"].astype(int).to_numpy())
        frame.insert(0, "settlementDate", served["settlementDate"].to_numpy())
        frame["origin"] = origin
        frames.append(frame.reset_index(drop=True))

        actual = served["apx_price"].to_numpy(dtype=float)
        scored = np.isfinite(actual)
        folds.append({
            "origin": origin.date().isoformat(),
            "predicts_until": (fold_end - pd.Timedelta(days=1)).date().isoformat(),
            "train_start": train["settlementDate"].min().date().isoformat(),
            "train_rows": int(len(train)),
            "share_below": {name: round(float(np.mean(actual[scored] <= values[scored, k])), 3)
                            for k, name in enumerate(names)},
            "pinball": {name: round(pinball(actual, values[:, k], q), 3)
                        for k, (name, q) in enumerate(zip(names, quantiles))},
        })
        if on_fold is not None:
            on_fold(folds[-1])

    columns = ["settlementDate", "settlementPeriod", *names, "origin"]
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    return table, folds
