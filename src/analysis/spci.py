"""
src/analysis/spci.py
====================
Sequential Predictive Conformal Inference (Xu & Xie 2023) for the bid-time guard
bands.

What it adds
------------
Split conformal bands (src/analysis/intervals.py) treat the next residual as one
more draw from the pool of past residuals. Price-forecast errors are not like that:
they cluster. A forecast that has been missing by £100 for three days is likely to
miss again, and a quarter-old calibration set cannot know that the forecast has
just started going wrong. SPCI fits a quantile regression forest of each residual on
the residuals just before it, so the band is the error's distribution *given how
the forecast has been doing lately*. It still wraps any point forecast, and needs
no calibration set held back from training.

As O'Connor et al. (2025) run it
--------------------------------
Their day-ahead SPCI keeps the reference implementation's settings: the last 300
residuals as the forest's inputs (w), 10 trees of depth 2, one forest per hour of
the next day (the "multi-step" variant, stride 24), refitted daily, and the
miscoverage split between the tails by searching β over five bins for the narrowest
interval.

Adapted to the bid deadline
---------------------------
  Information. Offers for day D close at 14:00 on D-1, when the newest complete
      prices are D-2's (price_forecast's information_lag_days=2). A day's inputs
      are the 300 residuals ending with D-2's last period, where the paper's end
      with D-1's.
  Resolution. GB settles half-hourly, so there are 48 forests, one per settlement
      period, and 300 residuals span 6¼ days rather than 12½.
  Training pairs. One per day and period, from windows ending where a bid-time
      window would. The reference slides its window a step at a time, pooling
      every phase of the day into each horizon's forest; aligning to the bid keeps
      each forest to one period and costs 48 times less to fit.
  Refits. Every `refit_days`, where the reference refits daily. The inputs still
      move every day, so every day's band responds to the newest errors;
      refit_days=1 reproduces the reference cadence.
  Tails. The plan trades each side of a band separately, so the bounds are the
      residual's α and 1 − α quantiles by default. optimise_beta=True restores the
      β search, which narrows a two-sided interval by moving miscoverage between
      its tails - measured, like every band here, after capping at the forecast.

What it sees and what it does not
---------------------------------
The forests are grown to predict the residual's *mean* (squared-error splits on
signed residuals, as in the reference), so what they condition on is persistence
in the forecast's bias: when the recent errors are running one way, the band moves
that way. A spell that is only noisier, with no bias, looks the same to them, and
gets nearly the same band - tests/test_spci.py pins that.

Residuals missing from the window (a clock-change day's short tail, a gap in the
forecast table) enter as zero, the value that says nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.intervals import residual_frame

WINDOW = 300                    # residual lags the forests condition on (the paper's w)
N_ESTIMATORS, MAX_DEPTH = 10, 2  # SPCI's own forests: deliberately small
BETA_BINS = 5
PERIODS = 48


def residual_matrix(predictions: pd.DataFrame, market_index: pd.DataFrame) -> pd.DataFrame:
    """
    Residuals (actual − forecast) as a calendar day × settlement period matrix.

    Rows run over every calendar day the prediction table spans, so a row's position
    is its date and a lag of one day is always 48 steps; NaN where a residual is
    missing, including days whose prices are not yet known.
    """
    days = pd.to_datetime(predictions["settlementDate"]).dt.normalize()
    calendar = pd.date_range(days.min(), days.max(), freq="D")
    residuals = residual_frame(predictions, market_index)
    matrix = residuals.pivot_table(index="settlementDate", columns="settlementPeriod", values="residual")
    return matrix.reindex(index=calendar, columns=range(1, PERIODS + 1))


def lagged_windows(flat: np.ndarray, ends: np.ndarray, window: int) -> np.ndarray:
    """
    The `window` values of `flat` ending at each index in `ends`, oldest first.

    Positions before the series starts, and missing residuals, are zero.
    """
    index = np.asarray(ends)[:, None] - np.arange(window - 1, -1, -1)[None, :]
    values = np.where(index >= 0, flat[np.clip(index, 0, len(flat) - 1)], np.nan)
    return np.nan_to_num(values, nan=0.0)


def levels(alpha: float, optimise_beta: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    The quantile levels a band at `alpha` reads: (α,) and (1 − α,), or with the β search
    the paired grids (β, 1 − 2α + β) for β from 0 to 2α - total miscoverage 2α, split
    however makes the band narrowest.
    """
    if not 0 < alpha < 0.5:
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")
    if not optimise_beta:
        return np.array([alpha]), np.array([round(1 - alpha, 10)])
    betas = np.round(np.linspace(0.0, 2 * alpha, BETA_BINS), 10)
    return betas, np.round(1 - 2 * alpha + betas, 10)


def walk_forward_spci_quantiles(
    predictions: pd.DataFrame,
    market_index: pd.DataFrame,
    quantile_levels,
    *,
    window: int = WINDOW,
    information_lag_days: int = 2,
    refit_days: int = 30,
    train_days: int | None = None,
    min_days: int = 120,
    n_estimators: int = N_ESTIMATORS,
    max_depth: int = MAX_DEPTH,
    random_state: int = 0,
    n_jobs: int = -1,
) -> tuple:
    """
    The residual's conditional quantiles, at each of `quantile_levels`, for every day
    with enough residual history behind it.

    One forest per settlement period is refitted every `refit_days` on the pairs
    (window of residuals ending at D - lag, residual on D) whose target was known at
    the first refitted day's bid deadline, and asked for every level at once - the
    forest does not depend on the level, so several bands cost one fit.

    Returns
    -------
    ({level: DataFrame days × 48 of residual quantiles}, folds). A period with too few
    training pairs is NaN - unbanded - rather than zero, so a score never mistakes it
    for a band of zero width.
    """
    from quantile_forest import RandomForestQuantileRegressor

    quantile_levels = sorted({round(float(q), 10) for q in quantile_levels})
    residuals = residual_matrix(predictions, market_index)
    if residuals.empty:
        return {q: pd.DataFrame(columns=range(1, PERIODS + 1)) for q in quantile_levels}, []
    days, R = residuals.index, residuals.to_numpy()
    n_days = len(days)
    # Day d's window ends with the last period of day d - lag
    ends = (np.arange(n_days) - information_lag_days) * PERIODS + PERIODS - 1
    complete = ends - window + 1 >= 0
    X = lagged_windows(R.ravel(), ends, window)
    forecast_days = set(pd.to_datetime(predictions["settlementDate"]).dt.normalize())

    served_days, values, folds = [], [], []
    for start in range(0, n_days, refit_days):
        served = [d for d in range(start, min(start + refit_days, n_days))
                  if complete[d] and days[d] in forecast_days]
        if not served:
            continue
        # Pairs whose target residual was known at the first served day's bid deadline
        last = start - information_lag_days
        first = 0 if train_days is None else max(0, last - train_days + 1)
        train = np.array([d for d in range(first, last + 1) if complete[d]], dtype=int)
        known = np.isfinite(R[train]) if len(train) else np.zeros((0, PERIODS), dtype=bool)
        record = {"refit": days[start].date().isoformat(), "until": days[served[-1]].date().isoformat(),
                  "train_days": int(known.any(axis=1).sum()), "days": len(served)}
        if record["train_days"] < min_days:
            folds.append({**record, "banded": False})
            continue

        block = np.full((len(served), PERIODS, len(quantile_levels)), np.nan)
        for p in range(PERIODS):
            rows = train[known[:, p]]
            if len(rows) < min_days:
                continue                        # too few pairs for this period: leave it unbanded
            forest = RandomForestQuantileRegressor(
                n_estimators=n_estimators, max_depth=max_depth, max_samples_leaf=None,
                random_state=random_state, n_jobs=n_jobs)
            forest.fit(X[rows], R[rows, p])
            block[:, p, :] = np.asarray(forest.predict(X[served], quantiles=quantile_levels),
                                        dtype=float).reshape(len(served), -1)
        served_days.extend(days[d] for d in served)
        values.append(block)
        folds.append({**record, "banded": True})

    stacked = np.concatenate(values) if values else np.zeros((0, PERIODS, len(quantile_levels)))
    index = pd.DatetimeIndex(served_days)
    tables = {q: pd.DataFrame(stacked[:, :, k], index=index, columns=range(1, PERIODS + 1))
              for k, q in enumerate(quantile_levels)}
    return tables, folds


def spci_bands(tables: dict, alpha: float, optimise_beta: bool = False) -> tuple[dict, dict]:
    """
    Guard bands from walk_forward_spci_quantiles' tables: the residual's α and 1 − α
    quantiles, or with the β search the pair giving the narrowest band as the plan
    will use it, capped at the forecast. Returns (low_by_date, high_by_date).
    """
    lows, highs = levels(alpha, optimise_beta)
    missing = [q for q in (*lows, *highs) if float(q) not in tables]
    if missing:
        raise ValueError(f"no residual quantiles at levels {missing}")
    lo = np.stack([tables[float(q)].to_numpy() for q in lows], axis=-1)
    hi = np.stack([tables[float(q)].to_numpy() for q in highs], axis=-1)
    width = np.maximum(hi, 0.0) - np.minimum(lo, 0.0)
    pick = np.nanargmin(np.where(np.isnan(width), np.inf, width), axis=-1)[..., None]
    low = np.minimum(np.take_along_axis(lo, pick, axis=-1)[..., 0], 0.0)
    high = np.maximum(np.take_along_axis(hi, pick, axis=-1)[..., 0], 0.0)
    template = tables[float(lows[0])]
    columns = template.columns.to_numpy()
    return ({day: pd.Series(low[k], index=columns) for k, day in enumerate(template.index)},
            {day: pd.Series(high[k], index=columns) for k, day in enumerate(template.index)})


def walk_forward_spci_bands(
    predictions: pd.DataFrame,
    market_index: pd.DataFrame,
    *,
    alpha: float = 0.2,
    optimise_beta: bool = False,
    **settings,
) -> tuple:
    """
    Guard bands from SPCI, for every day with enough residual history behind it.

    Parameters
    ----------
    predictions : long (settlementDate, settlementPeriod, prediction) forecasts - the
        early table, since the bands price offers made at the bid deadline.
    market_index : APXMIDP prices, as the rest of the engine reads them.
    alpha : miscoverage per tail; the band's bounds are the residual's α and 1−α
        quantiles given the recent residuals.
    optimise_beta : search the split of 2α between the tails for the narrowest band.
    settings : passed to walk_forward_spci_quantiles - window (how many of the latest
        residuals the forests condition on), information_lag_days (how old the newest
        of them must be), refit_days, train_days (a trailing limit on training pairs;
        None uses them all, as the reference does for SPCI), min_days, and the forests'
        n_estimators and max_depth.

    Returns
    -------
    (low_by_date, high_by_date, folds) as intervals.walk_forward_bands does: offsets
    to add to the forecast, low at most zero and high at least zero, and one record
    per refit with its training size and the band's median width.
    """
    lows, highs = levels(alpha, optimise_beta)
    tables, folds = walk_forward_spci_quantiles(predictions, market_index, [*lows, *highs], **settings)
    low_by_date, high_by_date = spci_bands(tables, alpha, optimise_beta)
    for fold in folds:
        spell = [(high_by_date[day] - low_by_date[day]).to_numpy() for day in low_by_date
                 if fold["banded"] and pd.Timestamp(fold["refit"]) <= day <= pd.Timestamp(fold["until"])]
        if spell:
            fold["median_width"] = round(float(np.nanmedian(np.concatenate(spell))), 1)
    return low_by_date, high_by_date, folds
