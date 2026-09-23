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

Bands that know about the day
-----------------------------
Split conformal widths depend on the hour and nothing else. Three builders here
and one in src/analysis/spci.py let them depend on the day as well, all returning
the same (low_by_date, high_by_date, folds) the engine takes:

  quantile_bands          : a quantile forecast's own α and 1−α quantiles
                            (src/analysis/quantile_forecast.py), less the point
                            forecast. Widths follow the features; nothing checks
                            that they cover.
  walk_forward_cqr_bands  : conformalised quantile regression (Romano, Patterson &
                            Candès 2019). The same quantiles, each moved by the
                            order statistic of how far past prices overshot it, so
                            the width keeps the features' shape and the coverage
                            comes back to the level asked for.
  spci.walk_forward_spci_bands : Sequential Predictive Conformal Inference (Xu & Xie
                            2023) - the residual's quantiles conditioned on the most
                            recent residuals, so the band widens the day after the
                            forecast starts going wrong rather than a quarter later.
  combine_bands           : the average of several, as O'Connor et al. (2025) do
                            with QR, EnbPI and SPCI ("Q-Ens").

`score_bands` measures any of them as the literature does - coverage, width,
pinball loss of each bound and the Winkler interval score - so a band can be judged
as a forecast before it is judged on revenue.

Reading the literature
----------------------
O'Connor et al. is the benchmark these methods come from, and its interval tables
need one caution. Its EnbPI and SPCI intervals are built with the reference
implementation's `alpha` of 0.1 and 0.3, which there is the *total* miscoverage -
[Q(β), Q(1−α+β)], a 90% and a 70% interval - and are then filed and compared as
the 10/90 and 30/70 quantile pairs, alongside quantile regression's genuine 80% and
40% intervals. On the paper's published random-forest forecasts the conformal
"80%" intervals in fact cover 71-75% of day-ahead prices against 86% for the
forest's own quantiles, with a worse Winkler score (55-56 against 47). Conformal
methods earn their place in that paper by repairing badly calibrated quantile
models (LightGBM, k-NN, LEAR), not by beating a quantile forest; here the point
model is a forest, so its quantiles are the first thing to try.
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


def band_frame(predictions: pd.DataFrame, market_index: pd.DataFrame, low_by_date: dict,
               high_by_date: dict) -> pd.DataFrame:
    """
    Every banded period with its residual and its two offsets: the frame each score
    is taken from.

    A period the band does not cover - a date a builder left unbanded, or a clock-change
    day's extra periods - is dropped rather than scored as an interval of zero width.
    """
    residuals = residual_frame(predictions, market_index)
    banded = residuals[residuals["settlementDate"].isin(low_by_date)].copy()
    if banded.empty:
        return banded.assign(low=pd.Series(dtype=float), high=pd.Series(dtype=float))
    keys = list(zip(banded["settlementDate"], banded["settlementPeriod"]))
    banded["low"] = [low_by_date[d].get(sp, np.nan) for d, sp in keys]
    banded["high"] = [high_by_date[d].get(sp, np.nan) for d, sp in keys]
    return banded.dropna(subset=["low", "high"]).reset_index(drop=True)


def coverage(predictions: pd.DataFrame, market_index: pd.DataFrame, low_by_date: dict,
             high_by_date: dict, group: str = "period") -> pd.DataFrame:
    """
    How often the interval contained the price, and how wide it had to be.

    Reported per group, because marginal coverage can hold while a particular hour
    is badly served - the reason for calibrating per group in the first place.
    """
    banded = band_frame(predictions, market_index, low_by_date, high_by_date)
    if banded.empty:
        return pd.DataFrame(columns=["group", "n", "coverage", "median_width"])
    banded["covered"] = (banded["residual"] >= banded["low"]) & (banded["residual"] <= banded["high"])
    banded["group"] = group_key(banded["settlementPeriod"], group).to_numpy()
    out = banded.groupby("group").agg(n=("covered", "size"), coverage=("covered", "mean"))
    out["median_width"] = (banded["high"] - banded["low"]).groupby(banded["group"]).median()
    return out.reset_index().round({"coverage": 3, "median_width": 1})


# --- Scoring --------------------------------------------------------------------------

SCORE_BY = ("year", "block", "period")


def score_bands(frame: pd.DataFrame, alpha: float, by: str | None = None) -> pd.DataFrame:
    """
    A band judged as a forecast, in the measures probabilistic price forecasting reports.

      coverage      share of prices inside [forecast + low, forecast + high]; 1 − 2α is
                    the target when the bounds are the α and 1 − α quantiles
      below, above  share outside on each side, each ideally α: a band can reach its
                    coverage while missing lopsidedly, and the plan trades each side
      mean_width, median_width   high − low, £/MWh
      pinball       mean pinball loss of the two bounds as the α and 1 − α quantiles
      winkler       the interval score (Winkler 1972; Gneiting & Raftery 2007): the
                    width plus 1/α times any miss, so narrowness pays only while the
                    band still covers

    Pinball and Winkler are proper scores - lower is better, and they cannot be
    improved by reporting anything but the true quantiles - which is why they, and
    not coverage alone, decide between methods. `frame` is band_frame's output;
    `by` splits the scores by "year", EFA "block" or settlement "period".
    """
    if by is not None and by not in SCORE_BY:
        raise ValueError(f"by must be one of {SCORE_BY}, got {by!r}")
    r, lo, hi = frame["residual"], frame["low"], frame["high"]
    below, above = lo - r, r - hi                         # positive when the price fell outside
    rows = pd.DataFrame({
        "below": below > 0,
        "above": above > 0,
        "width": hi - lo,
        # u = price − bound = −below at the lower bound and above at the upper one
        "pinball": 0.5 * (np.maximum(-alpha * below, (1 - alpha) * below)
                          + np.maximum((1 - alpha) * above, -alpha * above)),
        "winkler": (hi - lo) + (below.clip(lower=0) + above.clip(lower=0)) / alpha,
    })
    if by == "year":
        keys = pd.to_datetime(frame["settlementDate"]).dt.year.to_numpy()
    elif by in ("block", "period"):
        keys = group_key(frame["settlementPeriod"], by).to_numpy()
    else:
        keys = np.zeros(len(frame), dtype=int)
    grouped = rows.groupby(keys)
    out = pd.DataFrame({
        "n": grouped.size(),
        "coverage": 1 - grouped["below"].mean() - grouped["above"].mean(),
        "below": grouped["below"].mean(),
        "above": grouped["above"].mean(),
        "mean_width": grouped["width"].mean(),
        "median_width": grouped["width"].median(),
        "pinball": grouped["pinball"].mean(),
        "winkler": grouped["winkler"].mean(),
    })
    out.index.name = by or "all"
    return out.reset_index().round({"coverage": 3, "below": 3, "above": 3, "mean_width": 1,
                                    "median_width": 1, "pinball": 2, "winkler": 1})


# --- Bands from a quantile forecast ---------------------------------------------------

def _offsets_by_date(frame: pd.DataFrame, low: str, high: str) -> tuple[dict, dict]:
    """{date: Series by settlement period} for two offset columns, capped as every band is."""
    low_by_date, high_by_date = {}, {}
    for day, rows in frame.groupby("settlementDate"):
        index = rows["settlementPeriod"].astype(int).to_numpy()
        low_by_date[day] = pd.Series(np.minimum(rows[low].to_numpy(dtype=float), 0.0), index=index)
        high_by_date[day] = pd.Series(np.maximum(rows[high].to_numpy(dtype=float), 0.0), index=index)
    return low_by_date, high_by_date


def _quantile_offsets(predictions: pd.DataFrame, quantiles: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    The α and 1 − α quantiles as offsets from the point forecast the plan trades against:
    (settlementDate, settlementPeriod, prediction, lo, hi, origin).
    """
    from src.analysis.quantile_forecast import column

    if not 0 < alpha < 0.5:
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")
    lo, hi = column(alpha), column(1 - alpha)
    missing = [c for c in (lo, hi) if c not in quantiles.columns]
    if missing:
        raise ValueError(f"the quantile table has no {missing}; it holds "
                         f"{[c for c in quantiles.columns if c.startswith('q')]}")
    norm = lambda t: t.assign(settlementDate=pd.to_datetime(t["settlementDate"]).dt.normalize())
    joined = norm(predictions)[["settlementDate", "settlementPeriod", "prediction"]].merge(
        norm(quantiles)[["settlementDate", "settlementPeriod", lo, hi, "origin"]],
        on=["settlementDate", "settlementPeriod"], how="inner")
    return joined.assign(lo=joined[lo] - joined["prediction"], hi=joined[hi] - joined["prediction"])[
        ["settlementDate", "settlementPeriod", "prediction", "lo", "hi", "origin"]]


def quantile_bands(predictions: pd.DataFrame, quantiles: pd.DataFrame, alpha: float = 0.2) -> tuple:
    """
    Guard bands read straight off a quantile forecast.

    Each period's α and 1 − α quantiles less the point forecast, capped as every band
    is (low at most zero, high at least zero). Nothing is calibrated: the width is
    whatever the quantile model believes, and whether that covers is score_bands'
    question. The quantile table is itself walk-forward, so every date it holds is
    banded, from the first day of the backtest.

    Returns (low_by_date, high_by_date, folds), folds summarising each forecast origin.
    """
    offsets = _quantile_offsets(predictions, quantiles, alpha)
    low_by_date, high_by_date = _offsets_by_date(offsets, "lo", "hi")
    folds = [{"origin": pd.Timestamp(origin).date().isoformat(), "banded": True,
              "days": int(rows["settlementDate"].nunique()),
              "median_width": round(float((rows["hi"].clip(lower=0) - rows["lo"].clip(upper=0)).median()), 1)}
             for origin, rows in offsets.groupby("origin")]
    return low_by_date, high_by_date, folds


def _rank_statistic(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1−α))-th smallest score: the finite-sample conformal quantile."""
    n = len(scores)
    if not n:
        return 0.0
    return float(np.sort(scores)[min(n, int(np.ceil((n + 1) * (1 - alpha)))) - 1])


def walk_forward_cqr_bands(
    predictions: pd.DataFrame,
    quantiles: pd.DataFrame,
    market_index: pd.DataFrame,
    *,
    alpha: float = 0.2,
    group: str = "period",
    cadence_months: int = 3,
    window_days: int | None = None,
    min_days: int = 120,
    information_lag_days: int = 2,
) -> tuple:
    """
    Conformalised quantile regression: the quantile forecast's bounds, each moved by
    how far past prices overshot it.

    The score at the lower bound is q_α − price, positive when the price fell below it,
    and at the upper bound price − q_{1−α}. Each bound moves by the conformal order
    statistic of its own scores (Romano et al.'s asymmetric variant, since the plan
    trades the two sides separately), per group, so an over-wide quantile model is
    tightened and an over-confident one widened while every day keeps the width its
    features gave it.

    Walk-forward on the same origins as walk_forward_bands. Calibration history ends
    `information_lag_days` before each origin: offers for a day close at 14:00 the day
    before, when only prices to two days before are complete.

    Returns (low_by_date, high_by_date, folds) as walk_forward_bands does.
    """
    if group not in GROUPS:
        raise ValueError(f"group must be one of {GROUPS}, got {group!r}")
    offsets = _quantile_offsets(predictions, quantiles, alpha)
    if offsets.empty:
        return {}, {}, []
    residuals = residual_frame(predictions, market_index)
    scored = offsets.merge(residuals, on=["settlementDate", "settlementPeriod"], how="inner")
    scored = scored.assign(s_lo=scored["lo"] - scored["residual"], s_hi=scored["residual"] - scored["hi"])

    days = pd.DatetimeIndex(sorted(offsets["settlementDate"].unique()))
    origins = pd.date_range(days.min().normalize().replace(day=1), days.max(), freq=f"{cadence_months}MS")
    low_by_date, high_by_date, folds = {}, {}, []
    for k, origin in enumerate(origins):
        upto = origins[k + 1] if k + 1 < len(origins) else days.max() + pd.Timedelta(days=1)
        served = offsets[(offsets["settlementDate"] >= origin) & (offsets["settlementDate"] < upto)]
        if served.empty:
            continue
        cutoff = origin - pd.Timedelta(days=information_lag_days)
        history = scored[scored["settlementDate"] <= cutoff]
        if window_days is not None:
            history = history[history["settlementDate"] > cutoff - pd.Timedelta(days=window_days)]
        calibration_days = history["settlementDate"].nunique()
        record = {"origin": origin.date().isoformat(), "calibration_days": calibration_days,
                  "days": int(served["settlementDate"].nunique())}
        if calibration_days < min_days:
            folds.append({**record, "banded": False})
            continue

        pooled = (_rank_statistic(history["s_lo"].to_numpy(), alpha),
                  _rank_statistic(history["s_hi"].to_numpy(), alpha))
        adjust = {"pooled": pooled}
        for key, rows in history.groupby(group_key(history["settlementPeriod"], group).to_numpy()):
            adjust[key] = ((_rank_statistic(rows["s_lo"].to_numpy(), alpha),
                            _rank_statistic(rows["s_hi"].to_numpy(), alpha))
                           if len(rows) >= MIN_CALIBRATION else pooled)
        keys = group_key(served["settlementPeriod"], group).to_numpy()
        moves = np.array([adjust.get(key, pooled) for key in keys])
        adjusted = served.assign(lo=served["lo"].to_numpy() - moves[:, 0],
                                 hi=served["hi"].to_numpy() + moves[:, 1])
        low, high = _offsets_by_date(adjusted, "lo", "hi")
        low_by_date.update(low)
        high_by_date.update(high)
        width = adjusted["hi"].clip(lower=0) - adjusted["lo"].clip(upper=0)
        folds.append({**record, "banded": True,
                      "move_low": round(float(np.median(moves[:, 0])), 1),
                      "move_high": round(float(np.median(moves[:, 1])), 1),
                      "median_width": round(float(width.median()), 1)})
    return low_by_date, high_by_date, folds


def combine_bands(*bands: tuple) -> tuple:
    """
    The average of several builders' bands, date by date and period by period: the
    "Q-Ens" of O'Connor et al. (2025), who average quantile regression, EnbPI and SPCI.

    A date is banded only when every builder banded it, so the average never quietly
    becomes one member's band. Takes and returns (low_by_date, high_by_date, folds).
    """
    if not bands:
        return {}, {}, []
    dates = sorted(set.intersection(*(set(low) for low, _, _ in bands)))
    mean = lambda tables, day: pd.concat([t[day] for t in tables], axis=1).mean(axis=1)
    lows, highs = [b[0] for b in bands], [b[1] for b in bands]
    return ({day: mean(lows, day) for day in dates}, {day: mean(highs, day) for day in dates},
            [{"members": len(bands), "days": len(dates)}])
