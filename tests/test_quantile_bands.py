"""
Bands built from a quantile forecast, and the scores every band is judged by.

The quantile tables here are written by hand around a residual distribution the
test controls, so each check can say where a band ought to land: CQR should pull an
over-wide or over-narrow quantile model back to the true quantiles, the scores should
prefer the true quantiles to anything wider or narrower, and no day's band may be
calibrated on prices that were not known when its offers closed.
"""
import numpy as np
import pandas as pd
import pytest

from src.analysis.intervals import (
    band_frame,
    combine_bands,
    quantile_bands,
    score_bands,
    walk_forward_cqr_bands,
)
from src.analysis.quantile_forecast import column, pinball

PERIODS = [1, 2, 3, 4]
Z80 = 0.8416                     # standard normal 80th percentile


def _tables(n_days=700, start="2021-01-01", sigma=10.0, quantile_scale=1.0, seed=0):
    """
    A flat £80 forecast, prices with N(0, sigma) errors, and a quantile table that
    believes the errors are N(0, quantile_scale * sigma).
    """
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, periods=n_days, freq="D")
    rows_f, rows_a, rows_q = [], [], []
    for day in days:
        for sp in PERIODS:
            rows_f.append({"settlementDate": day, "settlementPeriod": sp, "prediction": 80.0})
            rows_a.append({"settlementDate": day, "settlementPeriod": sp, "dataProvider": "APXMIDP",
                           "price": 80.0 + rng.normal(0, sigma)})
            spread = Z80 * quantile_scale * sigma
            rows_q.append({"settlementDate": day, "settlementPeriod": sp, "q20": 80.0 - spread,
                           "q50": 80.0, "q80": 80.0 + spread, "origin": day.replace(day=1)})
    return pd.DataFrame(rows_f), pd.DataFrame(rows_a), pd.DataFrame(rows_q)


def _band(low, high, days, periods=PERIODS):
    return ({d: pd.Series(low, index=periods) for d in days},
            {d: pd.Series(high, index=periods) for d in days})


# --- Scores ---------------------------------------------------------------------------

def test_the_true_quantiles_cover_what_they_should_on_each_side():
    f, a, _ = _tables(n_days=2000)
    days = pd.to_datetime(f["settlementDate"]).unique()
    low, high = _band(-Z80 * 10, Z80 * 10, days)
    table = score_bands(band_frame(f, a, low, high), alpha=0.2)
    assert table["coverage"].iloc[0] == pytest.approx(0.6, abs=0.02)
    assert table["below"].iloc[0] == pytest.approx(0.2, abs=0.015)
    assert table["above"].iloc[0] == pytest.approx(0.2, abs=0.015)


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_the_scores_prefer_the_true_quantiles_to_a_wrong_width(scale):
    """Winkler and pinball are proper: neither a narrower nor a wider band beats the truth."""
    f, a, _ = _tables(n_days=2000)
    days = pd.to_datetime(f["settlementDate"]).unique()
    truth = score_bands(band_frame(f, a, *_band(-Z80 * 10, Z80 * 10, days)), alpha=0.2)
    wrong = score_bands(band_frame(f, a, *_band(-Z80 * 10 * scale, Z80 * 10 * scale, days)), alpha=0.2)
    assert truth["winkler"].iloc[0] < wrong["winkler"].iloc[0]
    assert truth["pinball"].iloc[0] < wrong["pinball"].iloc[0]


def test_pinball_is_the_quantile_loss():
    actual = np.array([10.0, 0.0])
    # forecast 5 for the 0.9 quantile: under by 5 once (0.9 x 5), over by 5 once (0.1 x 5)
    assert pinball(actual, np.array([5.0, 5.0]), 0.9) == pytest.approx((4.5 + 0.5) / 2)


def test_scores_split_by_year_and_block():
    f, a, _ = _tables(n_days=800)
    days = pd.to_datetime(f["settlementDate"]).unique()
    frame = band_frame(f, a, *_band(-8.0, 8.0, days))
    assert list(score_bands(frame, 0.2, by="year")["year"]) == [2021, 2022, 2023]
    assert score_bands(frame, 0.2, by="block")["n"].sum() == len(frame)
    with pytest.raises(ValueError, match="by must be one of"):
        score_bands(frame, 0.2, by="month")


def test_a_period_the_band_leaves_out_is_not_scored_as_zero_width():
    f, a, _ = _tables(n_days=50)
    days = pd.to_datetime(f["settlementDate"]).unique()
    low, high = _band(-8.0, 8.0, days, periods=[1, 2])
    assert set(band_frame(f, a, low, high)["settlementPeriod"]) == {1, 2}


# --- Quantile bands -------------------------------------------------------------------

def test_quantile_bands_are_the_quantiles_less_the_forecast():
    f, _, q = _tables(n_days=30)
    low, high, _ = quantile_bands(f, q, alpha=0.2)
    day = pd.Timestamp("2021-01-10")
    assert low[day][2] == pytest.approx(-Z80 * 10)
    assert high[day][2] == pytest.approx(Z80 * 10)


def test_a_quantile_on_the_wrong_side_of_the_forecast_is_capped():
    """A band never makes a sale look better than the forecast promised."""
    f, _, q = _tables(n_days=30)
    q["q20"] = 85.0                                     # above the £80 forecast
    low, _, _ = quantile_bands(f, q, alpha=0.2)
    assert (low[pd.Timestamp("2021-01-10")] == 0.0).all()


def test_a_level_the_table_does_not_hold_is_refused():
    f, _, q = _tables(n_days=30)
    with pytest.raises(ValueError, match="no \\['q10', 'q90'\\]"):
        quantile_bands(f, q, alpha=0.1)
    assert column(0.1) == "q10" and column(0.9) == "q90"


# --- CQR ------------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [0.4, 2.5])
def test_cqr_moves_a_miscalibrated_quantile_model_back_to_its_level(scale):
    """Over-confident quantiles are widened and over-cautious ones tightened."""
    f, a, q = _tables(n_days=900, quantile_scale=scale)
    raw = score_bands(band_frame(f, a, *quantile_bands(f, q, 0.2)[:2]), 0.2)
    low, high, folds = walk_forward_cqr_bands(f, q, a, alpha=0.2, group="period")
    fixed = score_bands(band_frame(f, a, low, high), 0.2)
    assert abs(raw["coverage"].iloc[0] - 0.6) > 0.15
    assert fixed["coverage"].iloc[0] == pytest.approx(0.6, abs=0.04)
    assert fixed["winkler"].iloc[0] < raw["winkler"].iloc[0]
    assert any(fold["banded"] for fold in folds)


def test_cqr_keeps_the_shape_the_quantile_model_gave_each_day():
    """A day the model calls twice as uncertain stays wider after calibration."""
    f, a, q = _tables(n_days=900)
    loud = pd.to_datetime(q["settlementDate"]).dt.dayofweek == 0
    q.loc[loud, "q20"] = 80.0 - 2 * Z80 * 10
    q.loc[loud, "q80"] = 80.0 + 2 * Z80 * 10
    low, high, _ = walk_forward_cqr_bands(f, q, a, alpha=0.2, group="day")
    monday, tuesday = pd.Timestamp("2023-05-01"), pd.Timestamp("2023-05-02")
    assert (high[monday] - low[monday])[1] > 1.5 * (high[tuesday] - low[tuesday])[1]


def test_cqr_is_not_calibrated_on_prices_unknown_at_the_bid():
    """The day before an origin is not complete when its offers close, so it cannot count."""
    f, a, q = _tables(n_days=900)
    origin = pd.Timestamp("2022-07-01")
    tampered = a.copy()
    eve = pd.to_datetime(tampered["settlementDate"]) == origin - pd.Timedelta(days=1)
    tampered.loc[eve, "price"] += 5000.0
    base = walk_forward_cqr_bands(f, q, a, alpha=0.2, group="day")[1]
    after = walk_forward_cqr_bands(f, q, tampered, alpha=0.2, group="day")[1]
    assert base[origin].equals(after[origin])
    # two days before is known, and does move the band
    tampered.loc[pd.to_datetime(tampered["settlementDate"]) == origin - pd.Timedelta(days=2), "price"] += 5000.0
    moved = walk_forward_cqr_bands(f, q, tampered, alpha=0.2, group="day")[1]
    assert not base[origin].equals(moved[origin])


def test_cqr_leaves_days_without_history_unbanded():
    f, a, q = _tables(n_days=200)
    low, _, folds = walk_forward_cqr_bands(f, q, a, alpha=0.2, min_days=120)
    assert pd.Timestamp("2021-02-01") not in low
    assert folds[0]["banded"] is False


# --- Combining ------------------------------------------------------------------------

def test_combined_bands_average_their_members_on_the_days_all_of_them_band():
    days = pd.date_range("2022-01-01", periods=3, freq="D")
    one = _band(-10.0, 10.0, days) + ([],)
    two = _band(-20.0, 30.0, days[1:]) + ([],)
    low, high, _ = combine_bands(one, two)
    assert sorted(low) == list(days[1:])
    assert low[days[1]][1] == pytest.approx(-15.0)
    assert high[days[2]][4] == pytest.approx(20.0)
