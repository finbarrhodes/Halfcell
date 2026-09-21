"""
Conformal guard bands: the offsets the plan trades against instead of the forecast.

The interval comes from an order statistic of past residuals, so these tests build
residuals with a known distribution and check the band lands where that distribution
says it should - and that no day's band is calibrated on its own future.
"""
import numpy as np
import pandas as pd
import pytest

from src.analysis.intervals import (
    MIN_CALIBRATION,
    conformal_offsets,
    coverage,
    fit_offsets,
    group_key,
    residual_frame,
    walk_forward_bands,
)

PERIODS = list(range(1, 49))


def _tables(n_days=900, start="2021-01-01", scale=lambda sp: 10.0, seed=0, bias=0.0):
    """Forecasts of a flat £80 with residuals whose spread can vary by period."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, periods=n_days, freq="D")
    rows_f, rows_a = [], []
    for day in days:
        for sp in PERIODS:
            error = rng.normal(bias, scale(sp))
            rows_f.append({"settlementDate": day, "settlementPeriod": sp, "prediction": 80.0})
            rows_a.append({"settlementDate": day, "settlementPeriod": sp, "price": 80.0 + error,
                           "dataProvider": "APXMIDP"})
    return pd.DataFrame(rows_f), pd.DataFrame(rows_a)


# --- The order statistic ------------------------------------------------------------

def test_the_band_lands_near_the_quantile_of_past_errors():
    residuals = np.random.default_rng(0).normal(0, 10, 5000)
    low, high = conformal_offsets(residuals, alpha=0.1)
    assert low == pytest.approx(np.quantile(residuals, 0.1), abs=0.5)
    assert high == pytest.approx(np.quantile(residuals, 0.9), abs=0.5)


def test_a_wider_band_is_asked_for_when_the_forecast_is_less_sure():
    rng = np.random.default_rng(1)
    tight = conformal_offsets(rng.normal(0, 5, 4000), alpha=0.1)
    loose = conformal_offsets(rng.normal(0, 40, 4000), alpha=0.1)
    assert (loose[1] - loose[0]) > 4 * (tight[1] - tight[0])


def test_a_smaller_alpha_widens_the_band():
    residuals = np.random.default_rng(2).normal(0, 10, 4000)
    wide, narrow = conformal_offsets(residuals, 0.05), conformal_offsets(residuals, 0.3)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_the_band_never_flatters_a_trade():
    """A one-sided error should not make the plan optimistic on the other side."""
    low, high = conformal_offsets(np.full(500, 25.0), alpha=0.1)   # always under-forecast
    assert low == 0.0 and high > 0


def test_no_history_means_no_band():
    assert conformal_offsets(np.array([]), alpha=0.1) == (0.0, 0.0)


# --- Grouping -----------------------------------------------------------------------

def test_each_grouping_splits_the_day_as_it_says():
    sp = pd.Series(PERIODS, index=PERIODS)
    assert group_key(sp, "period").nunique() == 48
    assert group_key(sp, "block").nunique() == 6
    assert group_key(sp, "day").nunique() == 1
    # EFA 1 runs from 23:00, so the last two periods of the day belong with it
    assert group_key(sp, "block")[47] == group_key(sp, "block")[48] == group_key(sp, "block")[1]


def test_an_unknown_grouping_is_refused():
    with pytest.raises(ValueError, match="group must be one of"):
        group_key(pd.Series(PERIODS, index=PERIODS), "hour")


def test_bands_follow_the_hour_when_calibrated_per_period():
    """Periods 1-24 are quiet, 25-48 are volatile: the bands should say so."""
    f, a = _tables(scale=lambda sp: 5.0 if sp <= 24 else 40.0)
    offsets = fit_offsets(residual_frame(f, a), alpha=0.1, group="period")
    quiet, loud = offsets[10], offsets[40]
    assert (loud[1] - loud[0]) > 4 * (quiet[1] - quiet[0])


def test_one_calibration_set_cannot_tell_the_hours_apart():
    f, a = _tables(scale=lambda sp: 5.0 if sp <= 24 else 40.0)
    offsets = fit_offsets(residual_frame(f, a), alpha=0.1, group="day")
    assert offsets[0] == offsets["pooled"]


def test_a_thin_group_borrows_the_pooled_band():
    f, a = _tables(n_days=MIN_CALIBRATION - 10)
    offsets = fit_offsets(residual_frame(f, a), alpha=0.1, group="period")
    assert offsets[10] == offsets["pooled"]


# --- Walk-forward and coverage ------------------------------------------------------

@pytest.fixture
def banded():
    f, a = _tables(n_days=900, scale=lambda sp: 5.0 if sp <= 24 else 40.0)
    low, high, folds = walk_forward_bands(f, a, alpha=0.1, group="period")
    return f, a, low, high, folds


def test_the_first_days_go_unbanded_until_there_is_history(banded):
    _, _, low, _, folds = banded
    assert pd.Timestamp("2021-01-05") not in low
    assert folds[0]["banded"] is False


def test_later_days_are_banded_per_period(banded):
    _, _, low, high, _ = banded
    day = pd.Timestamp("2022-06-01")
    assert (high[day] - low[day])[40] > 4 * (high[day] - low[day])[10]


def test_coverage_comes_out_near_the_level_it_asked_for(banded):
    f, a, low, high, _ = banded
    table = coverage(f, a, low, high, group="period")
    assert table["coverage"].mean() == pytest.approx(0.8, abs=0.03)
    assert table["coverage"].min() > 0.7          # and per hour, not only on average


def test_a_days_band_cannot_come_from_its_own_future():
    f, a = _tables(n_days=900)
    tampered = a.copy()
    late = tampered["settlementDate"] >= "2022-06-01"
    tampered.loc[late, "price"] = tampered.loc[late, "price"] + 500.0
    day = pd.Timestamp("2022-04-02")
    base = walk_forward_bands(f, a, alpha=0.1, group="day")[1]
    after = walk_forward_bands(f, tampered, alpha=0.1, group="day")[1]
    assert base[day].equals(after[day])
    assert not base[pd.Timestamp("2022-09-02")].equals(after[pd.Timestamp("2022-09-02")])


def test_a_trailing_window_forgets_the_old_regime():
    """Errors were huge in the first year and small since; a window should shrink the band."""
    quiet, loud = _tables(n_days=500, start="2022-06-01", scale=lambda sp: 4.0, seed=3)
    old_f, old_a = _tables(n_days=500, start="2021-01-17", scale=lambda sp: 60.0, seed=4)
    f = pd.concat([old_f, quiet], ignore_index=True)
    a = pd.concat([old_a, loud], ignore_index=True)
    day = pd.Timestamp("2023-06-05")
    everything = walk_forward_bands(f, a, alpha=0.1, group="day")
    windowed = walk_forward_bands(f, a, alpha=0.1, group="day", window_days=180)
    width = lambda bands: float((bands[1][day] - bands[0][day]).iloc[0])
    assert width(windowed) < 0.5 * width(everything)


def test_alpha_outside_the_open_interval_is_refused():
    f, a = _tables(n_days=200)
    with pytest.raises(ValueError, match="alpha"):
        walk_forward_bands(f, a, alpha=0.8)
