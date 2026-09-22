"""
Error bars for the forecast's value.

The statistics are only worth having if they widen when the data is dependent and
stay honest when it is not, so these tests build both kinds of series and check the
tools behave differently on them.
"""
import numpy as np
import pandas as pd
import pytest

from src.analysis.significance import (
    bootstrap_interval,
    diebold_mariano,
    moving_block_bootstrap,
    newey_west_variance,
    paired_daily,
)


def _ar1(n, rho=0.9, sd=1.0, mean=0.0, seed=0):
    """A series that remembers yesterday, as revenue and forecast errors both do."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + rng.normal(0, sd)
    return out + mean


# --- Long-run variance ---------------------------------------------------------------

def test_with_no_lags_it_is_the_plain_variance_of_the_mean():
    x = np.random.default_rng(0).normal(0, 1, 500)
    assert newey_west_variance(x, lags=0) == pytest.approx(x.var() / len(x), rel=1e-9)


def test_dependence_inflates_the_long_run_variance():
    independent = np.random.default_rng(1).normal(0, 1, 2000)
    dependent = _ar1(2000, rho=0.9, seed=1)
    ratio = newey_west_variance(dependent, lags=30) / newey_west_variance(dependent, lags=0)
    assert ratio > 3
    assert newey_west_variance(independent, lags=30) == pytest.approx(
        newey_west_variance(independent, lags=0), rel=0.4)


# --- Diebold-Mariano -----------------------------------------------------------------

def test_identical_forecasts_are_not_distinguishable():
    loss = np.abs(_ar1(800, seed=2)) + 1
    out = diebold_mariano(loss, loss.copy())
    assert out["mean_difference"] == 0
    assert np.isnan(out["t_stat"]) or out["p_value"] > 0.99


def test_a_clearly_better_forecast_is_detected_and_signed():
    rng = np.random.default_rng(3)
    good = np.abs(rng.normal(0, 1, 900))
    bad = good + np.abs(rng.normal(2, 1, 900))          # loses more on every day
    out = diebold_mariano(good, bad)
    assert out["mean_difference"] > 0                    # positive: the first argument wins
    assert out["p_value"] < 1e-6
    assert diebold_mariano(bad, good)["mean_difference"] < 0


def test_a_small_edge_on_dependent_data_is_not_oversold():
    """The same tiny edge looks significant if the days are treated as independent."""
    edge = 0.05
    loss_a = np.abs(_ar1(1200, rho=0.95, seed=4))
    loss_b = loss_a + edge
    honest = diebold_mariano(loss_a, loss_b)
    naive_lags = diebold_mariano(loss_a, loss_b, lags=0, harvey_correction=False)
    assert abs(honest["t_stat"]) < abs(naive_lags["t_stat"])


def test_mismatched_series_are_refused():
    with pytest.raises(ValueError, match="same length"):
        diebold_mariano(np.zeros(10), np.zeros(9))


def test_missing_days_are_dropped_pairwise():
    a = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0])
    b = np.array([2.0, 2.0, np.nan, 5.0, 6.0, 7.0])
    assert diebold_mariano(a, b)["n"] == 4


# --- Block bootstrap -----------------------------------------------------------------

def test_every_resample_is_the_length_of_the_series():
    x = np.arange(100.0)
    draws = moving_block_bootstrap(x, block=7, resamples=50)
    assert draws.shape == (50,)
    assert draws.min() >= x.min() and draws.max() <= x.max()


def test_a_block_of_one_is_the_ordinary_bootstrap():
    x = _ar1(300, seed=5)
    blocks = moving_block_bootstrap(x, block=1, resamples=2000, seed=1).std()
    assert blocks == pytest.approx(x.std() / np.sqrt(len(x)), rel=0.15)


def test_blocks_widen_the_interval_on_dependent_data():
    x = _ar1(1500, rho=0.95, seed=6)
    independent = bootstrap_interval(x, block=1, resamples=2000)
    blocked = bootstrap_interval(x, block=28, resamples=2000)
    assert (blocked["high"] - blocked["low"]) > 2 * (independent["high"] - independent["low"])


def test_the_interval_brackets_the_mean_and_scales():
    x = _ar1(1000, rho=0.5, mean=4.0, seed=7)
    out = bootstrap_interval(x, block=14, resamples=2000)
    assert out["low"] < out["mean"] < out["high"]
    scaled = bootstrap_interval(x, block=14, resamples=2000, scale=365.25)
    assert scaled["mean"] == pytest.approx(out["mean"] * 365.25, rel=1e-9)
    assert scaled["low"] == pytest.approx(out["low"] * 365.25, rel=1e-9)


def test_a_difference_that_is_all_noise_straddles_zero():
    x = _ar1(1200, rho=0.8, mean=0.0, seed=8)
    out = bootstrap_interval(x, block=28, resamples=4000)
    assert out["low"] < 0 < out["high"]
    # the realised path's mean can sit either side of zero; what matters is that the
    # interval does not commit to a sign
    assert 0.02 < out["share_below_zero"] < 0.98


def test_a_real_difference_clears_zero():
    x = _ar1(1200, rho=0.8, mean=3.0, seed=9)
    out = bootstrap_interval(x, block=28, resamples=4000)
    assert out["low"] > 0 and out["share_below_zero"] < 0.01


def test_an_empty_series_gives_no_interval_rather_than_an_error():
    out = bootstrap_interval(np.array([]))
    assert np.isnan(out["mean"]) and np.isnan(out["low"])


# --- Pairing ---------------------------------------------------------------------------

def test_pairing_keeps_the_days_both_strategies_covered():
    days = pd.date_range("2025-01-01", periods=10, freq="D")
    left = pd.DataFrame({"date": days, "net_revenue": np.arange(10.0)})
    right = pd.DataFrame({"date": days[2:], "net_revenue": np.arange(8.0)})
    paired = paired_daily(left, right)
    assert len(paired) == 8
    assert paired["difference"].iloc[0] == pytest.approx(2.0)
