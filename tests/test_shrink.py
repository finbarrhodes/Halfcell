"""
How far to believe a day's forecast shape.

The weight is the scaling that best matches what happened, `Σ f·s / Σ f²`, fitted
only on days that came before. These tests build days whose true scaling is known,
so the estimator can be checked against it rather than against itself.
"""
import numpy as np
import pandas as pd
import pytest

from src.analysis.shrink import (
    MAX_WEIGHT,
    amplitude,
    day_matrix,
    fit,
    naive_predictions,
    shapes,
    slope,
    walk_forward_slopes,
)

PERIODS = range(1, 49)


def _days(n, start="2021-01-01"):
    return pd.date_range(start, periods=n, freq="D")


def _forecast_shape(rng, n):
    """A daily shape: one sine wave per day at a random amplitude."""
    wave = np.sin(np.linspace(0, 2 * np.pi, 48, endpoint=False))
    return np.outer(rng.uniform(5, 60, n), wave)


def _tables(n=800, true_slope=0.6, noise=1.0, seed=0, start="2021-01-01"):
    """Forecast and actual tables where the actual shape is `true_slope` x the forecast's."""
    rng = np.random.default_rng(seed)
    days = _days(n, start)
    f = _forecast_shape(rng, n)
    a = true_slope * f + rng.normal(0, noise, f.shape)
    wide = lambda m, base: pd.DataFrame(m + base, index=days, columns=list(PERIODS))
    return wide(f, 80.0), wide(a, 80.0)


def _long(wide: pd.DataFrame, value: str) -> pd.DataFrame:
    out = wide.stack().rename(value).reset_index()
    out.columns = ["settlementDate", "settlementPeriod", value]
    return out


def _market_index(wide: pd.DataFrame) -> pd.DataFrame:
    return _long(wide, "price").assign(dataProvider="APXMIDP")


# --- The statistic ------------------------------------------------------------------

def test_the_weight_recovers_the_scaling_that_actually_happened():
    f, a = _tables(true_slope=0.6, noise=1.0)
    assert slope(*shapes(f, a)) == pytest.approx(0.6, abs=0.02)


def test_a_forecast_needing_no_rescaling_scores_one():
    f, a = _tables(true_slope=1.0, noise=0.5)
    assert slope(*shapes(f, a)) == pytest.approx(1.0, abs=0.02)


def test_noise_in_the_forecast_pulls_the_weight_below_one():
    """
    Shape the forecast draws that does not arrive should not be believed. Noise in the
    forecast does that; noise in the outcome alone leaves the scaling at one, which is
    why the weight answers "how much of this shape is real" rather than "how big is the
    error".
    """
    rng = np.random.default_rng(7)
    days, wave = _days(900), np.sin(np.linspace(0, 2 * np.pi, 48, endpoint=False))
    signal = np.outer(rng.uniform(5, 60, 900), wave)
    wide = lambda m: pd.DataFrame(m + 80.0, index=days, columns=list(PERIODS))
    noisy_forecast = wide(signal + rng.normal(0, 25, signal.shape))
    assert slope(*shapes(noisy_forecast, wide(signal))) < 0.9
    # the same signal, noise on the outcome instead
    assert slope(*shapes(wide(signal), wide(signal + rng.normal(0, 25, signal.shape)))) == pytest.approx(1.0, abs=0.03)


def test_shapes_remove_each_days_own_level():
    f, a = _tables(n=3)
    fs, _ = shapes(f, a)
    np.testing.assert_allclose(fs.mean(axis=1).to_numpy(), 0.0, atol=1e-9)


def test_amplitude_is_the_forecast_spread():
    f, _ = _tables(n=5)
    np.testing.assert_allclose(amplitude(f).to_numpy(), (f.max(axis=1) - f.min(axis=1)).to_numpy())


# --- Buckets ------------------------------------------------------------------------

def test_loud_days_earn_a_smaller_weight_than_quiet_ones():
    """Regression to the mean: build days where wide forecasts overstate the shape."""
    rng = np.random.default_rng(3)
    days, wave = _days(1200), np.sin(np.linspace(0, 2 * np.pi, 48, endpoint=False))
    width = rng.uniform(5, 80, 1200)
    f = np.outer(width, wave)
    # wide days realise half of their shape, narrow days all of it
    realised = np.where(width > 40, 0.5, 1.0)
    a = (realised[:, None] * f) + rng.normal(0, 2, f.shape)
    wide = lambda m: pd.DataFrame(m + 80.0, index=days, columns=list(PERIODS))
    buckets = fit(wide(f), wide(a))
    # the bucket is chosen by the day's spread, which is twice the wave's amplitude
    assert buckets.weight(20.0) > buckets.weight(150.0) + 0.3
    assert buckets.weight(150.0) == pytest.approx(0.5, abs=0.06)
    assert buckets.weight(20.0) == pytest.approx(1.0, abs=0.06)


def test_a_short_history_falls_back_to_one_pooled_weight():
    f, a = _tables(n=200, true_slope=0.7)
    buckets = fit(f, a)
    assert not buckets.slopes.size
    assert buckets.weight(10.0) == buckets.weight(500.0) == pytest.approx(0.7, abs=0.03)


# --- Walk-forward -------------------------------------------------------------------

@pytest.fixture
def walked():
    f, a = _tables(n=1200, true_slope=0.6, noise=8.0)
    return walk_forward_slopes(_long(f, "prediction"), _market_index(a), fallback=0.5)


def test_every_day_gets_a_weight(walked):
    weights, _ = walked
    assert len(weights) == 1200
    assert all(0.0 <= w <= MAX_WEIGHT for w in weights.values())


def test_the_first_days_take_the_fallback_until_there_is_history(walked):
    weights, folds = walked
    early = [w for d, w in weights.items() if d < pd.Timestamp("2021-04-01")]
    assert early and set(early) == {0.5}
    assert folds[0]["buckets"] is None


def test_later_days_are_fitted_near_the_true_scaling(walked):
    weights, _ = walked
    late = [w for d, w in weights.items() if d >= pd.Timestamp("2023-01-01")]
    assert np.mean(late) == pytest.approx(0.6, abs=0.1)


def test_a_days_weight_cannot_depend_on_days_after_it():
    f, a = _tables(n=1200, true_slope=0.6, noise=8.0)
    tampered = a.copy()
    tampered.loc[tampered.index >= "2023-06-01"] *= -3.0        # nonsense, but only later
    before = pd.Timestamp("2023-04-15")
    base = walk_forward_slopes(_long(f, "prediction"), _market_index(a))[0]
    after = walk_forward_slopes(_long(f, "prediction"), _market_index(tampered))[0]
    assert base[before] == after[before]
    assert base[pd.Timestamp("2023-09-15")] != after[pd.Timestamp("2023-09-15")]


def test_the_risk_factor_scales_every_weight():
    f, a = _tables(n=1200, true_slope=0.8, noise=4.0)
    full = walk_forward_slopes(_long(f, "prediction"), _market_index(a), fallback=0.5)[0]
    half = walk_forward_slopes(_long(f, "prediction"), _market_index(a), fallback=0.5,
                               risk_factor=0.5)[0]
    fitted = [d for d in full if full[d] != 0.5]
    assert fitted
    assert all(half[d] == pytest.approx(0.5 * full[d], rel=1e-9) for d in fitted)


def test_weights_are_capped_so_a_quiet_day_cannot_ask_for_the_moon():
    f, a = _tables(n=1200, true_slope=4.0, noise=1.0)
    weights = walk_forward_slopes(_long(f, "prediction"), _market_index(a))[0]
    assert max(weights.values()) == pytest.approx(MAX_WEIGHT)


def test_the_naive_forecast_is_the_day_before_yesterdays_prices():
    _, a = _tables(n=10)
    naive = day_matrix(naive_predictions(_market_index(a)), "prediction")
    day = pd.Timestamp("2021-01-05")
    np.testing.assert_allclose(naive.loc[day].to_numpy(),
                               a.loc[day - pd.Timedelta(days=2)].to_numpy())
