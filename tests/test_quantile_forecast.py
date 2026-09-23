"""
The quantile forest: conditional price quantiles, walk-forward.

A synthetic feature matrix whose price noise grows with one feature lets each check
say what the forest ought to find - quantiles that widen where the noise does, stay
ordered, land near their levels - and that no day is forecast by a forest that
trained on it.
"""
import numpy as np
import pandas as pd
import pytest

from src.analysis.quantile_forecast import QuantileForest, column, walk_forward_quantiles

SMALL = dict(n_estimators=40)


def _features(n_days=500, start="2021-01-01", seed=0):
    """Prices of 80 + 20·sp_sin with noise whose sd is 5 when apx_lag_1d < 0 and 40 above."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, periods=n_days, freq="D")
    rows = []
    for day in days:
        for sp in range(1, 49):
            driver = rng.normal()
            angle = 2 * np.pi * sp / 48
            noise = rng.normal(0, 40.0 if driver > 0 else 5.0)
            rows.append({"settlementDate": day, "settlementPeriod": sp, "apx_lag_1d": driver,
                         "sp_sin": np.sin(angle), "sp_cos": np.cos(angle),
                         "apx_price": 80.0 + 20.0 * np.sin(angle) + noise})
    return pd.DataFrame(rows)


def test_quantiles_widen_where_the_noise_does_and_stay_ordered():
    frame = _features(n_days=200)
    X, y = frame[["apx_lag_1d", "sp_sin", "sp_cos"]], frame["apx_price"]
    forest = QuantileForest(**SMALL).fit(X, y)
    probe = pd.DataFrame({"apx_lag_1d": [-1.0, 1.0], "sp_sin": [0.0, 0.0], "sp_cos": [1.0, 1.0]})
    q = forest.predict_quantiles(probe, (0.1, 0.5, 0.9))
    assert (np.diff(q, axis=1) >= 0).all()
    calm, loud = q[0, 2] - q[0, 0], q[1, 2] - q[1, 0]
    assert loud > 4 * calm


def test_walk_forward_quantiles_land_near_their_levels():
    frame = _features()
    table, folds = walk_forward_quantiles(frame, "2021-10-01", "2022-05-15", quantiles=(0.1, 0.5, 0.9),
                                          forest_params=SMALL)
    joined = table.merge(frame[["settlementDate", "settlementPeriod", "apx_price"]],
                         on=["settlementDate", "settlementPeriod"])
    for q in (0.1, 0.5, 0.9):
        assert np.mean(joined["apx_price"] <= joined[column(q)]) == pytest.approx(q, abs=0.05)
    assert [f["origin"] for f in folds] == ["2021-10-01", "2022-01-01", "2022-04-01"]
    assert set(folds[0]["share_below"]) == {"q10", "q50", "q90"}


def test_no_day_is_forecast_by_a_forest_that_saw_it():
    """Wrecking prices from 2022-01-01 on changes nothing before that origin."""
    frame = _features()
    tampered = frame.copy()
    tampered.loc[tampered["settlementDate"] >= "2022-01-01", "apx_price"] += 1000.0
    kwargs = dict(quantiles=(0.5,), forest_params=SMALL)
    base = walk_forward_quantiles(frame, "2021-10-01", "2022-03-31", **kwargs)[0]
    after = walk_forward_quantiles(tampered, "2021-10-01", "2022-03-31", **kwargs)[0]
    before = lambda t: t[t["settlementDate"] < "2022-01-01"].reset_index(drop=True)
    pd.testing.assert_frame_equal(before(base), before(after))


def test_a_price_below_zero_survives_the_transform():
    """The signed log is monotone, so a negative quantile maps back to a negative price."""
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"x": rng.normal(size=2000)})
    y = rng.normal(-30.0, 5.0, size=2000)
    q = QuantileForest(**SMALL).fit(X, y).predict_quantiles(X.iloc[:5], (0.5,))
    assert (q < 0).all() and np.abs(q + 30.0).max() < 5.0


def test_quantiles_come_back_in_the_order_they_were_asked_for():
    """Levels given high-to-low keep their columns: nothing is re-sorted under them."""
    frame = _features(n_days=120)
    X, y = frame[["apx_lag_1d", "sp_sin", "sp_cos"]], frame["apx_price"]
    forest = QuantileForest(**SMALL).fit(X, y)
    ascending = forest.predict_quantiles(X.iloc[:50], (0.1, 0.9))
    descending = forest.predict_quantiles(X.iloc[:50], (0.9, 0.1))
    np.testing.assert_allclose(descending, ascending[:, ::-1])
    assert (descending[:, 0] > descending[:, 1]).all()
