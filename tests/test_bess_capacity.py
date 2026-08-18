"""BESS fleet capacity feature construction.

Regression cover for a silent-wrong-value bug: months past the end of the
(quarterly, always-lagging) REPD extract were filled with 0 MW, so the model
was trained and evaluated on a national fleet that had apparently collapsed to
zero. Trailing months must carry the last known capacity forward; only months
before the fleet existed are genuinely zero.
"""
import numpy as np
import pandas as pd
import pytest

from src.analysis.features import build_feature_matrix


def _price_frame(start="2024-01-01", days=200):
    """Minimal APXMIDP-shaped frame: 48 periods a day, deterministic prices."""
    dates = pd.date_range(start, periods=days, freq="D")
    rows = []
    for i, d in enumerate(dates):
        for sp in range(1, 49):
            rows.append({
                "settlementDate": d, "settlementPeriod": sp, "dataProvider": "APXMIDP",
                "price": 50 + 20 * np.sin(sp / 48 * 2 * np.pi) + (i % 7),
            })
    return pd.DataFrame(rows)


def _generation_frame(start="2024-01-01", days=200):
    dates = pd.date_range(start, periods=days, freq="D")
    return pd.DataFrame([
        {"settlementDate": d, "fuelGroup": fuel, "generation": gen}
        for d in dates
        for fuel, gen in [("Wind", 8000), ("Gas", 12000), ("Nuclear", 5000)]
    ])


def _capacity_frame(months, start="2024-01-01"):
    idx = pd.date_range(start, periods=months, freq="MS")
    return pd.DataFrame({
        "month_start": idx,
        "bess_fleet_mw": [1000 + 50 * i for i in range(months)],
    })


def test_trailing_months_carry_last_capacity_forward():
    """Price data extends past the REPD extract; capacity must not drop to zero."""
    prices, generation = _price_frame(), _generation_frame()
    capacity = _capacity_frame(months=3)  # Jan-Mar, while prices run ~7 months

    fm = build_feature_matrix(prices, generation, capacity)
    last_known = capacity["bess_fleet_mw"].iloc[-1]

    trailing = fm[fm["settlementDate"] >= "2024-05-01"]
    assert not trailing.empty
    assert (trailing["bess_fleet_mw"] == last_known).all(), (
        "months past the REPD extract must carry the last measured capacity forward"
    )


def test_no_zero_capacity_once_the_fleet_exists():
    fm = build_feature_matrix(_price_frame(), _generation_frame(), _capacity_frame(months=3))
    assert (fm["bess_fleet_mw"] > 0).all()


def test_months_before_first_repd_entry_are_zero():
    """Leading gaps are genuinely zero — there was no fleet to speak of."""
    prices, generation = _price_frame(start="2024-01-01"), _generation_frame(start="2024-01-01")
    capacity = _capacity_frame(months=2, start="2024-04-01")  # starts after the prices do

    fm = build_feature_matrix(prices, generation, capacity)
    leading = fm[fm["settlementDate"] < "2024-04-01"]
    assert not leading.empty
    assert (leading["bess_fleet_mw"] == 0).all()


def test_capacity_is_never_from_the_future():
    """Each row uses the capacity for the month of D-1 — no look-ahead."""
    fm = build_feature_matrix(_price_frame(), _generation_frame(), _capacity_frame(months=8))
    march = fm[(fm["settlementDate"] >= "2024-03-02") & (fm["settlementDate"] < "2024-04-01")]
    assert (march["bess_fleet_mw"] == 1000 + 50 * 2).all()


def test_suppression_feature_is_finite_and_bounded():
    fm = build_feature_matrix(_price_frame(), _generation_frame(), _capacity_frame(months=8))
    s = fm["bess_spread_suppression"]
    assert np.isfinite(s).all()
    assert (s >= 0).all()


def test_missing_capacity_frame_is_tolerated():
    fm = build_feature_matrix(_price_frame(), _generation_frame(), None)
    assert "bess_fleet_mw" not in fm.columns
    assert len(fm) > 0
