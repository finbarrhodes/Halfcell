"""REPD trailing extrapolation.

REPD is published quarterly and always lags the half-hourly price data, so the
trailing months of the fleet series are projected. Evaluating the fitted OLS
line directly could place the first projected month *below* the last measured
one, which is meaningless for a cumulative capacity series — the projection is
anchored at the last measured value instead.
"""
import numpy as np
import pandas as pd
import pytest

from src.data_collection.repd_collector import EXTRAP_FIT_MONTHS, REPDCollector


def _projects(n_months=30, start="2023-01-01", per_month=100.0):
    """Synthetic battery projects: a steady build-out, one per month."""
    months = pd.date_range(start, periods=n_months, freq="MS")
    return pd.DataFrame({
        "operational_date": months,
        "installed_capacity_mw": [per_month] * n_months,
    })


def _series(n_months=30, end=None):
    c = REPDCollector(local_path="unused")
    return c.build_monthly_capacity_series(
        _projects(n_months), start_date="2023-01-01", end_date=end
    )


def test_fit_window_is_a_named_constant():
    assert isinstance(EXTRAP_FIT_MONTHS, int) and EXTRAP_FIT_MONTHS > 0


def test_series_is_monotonic_through_the_projection():
    """Installed capacity is cumulative — it cannot decrease, projected or not."""
    s = _series(n_months=30, end="2026-06-01")
    diffs = s["bess_fleet_mw"].diff().dropna()
    assert (diffs >= -1e-6).all(), (
        f"series decreases by up to {diffs.min():.1f} MW — the projection is not "
        "anchored at the last measured value"
    )


def test_projection_starts_at_or_above_last_measured_value():
    s = _series(n_months=30, end="2026-06-01")
    if "is_extrapolated" not in s.columns or not s["is_extrapolated"].any():
        pytest.skip("no months were projected for this range")
    last_measured = s.loc[~s["is_extrapolated"], "bess_fleet_mw"].iloc[-1]
    first_projected = s.loc[s["is_extrapolated"], "bess_fleet_mw"].iloc[0]
    assert first_projected >= last_measured - 1e-6


def test_extrapolation_flag_is_present_and_typed():
    s = _series(n_months=30, end="2026-06-01")
    assert "is_extrapolated" in s.columns, (
        "downstream needs to distinguish measured from projected capacity"
    )
    assert s["is_extrapolated"].dtype == bool


def test_measured_months_are_not_flagged():
    s = _series(n_months=30, end="2026-06-01")
    measured = s.loc[~s["is_extrapolated"]]
    assert len(measured) > 0
    assert measured["month"].max() < s.loc[s["is_extrapolated"], "month"].min()


def test_no_projection_when_range_ends_at_measured_data():
    s = _series(n_months=30, end="2025-06-01")
    assert not s["is_extrapolated"].any()


def test_capacity_never_negative():
    s = _series(n_months=30, end="2026-06-01")
    assert (s["bess_fleet_mw"] >= 0).all()
