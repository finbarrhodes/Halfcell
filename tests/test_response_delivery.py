"""Response delivery: NESO's response curves, direction, settlement periods and energy."""
import numpy as np
import pandas as pd
import pytest

from src.analysis.response_delivery import (
    DELIVERY_COLUMNS,
    build_delivery_table,
    delivery_by_settlement_period,
    product_shares,
    response_share,
    settlement_keys,
)


# --- Response curves (Response Service Terms, Table 1) ---------------------------------

@pytest.mark.parametrize("service", ["DC", "DM", "DR"])
def test_nothing_is_delivered_inside_the_deadband(service):
    assert np.all(response_share([0.0, 0.01, -0.015], service) == 0.0)


def test_dc_rises_to_five_percent_at_its_knee_then_to_full_at_half_a_hertz():
    shares = response_share([0.1075, 0.2, 0.35, 0.5, 0.8], "DC")
    np.testing.assert_allclose(shares, [0.025, 0.05, 0.525, 1.0, 1.0])


def test_dm_has_its_knee_at_a_tenth_of_a_hertz_and_saturates_at_a_fifth():
    shares = response_share([0.1, 0.15, 0.2, 0.3], "DM")
    np.testing.assert_allclose(shares, [0.05, 0.525, 1.0, 1.0])


def test_dr_is_one_straight_line_from_the_deadband_to_a_fifth_of_a_hertz():
    shares = response_share([0.1075, 0.2, 0.25], "DR")
    np.testing.assert_allclose(shares, [0.5, 1.0, 1.0])


def test_low_products_answer_under_frequency_and_high_products_over_frequency():
    shares = product_shares([49.8, 50.2])
    np.testing.assert_allclose(shares["dr_low"], [1.0, 0.0])
    np.testing.assert_allclose(shares["dr_high"], [0.0, 1.0])
    np.testing.assert_allclose(shares["dc_low"], [0.05, 0.0])


# --- Settlement periods --------------------------------------------------------------

def test_winter_periods_match_utc():
    dates, periods = settlement_keys(["2026-01-08 00:00:00", "2026-01-08 23:59:59"])
    assert list(dates) == [pd.Timestamp("2026-01-08")] * 2
    assert list(periods) == [1, 48]


def test_summer_time_shifts_utc_an_hour_later():
    dates, periods = settlement_keys(["2026-06-01 00:00:00", "2026-05-31 23:00:00"])
    assert list(dates) == [pd.Timestamp("2026-06-01")] * 2
    assert list(periods) == [3, 1]


def test_clock_change_days_have_fifty_and_forty_six_periods():
    dates, periods = settlement_keys(["2025-10-26 23:30:00", "2026-03-29 22:30:00"])
    assert list(dates) == [pd.Timestamp("2025-10-26"), pd.Timestamp("2026-03-29")]
    assert list(periods) == [50, 46]


# --- Energy per settlement period ------------------------------------------------------

def _seconds(start, n, hz):
    return pd.DataFrame({"dtm": pd.date_range(start, periods=n, freq="s", tz="UTC"), "f": hz})


def test_a_half_hour_at_49_8_hz_delivers_half_a_mwh_per_mw_of_dr_low():
    table = delivery_by_settlement_period(_seconds("2026-01-08 00:00", 1800, 49.8))
    row = table.iloc[0]
    assert (row.settlementDate, row.settlementPeriod) == (pd.Timestamp("2026-01-08"), 1)
    assert row.dr_low == pytest.approx(0.5)
    assert row.dm_low == pytest.approx(0.5)
    assert row.dc_low == pytest.approx(0.025)
    assert row.dr_high == 0.0 and row.coverage == pytest.approx(1.0)


def test_missing_seconds_count_against_coverage_not_energy():
    freq = _seconds("2026-01-08 00:00", 1800, 49.8)
    freq.loc[900:, "f"] = np.nan
    row = delivery_by_settlement_period(freq).iloc[0]
    assert row.dr_low == pytest.approx(0.5)
    assert row.coverage == pytest.approx(0.5)


def test_every_product_column_is_reported():
    table = delivery_by_settlement_period(_seconds("2026-01-08 00:00", 3600, 50.1))
    assert set(DELIVERY_COLUMNS) <= set(table.columns) and len(table) == 2


def test_a_period_split_across_files_keeps_the_fuller_reading(tmp_path):
    first = _seconds("2026-01-08 00:00", 600, 49.9)
    second = _seconds("2026-01-08 00:00", 1800, 49.8)
    for name, frame in (("a.csv", first), ("b.csv", second)):
        frame.assign(dtm=frame["dtm"].dt.strftime("%Y-%m-%d %H:%M:%S")).to_csv(tmp_path / name, index=False)
    table = build_delivery_table([tmp_path / "a.csv", tmp_path / "b.csv"])
    assert len(table) == 1
    assert table.iloc[0].coverage == pytest.approx(1.0)
    assert table.iloc[0].dr_low == pytest.approx(0.5)


# --- The prepare_data step ---------------------------------------------------------------

def _write_month(directory, frame):
    directory.mkdir(parents=True, exist_ok=True)
    frame.assign(dtm=frame["dtm"].dt.strftime("%Y-%m-%d %H:%M:%S")).to_csv(
        directory / "frequency_2026-01.csv", index=False)


def test_prepare_step_builds_the_table_and_later_pulls_win_on_append(tmp_path):
    from scripts.prepare_data import prepare_response_delivery

    raw, delta, processed = tmp_path / "raw", tmp_path / "delta", tmp_path / "processed"
    processed.mkdir()
    _write_month(raw / "frequency", _seconds("2026-01-08 00:00", 3600, 49.8))
    prepare_response_delivery(raw, processed, append=False)
    table = pd.read_parquet(processed / "response_delivery.parquet")
    assert len(table) == 2 and table.dr_low.tolist() == pytest.approx([0.5, 0.5])

    # A later pull covering only the second period replaces it and keeps the first
    _write_month(delta / "frequency", _seconds("2026-01-08 00:30", 1800, 50.2))
    prepare_response_delivery(delta, processed, append=True)
    table = pd.read_parquet(processed / "response_delivery.parquet")
    assert len(table) == 2
    assert table.dr_low.tolist() == pytest.approx([0.5, 0.0])
    assert table.dr_high.tolist() == pytest.approx([0.0, 0.5])


def test_prepare_step_leaves_the_table_alone_without_frequency_files(tmp_path):
    from scripts.prepare_data import prepare_response_delivery

    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame({"settlementDate": [pd.Timestamp("2026-01-08")], "settlementPeriod": [1]}).to_parquet(
        processed / "response_delivery.parquet", index=False)
    prepare_response_delivery(tmp_path / "empty", processed, append=True)
    assert len(pd.read_parquet(processed / "response_delivery.parquet")) == 1
