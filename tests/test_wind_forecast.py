"""
Day-ahead wind forecast: parsing, and the rule that a forecast is only usable if
it existed when the bid was made.
"""
from datetime import date

import pandas as pd
import pytest

from src.analysis.neso_rules import EAC_GO_LIVE
from src.analysis.wind_forecast import (
    COLUMNS,
    bid_deadline,
    build_wind_forecast_table,
    flag_published_before_deadline,
    read_forecast_file,
)

RAW_COLUMNS = ["Datetime_GMT", "Date", "Settlement_period", "Capacity",
               "Incentive_forecast", "Forecast_Timestamp"]


def _raw(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def _write_csv(directory, rows, name="day_ahead_wind_forecast.csv"):
    directory.mkdir(parents=True, exist_ok=True)
    _raw(rows).to_csv(directory / name, index=False)
    return directory / name


def _row(date_str, sp, forecast, made_at, capacity=20335):
    return [f"{date_str}T00:00:00", date_str, sp, capacity, forecast, made_at]


# --- The bid deadline -------------------------------------------------------------

def test_offers_close_at_two_the_afternoon_before_under_eac():
    assert bid_deadline("2026-01-05") == pd.Timestamp("2026-01-04 14:00")


def test_offers_closed_half_an_hour_later_before_eac():
    assert bid_deadline("2023-06-05") == pd.Timestamp("2023-06-04 14:30")
    assert bid_deadline(EAC_GO_LIVE) == pd.Timestamp("2023-11-01 14:00")


# --- Parsing ----------------------------------------------------------------------

def test_the_raw_columns_are_renamed_to_the_feature_matrix_shape(tmp_path):
    path = _write_csv(tmp_path, [_row("2024-06-05", 1, 1891, "2024-06-04T08:50:32")])
    frame = read_forecast_file(path)
    # published_before_deadline is added later, by flag_published_before_deadline
    assert list(frame.columns) == [c for c in COLUMNS if c != "published_before_deadline"]
    row = frame.iloc[0]
    assert row.settlementDate == pd.Timestamp("2024-06-05")
    assert row.settlementPeriod == 1
    assert row.wind_forecast_mw == 1891
    assert row.capacity_mw == 20335
    assert row.forecast_made_at == pd.Timestamp("2024-06-04 08:50:32")


def test_a_missing_column_is_reported_rather_than_guessed(tmp_path):
    frame = _raw([_row("2024-06-05", 1, 1891, "2024-06-04T08:50:32")]).drop(columns=["Incentive_forecast"])
    path = tmp_path / "broken.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Incentive_forecast"):
        read_forecast_file(path)


# --- Only forecasts that existed in time ------------------------------------------

def test_rows_are_flagged_by_whether_their_stamp_beat_the_deadline(tmp_path):
    path = _write_csv(tmp_path, [
        _row("2026-01-05", 1, 1891, "2026-01-04T08:50:00"),   # in time
        _row("2026-01-05", 2, 1917, "2026-01-04T15:30:00"),   # stamped late
    ])
    flagged = flag_published_before_deadline(read_forecast_file(path))
    assert flagged.published_before_deadline.tolist() == [True, False]


def test_a_forecast_with_no_timestamp_counts_as_in_time(tmp_path):
    """The resource is day-ahead by construction; a missing field should not lose a real forecast."""
    path = _write_csv(tmp_path, [_row("2026-01-05", 1, 1891, "")])
    assert flag_published_before_deadline(read_forecast_file(path)).published_before_deadline.all()


def test_late_stamped_rows_are_kept_by_default_rather_than_leaving_a_gap(tmp_path):
    """
    Forecast_Timestamp records portal publication, not forecast production: late-stamped
    rows are no closer to outturn than an independent day-ahead forecast, so dropping them
    would lose 12% of the resource for nothing. See the module docstring.
    """
    path = _write_csv(tmp_path, [_row("2026-01-05", sp, 1900 + sp, "2026-01-05T09:25:00")
                                 for sp in (1, 2)])
    table = build_wind_forecast_table([path])
    assert table.settlementPeriod.tolist() == [1, 2]
    assert not table.published_before_deadline.any()


def test_the_strict_variant_drops_them_instead(tmp_path):
    path = _write_csv(tmp_path, [
        _row("2026-01-05", 1, 1891, "2026-01-04T08:50:00"),
        _row("2026-01-05", 2, 1917, "2026-01-05T09:25:00"),
    ])
    table = build_wind_forecast_table([path], enforce_deadline=True)
    assert table.settlementPeriod.tolist() == [1]


def test_the_latest_forecast_that_beat_the_deadline_wins(tmp_path):
    path = _write_csv(tmp_path, [
        _row("2026-01-05", 1, 1800, "2026-01-04T08:50:00"),
        _row("2026-01-05", 1, 1950, "2026-01-04T12:10:00"),   # later, still in time
        _row("2026-01-05", 1, 2500, "2026-01-04T18:00:00"),   # later, too late
    ])
    table = build_wind_forecast_table([path])
    assert len(table) == 1
    assert table.iloc[0].wind_forecast_mw == 1950
    assert table.iloc[0].published_before_deadline


def test_clock_change_days_keep_the_periods_neso_published(tmp_path):
    """2024-03-31 lost an hour, so it has 46 periods rather than 48."""
    path = _write_csv(tmp_path, [_row("2024-03-31", sp, 1000 + sp, "2024-03-30T08:50:00")
                                 for sp in range(1, 47)])
    table = build_wind_forecast_table([path])
    assert len(table) == 46
    assert table.settlementPeriod.max() == 46


# --- The prepare step -------------------------------------------------------------

def test_prepare_step_writes_the_table_and_later_pulls_win_on_append(tmp_path):
    from scripts.prepare_data import prepare_wind_forecast

    raw, delta, processed = tmp_path / "raw", tmp_path / "delta", tmp_path / "processed"
    processed.mkdir()
    _write_csv(raw / "wind_forecast", [_row("2026-01-05", sp, 1000 + sp, "2026-01-04T08:50:00")
                                       for sp in (1, 2)])
    prepare_wind_forecast(raw, processed, append=False)
    table = pd.read_parquet(processed / "wind_forecast.parquet")
    assert table.wind_forecast_mw.tolist() == [1001, 1002]

    # A later collection revises period 2 and adds period 3
    _write_csv(delta / "wind_forecast", [_row("2026-01-05", 2, 1777, "2026-01-04T09:50:00"),
                                         _row("2026-01-05", 3, 1003, "2026-01-04T09:50:00")])
    prepare_wind_forecast(delta, processed, append=True)
    table = pd.read_parquet(processed / "wind_forecast.parquet").sort_values("settlementPeriod")
    assert table.wind_forecast_mw.tolist() == [1001, 1777, 1003]


def test_prepare_step_leaves_the_table_alone_without_a_csv(tmp_path):
    from scripts.prepare_data import prepare_wind_forecast

    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame({"settlementDate": [pd.Timestamp("2026-01-05")], "settlementPeriod": [1],
                  "wind_forecast_mw": [1891.0], "capacity_mw": [20335.0],
                  "forecast_made_at": [pd.Timestamp("2026-01-04 08:50")]}).to_parquet(
        processed / "wind_forecast.parquet", index=False)
    prepare_wind_forecast(tmp_path / "empty", processed, append=True)
    assert len(pd.read_parquet(processed / "wind_forecast.parquet")) == 1
