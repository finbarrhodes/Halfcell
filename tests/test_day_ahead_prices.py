"""
N2EX day-ahead auction prices: parsing, the UTC hour labels, settlement-period
mapping across the clock changes, and the rule that says when each price was public.
"""
import pandas as pd
import pytest

from src.analysis.day_ahead_prices import (
    COLUMNS,
    PUBLISH_HOUR_UTC,
    admissible_for_service_day,
    build_day_ahead_price_table,
    hourly_prices,
    published_at,
    read_price_file,
)
from src.analysis.neso_rules import bid_deadline

RAW_COLUMNS = ["Date", "Delivery Period", "Price"]


def _raw(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def _day(date: str, price=50.0, hours=range(24)) -> pd.DataFrame:
    return _raw([[date, f"{h:02d}:00 - {(h + 1) % 24:02d}:00", price + h] for h in hours])


def _write(tmp_path, frame: pd.DataFrame, name="n2ex.csv"):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


# --- parsing -------------------------------------------------------------

def test_delivery_period_is_read_as_the_hour_it_starts_in_utc(tmp_path):
    frame = read_price_file(_write(tmp_path, _raw([["2025-06-10", "14:00 - 15:00", 72.5]])))
    assert frame["hour_utc"].iloc[0] == pd.Timestamp("2025-06-10 14:00", tz="UTC")
    assert frame["day_ahead_price"].iloc[0] == 72.5


def test_a_missing_column_is_named_rather_than_raising_a_key_error(tmp_path):
    bad = pd.DataFrame([["2025-06-10", 72.5]], columns=["Date", "Price"])
    with pytest.raises(ValueError, match="delivery_period"):
        read_price_file(_write(tmp_path, bad))


def test_unparseable_rows_are_dropped_rather_than_poisoning_the_series(tmp_path):
    frame = read_price_file(_write(tmp_path, _raw([
        ["2025-06-10", "14:00 - 15:00", 72.5],
        ["2025-06-10", "not an hour", 80.0],
        ["", "15:00 - 16:00", 80.0],
        ["2025-06-10", "16:00 - 17:00", "n/a"],
    ])))
    assert len(frame) == 1


def test_a_later_file_wins_where_two_cover_the_same_hour(tmp_path):
    first = _write(tmp_path, _raw([["2025-06-10", "14:00 - 15:00", 10.0]]), "a.csv")
    second = _write(tmp_path, _raw([["2025-06-10", "14:00 - 15:00", 99.0]]), "b.csv")
    assert hourly_prices([first, second]).iloc[0] == 99.0


def test_no_files_gives_an_empty_table_rather_than_failing():
    assert list(build_day_ahead_price_table([]).columns) == COLUMNS
    assert build_day_ahead_price_table([]).empty


# --- settlement periods --------------------------------------------------

def test_each_hour_becomes_the_two_settlement_periods_inside_it(tmp_path):
    table = build_day_ahead_price_table([_write(tmp_path, _day("2025-01-15"))])
    winter = table[table.settlementDate == pd.Timestamp("2025-01-15")]
    assert len(winter) == 48
    # GMT: 00:00 UTC is SP 1, and the hour's price covers SP 1 and SP 2
    assert winter.day_ahead_price.iloc[0] == winter.day_ahead_price.iloc[1] == 50.0
    assert winter.settlementPeriod.tolist() == list(range(1, 49))


def test_summer_hours_shift_by_one_because_the_labels_are_utc(tmp_path):
    """BST is UTC+1, so 00:00 UTC is 01:00 local — settlement period 3, not 1."""
    table = build_day_ahead_price_table([_write(tmp_path, _day("2025-06-10"))])
    midnight_utc = table[table.day_ahead_price == 50.0]          # the 00:00 - 01:00 UTC hour
    assert midnight_utc.settlementPeriod.tolist() == [3, 4]
    assert (midnight_utc.settlementDate == pd.Timestamp("2025-06-10")).all()


def test_the_long_october_day_gets_fifty_settlement_periods(tmp_path):
    """The clock goes back, so the local day holds 25 hours. 2025-10-26 in GB."""
    files = [_write(tmp_path, pd.concat([_day("2025-10-25"), _day("2025-10-26"),
                                         _day("2025-10-27")]))]
    counts = build_day_ahead_price_table(files).groupby("settlementDate").size()
    assert counts[pd.Timestamp("2025-10-26")] == 50


def test_the_short_march_day_gets_forty_six_settlement_periods(tmp_path):
    """The clock goes forward, so the local day holds 23 hours. 2025-03-30 in GB."""
    files = [_write(tmp_path, pd.concat([_day("2025-03-29"), _day("2025-03-30"),
                                         _day("2025-03-31")]))]
    counts = build_day_ahead_price_table(files).groupby("settlementDate").size()
    assert counts[pd.Timestamp("2025-03-30")] == 46


def test_periods_are_numbered_from_local_midnight_not_from_the_first_row(tmp_path):
    """
    The real resource starts at 22:00 UTC, part-way through a BST day. Numbering
    from the first row available would call that settlement period 1; it is 47.
    """
    table = build_day_ahead_price_table([_write(tmp_path, _day("2025-06-10", hours=[22, 23]))])
    assert table.settlementPeriod.tolist() == [47, 48, 1, 2]
    assert table.settlementDate.dt.date.astype(str).tolist() == [
        "2025-06-10", "2025-06-10", "2025-06-11", "2025-06-11"]


# --- the publication rule ------------------------------------------------

def test_prices_publish_at_ten_the_morning_before_delivery():
    assert published_at("2025-06-10") == pd.Timestamp(f"2025-06-09 {PUBLISH_HOUR_UTC}:00")


def test_publication_beats_the_offer_deadline_for_the_service_day_it_prices():
    """The whole point: 10:00 on D-1 is four hours inside the 14:00 D-1 deadline."""
    assert published_at("2025-06-10") < pd.Timestamp(bid_deadline(pd.Timestamp("2025-06-10").date()))
    # and a day later than the service day is not yet public when offers close
    assert published_at("2025-06-11") > pd.Timestamp(bid_deadline(pd.Timestamp("2025-06-10").date()))


def test_the_service_day_itself_is_the_last_admissible_delivery_date():
    assert admissible_for_service_day("2025-06-10") == pd.Timestamp("2025-06-10")


def test_a_periods_delivery_date_never_runs_past_its_settlement_date(tmp_path):
    """
    London is never behind UTC, so the hour a settlement period sits in always
    belongs to that period's own date or the one before — which is what makes every
    price in a service day's blocks public before its offers close. Checked across
    both clock changes, where the risk of an off-by-one is highest.
    """
    days = pd.date_range("2025-03-28", "2025-03-31").union(pd.date_range("2025-10-25", "2025-10-28"))
    files = [_write(tmp_path, pd.concat([_day(str(d.date())) for d in days]))]
    table = build_day_ahead_price_table(files)
    assert (table["deliveryDate"] <= table["settlementDate"]).all()
    assert (table["settlementDate"] - table["deliveryDate"]).max() <= pd.Timedelta(days=1)
