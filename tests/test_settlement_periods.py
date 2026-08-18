"""Settlement period arithmetic and BST clock-change day lengths."""
from datetime import date, datetime

import pytest

from src.utils import calculate_settlement_period, settlement_periods_in_day


@pytest.mark.parametrize("clock,expected", [
    ("00:00", 1),   # SP1 starts at local midnight
    ("00:30", 2),
    ("01:00", 3),
    ("12:00", 25),
    ("22:30", 46),
    ("23:00", 47),  # Elexon reports SP47 at 23:00 on the same settlementDate
    ("23:30", 48),
])
def test_settlement_period_boundaries(clock, expected):
    hour, minute = map(int, clock.split(":"))
    assert calculate_settlement_period(datetime(2024, 1, 15, hour, minute)) == expected


def test_settlement_period_spans_full_day_exactly_once():
    """Every half-hour of a normal day maps to a distinct period 1..48."""
    seen = [
        calculate_settlement_period(datetime(2024, 1, 15, h, m))
        for h in range(24) for m in (0, 30)
    ]
    assert seen == list(range(1, 49))


@pytest.mark.parametrize("day,expected", [
    (date(2026, 3, 29), 46),   # spring forward — 01:00-02:00 does not exist
    (date(2026, 10, 25), 50),  # fall back — 01:00-02:00 happens twice
    (date(2026, 6, 1), 48),
    (date(2026, 1, 15), 48),
    (date(2025, 3, 30), 46),   # transition dates move year to year
    (date(2025, 10, 26), 50),
])
def test_clock_change_day_lengths(day, expected):
    assert settlement_periods_in_day(day) == expected


def test_only_two_irregular_days_per_year():
    """A year has exactly one 46-period day and one 50-period day."""
    counts = {}
    d = date(2026, 1, 1)
    while d.year == 2026:
        counts[settlement_periods_in_day(d)] = counts.get(settlement_periods_in_day(d), 0) + 1
        d = date.fromordinal(d.toordinal() + 1)
    assert counts[46] == 1
    assert counts[50] == 1
    assert counts[48] == 363
