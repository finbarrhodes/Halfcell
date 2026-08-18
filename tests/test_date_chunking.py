"""Elexon request chunking and date-range boundaries.

Regression cover for a bug where every chunk's final day came back with ~3 of
48 settlement periods: the API parses `to` as a datetime, so a bare date means
midnight and truncates the last day of the window.
"""
from datetime import date

import pytest

from src.data_collection.elexon_collector import _CHUNK_DAYS, _date_chunks, ElexonBMRSCollector


def _parse(s):
    return date.fromisoformat(s)


def test_chunks_cover_range_contiguously():
    chunks = _date_chunks("2026-01-01", "2026-03-01")
    assert _parse(chunks[0][0]) == date(2026, 1, 1)
    assert _parse(chunks[-1][1]) == date(2026, 3, 1)
    for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:]):
        assert (_parse(next_start) - _parse(prev_end)).days == 1, "gap or overlap between chunks"


def test_chunks_never_exceed_window_limit():
    for start, end in _date_chunks("2026-01-01", "2026-06-30"):
        span = (_parse(end) - _parse(start)).days + 1
        assert 1 <= span <= _CHUNK_DAYS


def test_single_day_range_yields_one_chunk():
    assert _date_chunks("2026-05-05", "2026-05-05") == [("2026-05-05", "2026-05-05")]


def test_every_day_in_range_appears_exactly_once():
    covered = []
    for start, end in _date_chunks("2026-01-01", "2026-02-14"):
        d = _parse(start)
        while d <= _parse(end):
            covered.append(d)
            d = date.fromordinal(d.toordinal() + 1)
    assert len(covered) == len(set(covered)) == 45


def test_market_index_request_covers_whole_final_day(monkeypatch):
    """`to` must pin end-of-day, or the last day of each chunk is truncated."""
    captured = []

    def fake_get(self, path, params=None):
        captured.append(params)
        return {"data": []}

    monkeypatch.setattr(ElexonBMRSCollector, "_get", fake_get, raising=True)
    collector = ElexonBMRSCollector(config={"apis": {"elexon": {
        "base_url": "https://example.invalid", "rate_limit": 60}}})
    collector.collect_imbalance_prices("2026-04-01", "2026-04-10", save=False)

    assert captured, "no requests were issued"
    for params in captured:
        assert "T23:59:59" in params["to"], (
            f"`to`={params['to']!r} is a bare date — the API reads that as midnight "
            "and returns only the first few settlement periods of that day"
        )
