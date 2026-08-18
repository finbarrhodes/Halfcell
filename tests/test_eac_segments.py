"""NESO EAC resource routing.

NESO splits EAC auction results into one resource per fiscal year plus a live
current-year feed, rotating the live feed into a new archive each April. The
collector previously modelled this as a single archive/current boundary, which
went stale on rotation and returned short *without erroring* — a from-scratch
collection would have silently lost roughly two years.
"""
from datetime import date

import pandas as pd
import pytest

from src.data_collection.neso_collector import _EAC_ARCHIVE_START, _EAC_SEGMENTS


def _seg_bounds(seg):
    _, start, end = seg
    return date.fromisoformat(start), (date.fromisoformat(end) if end else None)


def test_segments_are_chronological():
    starts = [_seg_bounds(s)[0] for s in _EAC_SEGMENTS]
    assert starts == sorted(starts)


def test_segments_have_unique_resource_ids():
    ids = [s[0] for s in _EAC_SEGMENTS]
    assert len(ids) == len(set(ids))


def test_exactly_one_open_ended_live_segment():
    open_ended = [s for s in _EAC_SEGMENTS if s[2] is None]
    assert len(open_ended) == 1, "expected exactly one live (current fiscal year) resource"
    assert open_ended[0] is _EAC_SEGMENTS[-1], "the live resource must be the last segment"


def test_segments_leave_no_gap_in_coverage():
    """Adjacent segments must touch or overlap — a gap is silent data loss."""
    for earlier, later in zip(_EAC_SEGMENTS, _EAC_SEGMENTS[1:]):
        _, earlier_end = _seg_bounds(earlier)
        later_start, _ = _seg_bounds(later)
        assert earlier_end is not None
        assert later_start <= earlier_end + pd.Timedelta(days=1).to_pytimedelta(), (
            f"gap between {earlier[0][:8]} (ends {earlier_end}) and "
            f"{later[0][:8]} (starts {later_start})"
        )


def test_coverage_starts_at_the_documented_archive_start():
    assert _seg_bounds(_EAC_SEGMENTS[0])[0] == date.fromisoformat(_EAC_ARCHIVE_START)


def _segments_for(start, end):
    """Mirror the routing in collect_eac_results."""
    start, end = date.fromisoformat(start), date.fromisoformat(end)
    hits = []
    for seg in _EAC_SEGMENTS:
        lo, hi = _seg_bounds(seg)
        hi = hi or end
        if not (end < lo or start > hi):
            hits.append(seg[0])
    return hits


def test_range_spanning_fiscal_years_hits_every_segment():
    hits = _segments_for("2023-11-02", "2026-08-17")
    assert len(hits) == len(_EAC_SEGMENTS)


def test_recent_range_still_reaches_the_previous_fy_archive():
    """Regression: Feb-Aug 2026 must include the FY2025 archive.

    Querying only the live resource for this range returned data starting
    2026-03-31, leaving a five-week hole against previously collected data.
    """
    hits = _segments_for("2026-02-15", "2026-08-17")
    assert len(hits) >= 2, "a range crossing the April rotation must span two resources"
    assert _EAC_SEGMENTS[-1][0] in hits, "must include the live resource"


def test_single_day_selects_exactly_one_segment():
    assert len(_segments_for("2025-06-01", "2025-06-01")) == 1


def test_range_before_any_data_selects_nothing():
    assert _segments_for("2020-01-01", "2020-06-01") == []
