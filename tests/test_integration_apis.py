"""Live API contract tests — deselected by default.

    pytest -m integration

These guard against upstream changes that unit tests cannot see: NESO rotating
its EAC resources each April, and Elexon's date-parameter semantics. Both were
sources of silent data loss in this project.
"""
import pandas as pd
import pytest

pytestmark = pytest.mark.integration

requests = pytest.importorskip("requests")

NESO_SQL = "https://api.neso.energy/api/3/action/datastore_search_sql"
ELEXON = "https://data.elexon.co.uk/bmrs/api/v1"


def _neso_bounds(resource_id):
    sql = f'SELECT MIN("deliveryStart") AS lo, MAX("deliveryStart") AS hi FROM "{resource_id}"'
    r = requests.get(NESO_SQL, params={"sql": sql}, timeout=60)
    r.raise_for_status()
    rec = r.json()["result"]["records"][0]
    return pd.Timestamp(rec["lo"]).date(), pd.Timestamp(rec["hi"]).date()


@pytest.mark.parametrize("resource_id,declared_start,declared_end",
                         [(s[0], s[1], s[2]) for s in
                          __import__("src.data_collection.neso_collector",
                                     fromlist=["_EAC_SEGMENTS"])._EAC_SEGMENTS])
def test_eac_segment_still_covers_its_declared_range(resource_id, declared_start, declared_end):
    """Each configured resource must still serve the range we claim it does.

    When NESO rotates the live feed into a new fiscal-year archive, the live
    resource stops serving older data and this fails — which is the point.
    """
    lo, hi = _neso_bounds(resource_id)
    assert lo <= pd.Timestamp(declared_start).date() + pd.Timedelta(days=1).to_pytimedelta()
    if declared_end is not None:
        assert hi >= pd.Timestamp(declared_end).date() - pd.Timedelta(days=1).to_pytimedelta()


def test_live_eac_resource_is_current():
    """The open-ended segment should be serving data from the last few days."""
    from src.data_collection.neso_collector import _EAC_SEGMENTS

    live = [s for s in _EAC_SEGMENTS if s[2] is None][0]
    _, hi = _neso_bounds(live[0])
    age = (pd.Timestamp.utcnow().date() - hi).days
    assert age <= 7, f"live EAC resource is {age} days stale — check for an April rotation"


def test_elexon_to_parameter_is_exclusive_at_midnight():
    """Documents the boundary semantics the collector depends on.

    A bare `to` date returns only the first few settlement periods of that day.
    If Elexon ever changes this to be inclusive, the end-of-day pin becomes
    redundant and this test tells us.
    """
    def count(to_value, day="2026-04-04"):
        r = requests.get(f"{ELEXON}/balancing/pricing/market-index",
                         params={"from": "2026-03-29", "to": to_value}, timeout=60)
        r.raise_for_status()
        return sum(1 for x in r.json().get("data", [])
                   if x.get("settlementDate") == day and x.get("dataProvider") == "APXMIDP")

    assert count("2026-04-04") < 10, "bare date no longer truncates — revisit the collector"
    assert count("2026-04-04T23:59:59Z") >= 46, "end-of-day pin no longer returns a full day"


def test_elexon_full_day_has_all_settlement_periods():
    r = requests.get(f"{ELEXON}/balancing/pricing/market-index",
                     params={"from": "2026-06-10", "to": "2026-06-10T23:59:59Z"}, timeout=60)
    r.raise_for_status()
    periods = {x["settlementPeriod"] for x in r.json()["data"]
               if x["dataProvider"] == "APXMIDP"}
    assert len(periods) == 48
