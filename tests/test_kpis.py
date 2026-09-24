"""Market snapshot computation.

Regression cover for a partial-day artifact: the peak-to-trough spread was
computed over whatever days existed, including a still-publishing current day
with 3 of 48 settlement periods. That produced a spread of £4.87 against a
30-day average near £97 — and passed the plausibility bounds, which only reject
values outside [0, 3000]. The cards now average a month, but a partial day would
still drag that average, so only complete days count.
"""
import importlib
import sys

import numpy as np
import pandas as pd
import pytest

START = pd.Timestamp("2025-01-01")


@pytest.fixture
def kpi_module(tmp_path, monkeypatch):
    sys.path.insert(0, str(tmp_path))
    mod = importlib.import_module("scripts.compute_kpis")
    importlib.reload(mod)
    processed = tmp_path / "processed"
    processed.mkdir()
    monkeypatch.setattr(mod, "PROCESSED", processed)
    monkeypatch.setattr(mod, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(mod, "OUT_FILE", tmp_path / "cache" / "latest_kpis.json")
    return mod, processed


def _write_auctions(processed, days=420):
    dates = pd.date_range(START, periods=days, freq="D")
    rows = [{"EFA Date": d, "Service": "DCH", "EFA": efa, "Clearing Price": 2.0, "Cleared Volume": 100.0}
            for d in dates for efa in range(1, 7)]
    pd.DataFrame(rows).to_parquet(processed / "auctions.parquet", index=False)


def _write_market(processed, days=420, truncate_last_to=None, amplitude=40.0, earlier_amplitude=None):
    """
    A sine through each day, so its peak-to-trough spread is twice the amplitude.
    With earlier_amplitude, days up to a year before the last use that instead.
    Optionally leaves the final day partly published.
    """
    dates = pd.date_range(START, periods=days, freq="D")
    year_end = dates[-1] - pd.DateOffset(years=1)
    rows = []
    for i, d in enumerate(dates):
        n = truncate_last_to if (truncate_last_to and i == len(dates) - 1) else 48
        amp = earlier_amplitude if (earlier_amplitude is not None and d <= year_end) else amplitude
        for sp in range(1, n + 1):
            rows.append({"settlementDate": d, "settlementPeriod": sp, "dataProvider": "APXMIDP",
                         "price": 60 + amp * np.sin(sp / 48 * 2 * np.pi)})
    pd.DataFrame(rows).to_parquet(processed / "market_index.parquet", index=False)


def _write_fleet(processed, measured_through="2026-01", first="2024-01", last="2026-06"):
    """Growing 100 MW a month; months after measured_through are projections."""
    months = pd.date_range(first, last, freq="MS")
    pd.DataFrame({"month": months, "bess_fleet_mw": 1000.0 + 100.0 * np.arange(len(months)),
                  "is_extrapolated": months > pd.Timestamp(measured_through)}
                 ).to_parquet(processed / "bess_fleet_capacity.parquet", index=False)


def _write_all(processed, **market):
    _write_auctions(processed)
    _write_market(processed, **market)
    _write_fleet(processed)


def test_partial_final_day_is_excluded_from_spread(kpi_module):
    mod, processed = kpi_module
    _write_all(processed, truncate_last_to=3)
    kpis = mod._compute()
    last = (START + pd.Timedelta(days=419)).strftime("%Y-%m-%d")
    assert kpis["spread_window_end"] < last, "the window ended on a partially published day"
    assert kpis["spread_30d_avg"] == pytest.approx(80.0, abs=1.0)


def test_complete_day_is_used_when_available(kpi_module):
    mod, processed = kpi_module
    _write_all(processed)
    kpis = mod._compute()
    assert kpis["spread_window_end"] == (START + pd.Timedelta(days=419)).strftime("%Y-%m-%d")
    assert kpis["spread_30d_avg"] == pytest.approx(80.0, abs=1.0)


def test_clock_change_day_still_counts_as_complete(kpi_module):
    """46 periods is a legitimate spring-forward day, not a truncated one."""
    mod, processed = kpi_module
    _write_all(processed, truncate_last_to=46)
    assert mod._compute()["spread_window_end"] == (START + pd.Timedelta(days=419)).strftime("%Y-%m-%d")


def test_the_last_month_is_compared_with_the_same_span_a_year_earlier(kpi_module):
    mod, processed = kpi_module
    _write_all(processed, earlier_amplitude=20.0)
    kpis = mod._compute()
    assert kpis["spread_30d_avg"] == pytest.approx(80.0, abs=1.0)
    assert kpis["spread_30d_avg_year_earlier"] == pytest.approx(40.0, abs=1.0)


def test_a_history_shorter_than_a_year_raises_rather_than_comparing_nothing(kpi_module):
    mod, processed = kpi_module
    _write_auctions(processed, days=60)
    _write_market(processed, days=60)
    _write_fleet(processed)
    with pytest.raises(ValueError, match="year-earlier"):
        mod._compute()


def test_the_fleet_card_takes_the_latest_measured_month_not_a_projection(kpi_module):
    mod, processed = kpi_module
    _write_all(processed)
    kpis = mod._compute()
    assert kpis["fleet_month"] == "2026-01"            # the projections run to 2026-06
    assert kpis["fleet_mw"] == 1000.0 + 100.0 * 24     # 24 months after 2024-01
    assert kpis["fleet_mw_year_earlier"] == 1000.0 + 100.0 * 12


def test_the_fleet_needs_a_measured_month_a_year_earlier(kpi_module):
    mod, processed = kpi_module
    _write_auctions(processed)
    _write_market(processed)
    _write_fleet(processed, first="2025-06", measured_through="2026-01")
    with pytest.raises(ValueError, match="a year before"):
        mod._compute()


def test_validation_rejects_implausible_values(kpi_module):
    mod, _ = kpi_module
    with pytest.raises(ValueError):
        mod._validate({**{k: 10 for k in mod.BOUNDS}, "fleet_mw": 10_000_000})


def test_validation_rejects_missing_keys(kpi_module):
    mod, _ = kpi_module
    with pytest.raises(ValueError):
        mod._validate({"fleet_mw": 4000.0})


def test_all_days_partial_raises_rather_than_reporting_nonsense(kpi_module):
    mod, processed = kpi_module
    _write_all(processed)
    df = pd.read_parquet(processed / "market_index.parquet")
    df = df[df["settlementPeriod"] <= 3]  # nothing complete anywhere
    df.to_parquet(processed / "market_index.parquet", index=False)
    with pytest.raises(ValueError):
        mod._compute()
