"""Market KPI snapshot computation.

Regression cover for a partial-day artifact: the peak-to-trough spread was
computed over whatever days existed, including a still-publishing current day
with 3 of 48 settlement periods. That produced a spread of £4.87 against a
30-day average near £97 — and passed the plausibility bounds, which only reject
values outside [0, 3000].
"""
import importlib
import sys

import numpy as np
import pandas as pd
import pytest


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


def _write_auctions(processed, days=120):
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    rows = [{"EFA Date": d, "Service": "DCH", "EFA": efa, "Clearing Price": 2.0 + (i % 5) * 0.1,
             "Cleared Volume": 100.0}
            for i, d in enumerate(dates) for efa in range(1, 7)]
    pd.DataFrame(rows).to_parquet(processed / "auctions.parquet", index=False)


def _write_market(processed, days=120, truncate_last_to=None):
    """Full 48-period days, optionally leaving the final day partly published."""
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    rows = []
    for i, d in enumerate(dates):
        n = truncate_last_to if (truncate_last_to and i == len(dates) - 1) else 48
        for sp in range(1, n + 1):
            rows.append({"settlementDate": d, "settlementPeriod": sp,
                         "dataProvider": "APXMIDP",
                         "price": 60 + 40 * np.sin(sp / 48 * 2 * np.pi)})
    pd.DataFrame(rows).to_parquet(processed / "market_index.parquet", index=False)


def test_partial_final_day_is_excluded_from_spread(kpi_module):
    mod, processed = kpi_module
    _write_auctions(processed)
    _write_market(processed, truncate_last_to=3)

    kpis = mod._compute()
    # The final day has 3 periods spanning a tiny slice of the sine curve
    assert kpis["spread_latest"] > 50, (
        f"spread_latest={kpis['spread_latest']} came from a partially published day"
    )
    assert kpis["spread_latest_date"] < "2026-04-30"


def test_complete_day_is_used_when_available(kpi_module):
    mod, processed = kpi_module
    _write_auctions(processed)
    _write_market(processed)
    kpis = mod._compute()
    assert kpis["spread_latest"] == pytest.approx(80.0, abs=1.0)


def test_clock_change_day_still_counts_as_complete(kpi_module):
    """46 periods is a legitimate spring-forward day, not a truncated one."""
    mod, processed = kpi_module
    _write_auctions(processed)
    _write_market(processed, truncate_last_to=46)
    kpis = mod._compute()
    assert kpis["spread_latest_date"] == "2026-04-30"


def test_validation_rejects_implausible_values(kpi_module):
    mod, _ = kpi_module
    with pytest.raises(ValueError):
        mod._validate({**{k: 0 for k in mod.BOUNDS}, "dch_latest": 10_000})


def test_validation_rejects_missing_keys(kpi_module):
    mod, _ = kpi_module
    with pytest.raises(ValueError):
        mod._validate({"dch_latest": 1.0})


def test_all_days_partial_raises_rather_than_reporting_nonsense(kpi_module):
    mod, processed = kpi_module
    _write_auctions(processed)
    _write_market(processed, days=5)
    df = pd.read_parquet(processed / "market_index.parquet")
    df = df[df["settlementPeriod"] <= 3]  # nothing complete anywhere
    df.to_parquet(processed / "market_index.parquet", index=False)
    with pytest.raises(ValueError):
        mod._compute()
