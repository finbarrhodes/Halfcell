"""Cache torn-state guard.

precompute_cache.py writes three strategy parquets sequentially over ~20 minutes
with the manifest last, so anything reading the cache mid-run can pick up a mix
of strategies from different runs. The numbers still look plausible, which is
what makes it worth failing a build over.
"""
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.check_cache_consistency import ENGINE_SOURCES, engine_fingerprint

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_cache_consistency.py"

STRATEGIES = ("pf_mpc", "naive_mpc", "ml_mpc")


def _entry(when, mtimes, engine=None, **params):
    base = {"power_mw": 50.0, "duration_h": 2.0, "efficiency_rt": 0.9,
            "cycling_cost_per_mwh": 3.0, "availability_factor": 0.95,
            "start_date": "2021-09-16", "end_date": "2026-08-17",
            "dispatch_method": "mpc", "pre_eac_rule": "d1", "auction_share_cap": 0.2}
    base.update(params)
    return {"computed_at": when.isoformat(), "git_sha": "abc123", "engine": engine,
            "data_mtimes": mtimes, "params": base, "summary": {}}


def _build_cache(tmp_path, spread_minutes=10, mismatch=None, drop_importances=False):
    # Stand-in engine sources, so the tree has code to fingerprint
    for relative in ENGINE_SOURCES:
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text("RATE = 1\n")
    engine = engine_fingerprint(tmp_path)

    cache = tmp_path / "data" / "cache"
    cache.mkdir(parents=True)
    for s in STRATEGIES:
        for name in (f"{s}.parquet", f"soc_{s}.parquet", f"{s}_arb_only.parquet"):
            (cache / name).write_bytes(b"x")
    (cache / "fr_only.parquet").write_bytes(b"x")

    t0 = datetime(2026, 8, 18, 11, 0, tzinfo=timezone.utc)
    mtimes = {"auctions.parquet": 1.0, "market_index.parquet": 2.0}

    manifest = {}
    for i, s in enumerate(STRATEGIES):
        m = dict(mtimes)
        if mismatch == "data" and s == "ml_mpc":
            m["market_index.parquet"] = 99.0
        extra = {}
        if mismatch == "params" and s == "ml_mpc":
            extra["power_mw"] = 100.0
        which = {"engines": "0123456789ab" if s == "ml_mpc" else engine, "no_engine": None}.get(mismatch, engine)
        manifest[s] = _entry(t0 + timedelta(minutes=spread_minutes * i), m, engine=which, **extra)

    if not drop_importances:
        manifest["ml_mpc"]["feature_importances"] = [{"feature": "apx_lag_1d", "importance": 0.25}]

    (cache / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def _run(root):
    """Run the guard against a synthetic tree by copying the script into it."""
    target = root / "scripts"
    target.mkdir(parents=True, exist_ok=True)
    (target / "check_cache_consistency.py").write_text(SCRIPT.read_text())
    return subprocess.run(
        [sys.executable, str(target / "check_cache_consistency.py")],
        capture_output=True, text=True,
    )


def test_passes_on_a_coherent_cache(tmp_path):
    result = _run(_build_cache(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "Cache consistent" in result.stdout


def test_rejects_strategies_built_from_different_source_data(tmp_path):
    result = _run(_build_cache(tmp_path, mismatch="data"))
    assert result.returncode == 1
    assert "different source data" in result.stderr


def test_rejects_strategies_from_separate_runs(tmp_path):
    result = _run(_build_cache(tmp_path, spread_minutes=60 * 5))
    assert result.returncode == 1
    assert "separate precompute runs" in result.stderr


def test_rejects_mismatched_asset_parameters(tmp_path):
    result = _run(_build_cache(tmp_path, mismatch="params"))
    assert result.returncode == 1
    assert "power_mw" in result.stderr


def test_rejects_missing_feature_importances(tmp_path):
    result = _run(_build_cache(tmp_path, drop_importances=True))
    assert result.returncode == 1
    assert "feature_importances" in result.stderr


def test_rejects_missing_parquet(tmp_path):
    root = _build_cache(tmp_path)
    (root / "data" / "cache" / "soc_ml_mpc.parquet").unlink()
    result = _run(root)
    assert result.returncode == 1
    assert "cache file missing" in result.stderr


def test_rejects_missing_scenario_parquet(tmp_path):
    root = _build_cache(tmp_path)
    (root / "data" / "cache" / "fr_only.parquet").unlink()
    result = _run(root)
    assert result.returncode == 1
    assert "fr_only.parquet" in result.stderr


def test_rejects_missing_manifest(tmp_path):
    (tmp_path / "data" / "cache").mkdir(parents=True)
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "not found" in result.stderr


def test_rejects_a_cache_the_current_engine_did_not_produce(tmp_path):
    """The cache agrees with itself, but the code has moved on since it was built."""
    root = _build_cache(tmp_path)
    (root / "src" / "optimisation" / "mpc.py").write_text("RATE = 2\n")
    result = _run(root)
    assert result.returncode == 1
    assert "fingerprints as" in result.stderr


def test_a_comment_or_blank_line_does_not_force_a_rebuild(tmp_path):
    root = _build_cache(tmp_path)
    (root / "src" / "optimisation" / "mpc.py").write_text("# why the rate is one\n\nRATE = 1   \n")
    result = _run(root)
    assert result.returncode == 0, result.stderr


def test_rejects_a_manifest_without_an_engine_fingerprint(tmp_path):
    result = _run(_build_cache(tmp_path, mismatch="no_engine"))
    assert result.returncode == 1
    assert "no engine fingerprint" in result.stderr


def test_rejects_strategies_from_different_engines(tmp_path):
    result = _run(_build_cache(tmp_path, mismatch="engines"))
    assert result.returncode == 1
    assert "different engines" in result.stderr
