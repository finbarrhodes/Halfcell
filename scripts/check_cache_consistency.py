"""
scripts/check_cache_consistency.py
==================================
Fail fast if data/cache/ holds a torn set of backtest results.

precompute_cache.py writes the three strategy parquets sequentially over roughly
twenty minutes, writing the manifest last. Anything that reads the cache while
that is in flight — a site build, a commit — can pick up a mix of strategies
from different runs. The numbers still look plausible, which is what makes it
dangerous: strategies are only comparable if they came from the same source data.

File mtimes cannot be used for this, because git does not preserve them across a
checkout. The manifest's own provenance fields are used instead.

Run from the project root:
    python scripts/check_cache_consistency.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
CACHE = ROOT / "data" / "cache"
MANIFEST = CACHE / "manifest.json"

STRATEGIES = ("pf_mpc", "naive_mpc", "ml_mpc")

# Strategies from one run finish within minutes of each other; a wider spread
# means the cache was assembled from separate runs.
MAX_SPREAD_HOURS = 6.0


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if not MANIFEST.exists():
        _fail(f"{MANIFEST} not found — run scripts/precompute_cache.py")

    manifest = json.loads(MANIFEST.read_text())

    missing = [s for s in STRATEGIES if s not in manifest]
    if missing:
        _fail(f"manifest missing strategies: {', '.join(missing)}")

    for strategy in STRATEGIES:
        for name in (f"{strategy}.parquet", f"soc_{strategy}.parquet"):
            if not (CACHE / name).exists():
                _fail(f"cache file missing: data/cache/{name}")

    # Every strategy must have been computed from the same source data.
    mtimes = {s: manifest[s].get("data_mtimes") for s in STRATEGIES}
    reference = mtimes[STRATEGIES[0]]
    for strategy, value in mtimes.items():
        if value != reference:
            _fail(
                f"{strategy} was computed from different source data than "
                f"{STRATEGIES[0]} — the cache is torn across precompute runs. "
                "Re-run scripts/precompute_cache.py to completion."
            )

    # And within the same run.
    stamps = []
    for strategy in STRATEGIES:
        raw = manifest[strategy].get("computed_at")
        if not raw:
            _fail(f"{strategy} has no computed_at timestamp")
        stamps.append(datetime.fromisoformat(raw))

    spread_hours = (max(stamps) - min(stamps)).total_seconds() / 3600
    if spread_hours > MAX_SPREAD_HOURS:
        _fail(
            f"strategies span {spread_hours:.1f}h (limit {MAX_SPREAD_HOURS}h) — "
            "they look like separate precompute runs"
        )

    # Params that define the asset and window must agree, or the strategies are
    # not comparable even if they ran together.
    shared_keys = ("power_mw", "duration_h", "efficiency_rt", "cycling_cost_per_mwh",
                   "availability_factor", "start_date", "end_date", "dispatch_method")
    ref_params = manifest[STRATEGIES[0]]["params"]
    for strategy in STRATEGIES[1:]:
        params = manifest[strategy]["params"]
        for key in shared_keys:
            if params.get(key) != ref_params.get(key):
                _fail(
                    f"{strategy} has {key}={params.get(key)!r} but "
                    f"{STRATEGIES[0]} has {ref_params.get(key)!r}"
                )

    if not manifest["ml_mpc"].get("feature_importances"):
        _fail("ml_mpc has no feature_importances — the app cannot render them")

    print("Cache consistent:")
    print(f"  strategies   : {', '.join(STRATEGIES)}")
    print(f"  window       : {ref_params['start_date']} → {ref_params['end_date']}")
    print(f"  asset        : {ref_params['power_mw']} MW / {ref_params['duration_h']}h")
    print(f"  computed     : {min(stamps).isoformat(timespec='seconds')} "
          f"(spread {spread_hours * 60:.0f} min)")


if __name__ == "__main__":
    main()
