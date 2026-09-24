"""
scripts/check_cache_consistency.py
==================================
Fail fast if data/cache/ holds a torn set of backtest results.

precompute_cache.py writes the strategy and scenario parquets sequentially over
roughly half an hour, writing the manifest last. Anything that reads the cache while
that is in flight — a site build, a commit — can pick up a mix of strategies
from different runs. The numbers still look plausible, which is what makes it
dangerous: strategies are only comparable if they came from the same source data.

File mtimes cannot be used for this, because git does not preserve them across a
checkout. The manifest's own provenance fields are used instead.

It also fails if the engine has changed since the cache was built. A cache from
older code still reads as coherent - every strategy agrees with every other - so
the site would go on publishing numbers the current engine no longer produces,
and nothing would say so. precompute_cache.py records a fingerprint of the code
that made the cache; this compares it with the code in the tree. Blank lines and
whole-line comments are ignored, so only a change to the code forces a rebuild.

Run from the project root:
    python scripts/check_cache_consistency.py
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
CACHE = ROOT / "data" / "cache"
MANIFEST = CACHE / "manifest.json"

STRATEGIES = ("pf_mpc", "naive_mpc", "ml_mpc")

# FR-only scenarios need no price forecast, so one copy serves every strategy
SHARED_FILES = ("fr_only.parquet",)

# Strategies from one run finish within minutes of each other; a wider spread
# means the cache was assembled from separate runs.
MAX_SPREAD_HOURS = 6.0

# The code that turns the processed data into the cache: the engine, the forecast
# path, and precompute_cache.py itself, whose constants set the shipped configuration
ENGINE_SOURCES = (
    "src/analysis/revenue_stack.py", "src/analysis/fr_allocation.py", "src/analysis/neso_rules.py",
    "src/analysis/response_delivery.py", "src/analysis/price_forecast.py", "src/analysis/features.py",
    "src/analysis/forecasting_models.py", "src/optimisation/mpc.py", "src/optimisation/day_ahead.py",
    "scripts/precompute_cache.py",
)


def engine_fingerprint(root: Path = ROOT) -> str:
    """A hash of ENGINE_SOURCES under `root`, blind to blank lines and whole-line comments."""
    digest = hashlib.sha256()
    for relative in ENGINE_SOURCES:
        path = root / relative
        if not path.exists():
            digest.update(f"{relative}: missing\n".encode())
            continue
        lines = (line.rstrip() for line in path.read_text().splitlines())
        code = "\n".join(line for line in lines if line.strip() and not line.lstrip().startswith("#"))
        digest.update(f"{relative}\n{code}\n".encode())
    return digest.hexdigest()[:12]


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

    names = list(SHARED_FILES)
    for strategy in STRATEGIES:
        names += [f"{strategy}.parquet", f"soc_{strategy}.parquet", f"{strategy}_arb_only.parquet"]
    for name in names:
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

    # Params that define the asset, the window and the engine must agree, or the
    # strategies are not comparable even if they ran together. price_shrink is
    # deliberately absent: it is tuned per signal, like the signal itself.
    shared_keys = ("power_mw", "duration_h", "efficiency_rt", "cycling_cost_per_mwh",
                   "availability_factor", "start_date", "end_date", "dispatch_method",
                   "pre_eac_rule", "auction_share_cap", "horizon", "delivery_modelled",
                   "offer_valuation", "offer_information", "forecast_vintages",
                   "credit_recovery", "block_start_margin")
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

    # And by the engine that is in the tree now
    engines = {manifest[s].get("engine") for s in STRATEGIES}
    if None in engines:
        _fail("the manifest records no engine fingerprint — re-run scripts/precompute_cache.py")
    if len(engines) > 1:
        _fail(f"strategies were computed by different engines ({', '.join(sorted(engines))}) — "
              "re-run scripts/precompute_cache.py to completion")
    current = engine_fingerprint()
    if engines != {current}:
        _fail(f"the cache was computed by engine {engines.pop()}, but the code in the tree fingerprints "
              f"as {current}: the site would publish numbers the current engine does not produce. "
              "Re-run scripts/precompute_cache.py, or the refresh workflow")

    print("Cache consistent:")
    print(f"  strategies   : {', '.join(STRATEGIES)}")
    print(f"  window       : {ref_params['start_date']} → {ref_params['end_date']}")
    print(f"  asset        : {ref_params['power_mw']} MW / {ref_params['duration_h']}h")
    print(f"  computed     : {min(stamps).isoformat(timespec='seconds')} "
          f"(spread {spread_hours * 60:.0f} min)")
    print(f"  engine       : {current}")


if __name__ == "__main__":
    main()
