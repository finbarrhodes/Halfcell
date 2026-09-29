# Reports

Each report records one experiment or check: what was run, and what it found. They are
the evidence behind decisions described on the site's Methodology and Research pages, not
measurements of the published model, and the engine has changed since most were written.
The published figures come from `data/cache/`, whose manifest records the engine that
produced them.

From 28 September 2026, every script below stamps its report under the title with the
command, the commit and the engine fingerprint. That fingerprint is the one in the published
manifest, so a report whose engine matches the manifest's measured the model the site shows.
The reports listed here predate the stamp, so the commands given for them are inferred
from their contents.

| Report | What it measured | Written by | Last written | Commit |
|---|---|---|---|---|
| `walk_forward_benchmark_forecast` | Forecast accuracy of RF, LightGBM, XGBoost and LEAR on walk-forward folds | `benchmark_walk_forward.py` | 2026-09-17 | `326f7a0` |
| `walk_forward_benchmark_full` | The same, plus a dispatch backtest per model over the whole window | `benchmark_walk_forward.py --revenue` | 2026-09-17 | `326f7a0` |
| `walk_forward_benchmark_confirmation` | The dispatch backtest over the confirmation folds only, from 2025-01-01 | `benchmark_walk_forward.py --revenue --revenue-from 2025-01-01` | 2026-09-17 | `326f7a0` |
| `feature_ablation` | Forecast accuracy with and without the wind forecast features | `ablate_features.py` | 2026-09-17 | `5284481` |
| `offer_valuation_day_ahead` | Offers priced by formula against the day-ahead trading plan (LP) | `compare_offer_valuation.py` | 2026-09-18 | `1e345ec` |
| `offer_valuation_bid_time` | Offers limited to what was known at 14:00 on D-1 | `compare_offer_valuation.py` | 2026-09-18 | `1e345ec` |
| `offer_valuation_vintages` | Dispatch planning on the early forecast (forecast vintages) | `compare_offer_valuation.py` | 2026-09-18 | `889ad8c` |
| `offer_curves` | The stepped price each MW slice would be offered at, one day | `offer_curves.py` | 2026-09-18 | `1e345ec` |
| `offer_valuation_conformal` | Conformal guard bands on the forecast | `compare_offer_valuation.py` | 2026-09-21 | `c42da23` |
| `offer_valuation_dynamic` | A forecast weight fitted per day | `compare_offer_valuation.py` | 2026-09-21 | `aa1f370` |
| `offer_valuation_smoothing` | Smoothing the offer and dispatch plans | `compare_offer_valuation.py` | 2026-09-21 | `9ed283d` |
| `offer_valuation_tilt` | A weight that leans on how volatile the day looks | `compare_offer_valuation.py` | 2026-09-21 | `60aee9a` |
| `interval_benchmark` | Guard bands (conformal, quantile, CQR, SPCI) scored as interval forecasts | `interval_benchmark.py` | 2026-09-23 | `c512ff0` |
| `offer_valuation_quantile` | Guard bands from those methods in dispatch | `compare_offer_valuation.py` | 2026-09-23 | `b9e6929` |
| `forecast_error_bars` | Bootstrap error bars on what the forecast is worth | `forecast_error_bars.py` | 2026-09-24 | `abec2a4` |
| `interval_literature_check` | Two claims about O'Connor et al. (2025), reproduced | `verify_interval_literature.py` | 2026-09-24 | `de84e90` |
| `offer_valuation_recovery` | Recovery credited through the Reserved Capacity | `compare_offer_valuation.py` | 2026-09-24 | `43268ab` |
| `offer_valuation_margin` | The block-start margin without the credit | `compare_offer_valuation.py` | 2026-09-24 | `c00909a` |

Scripts are under `scripts/`; `benchmark_walk_forward.py` takes `--report` to write under a
name of its own, and `compare_offer_valuation.py` and `interval_benchmark.py` do the same.
The benchmark and ablation scripts need `requirements-research.txt` for the models beyond
the Random Forest.

The commit is the last one that wrote the report. The experiments set aside since - the
per-day weight (`dynamic`, `tilt`), guard bands (`conformal`, `quantile`), smoothed plans
(`smoothing`) and the looser reading of recovery credit (the `rec` runs in `recovery`) -
left the engine on 2026-09-29, so they run only at that commit. The offer valuation
reports list their runs in the JSON with an engine hash of their own: before 2026-09-29,
`compare_offer_valuation.py` fingerprinted a different set of files from the manifest's, so
those hashes are not comparable with it. It now uses the manifest's.
