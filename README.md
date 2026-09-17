# Halfcell

**[halfcell.uk](https://halfcell.uk)** — live site

What can data science actually contribute to the energy transition? Halfcell answers that with
one concrete decision: a GB grid-scale battery deciding, every afternoon before prices are
known, how to divide its capacity between frequency response contracts and wholesale
arbitrage. The model backtests that decision over five years of public NESO and Elexon data,
under NESO's own participation rules, and measures what a better price forecast is worth.

Built as a static site so the offline pipeline stays the single source of truth: Python data
loaders read committed parquets at build time, so there is no server, no cold start and no
runtime compute.

## Results

50 MW / 2-hour battery, 90% round-trip efficiency, 16 September 2021 – 17 August 2026.
Annualised net revenue per MW:

| Strategy | Full stack | Arbitrage only | What it represents |
|---|---|---|---|
| Perfect Foresight | £104.1k | £62.5k | Theoretical ceiling; requires knowing day-D prices |
| **ML (Random Forest)** | **£97.0k** | **£48.9k** | Realistic case, using only data available at the bid deadline |
| Naive (D-1 prices) | £83.2k | £25.3k | Zero-skill floor; any real forecast must beat it |
| FR only | £82.2k | — | A site that ignores arbitrage entirely |

- **Foresight ratio 66%** — the share of the naive-to-perfect gap the forecast closes,
  `(ML − Naive) / (PF − Naive)`. On trading revenue alone the forecast captures 78% of what
  perfect foresight earns.
- **Frequency response dominates the stack.** FR-only earns £82.2k of the ML strategy's
  £97.0k, so arbitrage is the margin, not the business.
- **The battery stays compliant.** It starts 0.4–0.55% of settlement periods outside the
  state-of-energy range its contracts require, and almost every one is a single half-hour at
  an EFA block boundary.

## Scope

Halfcell is a modelling exercise; revenue is used as the yardstick because it is what real
battery sites rely on and makes the value of a forecast measurable in pounds where RMSE does
not, but it is not a commercial projection. This structure still answers the question *how
much is a better forecast worth*, and £/MW/yr can do that within a reasonably rigorous
modelling environment.

The figures describe one hypothetical 50 MW / 2-hour asset, under assumptions documented on
the methodology page, priced against historical public data. They are not a projection for
any real site, portfolio or market, and are not without methodological limitations: the model
excludes the Balancing Mechanism and the Capacity Market, assumes every offer clears at the
auction price, and rests on a single train/test split. These elements mark where I intend to
take Halfcell as much as where I am conscious of its current limits.

## The model

**Stage 1 — what to offer.** For each of the six EFA blocks, every product is offered at its
opportunity cost: the shadow arbitrage value of that MW plus the expected cost of the energy
the contract will deliver. The combination that earns most at the clearing prices is kept,
subject to NESO's rules — per-direction capacity including Reserved Capacity, the response
energy each contract needs in store or as headroom, whether the required range is reachable
from the battery's actual state of energy at the bid deadline, and the Maximum Sell Size.

**Stage 2 — dispatch.** A rolling linear programme (cvxpy/Clarabel) plans charge and
discharge over a 48-hour horizon at half-hourly resolution, re-solving every period and
executing only the first — model predictive control. State of energy is held inside NESO's
running Minimum State of Energy Requirement, which resets each block, falls as the battery
delivers response, and climbs back at the Energy Recovery rate. Reserved Capacity is
available to recover delivered energy but never to trade. Requirements are enforced as
penalised soft constraints, so a commitment the battery cannot physically reach degrades into
recorded unavailability instead of an infeasible solve.

**Response delivery.** Contracts are called on as GB frequency actually moved: NESO's
one-second frequency record is integrated along each service's response curve into the energy
every product delivers per MW contracted, per settlement period. That energy is neither paid
nor charged, so what a Low product gives away must be bought back and what a High product
absorbs can be sold on — which is why DR High clears at a negative price and is still worth
holding.

**Forecasts.** Three price signals drive the same engine: actual prices (ceiling), yesterday's
prices (floor), and a Random Forest on lagged prices, generation mix, cyclical time features
and BESS fleet capacity. A LEAR ensemble is implemented but not yet benchmarked against the
Random Forest.

## NESO rules, with citations

`src/analysis/neso_rules.py` holds every rule as a constant or feasibility check, each cited
inline to its source document *and clause* — so the model can be audited against NESO's
rulebooks rather than against itself. The tests reproduce NESO's own published worked example
for the state-of-energy requirement.

Rules changed during the backtest window, and the model applies each from its date: before
EAC go-live on 2 November 2023 a unit could offer only one service per EFA block; from then it
may split across all three. Reserved Capacity became binding in the Procurement Rules on
15 November 2024, though the model holds it from EAC go-live, because the energy requirement
it exists to serve applies throughout. Modelling choices that are *not* NESO rules — such as
holding no more than a fifth of any auction's cleared volume — are labelled as such on the
[methodology page](https://halfcell.uk/methodology), alongside the known limitations.

## Reproduce it

The processed data and the backtest cache are committed, so the published site rebuilds with
no API calls:

```bash
npm --prefix site install
npm --prefix site run build     # writes site/dist
npm --prefix site run dev       # local preview on :3000
```

Re-run the model itself:

```bash
pip install -r requirements.txt
python scripts/prepare_data.py          # raw -> data/processed/
python scripts/precompute_cache.py      # seven dispatch runs, ~60 min
python scripts/check_cache_consistency.py
```

Collect fresh data (both APIs are public, no key or registration):

```bash
python src/data_collection/collect_data.py --start 2026-01-01 --end 2026-01-31
python -m src.data_collection.frequency_collector --start 2026-01 --end 2026-01
```

One-second frequency is roughly 75 MB per month and stays out of the repo; only the derived
half-hourly delivery table is committed.

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -m "not integration"     # 201 tests, no network — what CI runs
pytest -m integration           # live NESO + Elexon contract checks
pytest --cov=src tests/         # with coverage
```

Unit tests cover settlement-period arithmetic and clock-change day lengths, the NESO rule
constants against their published values, allocation feasibility across all three rule eras,
the dispatch LP's mechanics and economics, response-curve integration, and the cache
consistency guard. Several tests deliberately pin known limitations so they stay visible.

Integration tests hit the live APIs and are deselected by default. They guard the two
upstream behaviours that have caused silent data loss here: NESO's annual EAC resource
rotation each April, and Elexon's date-parameter semantics.

## Automation

| Workflow | Trigger | Job |
|---|---|---|
| `tests.yml` | push, PR, weekly | Unit suite; the weekly run makes the upstream contract checks a hard failure |
| `refresh_data.yml` | 2nd of each month | Collect, append, rebuild the cache, verify freshness, commit |
| `deploy_site.yml` | push to `main` touching `site/`, `data/processed/` or `data/cache/` | Build and deploy to Cloudflare Pages |

The refresh workflow asserts data freshness before committing, because the failure mode it
guards is silent: a collection outage would leave the parquets untouched, reproduce identical
numbers, find nothing to commit and report green. `check_cache_consistency.py` gates the
deploy for the same reason — the strategy parquets are written sequentially over the better
part of an hour, so a build landing mid-run would publish a blend of strategies from
different runs.

Operational notes for the scheduled jobs are in `.github/workflows/RUNBOOK.md`.

## Deployment

Cloudflare Pages, deployed from GitHub Actions rather than Cloudflare's build image, because
the data loaders need pandas and pyarrow against the repo's parquets. First-time setup:

1. Create the Pages project as **Direct Upload** (Workers & Pages → Create → Pages → Upload
   assets), named `halfcell`. Cloudflare treats Direct Upload and Git-connected projects as
   different types, and `wrangler pages deploy` cannot target a Git-connected one — so if the
   repo was ever connected, delete that project and recreate it this way. The repo does not
   need connecting; the workflow uploads a prebuilt `dist/`.
2. Create an API token with **Cloudflare Pages: Edit**, and copy the Account ID from the
   Workers & Pages sidebar.
3. Add both as repository secrets: `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID`.

## Repository layout

```
src/
  analysis/
    neso_rules.py         NESO's rules as constants and feasibility checks, cited inline
    fr_allocation.py      Stage 1: allocation across the six products and arbitrage
    revenue_stack.py      The backtest engine: schedule, dispatch, settle
    response_delivery.py  Frequency -> energy delivered per MW contracted
    price_forecast.py     Feature matrix, model training, forecast backtests
    forecasting_models.py Random Forest, LEAR ensemble, DNN
    features.py           Feature engineering
  optimisation/mpc.py     The rolling dispatch LP
  data_collection/        NESO, Elexon, REPD and frequency collectors
scripts/                  prepare_data, precompute_cache, KPIs, OG card, guards, benchmarks
site/                     Observable Framework site (Market Overview, Backtester, Methodology)
tests/                    Unit and integration suites
data/
  raw/                    API responses (gitignored)
  processed/              Cleaned parquets (committed)
  cache/                  Precomputed backtest results + manifest (committed)
.github/workflows/        CI, monthly refresh, deploy, RUNBOOK
```

## Data sources

- **NESO Data Portal** (CKAN, no key) — DC/DR/DM auction clearing prices and volumes, EAC
  results from September 2021, and System Frequency at one-second resolution. 2 requests/min.
- **Elexon Insights** (no key) — Market Index Price (APXMIDP) as the wholesale reference,
  System Sell/Buy prices, and half-hourly generation by fuel type. 60 requests/min.
- **DESNZ Renewable Energy Planning Database** — operational GB battery capacity, used for the
  fleet-growth and price-suppression features.

## Licence and data

The code is under the [MIT Licence](LICENSE). The data is not mine to license, and each
source carries its own terms:

- **NESO** — DC/DR/DM auction results and System Frequency, under the
  [NESO Open Licence](https://www.neso.energy/data-portal/neso-open-licence), applied per
  dataset.
- **Elexon** — Market Index, system prices and generation by fuel, under Elexon's
  [licence for BMRS open data](https://www.elexon.co.uk/bsc/data/balancing-mechanism-reporting-agent/copyright-licence-bmrs-data/).
  Contains BMRS data © Elexon Limited copyright and database right 2026.
- **DESNZ** — Renewable Energy Planning Database, under the
  [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

The processed parquets under `data/` are derived from these sources and remain subject to
those terms. Nothing here is endorsed by NESO, Elexon or DESNZ.

## Contact

Finbar Rhodes — [LinkedIn](https://www.linkedin.com/in/finbar-rhodes-637650210/)
