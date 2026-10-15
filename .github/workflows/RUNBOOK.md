# Workflow runbook

Operational notes for the four scheduled/triggered workflows. This file is committed
deliberately — `TODOS.md` is gitignored and does not survive a fresh clone, so anything
needed while diagnosing a failed run belongs here.

## What runs when

| Workflow | Trigger | Duration | Writes to the repo? |
|---|---|---|---|
| `tests.yml` | push, PR, **Mondays 06:23 UTC** | ~1 min | no |
| `refresh_data.yml` | **2nd of month 04:41 UTC**, manual | ~80–100 min | pushes a `data-refresh/*` branch and opens a PR; **you merge it** |
| `deploy_site.yml` | push touching `site/`, `data/processed/`, `data/cache/`; manual; pull requests (build only, no deploy) | ~1 min | no |

`refresh_data.yml` is the only one that commits, and only to a `data-refresh/*` branch. The
data reaches `main`, and the live site, when you merge the PR it opens. **A green run is not
the finish line:** a PR titled `chore: monthly data refresh through …` is waiting for your
approval, and until it is merged the site is a month behind. `main`'s ruleset requires the
approval (see *The merge gate* below); merging deploys the site on its own.

**A scheduled time is when a run becomes eligible, not when it starts.** GitHub delays
scheduled runs under load and can drop them outright. The first weekly `tests.yml` run,
due 06:00 UTC on 2026-09-14 while the crons were still on the hour, started at 11:13. So
an absent run at its due time is not yet a failure — give it a few hours. If the monthly
refresh has not started by the evening of the 2nd, dispatch it by hand:
`gh workflow run refresh_data.yml --ref main`.

## Before the first scheduled refresh

Nothing outstanding. Both items that were here are resolved:

- ~~Confirm you receive failure emails~~ — **confirmed** (2026-09-10), from the era of the
  Streamlit keep-alive job.
- ~~Land the 13-row settlement correction locally first~~ — **deliberately deferred**
  (2026-09-10) to the first scheduled run. See below so its diff is not mistaken for a
  bug.

### Expect 13 unexplained-looking rows in the first refresh diff

The first scheduled run will change **13 rows that are not new data**, alongside the
weeks it actually collects. This is expected and was chosen rather than pre-empted.

Commit `6b600aa` taught `prepare_data.py` to resolve overlapping pulls to the most
recently collected one, because the trailing days of any pull are provisional — Elexon
moves system prices through several settlement runs and NESO restates embedded solar and
wind. The previous filename-ordered dedup kept the stale tail. Both methodology surfaces
already describe the corrected rule, but the committed parquets predate it.

What will move, and only this:

| File | Rows | Where |
|---|---|---|
| `system_prices.parquet` | 3 | 2026-03-18, SP 35–37 (e.g. £90.76 → £164.00) |
| `generation_daily.parquet` | 10 | embedded Solar and Wind, 2026-02-15→19 |

`market_index.parquet` is untouched, so **no revenue figure moves** — APXMIDP is the
arbitrage reference. Anything beyond these 13 rows in a non-collected date range is worth
investigating; these 13 are not.

### What the REPD tail is not

Checked 2026-09-10 so it does not get re-investigated: the fleet series shows five
projected months, and that is the **normal steady state**, not a missed download.

`data/raw/REPD_Publication_Q2_2026.xlsx` is byte-identical to the file currently on the
DESNZ page (5,185,558 bytes, the July 2026 publication, last modified 2026-07-31). Its
latest *confirmed operational* battery project is 2026-03-23. Two lags stack: REPD
publishes a quarter behind, and within any extract a project's operational date only
appears once confirmed, which trails further. So `check_repd_freshness.py` tolerates 6
months, not the 3 it originally used — at 3 it would have warned permanently.

One trap if you ever do compare files: publications are named by **data quarter**, so a
newer release can arrive under a filename you already have. Compare by size, not name.

## Triage: `refresh_data.yml`

Failures are grouped by the step that goes red.

**`Collect recent data`** — an upstream outage, or a NESO/Elexon contract change.
`collect_data.py` logs and swallows per-source failures, so this step often stays *green*
while collecting nothing; the next step is what catches that. Check the step log for
swallowed tracebacks.

**`Verify the data actually advanced`** — collection returned nothing usable. Nothing has
been committed. Look at the previous step's log. If the APIs are simply down, re-dispatch
later; if a contract changed, `tests.yml`'s contracts job will say which.

**`Check the REPD fleet series has not drifted`** — cannot fail, only warns. See above.

**`Rebuild the backtest cache`** — if this times out rather than errors, the cause is
growth, not a bug: `DEFAULT_TEST_START` is fixed so every refresh lengthens the backtest.
Raise `timeout-minutes`, or reconsider keeping the cache in git.

**`Guard against a torn cache`** — the gate did its job. It runs *before* anything is
staged, so a half-written cache is discarded with the runner. Nothing was committed;
just re-dispatch.

**`Commit and push to a refresh branch`** — nothing reached `main`. If it was the push, the
cause is the token or a ruleset on `data-refresh/*` branches; the 70-minute result is lost
with the runner, so fix the cause and re-dispatch.

**`Open PR and run tests`** — the data is safe on its `data-refresh/*` branch; do not re-run
the 70-minute job. Open the PR (or find it) and read its checks. If the failure is
unrelated to the data (a lint or test already red on `main`, a flaky upstream), fix that,
re-run `gh workflow run tests.yml --ref <branch>`, then approve and merge by hand. If
`tests.yml` itself failed on the data, treat it like a bad refresh and do not merge.

## Triage: `deploy_site.yml`

**`Guard against a torn cache`** failing with *the cache was computed by engine …, but the
code in the tree fingerprints as …* — the engine changed after the cache was last built, so
the site would publish numbers the current code does not produce. The data is fine and
nothing was deployed. Rebuild the cache: dispatch the refresh
(`gh workflow run refresh_data.yml --ref main`), which rebuilds and redeploys, or run
`python scripts/precompute_cache.py` locally and commit `data/cache/`. Blank lines and
whole-line comments in the engine do not trigger this; any other edit, docstrings included,
does.

## The merge gate

The first scheduled run (2026-10-02) did all its work and then failed at the push: a ruleset
on `main` ("Base rules", no bypass actors) requires a PR and a passing "Unit tests" check
(GH013). The refresh therefore pushes a `data-refresh/*` branch, opens a PR, and dispatches
`tests.yml` on the branch, because a `GITHUB_TOKEN`-opened PR raises no `pull_request`
event and the required check would never report.

A bot merge was then tried and refused ("base branch policy prohibits the merge") even with
both checks green. The ruleset requires 0 approving reviews, but also sets
`require_extra_approval_for_unattributed_changes`; no documentation for it turned up, so
"the bot's commit needs a person's approval" is inferred from that refusal, not confirmed.
Approving and merging the PR by hand worked. So a person is the gate.

To make the refresh fully automatic instead, that rule would have to go (or a bypass actor
be added, e.g. a GitHub App — `GITHUB_TOKEN` cannot be one), and the workflow would need to
merge and dispatch `deploy_site.yml` itself, since a `GITHUB_TOKEN` merge raises no push
event. That is a weaker `main`, so it is a decision, not a fix.

What the first runs settled, so it is not re-investigated:

- **Token permissions.** `default_workflow_permissions` is `read`, and the workflow's own
  `contents` / `actions` / `pull-requests: write` was enough for the branch push, the PR and
  the `tests.yml` dispatch. No repo-wide change is needed. "Allow GitHub Actions to create
  and approve pull requests" (Settings → Actions → General) must stay on.
- **Deploy.** A person's merge is a normal push, and `deploy_site.yml` fired on its own.
- **`${{ github.event.inputs.lookback_days || '60' }}` on a schedule trigger** fell through
  to 60 (the 2026-10-02 run computed its window).
- **GNU `date -u -d`** works on `ubuntu-latest`.

## Recovery: a bad refresh reached the site

Merging the refresh PR deploys it, so a bad merge is live until reverted. Both gates
(freshness, cache consistency) run before staging, and the tests run on the PR, so the
realistic bad outcome is data that is *plausible but wrong* rather than truncated or torn —
which is what the review before you merge is for.

```bash
git revert -m 1 --no-edit <merge-commit-sha>   # a merge-commit merge needs -m 1
git revert --no-edit <refresh-commit-sha>      # a squash or rebase merge does not
git push
```

A human push retriggers `deploy_site.yml` on its own — no dispatch needed. Confirm the site
recovered before investigating further.

To inspect what a run actually changed before deciding:

```bash
gh run list --workflow=refresh_data.yml
git show --stat <refresh-commit-sha>
```

## Triage: `tests.yml` contracts job

`continue-on-error` is `${{ github.event_name != 'schedule' }}` — informational on a push
so an upstream outage does not block one, and a hard failure on the weekly run, which is
what the cron exists for. A `continue-on-error: true` job reports the whole run green, so
without that condition the notification would never arrive.

The assertion messages carry the specifics. After an April NESO rotation,
`test_eac_naming_convention_is_unchanged` prints the exact `_EAC_SEGMENTS` line to paste,
including the real resource id. A different message means NESO renamed the resources
instead, which breaks `_EAC_LIVE_RESOURCE_NAME` / `_EAC_ARCHIVE_NAME_RE` and must be
fixed before the next refresh commits truncated data.
