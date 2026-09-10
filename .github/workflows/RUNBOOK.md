# Workflow runbook

Operational notes for the four scheduled/triggered workflows. This file is committed
deliberately — `TODOS.md` is gitignored and does not survive a fresh clone, so anything
needed while diagnosing a failed run belongs here.

## What runs when

| Workflow | Trigger | Duration | Writes to the repo? |
|---|---|---|---|
| `tests.yml` | push, PR, **Mondays 06:00 UTC** | ~1 min | no |
| `refresh_data.yml` | **2nd of month 04:00 UTC**, manual | ~35–45 min | yes — commits to `main` |
| `deploy_site.yml` | push touching `site/`, `data/processed/`, `data/cache/`; manual | ~1 min | no |

`refresh_data.yml` is the only one that commits. It publishes to the live site as a side
effect, so its gates matter — see *Recovery* below.

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

**`Commit and push`** — see *Known unknowns* below. Most likely the token permissions.

**`Trigger the site deploy`** — the data landed but the site was not rebuilt. Recover by
dispatching the deploy by hand: `gh workflow run deploy_site.yml --ref main`.

## Known unknowns on the first run

Every step has been verified locally against a clone with an empty `data/raw` — an exact
runner simulation — but four things only prove out in Actions itself. In rough order of
likelihood:

1. **Token permissions.** The repo's `default_workflow_permissions` is `read`
   (`gh api repos/finbarrhodes/Halfcell/actions/permissions/workflow`). The workflow
   declares `contents: write` and `actions: write` explicitly, which *should* override a
   restricted default — but that is unproven here, so it is suspect number one if either
   the push or the deploy dispatch fails with a 403.

   The narrow fix is to keep the restricted default and grant per-workflow, which is what
   the file already does. The blunt fix, only if the above turns out not to work, is
   Settings → Actions → General → Workflow permissions → "Read and write". That widens the
   default for *every* workflow, so prefer the narrow route.

2. **The deploy dispatch.** A `GITHUB_TOKEN` push raises no `push` event — GitHub
   suppresses it to prevent recursion — so `deploy_site.yml`'s push trigger will never see
   the refresh commit. The final step calls `gh workflow run` instead, because
   `workflow_dispatch` is exempt from that rule. If this proves awkward, the alternative is
   pushing with a PAT, which restores the push trigger at the cost of a secret to rotate.

3. **`${{ github.event.inputs.lookback_days || '60' }}` on a schedule trigger**, where
   `inputs` is null. Expected to fall through to 60; only the cron exercises it.

4. **GNU `date -u -d "N days ago"`** — correct on `ubuntu-latest`, would fail on macOS.
   Never runs locally.

Branch protection on `main` was checked and is absent, so the bot push is not blocked by
a required-PR rule.

## Recovery: a bad refresh reached the site

The refresh commits straight to `main` and the deploy follows, so a bad run is live until
reverted. Both gates (freshness, cache consistency) run before staging, so the realistic
bad outcome is data that is *plausible but wrong* rather than truncated or torn.

```bash
git revert --no-edit <bot-commit-sha>
git push
```

A human push retriggers `deploy_site.yml` on its own — no dispatch needed, unlike the
bot's. Confirm the site recovered before investigating further.

To inspect what a run actually changed before deciding:

```bash
gh run list --workflow=refresh_data.yml
git show --stat <bot-commit-sha>
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
