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

As of 2026-09-10, two things are worth clearing so the first run's output is signal
rather than noise.

- **The REPD warning already fires.** The fleet series is measured through 2026-03-01
  with five projected months after it, over the 3-month tolerance. DESNZ published the
  **July 2026** extract on 3 August 2026, which would move the frontier to roughly June
  and drop the tail to ~2 months. Download it from the
  [REPD quarterly extract page](https://www.gov.uk/government/publications/renewable-energy-planning-database-monthly-extract)
  into `data/raw/`, then:

  ```
  python src/data_collection/repd_collector.py data/raw/<extract>.xlsx
  python scripts/prepare_data.py            # full rebuild, local only
  python scripts/precompute_cache.py        # ~30 min
  python scripts/compute_kpis.py
  python scripts/check_cache_consistency.py
  ```

  Doing this locally also lands the 13-row settlement correction described in commit
  `6b600aa`, which the committed parquets do not yet carry.

- **Confirm you receive failure emails.** For scheduled workflows GitHub notifies the
  user who last modified the cron. The entire value of the freshness gate and the
  contract tests depends on somebody reading a red run.

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
