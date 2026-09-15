# MPC extensions — parked work, and where to pick it up

**Status: blocked.** This work sits on top of `neso-compliant-fr` and the
response-delivery branch. Do not start applying it until both are merged into
`main`. Written 2026-09-15 against `neso-compliant-fr` @ `5937983`.

Everything here was prototyped against the **pre-NESO-rules** MPC (the old
fixed 10–90% SoC band, single power bound). The scripts in this directory are
throwaway research code, not production code, and they import the *old*
`solve_mpc` signature. They are committed because the measurements cost about
an hour of compute and the conclusions are worth keeping; expect to port them
before re-running.

---

## 1. Read this first: what is already done, so you don't redo it

Three of the recommendations from the earlier review **have already been
implemented on `neso-compliant-fr`**. Check them off rather than proposing them
again:

| Recommendation | Status on `neso-compliant-fr` |
|---|---|
| Compile the LP once with `cp.Parameter` instead of rebuilding every period | **Done.** `_CompiledLP` + `@lru_cache` in `src/optimisation/mpc.py`. Their measurement: 7.7 → 3.1 ms, 2.5×. Mine independently: 8.07 → 2.90 ms, 2.8×. Agreed. |
| Stop the FR SoC band binding when no FR is committed | **Done, and better than proposed.** Limits are now per-period `lo`/`hi` parameters derived from the actual commitments, and *soft* — shortfalls are penalised at £50,000/MWh rather than made infeasible, so the LP always moves toward compliance instead of silently doing nothing. |
| Model FR participation limits instead of assuming unlimited stacked acceptance | **Done, properly sourced.** `src/analysis/neso_rules.py` encodes Delivery Duration (DC 0.25 h / DM 0.5 h / DR 1.0 h), Reserved Capacity shares (DC 10% / DM 20% / DR 40%), energy-recovery share, and per-direction capacity use, each cited to a NESO or Ofgem document. |

Consequence for the earlier analysis: **the crude −22.4% / −27.7% FR revenue
multipliers I estimated are superseded.** They were a hypothesis about
per-direction stacking that I explicitly flagged as needing verification against
the service terms. That verification has now been done from the source
documents. Use `neso_rules.py`, discard my multipliers.

---

## 2. What is still open

### 2a. The four quadratic/conic extensions — tested, and the answer is no

Prototyped and backtested over 365 days out-of-sample (2025-03-02 → 2026-03-02),
17,520 solves per configuration, plan-on-forecast / settle-on-actual, paired
daily *t*-tests against baseline.

**No setting of any extension was ever significantly better than baseline.**
Small coefficients were statistically indistinguishable; larger ones were
significantly worse, monotonically. Baseline Sharpe (0.462) was the highest in
the column.

| Extension | Best result | Verdict |
|---|---|---|
| E1 ramp penalty `λ‖u_t − u_{t−1}‖²` | λ=1: −£5,966/yr, p=0.34, **34% less ramping** | Only arguable one. See below. |
| E2 depth-of-discharge `γ‖soc − soc_mid‖²` | γ=0.02: −£8,274/yr, p=0.53, *raises* variance | No |
| E3 mean-variance `δ·u'Σu` | δ=1e-4: −£6,939/yr, p=0.41 | No |
| E4 robust `κ‖Σ^½u‖₂` | κ=0.5: −£13,771/yr, p=0.18, ~2× slower, solver inaccuracy warnings | No |

**Why they fail, and it is not the tuning.** E3's Sharpe is flat at
0.434 / 0.428 / 0.426 across a twentyfold range of δ while cash falls from £208k
to £155k. The term is not reallocating between periods — it is scaling the whole
position vector down. The forecast-error covariance is close to homoskedastic
(error sd ≈ £24–45/MWh, under 2× range), so there is nothing to reallocate
*between*, and mean-variance degenerates into a leverage dial. De-levering a
strategy with positive edge simply earns less of it.

I tested the obvious rescue — Σ re-estimated on a trailing 120-day window,
refreshed monthly, always strictly before the traded day. It recovers roughly
half the shortfall and still never crosses baseline (p = 0.70, 0.15, 0.23).
The staleness hypothesis was also wrong in direction: the trailing window was
*more* volatile (£52.2/MWh) than the long fit (£47.6).

**This conclusion should survive the merge.** The mechanism is a property of the
forecast-error covariance, not of the SoC band or the power bounds. Magnitudes
will shift — the NESO-compliant allocator gives a different arbitrage slice than
the flat 6 MW I used — but do not expect the sign to change. If you re-test,
re-test to confirm, not to go looking for a coefficient that works.

**E1 at λ=1 is the one thing worth considering**, and only if ramping costs you
something real (inverter duty, warranty terms, a ramp spec). A third less
ramping for a revenue cost indistinguishable from zero is a reasonable trade.
But if you have an actual ramp *limit*, express it as a hard constraint
`|u_t − u_{t−1}| ≤ R` — that stays a pure LP, is more honest, and is free.

### 2b. The one extension worth building — needs response-delivery

This is the real prize and it is why the work is parked rather than dropped.

The old fixed 10–90% band is gone, but its replacement still derives `lo`/`hi`
deterministically from contracted volumes. Actual delivered response energy is
*stochastic* — it depends on realised system frequency. Once response-delivery
lands you can replace a deterministic requirement with a probabilistic one:

```
soc[t+1] = soc[t] − p_dis·Δt + p_chg·η·Δt − δ[t]
δ[t]     = Σ_s f_s · ρ_s[t]                    (MWh delivered per MW committed)

require  P( soc[t] ≥ soc_floor ) ≥ 1 − ε
becomes  soc_plan[t] − m[t]ᵀf − z_{1−ε}·‖C[t]^½ f‖₂ ≥ soc_floor
```

where `m[t]` is cumulative mean delivery per MW and `C[t]` the cross-service
covariance of cumulative delivery. That is a **second-order cone constraint** —
the cone Clarabel already ships, and the first thing in this codebase that would
make Clarabel the right solver on capability rather than convenience.

Two regimes decide your solver requirements:

- **One aggregated commitment** → the norm collapses to `f·σ[t]`, constraint is
  linear, still an LP.
- **A vector of per-service commitments with imperfectly correlated delivery** →
  a genuine SOCP.

**What response-delivery must emit for this to be possible** (flag this early if
that branch is still in flight):

1. `ρ_s[t]` — net delivered energy per MW committed, per service, per settlement
   period.
2. `m[t]` — cumulative mean drift as a function of elapsed periods within a block.
3. **`C[t]` — the cross-service covariance of *cumulative* delivery.** This is the
   piece most likely to be skipped, and without it there is no cone.

One warning that matters more than it looks: `σ[t]` grows like `√t` **only if
delivery increments are independent**. Frequency excursions are strongly
autocorrelated, so cumulative variance grows faster than `√t`. Assume
independence and you will systematically under-reserve — precisely the failure
the constraint exists to prevent. Estimate cumulative covariance directly rather
than scaling a per-period variance. If the empirical distribution turns out
fat-tailed, drop the Gaussian `z` and use the empirical quantile (still linear
in `f` for the aggregated case).

---

## 3. Where to hook in, in the merged code

Line numbers are against `neso-compliant-fr` @ `5937983`; re-check after merge.

**`src/optimisation/mpc.py`**

- `_CompiledLP.__init__` (line 61) — builds all Parameters and Variables.
- Objective assembled at lines 75–77, combined at line 93:
  `cp.Maximize(revenue - wear - breach)`.
- A quadratic term goes in at line 93, e.g. `- delta * cp.quad_form(self.p_dis - self.p_chg, cp.psd_wrap(Sigma))`.
- **Σ and any coefficient must be compile-time constants, not Parameters**, or
  DPP breaks.
- Any new coefficient must join the `_compiled()` lru_cache key (line 96–97),
  otherwise you will silently share a cached LP across different settings.
- A delivery chance constraint goes in the `constraints` list (lines 78–92),
  alongside the existing soft `lo`/`hi` rows.

**`src/analysis/fr_allocation.py`** — `allocate_block` (line 100) and
`allocate_day` (line 166) are where per-service commitment `f` is decided. The
SOCP formulation above couples allocation and dispatch, so if you go that route
this is the other half of the change.

### The silent failure mode to watch for

The new LP is compiled once and cached. If an added term breaks DPP compliance,
cvxpy does **not** raise — it silently re-canonicalises on every solve, and you
lose the 2.5× without any error. Guard it explicitly:

```python
assert lp.problem.is_dpp(), "quadratic term broke DPP — the compile cache is now useless"
```

I verified DPP survives all four extension types on the *old* LP
(`is_dpp() == True` for ramp, DoD, mean-variance and robust). It should survive
on the new one too, but the new LP has slack variables and per-period bound
parameters that I never tested against, so verify rather than assume.

---

## 4. What of my measurements is stale

| Measurement | Still valid? |
|---|---|
| Cone structure: 289 vars, 675 rows, 97 zero + 578 nonneg | **Stale.** New LP adds `below`/`above` slacks (H+1 each) → roughly 5H+3 variables. Re-dump with `prob.get_problem_data(cp.CLARABEL)`. |
| Solve timings (8.07 ms rebuild / 2.90 ms DPP) | **Stale but directionally confirmed** — the branch measured 7.7 / 3.1 on its own LP. |
| Native quadratic vs epigraph: 16.0 ms vs 43.1 ms, **2.7× faster native** | **Valid and still the key argument.** This is the Clarabel paper's headline claim, confirmed on this problem. It only starts paying once P is non-empty. |
| IPM iterations 12 → 12–14 when adding quadratic terms | Should hold; cheap to re-check. |
| Dense-P scaling H=48/96/192 → 10.2/16.0/26.1 ms | Should hold. |
| Risk-term backtest results (§2a) | **Conclusion holds, magnitudes stale.** Different arbitrage slice under NESO rules. |
| FR revenue multipliers 0.776 / 0.723 | **Superseded** by `neso_rules.py`. Discard. |

---

## 5. Re-running the harness

```bash
python -m venv .venv && . .venv/bin/activate
pip install cvxpy clarabel pandas pyarrow numpy scipy
python experiments/mpc-extensions/run_clean.py   # 365-day paired backtest, ~30 min
python experiments/mpc-extensions/run_ewma.py    # adaptive-covariance variant
python experiments/mpc-extensions/run_e1.py      # ramp penalty only
```

Run `run_clean.py` first — it writes `clean_res.json`, which the other two read
for their baseline. Result JSON is not committed.

`ext_lab.py` builds the LP and the forecast-error covariance; `rolling.py` is
the plan-on-forecast / settle-on-actual loop. **Porting note:** both hard-code
the old flat `ARB_MW = 6.0` and the old `FR_SOC_LOWER/UPPER` band. To re-run
post-merge, drive them from `fr_allocation` output and the new per-period
`lo`/`hi` parameters instead.

### The methodology trap in this harness

My first pass estimated Σ from the most recent 520 days — which **overlapped the
test window**. That lookahead made E3 look like a 13% revenue uplift with Sharpe
0.59 against a baseline 0.47. Re-fitting Σ strictly before the window and
extending to a full year reversed it completely.

It is the same failure mode as the in-sample ML backtest flagged in the
methodology review. If you extend this work, fit **every** statistic — covariance
included — on data strictly preceding the day being traded, and use paired daily
tests rather than comparing totals. Unpaired totals cannot distinguish a 3%
effect from noise here: daily revenue sd is roughly twice the daily mean.

---

## 6. Background

Four write-ups, in dependency order:

1. [Methodology review](https://claude.ai/code/artifact/fd34923d-4211-459a-86c6-a5342178c737) — full read of the backtester, ML stack and dispatch logic, plus the three findings.
2. [Three fixes for the backtester](https://claude.ai/code/artifact/6b922cb8-685d-4169-bae8-7e211cad8ec5) — costed remediation options for those findings.
3. [Cones, barriers and Clarabel](https://claude.ai/code/artifact/51089a28-603e-4789-af9a-796e237674da) — how conic optimisation and the solver work; Parts 8 and 9 are what this directory tests.
4. [What the P block buys](https://claude.ai/artifact/3JA12zu5bWNt4gnvhfH1CR) — the experiments summarised in §2a, with the full results table and mean-variance frontier.

Still open from (1) and (2), unrelated to this directory and not yet addressed
on any branch I can see:

- The ML backtest still reads its own training data (42 of 60 months in-sample;
  out-of-sample arbitrage foresight ratio was 0.005 against a published 0.837).
  The fix is walk-forward with recalibration and an append-only forecast cache.
  LightGBM fits 9.4× faster than the production Random Forest on identical data,
  which is what makes daily recalibration affordable.
- No `PF ≥ ML ≥ Naive` assertion in the test suite. That guard is what would have
  caught the broken ceiling on its own.
