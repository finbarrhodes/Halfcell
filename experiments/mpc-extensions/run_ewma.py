"""Does the mean-variance term fail because the idea is wrong, or because the
covariance is stale?

The pre-split Sigma is fitted on 2019-2025, which includes the 2021-22 gas crisis:
mean forecast-error sd GBP47.6/MWh against roughly GBP34/MWh in the test period.
An over-large Sigma over-shrinks every position, which would depress the mean
faster than it depresses the variance -- exactly the observed failure.

This re-runs the risk configs with Sigma re-estimated on a trailing 120-day
window, refreshed monthly, using only data strictly before each refresh date.
If E3 still loses, the term is a preference dial, not an improvement.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats
from ext_lab import load_prices, DT, ETA, WEAR, E_MWH, SOC_LO, SOC_HI, ARB_MW
from rolling import make_solver

SPLIT = pd.Timestamp("2025-03-02")
LOOKBACK = 120
H = 96
wide = load_prices()


def sigma_before(date, lookback=LOOKBACK):
    pre = wide[wide.index < date]
    arr = pre.values[-(lookback + 2):]
    err = arr[1:] - arr[:-1]
    pairs = np.stack([np.concatenate([err[i], err[i + 1]]) for i in range(len(err) - 1)])
    S = np.cov(pairs, rowvar=False)
    S = 0.5 * (S + S.T)
    S += 1e-6 * np.eye(S.shape[0]) * np.trace(S) / S.shape[0]
    w, V = np.linalg.eigh(S)
    return (V * np.clip(w, 1e-9, None)) @ V.T


def rolling_adaptive(dates, ext, coef):
    """Same rolling loop, but Sigma refreshed at the start of each month."""
    soc = 0.5 * E_MWH
    daily, thru = [], 0.0
    cur_month, solve = None, None
    for d in dates:
        if (d.year, d.month) != cur_month:
            cur_month = (d.year, d.month)
            S = sigma_before(d)
            Sh = np.linalg.cholesky(S).T
            solve, _ = make_solver(ext, coef, S, Sh)
        actual = wide.loc[d].values
        prev = wide.iloc[wide.index.get_loc(d) - 1].values
        day = 0.0
        for sp in range(48):
            fc = np.concatenate([prev[sp:], prev, prev])[:H]
            d_mw, c_mw = solve(fc, soc)
            e_dis, e_chg = d_mw * DT, c_mw * DT
            soc = float(np.clip(soc - e_dis + e_chg * ETA, 0.0, E_MWH))
            day += actual[sp] * (e_dis - e_chg) - WEAR * e_dis
            thru += e_dis
        daily.append(day)
    daily = np.array(daily)
    return dict(cash=float(daily.sum()), day_sd=float(daily.std(ddof=1)),
                sharpe=float(daily.mean() / daily.std(ddof=1)),
                throughput=thru, daily=daily.tolist())


oos = wide.index[wide.index >= SPLIT]
dates = list(oos[:365])
s0 = sigma_before(dates[0])
print(f"trailing-{LOOKBACK}d Sigma at window open: mean sd GBP"
      f"{np.mean(np.sqrt(np.diag(s0))):.1f}/MWh  (vs GBP47.6 for the 2019-2025 fit)", flush=True)

base = json.load(open("clean_res.json"))[0]
b = np.array(base["daily"])
print(f"baseline (from clean run): cash {base['cash']:.0f}  sharpe {base['sharpe']:.2f}\n", flush=True)

out = []
for lbl, ext, coef in [("E3 adaptive d=1e-4", "mv", 1e-4),
                       ("E3 adaptive d=5e-4", "mv", 5e-4),
                       ("E4 adaptive k=0.5", "robust", 0.5)]:
    t0 = time.time()
    r = rolling_adaptive(dates, ext, coef)
    r.update(label=lbl)
    out.append(r)
    json.dump(out, open("ewma_res.json", "w"), indent=1)
    x = np.array(r["daily"]); d = x - b
    t, p = stats.ttest_rel(x, b)
    verdict = ("BETTER" if d.mean() > 0 else "worse") if p < 0.05 else "indistinguishable"
    print(f"{lbl:20s} cash {r['cash']:9.0f}  sd/day {r['day_sd']:7.1f}  sharpe {r['sharpe']:5.2f}  "
          f"| vs base D {d.sum():+8.0f}  t {t:6.2f}  p {p:7.4f}  {verdict}  ({time.time()-t0:.0f}s)",
          flush=True)
print("DONE-EWMA", flush=True)
