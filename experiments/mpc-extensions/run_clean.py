"""Re-run the risk configs with Sigma estimated ONLY on data strictly before the
test window, and over a longer window, with paired significance tests.

The first sweep estimated the forecast-error covariance from the most recent 520
days, which overlaps the rolling test period — lookahead. Sigma here is fitted on
errors ending before the window opens, so the risk model uses only information an
operator would have had.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats
from ext_lab import load_prices
from rolling import make_solver, rolling

SPLIT = pd.Timestamp("2025-03-02")
wide = load_prices()

# ---- Sigma from errors STRICTLY BEFORE the split -------------------------
pre = wide[wide.index < SPLIT]
arr = pre.values
err = arr[1:] - arr[:-1]
pairs = np.stack([np.concatenate([err[i], err[i + 1]]) for i in range(len(err) - 1)])
S = np.cov(pairs, rowvar=False)
S = 0.5 * (S + S.T)
S += 1e-6 * np.eye(S.shape[0]) * np.trace(S) / S.shape[0]
w, V = np.linalg.eigh(S)
Sigma = (V * np.clip(w, 1e-9, None)) @ V.T
Shalf = np.linalg.cholesky(Sigma).T
print(f"Sigma fitted on {len(pairs)} pre-split day-pairs "
      f"({pre.index.min().date()} -> {pre.index.max().date()}), "
      f"mean sd GBP{np.mean(np.sqrt(np.diag(Sigma))):.1f}/MWh", flush=True)

oos = wide.index[wide.index >= SPLIT]
dates = list(oos[:365])
print(f"test window: {dates[0].date()} -> {dates[-1].date()} ({len(dates)} days, "
      f"{len(dates)*48} solves per config)\n", flush=True)

configs = [("baseline", None, 0.0)]
configs += [(f"E2 DoD  g={c}", "dod", c) for c in (0.02, 0.05)]
configs += [(f"E3 mvar d={c}", "mv", c) for c in (1e-4, 5e-4, 2e-3, 8e-3)]
configs += [(f"E4 rob  k={c}", "robust", c) for c in (0.5, 1.0, 2.0)]

out = []
for lbl, ext, coef in configs:
    solve, _ = make_solver(ext, coef, Sigma, Shalf)
    t0 = time.time()
    r = rolling(wide, dates, solve)
    r.update(label=lbl, ext=ext or "none", coef=coef)
    out.append(r)
    json.dump(out, open("clean_res.json", "w"), indent=1)
    print(f"{lbl:16s} cash {r['cash']:9.0f}  sd/day {r['day_sd']:7.1f}  sharpe {r['sharpe']:5.2f}  "
          f"MWh {r['throughput']:8.1f}  mean|du| {r['mean_ramp']:4.2f}  ({time.time()-t0:.0f}s)",
          flush=True)

b = np.array(out[0]["daily"])
print(f"\n=== paired daily test vs baseline, n={len(b)} ===")
print(f"{'config':16s} {'D total':>10s} {'D/day':>8s} {'t':>6s} {'p':>8s}  verdict")
print("-" * 68)
for r in out[1:]:
    x = np.array(r["daily"]); d = x - b
    t, p = stats.ttest_rel(x, b)
    verdict = ("BETTER" if d.mean() > 0 else "worse") if p < 0.05 else "indistinguishable"
    print(f"{r['label']:16s} {d.sum():10.0f} {d.mean():8.1f} {t:6.2f} {p:8.4f}  {verdict}", flush=True)
print("DONE-CLEAN", flush=True)
