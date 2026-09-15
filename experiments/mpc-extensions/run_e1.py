"""E1 (ramp penalty) on the same clean 365-day window, for table consistency.

E1 does not use Sigma at all, so the lookahead flaw never applied to it -- but it
was only ever run on the 45-day window, which has far less statistical power.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats
from ext_lab import load_prices
from rolling import make_solver, rolling

wide = load_prices()
Sigma = np.load("Sigma.npy"); Shalf = np.load("Shalf.npy")   # unused by E1
oos = wide.index[wide.index >= pd.Timestamp("2025-03-02")]
dates = list(oos[:365])
b = np.array(json.load(open("clean_res.json"))[0]["daily"])

out = []
for lbl, coef in [("E1 ramp L=1", 1.0), ("E1 ramp L=5", 5.0), ("E1 ramp L=20", 20.0)]:
    solve, _ = make_solver("ramp", coef, Sigma, Shalf)
    t0 = time.time()
    r = rolling(wide, dates, solve)
    r.update(label=lbl, coef=coef)
    out.append(r)
    json.dump(out, open("e1_res.json", "w"), indent=1)
    x = np.array(r["daily"]); d = x - b
    t, p = stats.ttest_rel(x, b)
    verdict = ("BETTER" if d.mean() > 0 else "worse") if p < 0.05 else "indistinguishable"
    print(f"{lbl:14s} cash {r['cash']:9.0f}  sd/day {r['day_sd']:7.1f}  sharpe {r['sharpe']:5.2f}  "
          f"mean|du| {r['mean_ramp']:4.2f}  | D {d.sum():+8.0f}  t {t:6.2f}  p {p:7.4f}  {verdict}  "
          f"({time.time()-t0:.0f}s)", flush=True)
print("DONE-E1", flush=True)
