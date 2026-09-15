"""Paired significance tests + a longer-window re-run for the promising configs.

The 45-day sweep cannot distinguish a 3% revenue change from noise on unpaired
totals (sd/day is roughly 2x mean/day). Every config sees identical prices on
identical days, so the paired daily difference is the right test and is far
more powerful than comparing totals.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats
from ext_lab import load_prices
from rolling import make_solver, rolling

res = json.load(open("rolling_res.json"))
base = next(r for r in res if r["label"] == "baseline")
b = np.array(base["daily"])

print(f"=== paired daily comparison vs baseline, n={len(b)} days ===")
print(f"{'config':18s} {'Δ total':>9s} {'Δ/day':>8s} {'sd(Δ)':>8s} {'t':>6s} {'p':>7s}  verdict")
print("-" * 76)
for r in res:
    if r["label"] == "baseline":
        continue
    x = np.array(r["daily"])
    d = x - b
    t, p = stats.ttest_rel(x, b)
    verdict = ("better" if d.mean() > 0 else "worse") if p < 0.05 else "indistinguishable"
    print(f"{r['label']:18s} {d.sum():9.0f} {d.mean():8.1f} {d.std(ddof=1):8.1f} "
          f"{t:6.2f} {p:7.4f}  {verdict}")

# ---------------------------------------------------------------- long window
SEL = json.load(open("selection.json")) if os.path.exists("selection.json") else []
if SEL:
    wide = load_prices()
    Sigma = np.load("Sigma.npy"); Shalf = np.load("Shalf.npy")
    oos = wide.index[wide.index >= pd.Timestamp("2025-03-02")]
    dates = list(oos[:400])
    print(f"\n=== longer window: {dates[0].date()} -> {dates[-1].date()} ({len(dates)} days) ===")
    out = []
    for lbl, ext, coef in SEL:
        solve, _ = make_solver(ext, coef, Sigma, Shalf)
        t0 = time.time()
        r = rolling(wide, dates, solve)
        r.update(label=lbl, ext=ext or "none", coef=coef)
        out.append(r)
        json.dump(out, open("long_res.json", "w"), indent=1)
        print(f"{lbl:18s} cash {r['cash']:9.0f}  sd/day {r['day_sd']:7.1f}  "
              f"sharpe {r['sharpe']:5.2f}  MWh {r['throughput']:8.1f}  ({time.time()-t0:.0f}s)",
              flush=True)
    lb = np.array(out[0]["daily"])
    print(f"\n{'config':18s} {'Δ total':>10s} {'Δ/day':>8s} {'t':>6s} {'p':>8s}  verdict")
    print("-" * 70)
    for r in out[1:]:
        x = np.array(r["daily"]); d = x - lb
        t, p = stats.ttest_rel(x, lb)
        verdict = ("better" if d.mean() > 0 else "worse") if p < 0.05 else "indistinguishable"
        print(f"{r['label']:18s} {d.sum():10.0f} {d.mean():8.1f} {t:6.2f} {p:8.4f}  {verdict}")
    print("DONE-LONG", flush=True)
