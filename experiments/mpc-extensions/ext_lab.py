"""
Prototype harness for the Part 8 / Part 9 extensions of the Clarabel write-up.
Runs entirely outside src/ so it cannot collide with the dispatch-engine work
happening in the other session.

Builds the real Halfcell MPC LP at H=96 and adds, one at a time:
  E1  ramping penalty          lambda * ||u_t - u_{t-1}||^2        -> tridiagonal P
  E2  depth-of-discharge       gamma  * ||soc_t - soc_mid||^2      -> diagonal P
  E3  mean-variance risk       delta  * u' Sigma u                 -> dense P
  E4  robust / chance          kappa  * ||Sigma^(1/2) u||_2        -> second-order cone

and reports: solve status, IPM iterations, solve time, cone structure, and what
each one actually does to the dispatch.
"""
import sys, time, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np, pandas as pd, cvxpy as cp

DT, ETA, WEAR = 0.5, 0.90, 3.0
P_MW, DUR = 50.0, 2.0
E_MWH = P_MW * DUR
SOC_LO, SOC_HI = 0.10 * E_MWH, 0.90 * E_MWH
H = 96
ARB_MW = 6.0          # representative residual after FR commitment (cache: arb_mw ~ 4.3-6.3)

# ---------------------------------------------------------------- data
def load_prices():
    mi = pd.read_parquet(REPO / "data" / "processed" / "market_index.parquet")
    apx = mi[mi["dataProvider"] == "APXMIDP"].copy()
    apx["settlementDate"] = pd.to_datetime(apx["settlementDate"]).dt.normalize()
    apx = apx[apx["settlementPeriod"] <= 48]
    wide = apx.pivot_table(index="settlementDate", columns="settlementPeriod", values="price")
    wide = wide.dropna()
    return wide  # index=date, cols=1..48

def naive_error_covariance(wide, n_days=520, ridge=1e-6):
    """96-dim covariance of D-1 naive forecast errors over consecutive 2-day blocks."""
    arr = wide.values                       # (days, 48)
    err = arr[1:] - arr[:-1]                # forecast D from D-1 -> error
    # stack consecutive pairs into 96-vectors
    pairs = np.stack([np.concatenate([err[i], err[i + 1]]) for i in range(len(err) - 1)])
    pairs = pairs[-n_days:]
    S = np.cov(pairs, rowvar=False)
    S = S + ridge * np.eye(S.shape[0]) * np.trace(S) / S.shape[0]
    # symmetrise + PSD-clip for numerical safety
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-9, None)
    return (V * w) @ V.T, pairs

# ---------------------------------------------------------------- model
def build(prices, soc0, extension=None, coef=0.0, Sigma=None, Shalf=None, epigraph=False):
    p_dis = cp.Variable(H, nonneg=True)
    p_chg = cp.Variable(H, nonneg=True)
    soc   = cp.Variable(H + 1)
    u     = p_dis - p_chg                      # net export power, MW

    obj = cp.sum(cp.multiply(prices, u)) * DT - WEAR * cp.sum(p_dis) * DT
    cons = [
        soc[0] == soc0,
        soc[1:] == soc[:-1] - p_dis * DT + cp.multiply(p_chg, ETA) * DT,
        soc >= SOC_LO, soc <= SOC_HI,
        p_dis <= ARB_MW, p_chg <= ARB_MW,
    ]

    if extension == "ramp":
        obj = obj - coef * cp.sum_squares(cp.diff(u))
    elif extension == "dod":
        obj = obj - coef * cp.sum_squares(soc - 0.5 * E_MWH)
    elif extension == "mv":
        if epigraph:                            # force the reformulation Clarabel avoids
            t = cp.Variable()
            cons.append(cp.quad_form(u, cp.psd_wrap(Sigma)) <= t)
            obj = obj - coef * (DT ** 2) * t
        else:                                   # native quadratic objective -> populates P
            obj = obj - coef * (DT ** 2) * cp.quad_form(u, cp.psd_wrap(Sigma))
    elif extension == "robust":
        obj = obj - coef * DT * cp.norm(Shalf @ u, 2)

    return cp.Problem(cp.Maximize(obj), cons), (p_dis, p_chg, soc)


def run(label, prices, soc0, **kw):
    prob, (p_dis, p_chg, soc) = build(prices, soc0, **kw)
    data, _, _ = prob.get_problem_data(cp.CLARABEL)
    dims = data["dims"]
    Pm = data.get("P")
    t0 = time.perf_counter()
    prob.solve(solver=cp.CLARABEL)
    wall = time.perf_counter() - t0
    st = prob.solver_stats
    d, c, s = p_dis.value, p_chg.value, soc.value
    u = d - c
    return dict(
        label=label, status=prob.status,
        obj=float(prob.value),
        cash=float(np.sum(prices * u) * DT - WEAR * np.sum(d) * DT),
        iters=getattr(st, "num_iters", None),
        ms=1e3 * wall,
        n=data["c"].shape[0], m=data["A"].shape[0],
        zero=dims.zero, nonneg=dims.nonneg,
        soc_dims=list(dims.soc) if hasattr(dims, "soc") else [],
        P_nnz=(0 if Pm is None else int(Pm.nnz)),
        throughput=float(np.sum(d) * DT),
        max_ramp=float(np.max(np.abs(np.diff(u)))) if H > 1 else 0.0,
        soc_range=float((s.max() - s.min()) / E_MWH),
        mean_abs_dev=float(np.mean(np.abs(s - 0.5 * E_MWH)) / E_MWH),
    )
