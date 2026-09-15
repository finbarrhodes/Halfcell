"""
Rolling MPC, plan-on-forecast / settle-on-actual, over the out-of-sample window.

This is the only test that can tell whether the Part 8 extensions are worth
anything: under perfect foresight every penalty term can only destroy revenue,
because there is no forecast error to hedge. Here the LP plans on a naive D-1
forecast and cash is realised at actual prices, exactly as the production
backtester does.
"""
import sys, time
sys.path.insert(0, ".")
import numpy as np, cvxpy as cp
from ext_lab import DT, ETA, WEAR, P_MW, DUR, E_MWH, SOC_LO, SOC_HI, ARB_MW, load_prices

H = 96


def make_solver(extension=None, coef=0.0, Sigma=None, Shalf=None):
    """Build one DPP-compiled problem; returns a closure that solves it fast."""
    pP = cp.Parameter(H)
    pS = cp.Parameter()
    p_dis = cp.Variable(H, nonneg=True)
    p_chg = cp.Variable(H, nonneg=True)
    soc = cp.Variable(H + 1)
    u = p_dis - p_chg

    obj = cp.sum(cp.multiply(pP, u)) * DT - WEAR * cp.sum(p_dis) * DT
    cons = [
        soc[0] == pS,
        soc[1:] == soc[:-1] - p_dis * DT + cp.multiply(p_chg, ETA) * DT,
        soc >= SOC_LO, soc <= SOC_HI,
        p_dis <= ARB_MW, p_chg <= ARB_MW,
    ]
    if extension == "ramp":
        obj = obj - coef * cp.sum_squares(cp.diff(u))
    elif extension == "dod":
        obj = obj - coef * cp.sum_squares(soc - 0.5 * E_MWH)
    elif extension == "mv":
        obj = obj - coef * (DT ** 2) * cp.quad_form(u, cp.psd_wrap(Sigma))
    elif extension == "robust":
        obj = obj - coef * DT * cp.norm(Shalf @ u, 2)

    prob = cp.Problem(cp.Maximize(obj), cons)

    def solve(forecast, soc_now):
        pP.value = forecast
        pS.value = float(np.clip(soc_now, SOC_LO, SOC_HI))
        try:
            prob.solve(solver=cp.CLARABEL)
        except Exception:
            return 0.0, 0.0
        if prob.status not in ("optimal", "optimal_inaccurate") or p_dis.value is None:
            return 0.0, 0.0
        return max(0.0, float(p_dis.value[0])), max(0.0, float(p_chg.value[0]))

    return solve, prob


def rolling(wide, dates, solve):
    """Walk every settlement period; plan on D-1 profile, settle at actual."""
    soc = 0.5 * E_MWH
    cash = 0.0
    thru = 0.0
    u_prev = None
    ramps = []
    socs = []
    daily = []
    for di, d in enumerate(dates):
        actual = wide.loc[d].values                      # 48 actual prices for D
        prev = wide.iloc[wide.index.get_loc(d) - 1].values  # D-1 profile = the forecast
        day_cash = 0.0
        for sp in range(48):
            # horizon forecast: rest of D then all of D+1, both from the D-1 profile
            fc = np.concatenate([prev[sp:], prev, prev])[:H]
            d_mw, c_mw = solve(fc, soc)
            e_dis, e_chg = d_mw * DT, c_mw * DT
            soc = float(np.clip(soc - e_dis + e_chg * ETA, 0.0, E_MWH))
            step_cash = actual[sp] * (e_dis - e_chg) - WEAR * e_dis
            cash += step_cash
            day_cash += step_cash
            thru += e_dis
            u = d_mw - c_mw
            if u_prev is not None:
                ramps.append(abs(u - u_prev))
            u_prev = u
            socs.append(soc / E_MWH)
        daily.append(day_cash)
    socs = np.array(socs)
    daily = np.array(daily)
    return dict(cash=cash, throughput=thru,
                daily=daily.tolist(),
                day_mean=float(daily.mean()),
                day_sd=float(daily.std(ddof=1)),
                day_p5=float(np.percentile(daily, 5)),
                day_min=float(daily.min()),
                sharpe=float(daily.mean() / daily.std(ddof=1)) if daily.std(ddof=1) > 0 else 0.0,
                mean_ramp=float(np.mean(ramps)) if ramps else 0.0,
                p95_ramp=float(np.percentile(ramps, 95)) if ramps else 0.0,
                soc_p5=float(np.percentile(socs, 5)),
                soc_p95=float(np.percentile(socs, 95)),
                mean_abs_dev=float(np.mean(np.abs(socs - 0.5))))
