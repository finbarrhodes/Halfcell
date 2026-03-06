"""
src/optimisation/mpc.py
=======================
Rolling MPC LP solver for BESS dispatch.

Re-solves at every settlement period (30 min). Only the first period's result
is executed; the rest of the horizon is discarded and re-optimised next period.

The LP enforces the FR SoC feasibility band [soc_min, soc_max] as a hard
constraint on all H+1 SoC variables, forcing the battery to pre-condition its
state of charge for upcoming FR delivery obligations.

Approximations (documented):
- Rolling horizon is not globally optimal; a single LP over the full backtest
  would be, but rolling MPC reflects real operational constraints.
- Mutual exclusion of charge/discharge is handled via LP relaxation: since
  the objective penalises cycling, simultaneous charge+discharge is never
  optimal at positive spread.

Solver: CLARABEL (bundled with cvxpy >= 1.4.0, no separate install needed).
"""

import numpy as np
import cvxpy as cp

DT = 0.5  # hours per settlement period


def solve_mpc(
    soc_current: float,
    price_forecast: np.ndarray,
    arb_mw_schedule: np.ndarray,
    soc_min: float,
    soc_max: float,
    energy_mwh: float,
    efficiency_rt: float,
    cycling_cost_per_mwh: float,
    horizon: int = 96,
) -> tuple[float, float]:
    """
    Solve the rolling MPC LP for the current settlement period.

    Parameters
    ----------
    soc_current : float
        Battery state of charge at the start of this period (MWh).
    price_forecast : np.ndarray, shape (H,)
        Forecast (or actual, for perfect-foresight) price per period (£/MWh).
    arb_mw_schedule : np.ndarray, shape (H,)
        Maximum MW available for arbitrage in each period (power_mw - fr_mw,
        constant within each EFA block, steps every 8 periods).
    soc_min : float
        SoC floor in MWh = FR_SOC_LOWER × energy_mwh.
    soc_max : float
        SoC ceiling in MWh = FR_SOC_UPPER × energy_mwh.
    energy_mwh : float
        Total battery capacity in MWh (for guard clipping).
    efficiency_rt : float
        Round-trip efficiency, e.g. 0.90.
    cycling_cost_per_mwh : float
        Degradation cost per MWh discharged (£/MWh).
    horizon : int
        Number of periods to include in the LP (capped by array lengths).

    Returns
    -------
    (energy_dis_mwh, energy_chg_mwh) : tuple[float, float]
        Energy to dispatch in period 0 only (MWh).
        Returns (0.0, 0.0) on LP failure or if no trade is optimal.
    """
    H = min(horizon, len(price_forecast), len(arb_mw_schedule))
    if H == 0:
        return 0.0, 0.0

    # Clip soc_current into the feasible band — prevents infeasibility on the
    # soc[0] == soc_current equality constraint if SoC drifts marginally out
    # of band due to floating-point accumulation.
    soc_current = float(np.clip(soc_current, soc_min, soc_max))

    prices = np.asarray(price_forecast[:H], dtype=float)
    arb_mw = np.asarray(arb_mw_schedule[:H], dtype=float)

    # Decision variables
    p_dis = cp.Variable(H, nonneg=True)  # discharge power (MW)
    p_chg = cp.Variable(H, nonneg=True)  # charge power (MW)
    soc   = cp.Variable(H + 1)           # SoC at start of each period (MWh)

    # Objective: maximise net arbitrage revenue minus cycling degradation cost
    revenue   = cp.sum(cp.multiply(prices, (p_dis - p_chg))) * DT
    wear      = cycling_cost_per_mwh * cp.sum(p_dis) * DT
    objective = cp.Maximize(revenue - wear)

    constraints = [
        # Initial condition
        soc[0] == soc_current,
        # SoC state equation (charge input reduced by round-trip loss)
        soc[1:] == soc[:-1] - p_dis * DT + cp.multiply(p_chg, efficiency_rt) * DT,
        # FR feasibility band enforced at ALL H+1 periods (hard constraint).
        # This forces the LP to pre-condition SoC for future FR delivery blocks.
        soc >= soc_min,
        soc <= soc_max,
        # Power limits: arb MW is the residual after FR commitment
        p_dis <= arb_mw,
        p_chg <= arb_mw,
    ]

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL, warm_start=True)
    except Exception:
        return 0.0, 0.0

    if prob.status not in ("optimal", "optimal_inaccurate") or p_dis.value is None:
        return 0.0, 0.0

    e_dis = max(0.0, float(p_dis.value[0])) * DT
    e_chg = max(0.0, float(p_chg.value[0])) * DT
    return e_dis, e_chg
