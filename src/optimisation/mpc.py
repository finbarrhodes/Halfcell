"""
src/optimisation/mpc.py
=======================
Rolling MPC LP solver for BESS arbitrage dispatch around frequency response
commitments.

Re-solves at every settlement period (30 min). Only the first period's result
is executed; the rest of the horizon is discarded and re-optimised next period.

State of energy is held inside the range that keeps every contracted response
product deliverable (Response Service Terms 6.11): at least the Low products'
response energy in store, and at least the High products' as headroom. Those
limits change from block to block with the commitments, so they are passed per
period rather than as one fixed band.

The limits are soft. A unit can arrive at a block already outside them - a
commitment it could not physically reach in time - and demanding otherwise
makes the LP infeasible, which used to mean silently doing nothing. Shortfalls
are instead allowed at a penalty far above any energy price, so the solver
always moves towards compliance and never breaches to earn a spread. The caller
records any breach, which NESO treats as unavailability for that settlement
period (Service Terms 6.12).

Trading power is limited separately on the discharge and charge sides, because
response commitments and their reserved capacity use the two sides unequally.

Each horizon length is compiled once with cvxpy Parameters and re-solved with
new values, rather than rebuilt every period. That is 2.5x faster (7.7 to
3.1 ms per solve, measured 2026-09-14) with identical solutions, which keeps the
scenario runs inside the refresh workflow's time limit.

Approximations (documented):
- Rolling horizon is not globally optimal; a single LP over the full backtest
  would be, but rolling MPC reflects real operational constraints.
- Stored energy has no terminal value, so energy left at the end of the horizon
  is worth nothing to the LP. The 48-hour horizon keeps this from dominating.
- Mutual exclusion of charge/discharge is handled via LP relaxation: since
  the objective penalises cycling, simultaneous charge+discharge is never
  optimal at positive spread.

Solver: CLARABEL (bundled with cvxpy >= 1.4.0, no separate install needed).
"""

from functools import lru_cache

import cvxpy as cp
import numpy as np

DT = 0.5  # hours per settlement period

# £ per MWh by which state of energy misses a response requirement. It only has
# to exceed what breaching could ever earn: APXMIDP peaks at £1,984/MWh in this
# dataset (99.99th percentile £1,340), so this is ~25x the worst case while
# staying well-scaled for the solver.
SOE_BREACH_PENALTY_GBP_PER_MWH = 50_000.0


class _CompiledLP:
    """The dispatch LP for one horizon length and battery, built once."""

    def __init__(self, H: int, energy_mwh: float, efficiency_rt: float, cycling_cost_per_mwh: float):
        self.prices  = cp.Parameter(H)
        self.dis_max = cp.Parameter(H, nonneg=True)
        self.chg_max = cp.Parameter(H, nonneg=True)
        self.lo      = cp.Parameter(H + 1)
        self.hi      = cp.Parameter(H + 1)
        self.soc0    = cp.Parameter()

        self.p_dis = cp.Variable(H, nonneg=True)       # discharge power (MW)
        self.p_chg = cp.Variable(H, nonneg=True)       # charge power (MW)
        self.soc   = cp.Variable(H + 1)                # state of energy at start of each period (MWh)
        self.below = cp.Variable(H + 1, nonneg=True)   # shortfall under the stored-energy requirement
        self.above = cp.Variable(H + 1, nonneg=True)   # shortfall under the headroom requirement

        revenue = self.prices @ (self.p_dis - self.p_chg) * DT
        wear    = cycling_cost_per_mwh * cp.sum(self.p_dis) * DT
        breach  = SOE_BREACH_PENALTY_GBP_PER_MWH * cp.sum(self.below + self.above)

        constraints = [
            self.soc[0] == self.soc0,
            # State equation: charge input reduced by the round-trip loss
            self.soc[1:] == self.soc[:-1] - self.p_dis * DT + (efficiency_rt * DT) * self.p_chg,
            # Physical limits of the store
            self.soc >= 0.0,
            self.soc <= energy_mwh,
            # Response requirements, softened by the penalised shortfalls
            self.soc + self.below >= self.lo,
            self.soc - self.above <= self.hi,
            # Trading power left around the response commitments, per side
            self.p_dis <= self.dis_max,
            self.p_chg <= self.chg_max,
        ]
        self.problem = cp.Problem(cp.Maximize(revenue - wear - breach), constraints)


@lru_cache(maxsize=16)
def _compiled(H: int, energy_mwh: float, efficiency_rt: float, cycling_cost_per_mwh: float) -> _CompiledLP:
    return _CompiledLP(H, energy_mwh, efficiency_rt, cycling_cost_per_mwh)


def _per_period(value, length: int) -> np.ndarray:
    """Broadcast a scalar, or take the first `length` entries of a sequence."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(length, float(arr))
    return arr[:length]


def solve_mpc(
    soc_current: float,
    price_forecast: np.ndarray,
    arb_mw_schedule: np.ndarray,
    soc_min,
    soc_max,
    energy_mwh: float,
    efficiency_rt: float,
    cycling_cost_per_mwh: float,
    horizon: int = 96,
    charge_mw_schedule: np.ndarray | None = None,
    return_plan: bool = False,
):
    """
    Solve the rolling MPC LP for the current settlement period.

    Parameters
    ----------
    soc_current : float
        State of energy at the start of this period (MWh).
    price_forecast : np.ndarray, shape (H,)
        Forecast (or actual, for perfect foresight) price per period (£/MWh).
    arb_mw_schedule : np.ndarray, shape (H,)
        Maximum discharge power for trading in each period (MW). Also the
        charge limit unless charge_mw_schedule is given.
    soc_min, soc_max : float or array of shape (H+1,)
        Required state-of-energy range at the start of each period (MWh). A
        scalar applies to every period.
    energy_mwh : float
        Usable energy capacity (MWh); a hard physical bound.
    efficiency_rt : float
        Round-trip efficiency, e.g. 0.90.
    cycling_cost_per_mwh : float
        Degradation cost per MWh discharged (£/MWh).
    horizon : int
        Number of periods to include in the LP (capped by array lengths).
    charge_mw_schedule : np.ndarray, shape (H,), optional
        Maximum charge power for trading in each period (MW).
    return_plan : bool
        Return the full horizon plan instead of the first period's energies.

    Returns
    -------
    (energy_dis_mwh, energy_chg_mwh) for period 0, or with return_plan a dict
    of arrays {"p_dis", "p_chg", "soc", "shortfall_mwh"}. Returns (0.0, 0.0),
    or None with return_plan, if the solver fails.
    """
    charge_schedule = arb_mw_schedule if charge_mw_schedule is None else charge_mw_schedule
    H = min(horizon, len(price_forecast), len(arb_mw_schedule), len(charge_schedule))
    if H == 0:
        return None if return_plan else (0.0, 0.0)

    lp = _compiled(H, float(energy_mwh), float(efficiency_rt), float(cycling_cost_per_mwh))
    lp.soc0.value    = float(np.clip(soc_current, 0.0, energy_mwh))
    lp.prices.value  = np.asarray(price_forecast[:H], dtype=float)
    lp.dis_max.value = np.clip(np.asarray(arb_mw_schedule[:H], dtype=float), 0.0, None)
    lp.chg_max.value = np.clip(np.asarray(charge_schedule[:H], dtype=float), 0.0, None)
    lp.lo.value      = np.clip(_per_period(soc_min, H + 1), 0.0, energy_mwh)
    lp.hi.value      = np.clip(_per_period(soc_max, H + 1), 0.0, energy_mwh)

    try:
        lp.problem.solve(solver=cp.CLARABEL)
    except Exception:
        return None if return_plan else (0.0, 0.0)

    if lp.problem.status not in ("optimal", "optimal_inaccurate") or lp.p_dis.value is None:
        return None if return_plan else (0.0, 0.0)

    if return_plan:
        return {
            "p_dis": np.maximum(lp.p_dis.value, 0.0),
            "p_chg": np.maximum(lp.p_chg.value, 0.0),
            "soc": lp.soc.value.copy(),
            "shortfall_mwh": lp.below.value + lp.above.value,
        }

    e_dis = max(0.0, float(lp.p_dis.value[0])) * DT
    e_chg = max(0.0, float(lp.p_chg.value[0])) * DT
    return e_dis, e_chg
