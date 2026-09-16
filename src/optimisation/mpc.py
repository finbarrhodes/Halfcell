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

Reserved Capacity is kept for state-of-energy management (SOE guidance §3), so
it is a separate pair of flows usable in any period of the horizon. The plan
counts what those flows cost at the price - discharging into a negative price,
charging at a positive one - but never what they earn, so it draws on the
reserve to keep a requirement reachable and not to make money. Executed energy
settles at actual prices whichever route it took.

Each horizon length is compiled once with cvxpy Parameters and re-solved with
new values, rather than rebuilt every period. That is 2.5x faster (7.7 to
3.1 ms per solve, measured 2026-09-14) with identical solutions, which keeps the
scenario runs inside the refresh workflow's time limit. The reserve flows add
half again to each solve even when bounded at zero (3.1 to 4.6 ms, measured
2026-09-15), so a horizon holding no reserve compiles without them.

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
# dataset. Larger is not safer: the penalty shares a cost vector with the prices,
# and at £50,000 Clarabel needed a median 23 iterations and returned 1% of
# January 2026 solves inaccurate, against 18 and none at £5,000 (2026-09-15).
SOE_BREACH_PENALTY_GBP_PER_MWH = 5_000.0

# £ per MWh moved through Reserved Capacity. Nominal: it only stops the plan
# moving energy through the reserve when no requirement needs it.
RESERVE_USE_COST_GBP_PER_MWH = 1.0


class _CompiledLP:
    """The dispatch LP for one horizon length and battery, built once."""

    def __init__(self, H: int, energy_mwh: float, efficiency_rt: float, cycling_cost_per_mwh: float,
                 trade_cost_per_mwh: float = 0.0, with_reserve: bool = True):
        self.with_reserve = with_reserve
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
        trade   = trade_cost_per_mwh * cp.sum(self.p_dis + self.p_chg) * DT
        breach  = SOE_BREACH_PENALTY_GBP_PER_MWH * cp.sum(self.below + self.above)
        discharged, charged = self.p_dis, self.p_chg
        constraints = [
            # Trading power left around the response commitments, per side
            self.p_dis <= self.dis_max,
            self.p_chg <= self.chg_max,
        ]
        objective = revenue - trade - breach

        if with_reserve:
            self.falls = cp.Parameter(H, nonpos=True)    # prices where negative, else zero
            self.rises = cp.Parameter(H, nonneg=True)    # prices where positive, else zero
            self.rec_dis_max = cp.Parameter(H, nonneg=True)
            self.rec_chg_max = cp.Parameter(H, nonneg=True)
            self.r_dis = cp.Variable(H, nonneg=True)     # discharge through Reserved Capacity (MW)
            self.r_chg = cp.Variable(H, nonneg=True)     # charge through Reserved Capacity (MW)
            recovery = (self.falls @ self.r_dis - self.rises @ self.r_chg) * DT   # never positive
            reserve  = RESERVE_USE_COST_GBP_PER_MWH * cp.sum(self.r_dis + self.r_chg) * DT
            objective = objective + recovery - reserve
            discharged, charged = discharged + self.r_dis, charged + self.r_chg
            constraints += [self.r_dis <= self.rec_dis_max, self.r_chg <= self.rec_chg_max]

        wear = cycling_cost_per_mwh * cp.sum(discharged) * DT
        constraints += [
            self.soc[0] == self.soc0,
            # State equation: charge input reduced by the round-trip loss
            self.soc[1:] == self.soc[:-1] - discharged * DT + (efficiency_rt * DT) * charged,
            # Physical limits of the store
            self.soc >= 0.0,
            self.soc <= energy_mwh,
            # Response requirements, softened by the penalised shortfalls
            self.soc + self.below >= self.lo,
            self.soc - self.above <= self.hi,
        ]
        self.problem = cp.Problem(cp.Maximize(objective - wear), constraints)


@lru_cache(maxsize=32)
def _compiled(H: int, energy_mwh: float, efficiency_rt: float, cycling_cost_per_mwh: float,
              trade_cost_per_mwh: float = 0.0, with_reserve: bool = True) -> _CompiledLP:
    return _CompiledLP(H, energy_mwh, efficiency_rt, cycling_cost_per_mwh, trade_cost_per_mwh, with_reserve)


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
    trade_cost_per_mwh: float = 0.0,
    reserve_dis_mw: np.ndarray | None = None,
    reserve_chg_mw: np.ndarray | None = None,
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
    trade_cost_per_mwh : float
        Cost per MWh traded in either direction. Zero for arbitrage; a small value
        makes a price-blind plan move no more energy than its requirements need.
    reserve_dis_mw, reserve_chg_mw : np.ndarray, shape (H,), optional
        Reserved Capacity available to discharge and to charge in each period
        (MW), for state-of-energy recovery only. None means no reserve.

    Returns
    -------
    (energy_dis_mwh, energy_chg_mwh) for period 0, trading and reserve together,
    or with return_plan a dict of arrays {"p_dis", "p_chg", "r_dis", "r_chg",
    "soc", "shortfall_mwh"}. Returns (0.0, 0.0), or None with return_plan, if
    the solver fails.
    """
    charge_schedule = arb_mw_schedule if charge_mw_schedule is None else charge_mw_schedule
    H = min(horizon, len(price_forecast), len(arb_mw_schedule), len(charge_schedule))
    if H == 0:
        return None if return_plan else (0.0, 0.0)

    def reserve(schedule):
        return np.zeros(H) if schedule is None else np.clip(_per_period(schedule, H), 0.0, None)

    rec_dis, rec_chg = reserve(reserve_dis_mw), reserve(reserve_chg_mw)
    with_reserve = bool(rec_dis.any() or rec_chg.any())
    prices = np.asarray(price_forecast[:H], dtype=float)
    lp = _compiled(H, float(energy_mwh), float(efficiency_rt), float(cycling_cost_per_mwh),
                   float(trade_cost_per_mwh), with_reserve)
    lp.soc0.value    = float(np.clip(soc_current, 0.0, energy_mwh))
    lp.prices.value  = prices
    lp.dis_max.value = np.clip(np.asarray(arb_mw_schedule[:H], dtype=float), 0.0, None)
    lp.chg_max.value = np.clip(np.asarray(charge_schedule[:H], dtype=float), 0.0, None)
    lp.lo.value      = np.clip(_per_period(soc_min, H + 1), 0.0, energy_mwh)
    lp.hi.value      = np.clip(_per_period(soc_max, H + 1), 0.0, energy_mwh)
    if with_reserve:
        lp.falls.value       = np.minimum(prices, 0.0)
        lp.rises.value       = np.maximum(prices, 0.0)
        lp.rec_dis_max.value = rec_dis
        lp.rec_chg_max.value = rec_chg

    try:
        lp.problem.solve(solver=cp.CLARABEL)
    except Exception:
        return None if return_plan else (0.0, 0.0)

    if lp.problem.status not in ("optimal", "optimal_inaccurate") or lp.p_dis.value is None:
        return None if return_plan else (0.0, 0.0)

    r_dis = np.maximum(lp.r_dis.value, 0.0) if with_reserve else np.zeros(H)
    r_chg = np.maximum(lp.r_chg.value, 0.0) if with_reserve else np.zeros(H)
    if return_plan:
        return {
            "p_dis": np.maximum(lp.p_dis.value, 0.0),
            "p_chg": np.maximum(lp.p_chg.value, 0.0),
            "r_dis": r_dis,
            "r_chg": r_chg,
            "soc": lp.soc.value.copy(),
            "shortfall_mwh": lp.below.value + lp.above.value,
        }

    e_dis = (max(0.0, float(lp.p_dis.value[0])) + r_dis[0]) * DT
    e_chg = (max(0.0, float(lp.p_chg.value[0])) + r_chg[0]) * DT
    return e_dis, e_chg
