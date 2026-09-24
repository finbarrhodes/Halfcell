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

That leaves recovery untimed. Where Low holdings and the reserve for High ones
take the whole discharge rating, the reserve is the only way out for energy a
High product absorbed, and a plan that earns nothing from it sells whenever
headroom runs short, whatever the price. Passing a `recovery_allowance` adds a
second, credited pair of flows through the same reserve: they earn at the price,
but in total may move no more energy than the caller says delivery has put in
play and not yet recovered. Recovery can then wait for a better price without
the reserve becoming trading capacity. The uncredited flows stay, so the reserve
can still pre-position for a block whatever the allowance.

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
                 trade_cost_per_mwh: float = 0.0, with_reserve: bool = True, credit_recovery: bool = False):
        self.with_reserve = with_reserve
        self.credit_recovery = credit_recovery and with_reserve
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
            through_dis, through_chg = self.r_dis, self.r_chg
            if self.credit_recovery:
                # Recovery that earns at the price, within what delivery has put in play:
                # MWh taken out of store for the discharge side, restored for the charge side
                self.allow_out = cp.Parameter(nonneg=True)
                self.allow_in  = cp.Parameter(nonneg=True)
                self.c_dis = cp.Variable(H, nonneg=True)
                self.c_chg = cp.Variable(H, nonneg=True)
                credited = (self.prices @ (self.c_dis - self.c_chg)) * DT
                objective = (objective + credited
                             - RESERVE_USE_COST_GBP_PER_MWH * cp.sum(self.c_dis + self.c_chg) * DT)
                discharged, charged = discharged + self.c_dis, charged + self.c_chg
                through_dis, through_chg = through_dis + self.c_dis, through_chg + self.c_chg
                constraints += [cp.sum(self.c_dis) * DT <= self.allow_out,
                                efficiency_rt * cp.sum(self.c_chg) * DT <= self.allow_in]
            constraints += [through_dis <= self.rec_dis_max, through_chg <= self.rec_chg_max]

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


# Vintage-limited plans run from 49 to 96 periods, each length compiled once
# (about 25 ms), with and without reserve flows
@lru_cache(maxsize=256)
def _compiled(H: int, energy_mwh: float, efficiency_rt: float, cycling_cost_per_mwh: float,
              trade_cost_per_mwh: float = 0.0, with_reserve: bool = True,
              credit_recovery: bool = False) -> _CompiledLP:
    return _CompiledLP(H, energy_mwh, efficiency_rt, cycling_cost_per_mwh, trade_cost_per_mwh, with_reserve,
                       credit_recovery)


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
    recovery_allowance: tuple[float, float] | None = None,
    return_reserve: bool = False,
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
    recovery_allowance : (out_mwh, in_mwh), optional
        Credit recovery through the reserve at the price, up to out_mwh taken
        out of store and in_mwh put back over the horizon: the energy delivery
        has put in play that the caller has not yet counted as recovered. None
        leaves the reserve uncredited, as before.
    return_reserve : bool
        Also return the period-0 MWh through the reserve, as with a
        recovery_allowance, without crediting it.

    Returns
    -------
    (energy_dis_mwh, energy_chg_mwh) for period 0, trading and reserve together,
    or with return_plan a dict of arrays {"p_dis", "p_chg", "r_dis", "r_chg",
    "c_dis", "c_chg", "soc", "shortfall_mwh"}. With a recovery_allowance or
    return_reserve, also
    the period-0 MWh through the reserve on each side, credited or not, for the
    caller's account: (energy_dis, energy_chg, reserve_dis, reserve_chg).
    Returns zeros, or None with return_plan, if the solver fails.
    """
    charge_schedule = arb_mw_schedule if charge_mw_schedule is None else charge_mw_schedule
    accounting = return_reserve or recovery_allowance is not None
    failed = None if return_plan else ((0.0, 0.0, 0.0, 0.0) if accounting else (0.0, 0.0))
    H = min(horizon, len(price_forecast), len(arb_mw_schedule), len(charge_schedule))
    if H == 0:
        return failed

    def reserve(schedule):
        return np.zeros(H) if schedule is None else np.clip(_per_period(schedule, H), 0.0, None)

    rec_dis, rec_chg = reserve(reserve_dis_mw), reserve(reserve_chg_mw)
    with_reserve = bool(rec_dis.any() or rec_chg.any())
    credit = recovery_allowance is not None and with_reserve
    prices = np.asarray(price_forecast[:H], dtype=float)
    lp = _compiled(H, float(energy_mwh), float(efficiency_rt), float(cycling_cost_per_mwh),
                   float(trade_cost_per_mwh), with_reserve, credit)
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
    if credit:
        lp.allow_out.value = max(0.0, float(recovery_allowance[0]))
        lp.allow_in.value  = max(0.0, float(recovery_allowance[1]))

    try:
        lp.problem.solve(solver=cp.CLARABEL)
    except Exception:
        return failed

    if lp.problem.status not in ("optimal", "optimal_inaccurate") or lp.p_dis.value is None:
        return failed

    def flow(var):
        return np.maximum(var.value, 0.0)

    r_dis = flow(lp.r_dis) if with_reserve else np.zeros(H)
    r_chg = flow(lp.r_chg) if with_reserve else np.zeros(H)
    c_dis = flow(lp.c_dis) if credit else np.zeros(H)
    c_chg = flow(lp.c_chg) if credit else np.zeros(H)
    if return_plan:
        return {
            "p_dis": flow(lp.p_dis),
            "p_chg": flow(lp.p_chg),
            "r_dis": r_dis,
            "r_chg": r_chg,
            "c_dis": c_dis,
            "c_chg": c_chg,
            "soc": lp.soc.value.copy(),
            "shortfall_mwh": lp.below.value + lp.above.value,
        }

    reserve_dis = (r_dis[0] + c_dis[0]) * DT
    reserve_chg = (r_chg[0] + c_chg[0]) * DT
    e_dis = max(0.0, float(lp.p_dis.value[0])) * DT + reserve_dis
    e_chg = max(0.0, float(lp.p_chg.value[0])) * DT + reserve_chg
    return (e_dis, e_chg, reserve_dis, reserve_chg) if accounting else (e_dis, e_chg)
