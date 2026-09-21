"""
src/optimisation/day_ahead.py
=============================
The bid-deadline decision made the way a dispatcher would make it: which
response products to hold in each EFA block of a service day, chosen together
with a half-hourly trading plan for every period from the deadline to the end
of that day.

Why together
------------
Holding response gives up the trading that capacity would otherwise do. The
formula allocator (fr_allocation.allocate_day with revenue_stack's
_shadow_arb_value_per_mw) prices that as one number per block, from the block's
own forecast prices. But the trade a battery actually makes spans blocks - buy
overnight, sell into the evening peak - and it has one store to trade with, so
the value of keeping capacity free is not a price per block:

  - it is not additive: two evening blocks are substitutes, so keeping both
    free is worth one sale of the store rather than two; and where a holding
    carries an energy requirement, blocks can instead be complements;
  - it depends on direction and hour: overnight, trading needs the charge side,
    which High products take; at the evening peak it needs the discharge side,
    which Low products take;
  - it depends on where state of energy is when the offers are made.

Maximising FR revenue plus the value of the trading plan in one LP gets each of
these from the constraints rather than approximating them. Against realised
clearing prices it gives the holding a price-taker ends up with when it offers
each product along its marginal cost of supply - fr_allocation's argument, with
the linear cost generalised to the falling, stepped one that the plan implies.

The plan is priced on a forecast and is only as good as the forecast's shape
across the day. `price_shrink` pulls the forecast towards its mean before
planning: 1 takes it as given, 0 gives trading no value.

Leftover energy
---------------
The plan ends when the service day does. Energy still in store is valued at
the day's mean forecast price less wear, a stand-in for what it would sell for
later. With no value the plan would empty the store in the last block, and
holding Low response there would look expensive for no real reason.

Solver
------
HiGHS (dual simplex, through scipy) rather than Clarabel, which dispatch uses.
Holdings are chosen among options that often tie - two products at the same
net value, response worth exactly what trading is - and ties are broken by a
tiny cost per MW held (fr_allocation._TIE_BREAK_GBP_PER_MW, far inside an
interior-point solver's tolerance). A simplex solver returns a vertex that
respects it; an interior-point one returns the middle of the tied set, splitting
holdings fractionally and leaving dust that dispatch would count as held. HiGHS
also solves the pre-EAC variant, one service per block, as a small MIP.

Each shape (lead-in length, number of blocks, battery, reserve rule, whether one
service per block) is compiled once with cvxpy Parameters, as in mpc.py.
"""

from functools import lru_cache
from typing import Mapping, Sequence

import cvxpy as cp
import numpy as np

from src.analysis.fr_allocation import (
    EFA_HOURS,
    EFA_SETTLEMENT_PERIODS,
    SETTLEMENT_PERIOD_H,
    _SOE_SLACK_PENALTY_GBP_PER_MWH,
    _TIE_BREAK_GBP_PER_MW,
    _export_use,
    _import_use,
    _offered,
    _upper,
)
from src.analysis.neso_rules import DELIVERY_DURATION_H, FAMILIES, PRODUCTS, family, is_low

_HIGHS_LP = {"method": "highs-ds"}


class _DayAheadLP:
    """The joint allocation and trading LP for one shape, built once."""

    def __init__(self, n_lead: int, n_blocks: int, power_mw: float, energy_mwh: float,
                 efficiency_rt: float, cycling_cost_per_mwh: float, apply_reserve: bool, one_service: bool):
        T = n_lead + n_blocks * EFA_SETTLEMENT_PERIODS
        P, E, eta, dt = power_mw, energy_mwh, efficiency_rt, SETTLEMENT_PERIOD_H
        shape = (n_blocks, len(PRODUCTS))
        self.n_lead, self.one_service = n_lead, one_service

        self.value  = cp.Parameter(shape)               # £ per MW held for the block, net of costs
        self.lower  = cp.Parameter(shape, nonneg=True)  # forced holdings, for pricing; zero to decide
        self.upper  = cp.Parameter(shape, nonneg=True)  # zero where a product is not offered
        self.prices = cp.Parameter(T)
        self.soc0   = cp.Parameter(nonneg=True)
        self.terminal = cp.Parameter(nonneg=True)       # £ per MWh left in store at the end

        self.q = cp.Variable(shape, nonneg=True)        # MW held, per block and product
        self.d = cp.Variable(T, nonneg=True)            # trading discharge (MW)
        self.c = cp.Variable(T, nonneg=True)            # trading charge (MW)
        self.s = cp.Variable(T + 1)                     # state of energy at the start of each period

        export = np.array([_export_use(p, apply_reserve) for p in PRODUCTS])
        imports = np.array([_import_use(p, apply_reserve) for p in PRODUCTS])
        stored = np.array([DELIVERY_DURATION_H[family(p)] if is_low(p) else 0.0 for p in PRODUCTS])
        headroom = np.array([0.0 if is_low(p) else DELIVERY_DURATION_H[family(p)] for p in PRODUCTS])
        # Spreads a per-block quantity over the block's settlement periods
        by_period = np.kron(np.eye(n_blocks), np.ones((EFA_SETTLEMENT_PERIODS, 1)))

        held = slice(n_lead, T)
        constraints = [
            self.q >= self.lower,
            self.q <= self.upper,
            self.s[0] == self.soc0,
            self.s[1:] == self.s[:-1] - dt * self.d + (eta * dt) * self.c,
            self.s >= 0.0,
            self.s <= E,
            # Trading uses only the power the block's holdings leave on each side
            self.d[held] + by_period @ (self.q @ export) <= P,
            self.c[held] + by_period @ (self.q @ imports) <= P,
            # Service Terms 6.11: Low energy in store, High energy as headroom, at
            # the start of every settlement period of the block
            self.s[held] >= by_period @ (self.q @ stored),
            self.s[held] <= E - by_period @ (self.q @ headroom),
        ]

        breach = 0
        if n_lead:
            # Commitments already held from earlier offers. Soft, because the unit
            # may already be out of position; see fr_allocation.allocate_day.
            self.lead_lo, self.lead_hi = cp.Parameter(n_lead, nonneg=True), cp.Parameter(n_lead, nonneg=True)
            self.lead_dis, self.lead_chg = cp.Parameter(n_lead, nonneg=True), cp.Parameter(n_lead, nonneg=True)
            below, above = cp.Variable(n_lead, nonneg=True), cp.Variable(n_lead, nonneg=True)
            constraints += [
                self.d[:n_lead] <= self.lead_dis,
                self.c[:n_lead] <= self.lead_chg,
                self.s[:n_lead] + below >= self.lead_lo,
                self.s[:n_lead] - above <= self.lead_hi,
            ]
            breach = _SOE_SLACK_PENALTY_GBP_PER_MWH * cp.sum(below + above)

        if one_service:
            # Before EAC a unit could offer only one of DC, DM or DR into a block
            member = np.array([[1.0 if family(p) == f else 0.0 for f in FAMILIES] for p in PRODUCTS])
            self.z = cp.Variable((n_blocks, len(FAMILIES)), boolean=True)
            constraints += [self.q <= P * (self.z @ member.T), cp.sum(self.z, axis=1) <= 1]

        self.fr = cp.sum(cp.multiply(self.value, self.q))
        self.trading = (self.prices @ (self.d - self.c) * dt
                        - cycling_cost_per_mwh * cp.sum(self.d) * dt
                        + self.terminal * self.s[T])
        self.problem = cp.Problem(cp.Maximize(self.fr + self.trading - breach), constraints)

    def solve(self) -> bool:
        options = {} if self.one_service else {"scipy_options": _HIGHS_LP}
        try:
            self.problem.solve(solver=cp.SCIPY, **options)
        except cp.error.SolverError:
            return False
        return self.problem.status == "optimal" and self.q.value is not None


@lru_cache(maxsize=32)
def _compiled(n_lead: int, n_blocks: int, power_mw: float, energy_mwh: float, efficiency_rt: float,
              cycling_cost_per_mwh: float, apply_reserve: bool, one_service: bool) -> _DayAheadLP:
    return _DayAheadLP(n_lead, n_blocks, power_mw, energy_mwh, efficiency_rt, cycling_cost_per_mwh,
                       apply_reserve, one_service)


def planning_prices(prices: Sequence[float], n_service: int, cycling_cost_per_mwh: float,
                    price_shrink=1.0) -> tuple[np.ndarray, float]:
    """
    The price path the plan trades against, and the value of energy left at its end.

    Missing periods take the mean of the rest, so a gap neither invents nor
    removes a spread; with no prices at all, trading has no value. The path is
    then pulled towards its mean by `price_shrink`, a constant or one weight per
    period. Leftover energy is worth the mean over the last `n_service` periods
    (the service day) less wear, and never less than nothing.
    """
    path = np.asarray(prices, dtype=float)
    known = np.isfinite(path)
    if not known.any():
        return np.zeros(len(path)), 0.0
    mean = float(path[known].mean())
    path = np.where(known, path, mean)
    path = mean + np.asarray(price_shrink, dtype=float) * (path - mean)
    terminal = max(0.0, float(path[-n_service:].mean()) - cycling_cost_per_mwh)
    return path, terminal


def plan_day(
    blocks: Sequence[Mapping],
    power_mw: float,
    energy_mwh: float,
    efficiency_rt: float,
    cycling_cost_per_mwh: float,
    soc_now_mwh: float,
    prices: Sequence[float],
    *,
    apply_reserve: bool,
    lead_in: Sequence[Mapping] = (),
    one_service: bool = False,
    price_shrink: float = 1.0,
    terminal_value_per_mwh: float | None = None,
    fixed: Sequence[Mapping[str, float]] | None = None,
) -> dict:
    """
    The holdings that earn most across a service day, valued together with the
    trading the capacity they leave free can do.

    Parameters
    ----------
    blocks : sequence of {"prices": {product: £/MW/h}, "families": iterable}
        The EFA blocks being bid, in delivery order, each optionally with "caps"
        and "costs" as in fr_allocation.allocate_block. Any "arb_value" is
        ignored: the plan replaces it.
    power_mw, energy_mwh, efficiency_rt, cycling_cost_per_mwh : float
        The battery. Efficiency applies on charge, as in dispatch.
    soc_now_mwh : float
        State of energy at bid time.
    prices : sequence of £/MWh
        Forecast price of every settlement period from bid time to the end of
        the last block: the lead-in periods, then eight per block. NaN is allowed.
    apply_reserve : bool
        Whether the Reserved Capacity rule binds.
    lead_in : sequence of {"lo", "hi", "dis", "chg", "n_sp"}
        Commitments already held between bid time and the first block, as in
        fr_allocation.allocate_day.
    one_service : bool
        Offer at most one of DC, DM or DR per block (before EAC), chosen jointly.
    price_shrink : float or sequence
        Weight on the forecast's deviations from its mean, one value or one per
        period; see planning_prices.
    terminal_value_per_mwh : float, optional
        Value of energy left at the end. Default: the service day's mean
        forecast price less wear.
    fixed : sequence of {product: MW} per block, optional
        Hold exactly these quantities instead of choosing, earning no FR revenue.
        For pricing a holding: the plan's value with and without it is what the
        holding costs in trading.

    Returns
    -------
    {"blocks": [{"q", "arb_mw", "fr_revenue_gbp"}] per block,
     "soc_plan_mwh": [(SoE at the block's first period, at its last)],
     "soc_path_mwh": SoE at the start of every period, and at the end,
     "discharge_mw", "charge_mw": the trading plan per period,
     "trading_value_gbp": the plan's trading profit including leftover energy,
     "families": the services held in each block, as a sorted tuple,
     "solved": whether the solver succeeded}
    Holding nothing and a flat SoE, with solved False, if the solver fails - which
    with fixed holdings means they cannot be held from this position.
    """
    P, E = float(power_mw), float(energy_mwh)
    n_blocks = len(blocks)
    n_lead = int(sum(int(seg["n_sp"]) for seg in lead_in))
    T = n_lead + n_blocks * EFA_SETTLEMENT_PERIODS
    if len(prices) != T:
        raise ValueError(f"expected {T} prices ({n_lead} lead-in + {n_blocks} blocks), got {len(prices)}")
    soc_now = min(max(float(soc_now_mwh), 0.0), E)

    path, terminal = planning_prices(prices, n_blocks * EFA_SETTLEMENT_PERIODS, cycling_cost_per_mwh, price_shrink)
    if terminal_value_per_mwh is not None:
        terminal = max(0.0, float(terminal_value_per_mwh))

    value = np.zeros((n_blocks, len(PRODUCTS)))
    upper = np.zeros_like(value)
    lower = np.zeros_like(value)
    for b, block in enumerate(blocks):
        caps, costs = block.get("caps") or {}, block.get("costs") or {}
        for p in _offered(block["prices"], block["families"]):
            j = PRODUCTS.index(p)
            value[b, j] = block["prices"][p] * EFA_HOURS - costs.get(p, 0.0) - _TIE_BREAK_GBP_PER_MW
            upper[b, j] = _upper(p, P, caps)
    if fixed is not None:
        value[:] = 0.0
        for b, holding in enumerate(fixed):
            for p, mw in holding.items():
                upper[b, PRODUCTS.index(p)] = lower[b, PRODUCTS.index(p)] = max(0.0, float(mw))

    lp = _compiled(n_lead, n_blocks, P, E, float(efficiency_rt), float(cycling_cost_per_mwh),
                   bool(apply_reserve), bool(one_service))
    lp.value.value, lp.lower.value, lp.upper.value = value, lower, upper
    lp.prices.value, lp.soc0.value, lp.terminal.value = path, soc_now, terminal
    if n_lead:
        def per_period(key, floor=0.0):
            return np.concatenate([np.full(int(seg["n_sp"]), max(floor, float(seg[key]))) for seg in lead_in])
        lp.lead_lo.value, lp.lead_hi.value = per_period("lo"), per_period("hi")
        lp.lead_dis.value, lp.lead_chg.value = per_period("dis"), per_period("chg")

    if not lp.solve():
        return {
            "blocks": [{"q": {}, "arb_mw": 0.0, "fr_revenue_gbp": 0.0} for _ in blocks],
            "soc_plan_mwh": [(soc_now, soc_now) for _ in blocks],
            "soc_path_mwh": np.full(T + 1, soc_now),
            "discharge_mw": np.zeros(T),
            "charge_mw": np.zeros(T),
            "trading_value_gbp": 0.0,
            "families": [()] * n_blocks,
            "solved": False,
        }

    q = np.maximum(lp.q.value, 0.0)
    soc = lp.s.value.copy()
    allocations, families, plan = [], [], []
    for b, block in enumerate(blocks):
        held = {p: float(q[b, j]) for j, p in enumerate(PRODUCTS) if q[b, j] > 1e-9}
        export = sum(_export_use(p, apply_reserve) * mw for p, mw in held.items())
        imports = sum(_import_use(p, apply_reserve) * mw for p, mw in held.items())
        allocations.append({
            "q": held,
            # The power left for trading on the more committed side
            "arb_mw": max(0.0, P - max(export, imports)),
            "fr_revenue_gbp": 0.0 if fixed is not None else
                sum(block["prices"][p] * EFA_HOURS * mw for p, mw in held.items()),
        })
        families.append(tuple(sorted({family(p) for p in held})))
        first = n_lead + b * EFA_SETTLEMENT_PERIODS
        plan.append((float(soc[first]), float(soc[first + EFA_SETTLEMENT_PERIODS - 1])))
    return {
        "blocks": allocations,
        "soc_plan_mwh": plan,
        "soc_path_mwh": soc,
        "discharge_mw": np.maximum(lp.d.value, 0.0),
        "charge_mw": np.maximum(lp.c.value, 0.0),
        "trading_value_gbp": float(lp.trading.value),
        "families": families,
        "solved": True,
    }
