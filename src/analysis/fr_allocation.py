"""
src/analysis/fr_allocation.py
=============================
Allocation of a battery between the six Dynamic Response products and
wholesale arbitrage, within NESO's participation rules: for one EFA block, and
for a service day's six blocks together.

How this mirrors the auction
----------------------------
NESO accepts a sell order when the clearing price covers its offer price: an
order's surplus, (clearing price - offer price) x accepted MW summed over its
products, must be non-negative (Procurement Rules 9.2.9-9.2.10). A battery
that offers each product at its opportunity cost - the arbitrage value it
would give up - therefore ends up holding whichever permitted combination
maximises clearing-price revenue plus the arbitrage value it keeps. Since EAC,
the auction chooses among a unit's mutually exclusive baskets on exactly that
basis (EAC Detailed Market Design, co-optimisation). So solving this LP
against realised clearing prices reproduces the outcome of bidding at cost; it
is not hindsight about prices.

Why whole days
--------------
Holding a product fixes the range state of energy must stay inside for the
block (Service Terms 6.11), and uses power that would otherwise be free to move
it. Some holdings use every MW on both sides, so state of energy cannot move at
all while they run. Whether a block's holding is deliverable therefore depends
on where state of energy can be when it starts, which depends on the blocks
before it. Providers bid a service day's six blocks together at the day-ahead
deadline, knowing their state of energy and the commitments they already hold,
and should offer only what they could deliver if accepted. allocate_day solves
the six blocks jointly from that position.

Offers can carry two more inputs from the caller: a cap on the MW each product
may be held at (the auction's size), and a cost per MW held on top of the
arbitrage forgone (the energy the product is expected to deliver).

What is not modelled: the cap on baskets per unit per day, integer MW
quantities, and any effect of the battery's own bids on clearing prices beyond
a cap on its share of each auction (the battery is otherwise a price-taker).

Which services a unit may offer, and whether the Reserved Capacity rule
applies, depend on the date. Callers decide those per block from
src/analysis/neso_rules.py; this module is date-agnostic.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog

from src.analysis.neso_rules import (
    DELIVERY_DURATION_H,
    FAMILIES,
    MAX_SELL_SIZE_MW,
    PRODUCTS,
    RESERVED_CAPACITY_SHARE,
    family,
    is_low,
)

EFA_HOURS = 4
SETTLEMENT_PERIOD_H = 0.5
EFA_SETTLEMENT_PERIODS = 8

# Holding response capacity at a zero clearing price earns nothing but still
# binds state of charge, so ties are broken towards committing less. Small
# enough never to outweigh a real price difference of a penny per MW-block.
_TIE_BREAK_GBP_PER_MW = 1e-6

# £ per MWh by which the plan misses a range the unit is already committed to.
# Only needed when the unit is out of position at bid time; far above anything
# a block of response could earn, so a new holding never buys its way in.
_SOE_SLACK_PENALTY_GBP_PER_MWH = 1e5


def _offered(clearing_prices: Mapping[str, float], families: Iterable[str]) -> list[str]:
    allowed = set(families)
    return [
        p for p in PRODUCTS
        if family(p) in allowed
        and p in clearing_prices
        and clearing_prices[p] is not None
        and not math.isnan(clearing_prices[p])
    ]


def _reserve(p: str, apply_reserve: bool) -> float:
    return RESERVED_CAPACITY_SHARE[family(p)] if apply_reserve else 0.0


def _export_use(p: str, apply_reserve: bool) -> float:
    """MW of the discharge side one MW of this product uses (Procurement Rules 8.3.3.2)."""
    return 1.0 if is_low(p) else _reserve(p, apply_reserve)


def _import_use(p: str, apply_reserve: bool) -> float:
    """MW of the charge side one MW of this product uses."""
    return _reserve(p, apply_reserve) if is_low(p) else 1.0


def _upper(p: str, power_mw: float, caps: Mapping[str, float]) -> float:
    """Most MW a product can be held at: the rating, the Maximum Sell Size and any cap."""
    return max(0.0, min(MAX_SELL_SIZE_MW, power_mw, caps.get(p, math.inf)))


def allocate_block(
    clearing_prices: Mapping[str, float],
    arb_value_per_mw: float,
    power_mw: float,
    energy_mwh: float,
    duration_h: float,
    *,
    apply_reserve: bool,
    families: Iterable[str] = FAMILIES,
    caps: Mapping[str, float] | None = None,
    offer_costs: Mapping[str, float] | None = None,
) -> dict:
    """
    The permitted holding that earns most in one EFA block.

    Parameters
    ----------
    clearing_prices : {product: £/MW/h}
        Missing or NaN products are not offered.
    arb_value_per_mw : float
        £ that one MW kept back for trading is expected to earn in this block.
        Negative values are treated as zero: holding capacity back costs nothing.
    power_mw, energy_mwh, duration_h : float
        The battery.
    apply_reserve : bool
        Whether the Reserved Capacity rule binds in this block.
    families : iterable of "DC", "DM", "DR"
        Services the unit may offer into this block - all three since EAC,
        a single one before it.
    caps : {product: MW}, optional
        Most MW each product may be held at, on top of the rating and Maximum Sell Size.
    offer_costs : {product: £ per MW held for the block}, optional
        Costs of holding a product beyond the arbitrage forgone. Negative is a benefit.

    Returns
    -------
    {"q": {product: MW}, "arb_mw": float, "fr_revenue_gbp": float}
        arb_mw is the capacity the allocation values for trading. Dispatch uses
        the physical limits from neso_rules.arbitrage_power_limits_mw instead,
        which can be larger on one side.
    """
    offered = _offered(clearing_prices, families)
    arb_value = max(0.0, float(arb_value_per_mw))
    caps, offer_costs = caps or {}, offer_costs or {}

    # Decision vector: [q_p for p in offered] + [arb_mw]. linprog minimises.
    cost = ([-(clearing_prices[p] * EFA_HOURS) + offer_costs.get(p, 0.0) + _TIE_BREAK_GBP_PER_MW for p in offered]
            + [-arb_value])

    export_row = [_export_use(p, apply_reserve) for p in offered] + [1.0]
    import_row = [_import_use(p, apply_reserve) for p in offered] + [1.0]
    # Response energy in both directions plus the swing arbitrage needs must fit
    # in the store together (Service Terms 6.11; arbitrage one full cycle).
    energy_row = [DELIVERY_DURATION_H[family(p)] for p in offered] + [duration_h]

    result = linprog(
        cost,
        A_ub=[export_row, import_row, energy_row],
        b_ub=[power_mw, power_mw, energy_mwh],
        bounds=[(0.0, _upper(p, power_mw, caps)) for p in offered] + [(0.0, power_mw)],
        method="highs",
    )
    if not result.success:
        return {"q": {}, "arb_mw": 0.0, "fr_revenue_gbp": 0.0}

    x = [max(0.0, v) for v in result.x]
    q = {p: x[i] for i, p in enumerate(offered) if x[i] > 1e-9}
    return {
        "q": q,
        "arb_mw": x[-1],
        "fr_revenue_gbp": sum(clearing_prices[p] * EFA_HOURS * mw for p, mw in q.items()),
    }


def allocate_day(
    blocks: Sequence[Mapping],
    power_mw: float,
    energy_mwh: float,
    duration_h: float,
    efficiency_rt: float,
    soc_now_mwh: float,
    *,
    apply_reserve: bool,
    lead_in: Sequence[Mapping] = (),
) -> dict:
    """
    The permitted holdings across consecutive EFA blocks that earn most while
    leaving state of energy a deliverable path through all of them.

    Parameters
    ----------
    blocks : sequence of {"prices": {product: £/MW/h}, "arb_value": £/MW, "families": iterable}
        The blocks being bid, in delivery order, each optionally with "caps" and
        "costs" as in allocate_block's caps and offer_costs.
    power_mw, energy_mwh, duration_h, efficiency_rt : float
        The battery. Efficiency applies on charge, as in dispatch.
    soc_now_mwh : float
        State of energy at bid time.
    apply_reserve : bool
        Whether the Reserved Capacity rule binds.
    lead_in : sequence of {"lo", "hi", "dis", "chg", "n_sp"}
        Commitments already held between bid time and the first block, in
        order: the required state-of-energy range (MWh), the discharge and
        charge power they leave free (MW), and their length in settlement
        periods. The first starts at bid time. Empty if bidding at the first
        block's start.

    Returns
    -------
    {"blocks": [allocate_block-style result per block],
     "soc_plan_mwh": [(SoE at the block's first period, SoE at its last period)]}

    State of energy is tracked at the start of each segment's first and last
    settlement period. Within a segment the range and free power are constant,
    so a straight path between those points stays in range, and one more
    half-hour at the same power reaches the next segment's first point.
    """
    P, E, eta, dt = float(power_mw), float(energy_mwh), float(efficiency_rt), SETTLEMENT_PERIOD_H
    cost, bounds, ub_rows, ub_rhs = [], [], [], []

    def var(c: float, lo: float, hi: float | None) -> int:
        cost.append(c)
        bounds.append((lo, hi))
        return len(cost) - 1

    def at_most(coefs: dict, rhs: float) -> None:
        ub_rows.append(coefs)
        ub_rhs.append(rhs)

    def scaled(terms: dict, k: float) -> dict:
        return {i: k * c for i, c in terms.items()}

    # Every limit is a constant plus linear terms in the holdings:
    # range lo = c + terms, hi = c - terms; free power dis/chg = c - terms.
    segments = []
    for seg in lead_in:
        segments.append({
            "first": var(0.0, 0.0, E), "last": var(0.0, 0.0, E), "n_sp": int(seg["n_sp"]),
            "lo": (float(seg["lo"]), {}), "hi": (float(seg["hi"]), {}),
            "dis": (max(0.0, float(seg["dis"])), {}), "chg": (max(0.0, float(seg["chg"])), {}),
            "soft": True,
        })

    held = []
    for block in blocks:
        prices = block["prices"]
        caps, costs = block.get("caps") or {}, block.get("costs") or {}
        offered = _offered(prices, block["families"])
        q_idx = [var(-(prices[p] * EFA_HOURS) + costs.get(p, 0.0) + _TIE_BREAK_GBP_PER_MW, 0.0, _upper(p, P, caps))
                 for p in offered]
        arb_idx = var(-max(0.0, float(block["arb_value"])), 0.0, P)
        export_terms = {i: _export_use(p, apply_reserve) for i, p in zip(q_idx, offered)}
        import_terms = {i: _import_use(p, apply_reserve) for i, p in zip(q_idx, offered)}
        response_energy = {i: DELIVERY_DURATION_H[family(p)] for i, p in zip(q_idx, offered)}

        at_most({**export_terms, arb_idx: 1.0}, P)
        at_most({**import_terms, arb_idx: 1.0}, P)
        at_most({**response_energy, arb_idx: duration_h}, E)

        segments.append({
            "first": var(0.0, 0.0, E), "last": var(0.0, 0.0, E), "n_sp": EFA_SETTLEMENT_PERIODS,
            "lo": (0.0, {i: v for (i, v), p in zip(response_energy.items(), offered) if is_low(p)}),
            "hi": (E, {i: v for (i, v), p in zip(response_energy.items(), offered) if not is_low(p)}),
            "dis": (P, export_terms), "chg": (P, import_terms),
            "soft": False,
        })
        held.append((offered, q_idx, arb_idx, prices))

    if not segments:
        return {"blocks": [], "soc_plan_mwh": []}

    for n, seg in enumerate(segments):
        lo_c, lo_terms = seg["lo"]
        hi_c, hi_terms = seg["hi"]
        for point in (seg["first"], seg["last"]):
            below = {var(_SOE_SLACK_PENALTY_GBP_PER_MWH, 0.0, None): -1.0} if seg["soft"] else {}
            above = {var(_SOE_SLACK_PENALTY_GBP_PER_MWH, 0.0, None): -1.0} if seg["soft"] else {}
            at_most({point: -1.0, **lo_terms, **below}, -lo_c)    # lo <= SoE
            at_most({point: 1.0, **hi_terms, **above}, hi_c)      # SoE <= hi

        dis_c, dis_terms = seg["dis"]
        chg_c, chg_terms = seg["chg"]
        moves = [(seg["first"], seg["last"], (seg["n_sp"] - 1) * dt)]
        if n + 1 < len(segments):
            moves.append((seg["last"], segments[n + 1]["first"], dt))
        for start, end, hours in moves:
            at_most({end: 1.0, start: -1.0, **scaled(chg_terms, hours * eta)}, hours * eta * chg_c)
            at_most({start: 1.0, end: -1.0, **scaled(dis_terms, hours)}, hours * dis_c)

    A_ub = np.zeros((len(ub_rows), len(cost)))
    for r, coefs in enumerate(ub_rows):
        for i, c in coefs.items():
            A_ub[r, i] += c
    A_eq = np.zeros((1, len(cost)))
    A_eq[0, segments[0]["first"]] = 1.0
    soc_now = min(max(float(soc_now_mwh), 0.0), E)

    result = linprog(cost, A_ub=A_ub, b_ub=ub_rhs, A_eq=A_eq, b_eq=[soc_now],
                     bounds=bounds, method="highs")
    if not result.success:
        return {
            "blocks": [{"q": {}, "arb_mw": 0.0, "fr_revenue_gbp": 0.0} for _ in blocks],
            "soc_plan_mwh": [(soc_now, soc_now) for _ in blocks],
        }

    x = result.x
    allocations = []
    for offered, q_idx, arb_idx, prices in held:
        q = {p: float(x[i]) for p, i in zip(offered, q_idx) if x[i] > 1e-9}
        allocations.append({
            "q": q,
            "arb_mw": float(max(0.0, x[arb_idx])),
            "fr_revenue_gbp": sum(prices[p] * EFA_HOURS * mw for p, mw in q.items()),
        })
    plan = [(float(x[s["first"]]), float(x[s["last"]])) for s in segments[len(lead_in):]]
    return {"blocks": allocations, "soc_plan_mwh": plan}


def choose_family(
    reference_prices: Mapping[str, float],
    arb_value_per_mw: float,
    power_mw: float,
    energy_mwh: float,
    duration_h: float,
    *,
    apply_reserve: bool,
    caps: Mapping[str, float] | None = None,
    offer_costs: Mapping[str, float] | None = None,
) -> str | None:
    """
    The single service a pre-EAC unit would offer into, judged on reference prices.

    Before EAC a unit could offer only one service per EFA block, and had to
    choose before the auction cleared. Pass prices the operator could actually
    have known at bid time; the choice is then scored against realised prices
    by allocating with families=[chosen].

    Returns None when no service beats keeping the capacity for trading.
    """
    best, best_value = None, 0.0
    for fam in FAMILIES:
        alloc = allocate_block(
            reference_prices, 0.0, power_mw, energy_mwh, duration_h,
            apply_reserve=apply_reserve, families=[fam], caps=caps, offer_costs=offer_costs,
        )
        # Compare like for like: FR revenue from this family against the
        # arbitrage the same capacity would otherwise earn.
        used = max(
            sum(mw for p, mw in alloc["q"].items() if is_low(p)),
            sum(mw for p, mw in alloc["q"].items() if not is_low(p)),
        )
        costs = sum((offer_costs or {}).get(p, 0.0) * mw for p, mw in alloc["q"].items())
        value = alloc["fr_revenue_gbp"] - costs - max(0.0, arb_value_per_mw) * used
        if value > best_value:
            best, best_value = fam, value
    return best
