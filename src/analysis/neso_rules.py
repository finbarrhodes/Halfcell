"""
src/analysis/neso_rules.py
==========================
The NESO rules that bound how a battery can take part in the Dynamic Response
Services (DC, DM, DR), expressed as constants and feasibility checks.

Every number here is taken from a NESO or Ofgem document and cited inline, so
the model can be audited against its sources rather than against itself. The
revenue model imports these; nothing in this module chooses a strategy.

Sources
-------
[ST]  Response Service Terms, effective 31 July 2026.
      https://www.neso.energy/document/384606/download
[PR]  Response Services Procurement Rules v5, effective 31 March 2026.
      https://www.neso.energy/document/378246/download
[SOE] SOE Monitoring Guidance for Energy Limited DC/DM/DR, V2, March 2025.
      https://www.neso.energy/document/347241/download
[EAC] Enduring Auction Capability Detailed Market Design, 17 February 2023.
      https://www.neso.energy/document/276866/download
[R3]  Dynamic Response Services August 2024 Submission, which moved the
      Reserved Capacity percentages from guidance into the Procurement Rules.
      Ofgem's decision of 30 October 2024 approved it apart from submissions
      5 and 6. https://www.neso.energy/document/323326/download
[PR2] Response Services Procurement Rules v2.0, effective 15 November 2024,
      the first version carrying the Reserved Capacity rule.
      https://www.neso.energy/document/347456/download
[DRA] Dynamic Regulation Auction Rules v1.0, effective 8 April 2022 (pre-EAC).
      https://www.neso.energy/document/246746/download
[DMA] Dynamic Moderation Auction Rules v1.0, effective 6 May 2022 (pre-EAC).
      https://www.neso.energy/document/246721/download
[DCG] Dynamic Containment Glossary of Terms and Rules of Interpretation v4.0,
      effective 7 October 2021 (pre-EAC).
      https://www.neso.energy/document/177116/download
"""

from __future__ import annotations

from datetime import date
from typing import Mapping

PRODUCTS = ("DCH", "DCL", "DMH", "DML", "DRH", "DRL")
FAMILIES = ("DC", "DM", "DR")

# Delivery Duration: how long a contracted quantity must be sustainable.
# [ST] Schedule 2 (Tsus) sets 15, 30 and 60 minutes, and [SOE] §2 states these
# values are unchanged since the services launched, so they hold across the
# whole backtest.
DELIVERY_DURATION_H = {"DC": 0.25, "DM": 0.5, "DR": 1.0}

# A unit must be able to recover 20% of its Contracted Response Energy Volume
# within one settlement period. [ST] "Energy Recovery Adjustment Volume";
# [SOE] Table 2.
ENERGY_RECOVERY_SHARE = 0.20

# Reserved Capacity: the share of each offered quantity that must be kept free
# in the *opposite* direction, so the unit can recover energy after activation.
# [PR] Schedule 1 "Reserved Capacity". [SOE] §3 derives each figure as
# duration_h x ENERGY_RECOVERY_SHARE x 2, since recovering an energy volume
# inside a 30-minute settlement period takes twice that volume in MW.
# tests/test_neso_rules.py checks the published figures against the derivation.
RESERVED_CAPACITY_SHARE = {"DC": 0.10, "DM": 0.20, "DR": 0.40}

# Largest Offered Quantity allowed per product in a Sell Order.
# [PR] Schedule 1 "Maximum Sell Size": 100 MW unless NESO notifies otherwise.
MAX_SELL_SIZE_MW = 100.0

# First Response auction run on the Enduring Auction Capability, per NESO's
# published EAC results. Before it a unit could offer only one of DC, DM or DR
# in a window: [DRA] 7.3.1 bars a DR sell order for any EFA block already
# carrying a sell order for another service, while 7.3.2 allows both DR High
# and DR Low in the same block ([EAC] slide 7 describes the same limit). From
# it, one unit may split its capacity across all three ([EAC] "Splitting
# (Revenue Stacking)"; [ST] Schedule 2 gives the stacked delivery curve).
EAC_GO_LIVE = date(2023, 11, 2)

# The Reserved Capacity rule became binding in the Procurement Rules from this
# date ([PR2] cover; Ofgem decision 30 October 2024). Before it the percentages
# existed only in NESO guidance ([R3] §4). The revenue model holds the reserve
# from EAC go-live regardless; see revenue_stack._Scheduler.allocate.
RESERVE_RULE_EFFECTIVE = date(2024, 11, 15)

# Sell orders for a service day close, and results publish, on the day before.
# Clearing prices for day D are therefore unknown when day D's bids are made;
# the latest a bidder can know are day D-1's, published the afternoon before
# that. Pre-EAC times are from [DCG]; EAC times from [PR] Schedule 1.
PRE_EAC_BID_CLOSE, PRE_EAC_RESULTS = (14, 30), (15, 0)
EAC_BID_CLOSE, EAC_RESULTS = (14, 0), (16, 0)


def splitting_allowed(service_day: date) -> bool:
    """Whether a unit may hold more than one of DC, DM and DR in one block."""
    return service_day >= EAC_GO_LIVE


def reserve_rule_applies(service_day: date) -> bool:
    """Whether the Reserved Capacity rule binds on this service day."""
    return service_day >= RESERVE_RULE_EFFECTIVE


def family(product: str) -> str:
    """DC, DM or DR."""
    return product[:2]


def is_low(product: str) -> bool:
    """Low products answer low frequency by exporting; High products import."""
    return product[2] == "L"


def response_energy_mwh(q: Mapping[str, float]) -> tuple[float, float]:
    """
    Contracted Response Energy Volume per direction, as (low, high) in MWh.

    REV is each contracted quantity times its Delivery Duration, summed across
    every product in that direction when services are stacked ([ST] definition;
    [SOE] Table 2 note). The two directions are assessed separately, so unequal
    quantities give an asymmetric requirement ([ST] 6.11 iv).
    """
    low = sum(mw * DELIVERY_DURATION_H[family(p)] for p, mw in q.items() if is_low(p))
    high = sum(mw * DELIVERY_DURATION_H[family(p)] for p, mw in q.items() if not is_low(p))
    return low, high


def capacity_use_mw(q: Mapping[str, float], apply_reserve: bool = True) -> tuple[float, float]:
    """
    Power committed on each side of the unit, as (export, import) in MW.

    The export side carries every Low offer plus the reserve that High offers
    need in order to recover by discharging; the import side mirrors it.
    [PR] 8.3.3.2 caps these against the unit's registered capacity.

    The cap is applied per direction. That is consistent with NESO's worked
    example in [SOE] §3 Figure 2 (40 MW of DM each way on a 50 MW unit, with a
    20% reserve for each), and it is the only reading under which a unit is
    never committed beyond its rating on one side.
    """
    def reserve(p: str) -> float:
        return RESERVED_CAPACITY_SHARE[family(p)] if apply_reserve else 0.0

    export = sum(mw for p, mw in q.items() if is_low(p)) + sum(
        mw * reserve(p) for p, mw in q.items() if not is_low(p)
    )
    imp = sum(mw for p, mw in q.items() if not is_low(p)) + sum(
        mw * reserve(p) for p, mw in q.items() if is_low(p)
    )
    return export, imp


def is_feasible(
    q: Mapping[str, float],
    power_mw: float,
    energy_mwh: float,
    apply_reserve: bool = True,
    single_family: bool = False,
    tol: float = 1e-6,
) -> bool:
    """
    Whether a set of contracted quantities is permitted for this unit.

    single_family enforces the pre-EAC limit of one service per window.
    """
    if any(mw < -tol or mw > MAX_SELL_SIZE_MW + tol for mw in q.values()):
        return False
    if single_family and len({family(p) for p, mw in q.items() if mw > tol}) > 1:
        return False
    export, imp = capacity_use_mw(q, apply_reserve)
    if export > power_mw + tol or imp > power_mw + tol:
        return False
    # Some starting state of energy must hold the Low requirement in store and
    # the High requirement as headroom at the same time ([ST] 6.11 i-ii).
    low, high = response_energy_mwh(q)
    return low + high <= energy_mwh + tol


def soc_bounds_mwh(q: Mapping[str, float], energy_mwh: float) -> tuple[float, float]:
    """
    State-of-energy range that keeps every contract deliverable, as (min, max).

    The Minimum State of Energy Requirement equals REV in a block's first
    settlement period and falls only as delivery is activated ([ST] 6.11 ii).
    Activations are not simulated, so the requirement is held at REV for the
    whole block.
    """
    low, high = response_energy_mwh(q)
    return low, energy_mwh - high


def arbitrage_power_limits_mw(
    q: Mapping[str, float], power_mw: float, apply_reserve: bool = True
) -> tuple[float, float]:
    """
    Power left for trading around the response commitments, as (discharge, charge).

    The unit must operate within a range that still allows full delivery
    ([ST] 6.11 i). Reserved Capacity is kept out of trading as well: it exists
    so the unit can recover energy after an activation, which a baseline already
    using it could not do. That is the conservative reading.
    """
    export, imp = capacity_use_mw(q, apply_reserve)
    return max(0.0, power_mw - export), max(0.0, power_mw - imp)


# Energy Recovery [ST 6.11 ii-iii]. Delivering response lowers a contract's Minimum
# State of Energy Requirement; the Energy Recovery Adjustment Volume then raises it
# again. The Service Terms' worked example sets the adjustment for the fifth
# settlement period from the shortfall at the start of the second, and applies it
# to the sixth: a lag of three periods between assessment and adjustment.
ENERGY_RECOVERY_LAG_PERIODS = 3


def energy_recovery_adjustment(k: int, block_start: int, rev, requirement, adjustments) -> float:
    """
    Energy Recovery Adjustment Volume for settlement period k (MWh), which raises
    the requirement for period k + 1 [ST 6.11 ii-iii].

    It is the shortfall below the Contracted Response Energy Volume (`rev`) at the
    start of period k - 3, less the adjustments made since then, and no more than
    ENERGY_RECOVERY_SHARE of that volume. Sequences are indexed by settlement
    period; nothing is carried across from before `block_start`, the first period
    of the Contracted Service Period.
    """
    assessed = k - ENERGY_RECOVERY_LAG_PERIODS
    if assessed < block_start:
        return 0.0
    shortfall = rev[k] - requirement[assessed] - sum(adjustments[assessed:k])
    return min(ENERGY_RECOVERY_SHARE * rev[k], max(0.0, shortfall))


def minimum_soe_requirement(rev: float, delivered) -> list[float]:
    """
    Minimum State of Energy Requirement at the start of each settlement period of
    one Contracted Service Period, in one direction [ST 6.11 ii].

    It starts at the Contracted Response Energy Volume, falls by the energy the
    direction's products delivered in the previous period, and rises by that
    period's Energy Recovery Adjustment Volume, never above the starting volume.
    At or below zero the unit is allowed to be unavailable [SOE].
    """
    revs = [rev] * len(delivered)
    requirement, adjustments = [rev], []
    for j in range(1, len(delivered)):
        k = j - 1
        adjustments.append(energy_recovery_adjustment(k, 0, revs, requirement, adjustments))
        requirement.append(min(rev, requirement[k] - delivered[k] + adjustments[k]))
    return requirement
