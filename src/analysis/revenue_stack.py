"""
BESS Revenue Stack Backtester
==============================
Models the revenue a grid-scale battery could have earned from:

  1. Frequency response availability payments (DC, DM and DR, High and Low).
     Revenue = clearing price (£/MW/h) x contracted MW x 4 hours per EFA block,
     for each product the battery holds.

  2. Wholesale energy arbitrage, dispatched by a rolling MPC linear programme
     against APXMIDP prices around those commitments.

How much capacity goes to each product is decided within NESO's participation
rules (src/analysis/neso_rules.py) by the allocator in
src/analysis/fr_allocation.py:

  - offers in one direction, plus the Reserved Capacity that opposite-direction
    offers need, must fit the battery's rating on that side;
  - the response energy both directions require must fit in the store;
  - before EAC go-live (2 Nov 2023) a unit could offer only one of DC, DM or DR
    per block, chosen before the auction - by default on the previous day's
    clearing prices, the latest known at the bid deadline;
  - Reserved Capacity is held from EAC go-live (2 Nov 2023), when stacking began,
    though the rule binds only from 15 Nov 2024 (see _Scheduler.allocate).

Each service day is allocated at its bid deadline (14:00 the day before under
EAC, 14:30 before it), from the state of energy the battery has at that moment
and the commitments it already holds, so that every block it takes on can
actually be reached and held. Dispatch plans around a day's commitments from
that deadline: an operator must be able to deliver everything it offered, so
it positions for its offers before results publish. In this price-taker model
the offers are exactly what clears.

Dispatch trades only the power left on each side, and holds state of energy
inside the range every contract needs (Service Terms 6.11). Starting a
settlement period outside that range makes the unit unavailable for it, so the
block loses that period's availability payment (Service Terms 6.12).

Contracts are called on as GB frequency actually moved (response_delivery).
Delivered energy is neither paid nor charged (Service Terms 16), but it moves
state of energy and has to be made good by trading, so each offer carries its
expected cost and dispatch recovers from what is actually delivered.

Key assumptions:
  - The battery is a price-taker, and never holds more than a fifth of any
    auction's cleared volume.
  - It offers each product at its opportunity cost, so it holds a product only
    when the clearing price beats the arbitrage value the capacity would forgo.
  - Baseline changes take effect in the settlement period they are planned for;
    NESO's baseline gate closure is not modelled.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analysis.fr_allocation import allocate_day, choose_family
from src.optimisation.day_ahead import plan_day
from src.analysis.response_delivery import DELIVERY_COLUMNS
from src.analysis.neso_rules import (
    EAC_BID_CLOSE,
    FAMILIES,
    PRE_EAC_BID_CLOSE,
    PRODUCTS,
    RESERVED_CAPACITY_SHARE,
    arbitrage_power_limits_mw,
    energy_recovery_adjustment,
    family,
    is_low,
    soc_bounds_mwh,
    splitting_allowed,
)

# Each EFA block is 4 hours in GB
EFA_HOURS = 4

# Accurate GB EFA block → settlement period mapping.
# EFA 1 spans two calendar dates: periods 47–48 of D-1 plus periods 1–6 of D.
# EFA 2–6 fall entirely within calendar date D.
# Settlement periods are 1–48 (half-hourly, 00:00–24:00).
# Note: DST changeover days may have 50 periods; callers should clip to ≤ 48 first.
EFA_PERIODS = {
    1: {"prev": [47, 48], "curr": [1, 2, 3, 4, 5, 6]},   # 23:00–03:00
    2: {"prev": [],       "curr": list(range(7,  15))},    # 03:00–07:00
    3: {"prev": [],       "curr": list(range(15, 23))},    # 07:00–11:00
    4: {"prev": [],       "curr": list(range(23, 31))},    # 11:00–15:00
    5: {"prev": [],       "curr": list(range(31, 39))},    # 15:00–19:00
    6: {"prev": [],       "curr": list(range(39, 47))},    # 19:00–23:00
}

SERVICE_LABELS = {
    "DCH": "DC High",
    "DCL": "DC Low",
    "DRH": "DR High",
    "DRL": "DR Low",
    "DMH": "DM High",
    "DML": "DM Low",
}

ALL_SERVICES = list(SERVICE_LABELS.keys())

# Colour map for dashboard charts (revenue streams)
SERVICE_COLOURS = {
    "DCH": "#0D7680",   # FT teal
    "DCL": "#5BA8AE",   # lighter teal
    "DRH": "#4E8A3C",   # dark green
    "DRL": "#8AB87F",   # lighter green
    "DMH": "#7B3FA0",   # purple
    "DML": "#B08FC8",   # lighter purple
    "Arbitrage": "#C9400A",  # warm orange-red
    "Cycling cost": "#8B2020",  # dark red
}

# How a pre-EAC unit picks its one service per block, back when it could hold only
# one. Only "d1" remains: an "always_dc" sensitivity mirroring the 2022-23 fleet's
# habit was dropped 2026-09-16, because a full-power DC stack in both directions
# cannot recover what it delivers and spent a fifth of the era unavailable.
PRE_EAC_RULES = ("d1",)

# How an offer prices the trading it gives up. "formula" values each block alone,
# as one cycle between its own cheapest and dearest periods
# (_shadow_arb_value_per_mw). "lp" chooses the day's holdings together with a
# half-hourly trading plan (src/optimisation/day_ahead.py), so the value of free
# capacity depends on direction, hour, state of energy and the other blocks.
OFFER_VALUATIONS = ("formula", "lp")

# A modelling limit, not a NESO rule: the battery never holds more than this share
# of an auction's cleared volume. Before EAC the DM and DR auctions were often
# smaller than the battery itself (median DM Low cleared 4 MW), and beyond about a
# fifth of an auction one unit's offer could plausibly set the price, which a
# price-taker cannot. Chosen 2026-09-15; re-clearing real order books would replace it.
AUCTION_SHARE_CAP = 0.20

# Offers price delivery on what an operator knows at the bid deadline: delivery over
# the previous four weeks and prices over the previous week, each ending on the last
# complete day before 14:00 on D-1.
DELIVERY_TRAILING_DAYS = 28
PRICE_TRAILING_DAYS = 7

# Frequency cannot be forecast, but an operator sees it live. Where a new block will
# restore the full requirement, dispatch plans on delivery carrying on at its average
# over this many preceding settlement periods (one day).
DELIVERY_PLANNING_WINDOW = 48

# A site with no interest in arbitrage still trades to recover delivered energy. This
# small cost per MWh stops its price-blind plan moving more energy than it needs to.
RECOVERY_TRADE_COST_GBP_PER_MWH = 1.0

_LOW_PRODUCTS, _HIGH_PRODUCTS = ("DCL", "DML", "DRL"), ("DCH", "DMH", "DRH")
_LOW_COLUMNS = [DELIVERY_COLUMNS.index(c) for c in ("dc_low", "dm_low", "dr_low")]
_HIGH_COLUMNS = [DELIVERY_COLUMNS.index(c) for c in ("dc_high", "dm_high", "dr_high")]

_SETTLEMENT_PERIODS_PER_BLOCK = 8

TRAJECTORY_COLUMNS = ["date", "sp", "soc_frac", "soc_min_frac", "soc_max_frac"]

# Every cached table carries the same columns, whichever streams a scenario has
REVENUE_COLUMNS = [f"{p}_rev" for p in PRODUCTS] + ["imbalance_revenue_gbp", "cycling_cost_gbp", "mwh_cycled",
                                                    "delivery_mwh", "delivery_cycling_cost_gbp"]


@dataclass
class BatterySpec:
    power_mw: float = 50.0
    duration_h: float = 2.0
    efficiency_rt: float = 0.90       # Round-trip, expressed as a fraction (e.g. 0.90)
    cycling_cost_per_mwh: float = 3.0 # £ per MWh of usable energy throughput
    availability_factor: float = 0.95 # Fraction of periods the asset is available (maintenance,
                                      # faults, curtailment). 0.95 reflects the 95% minimum
                                      # availability threshold specified in NESO's Dynamic
                                      # Containment and EAC service agreements, and is consistent
                                      # with observed GB BESS fleet performance (Modo Energy,
                                      # "GB Battery Storage Report", 2024).

    @property
    def energy_mwh(self) -> float:
        return self.power_mw * self.duration_h


# Representative GB BESS asset used for all pre-computed backtests.
# Defined here — next to BatterySpec — so scripts and the UI import from
# one place and parameter changes only need to be made once.
REFERENCE_BATTERY = BatterySpec(
    power_mw=50.0,
    duration_h=2.0,
    efficiency_rt=0.90,       # Industry standard for modern Li-ion (NESO/Modo fleet data)
    cycling_cost_per_mwh=3.0, # Mid-range estimate consistent with Li-ion degradation literature
    availability_factor=0.95, # Min threshold in DC/EAC service agreements; GB fleet consistent
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _filter_dates(df: pd.DataFrame, date_col: str, start_date, end_date) -> pd.DataFrame:
    if start_date is not None:
        df = df[df[date_col] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df[date_col] <= pd.Timestamp(end_date)]
    return df


def _in_range(date: pd.Timestamp, start_date, end_date) -> bool:
    return (start_date is None or date >= pd.Timestamp(start_date)) and (
        end_date is None or date <= pd.Timestamp(end_date)
    )


def _apx_by_date(market_index: pd.DataFrame) -> dict:
    """{date: Series(index=settlementPeriod, values=APXMIDP price)} over the full table."""
    if market_index.empty:
        return {}
    apx = market_index[market_index["dataProvider"] == "APXMIDP"].copy()
    apx["settlementDate"] = pd.to_datetime(apx["settlementDate"]).dt.normalize()
    apx = apx[apx["settlementPeriod"] <= 48]  # drop DST extra periods
    return {
        date: grp.set_index("settlementPeriod")["price"]
        for date, grp in apx.groupby("settlementDate")
    }


def _efa_prices(apx_by_date: dict, date: pd.Timestamp, efa: int) -> pd.Series:
    """
    Return APXMIDP prices for the given EFA block, spanning D-1/D where needed.

    Parameters
    ----------
    apx_by_date : dict
        {pd.Timestamp (normalised): pd.Series(index=settlementPeriod, values=price)}
        Built from the full (unfiltered) market_index so D-1 lookups succeed on the
        first day of the backtest period.
    date : pd.Timestamp
        Calendar date D (normalised to midnight) for which the EFA block is required.
    efa : int
        EFA block number 1–6.

    Returns
    -------
    pd.Series indexed by settlement period (values from the correct calendar date(s)).
    For EFA 1: D-1 periods 47–48 followed by D periods 1–6 (up to 8 values).
    For EFA 2–6: 8 values from D only.
    """
    curr = apx_by_date.get(date, pd.Series(dtype=float))
    slices = [curr.reindex(EFA_PERIODS[efa]["curr"]).dropna()]
    if EFA_PERIODS[efa]["prev"]:
        prev = apx_by_date.get(date - pd.Timedelta(days=1), pd.Series(dtype=float))
        slices.insert(0, prev.reindex(EFA_PERIODS[efa]["prev"]).dropna())
    valid = [s for s in slices if not s.empty]
    return pd.concat(valid) if valid else pd.Series(dtype=float)


# Settlement period on a calendar date -> (days to add to reach its service day, EFA block)
_SP_TO_BLOCK = {}
for _efa, _spec in EFA_PERIODS.items():
    for _sp in _spec["curr"]:
        _SP_TO_BLOCK[_sp] = (0, _efa)
    for _sp in _spec["prev"]:
        _SP_TO_BLOCK[_sp] = (1, _efa)


def _period_block(date: pd.Timestamp, sp: int) -> tuple[pd.Timestamp, int]:
    """The (service day, EFA block) a calendar settlement period belongs to."""
    offset, efa = _SP_TO_BLOCK[int(sp)]
    return date + pd.Timedelta(days=offset), efa


def _block_start(service_date: pd.Timestamp, efa: int) -> pd.Timestamp:
    """EFA 1 starts at 23:00 the evening before its service day."""
    return service_date + pd.Timedelta(hours=-1 + EFA_HOURS * (efa - 1))


def _settlement_period(ts: pd.Timestamp) -> int:
    return ts.hour * 2 + ts.minute // 30 + 1


def _bid_close(service_date: pd.Timestamp) -> pd.Timestamp:
    """The deadline for offers into a service day's auctions."""
    hh, mm = EAC_BID_CLOSE if splitting_allowed(service_date.date()) else PRE_EAC_BID_CLOSE
    return service_date - pd.Timedelta(days=1) + pd.Timedelta(hours=hh, minutes=mm)


def _shadow_arb_value_per_mw(block_prices: pd.Series, battery: "BatterySpec") -> float:
    """
    Expected net arbitrage profit per MW kept back for one EFA block, from the
    price signal: one full cycle between the block's cheapest and dearest periods,
    after round-trip loss and wear.
    """
    n_periods = max(1, int(battery.duration_h * 2))
    if len(block_prices) < n_periods * 2:
        return 0.0
    net_per_mw = (
        block_prices.nlargest(n_periods).mean()
        - block_prices.nsmallest(n_periods).mean() / battery.efficiency_rt
        - battery.cycling_cost_per_mwh
    ) * battery.duration_h
    return max(0.0, float(net_per_mw))


def _clearing_prices(auctions: pd.DataFrame, services: list) -> dict:
    """{(service day, EFA): {product: £/MW/h}} for the selected products."""
    df = auctions[auctions["Service"].isin(services)]
    if df.empty:
        return {}
    days = pd.to_datetime(df["EFA Date"]).dt.normalize()
    return {
        (day, int(efa)): dict(zip(grp["Service"], grp["Clearing Price"].astype(float)))
        for (day, efa), grp in df.groupby([days, df["EFA"]])
    }


def _cleared_volumes(auctions: pd.DataFrame, services: list) -> dict:
    """{(service day, EFA): {product: MW NESO cleared}} for the selected products."""
    df = auctions[auctions["Service"].isin(services)]
    if df.empty:
        return {}
    days = pd.to_datetime(df["EFA Date"]).dt.normalize()
    return {
        (day, int(efa)): dict(zip(grp["Service"], grp["Cleared Volume"].astype(float)))
        for (day, efa), grp in df.groupby([days, df["EFA"]])
    }


def _caps(volumes: dict) -> dict:
    """Most MW of each product the battery may hold: its share of what the auction cleared."""
    return {product: AUCTION_SHARE_CAP * mw for product, mw in volumes.items() if np.isfinite(mw)}


def _service_block(dates: pd.Series, periods: pd.Series) -> tuple[pd.Series, np.ndarray]:
    """Service day and EFA block of settlement periods 1-48, as _period_block does one at a time."""
    late = (periods >= 47).to_numpy()
    service_day = pd.to_datetime(dates).dt.normalize() + pd.to_timedelta(late.astype(int), unit="D")
    efa = np.where(late, 1, (periods.to_numpy() + 1) // 8 + 1)
    return service_day, efa


def _trailing_by_block(per_block, window: int, min_periods: int) -> dict:
    """
    {(service day, EFA): mean} over the `window` days ending two days before each
    service day, the last complete day before its bid deadline. `per_block` is a
    Series or DataFrame indexed by (service day, EFA).
    """
    out = {}
    for efa, values in per_block.groupby(level=1):
        values = values.droplevel(1)
        days = pd.date_range(values.index.min(), values.index.max() + pd.Timedelta(days=2))
        trailing = values.reindex(days).rolling(window, min_periods=min_periods).mean().shift(2)
        if isinstance(trailing, pd.DataFrame):
            for day, row in trailing.dropna(how="all").iterrows():
                out[(day, int(efa))] = row.to_dict()
        else:
            for day, value in trailing.dropna().items():
                out[(day, int(efa))] = float(value)
    return out


def _expected_delivery(delivery: pd.DataFrame) -> dict:
    """{(service day, EFA): {delivery column: MWh per MW}} expected over each block at its bid deadline."""
    d = delivery[delivery["settlementPeriod"] <= 48]
    day, efa = _service_block(d["settlementDate"], d["settlementPeriod"])
    per_block = d[DELIVERY_COLUMNS].groupby([day.to_numpy(), efa]).sum()
    return _trailing_by_block(per_block, DELIVERY_TRAILING_DAYS, min_periods=7)


def _expected_block_prices(market_index: pd.DataFrame) -> dict:
    """{(service day, EFA): £/MWh} recent average APXMIDP price for each block at its bid deadline."""
    apx = market_index[(market_index["dataProvider"] == "APXMIDP") & (market_index["settlementPeriod"] <= 48)]
    if apx.empty:
        return {}
    day, efa = _service_block(apx["settlementDate"], apx["settlementPeriod"])
    per_block = apx["price"].groupby([day.to_numpy(), efa]).mean()
    return _trailing_by_block(per_block, PRICE_TRAILING_DAYS, min_periods=3)


def _delivery_offer_costs(expected: dict | None, price: float | None, battery: "BatterySpec") -> dict:
    """
    £ per MW held for a block that making good its expected delivery costs.

    Delivered energy itself is neither paid nor charged (Service Terms 16). Low
    delivery empties the store, so replacing it means buying E/η at the block's
    price, plus wear on the discharge. High delivery fills it for free, so selling
    η·E back earns the price, less wear: a benefit, entered as a negative cost.
    """
    if not expected or price is None or not np.isfinite(price):
        return {}
    eta, wear = battery.efficiency_rt, battery.cycling_cost_per_mwh
    costs = {}
    for p in PRODUCTS:
        e = float(expected.get(f"{family(p).lower()}_{'low' if is_low(p) else 'high'}", 0.0))
        e = e if np.isfinite(e) else 0.0
        costs[p] = e * (price / eta + wear) if is_low(p) else -eta * e * (price - wear)
    return costs


def _build_result(
    anc_wide: pd.DataFrame,
    imb_wide: pd.DataFrame,
    battery: "BatterySpec",
    avg_fr_mw: float,
    avg_arb_mw: float,
    soc_traj: list,
    extras: dict | None = None,
) -> dict:
    """
    Merge ancillary and arbitrage monthly streams, apply availability factor,
    compute net revenue columns, and return the standard result dict.

    Returns {"monthly": DataFrame, "summary": dict, "soc_trajectory": DataFrame | None}.
    """
    frames = [f for f in [anc_wide, imb_wide] if not f.empty]
    if not frames:
        return {"monthly": pd.DataFrame(), "summary": dict(extras or {}), "soc_trajectory": None}

    monthly = frames[0].join(frames[1:], how="outer").fillna(0).reset_index()
    for col in REVENUE_COLUMNS:
        if col not in monthly.columns:
            monthly[col] = 0.0
    monthly["month_dt"] = monthly["month"].dt.to_timestamp()

    rev_cols  = [c for c in monthly.columns if c.endswith("_rev") or c == "imbalance_revenue_gbp"]
    cost_cols = [c for c in ("cycling_cost_gbp", "delivery_cycling_cost_gbp") if c in monthly.columns]

    # Apply availability factor to every revenue stream and cycling cost proportionally.
    # This models the fraction of committed periods/days where the asset is actually
    # available — accounting for planned maintenance, unplanned faults, and curtailment.
    for col in rev_cols + cost_cols:
        monthly[col] = monthly[col] * battery.availability_factor
    for col in ("mwh_cycled", "delivery_mwh"):
        if col in monthly.columns:
            monthly[col] = monthly[col] * battery.availability_factor

    monthly["gross_revenue"] = monthly[rev_cols].sum(axis=1)
    monthly["cycling_cost"]  = monthly[cost_cols].sum(axis=1) if cost_cols else 0.0
    monthly["net_revenue"]   = monthly["gross_revenue"] - monthly["cycling_cost"]

    years     = len(monthly) / 12
    net_total = monthly["net_revenue"].sum()

    breakdown = {}
    for col in rev_cols:
        label = col.replace("_rev", "") if col.endswith("_rev") else "Arbitrage"
        breakdown[label] = round(monthly[col].sum(), 0)

    summary = {
        "total_gross":        round(monthly["gross_revenue"].sum(), 0),
        "total_cycling_cost": round(monthly["cycling_cost"].sum(), 0),
        "total_net":          round(net_total, 0),
        "years_covered":      round(years, 2),
        "annualised_net":     round(net_total / years, 0) if years > 0 else 0,
        "annualised_per_mw":  round(net_total / years / battery.power_mw, 0) if years > 0 and battery.power_mw > 0 else 0,
        "breakdown":          breakdown,
        "top_service":        max(breakdown, key=breakdown.get) if breakdown else "N/A",
        "fr_mw":              avg_fr_mw,
        "arb_mw":             avg_arb_mw,
    }
    if "mwh_cycled" in monthly.columns:
        summary["total_mwh_cycled"] = round(monthly["mwh_cycled"].sum(), 1)
    if "delivery_mwh" in monthly.columns:
        summary["total_delivery_mwh"] = round(monthly["delivery_mwh"].sum(), 1)
    summary.update(extras or {})

    soc_trajectory_df = (
        pd.DataFrame(soc_traj, columns=TRAJECTORY_COLUMNS)
        if soc_traj else None
    )
    return {"monthly": monthly, "summary": summary, "soc_trajectory": soc_trajectory_df}


# ---------------------------------------------------------------------------
# Stage 1 — what the battery holds in each EFA block
# ---------------------------------------------------------------------------

_SCHEDULE_COLUMNS = ["date", "efa", "family", "apply_reserve", "soc_min_mwh", "soc_max_mwh",
                     "discharge_max_mw", "charge_max_mw", "arb_mw"] + [f"q_{p}" for p in PRODUCTS]


class _Scheduler:
    """
    Decides each service day's holdings at its bid deadline.

    For each block the battery offers every allowed product at its opportunity
    cost - the arbitrage value of keeping that capacity, estimated from the
    price signal - and holds the combination that earns most across the day
    while leaving state of energy a deliverable path (see fr_allocation). Which
    services it may offer, and whether Reserved Capacity is held, follow
    the service day's date. Decisions are kept, because the previous day's
    commitments limit how state of energy can move before the next day starts.

    Each product is capped at AUCTION_SHARE_CAP of its auction's cleared volume.
    Given expected delivery and prices, each offer also carries the expected cost
    of making good the energy the product will deliver.

    offer_valuation="lp" replaces the per-block arbitrage value with a trading plan
    for every period from the deadline to the end of the service day, solved
    together with the holdings (day_ahead.plan_day), and picks each pre-EAC
    block's one service jointly across the day. price_shrink pulls that plan's
    forecast towards its mean, by price_shrink or, where price_shrink_by_date has
    a weight for the service day, by that. With include_arbitrage=False there is
    nothing to plan for, and both valuations hold the same.

    offer_forecast_prices_by_date is the forecast of the service day as it stood
    at the bid deadline, for strategies whose day-ahead forecast needs all of D-1
    (see price_forecast.run_forecast_backtest's offer_information). The periods of
    D-1 itself still come from forecast_prices_by_date, which was complete by then.
    """

    def __init__(self, auctions, battery, forecast_prices_by_date=None, services=None, *,
                 include_arbitrage=True, pre_eac_rule="d1", expected_delivery=None, expected_prices=None,
                 offer_valuation="formula", price_shrink=1.0, offer_forecast_prices_by_date=None,
                 price_shrink_by_date=None, guard_low_by_date=None, guard_high_by_date=None):
        if pre_eac_rule not in PRE_EAC_RULES:
            raise ValueError(f"pre_eac_rule must be one of {PRE_EAC_RULES}, got {pre_eac_rule!r}")
        if offer_valuation not in OFFER_VALUATIONS:
            raise ValueError(f"offer_valuation must be one of {OFFER_VALUATIONS}, got {offer_valuation!r}")
        services = ALL_SERVICES if services is None else list(services)
        self.table = _clearing_prices(auctions, services)
        self.volumes = _cleared_volumes(auctions, services)
        self.expected_delivery = expected_delivery or {}
        self.expected_prices = expected_prices or {}
        self.forecast = {pd.Timestamp(d).normalize(): s for d, s in (forecast_prices_by_date or {}).items()}
        self.offer_forecast = (self.forecast if offer_forecast_prices_by_date is None else
                               {pd.Timestamp(d).normalize(): s for d, s in offer_forecast_prices_by_date.items()})
        self.battery = battery
        self.include_arbitrage = include_arbitrage
        self.pre_eac_rule = pre_eac_rule
        self.plan_trading = offer_valuation == "lp" and include_arbitrage
        self.price_shrink = float(price_shrink)
        # How far to believe each day's forecast shape, when it is estimated per day
        # (src/analysis/shrink.py) rather than held constant. The service day's weight
        # applies to the whole plan, including the lead-in hours of D-1.
        self.price_shrink_by_date = {pd.Timestamp(d).normalize(): float(w)
                                     for d, w in (price_shrink_by_date or {}).items()}
        # Conformal guard bands: how much worse than forecast each side plans to trade
        # at, by settlement period (src/analysis/intervals.py). Empty plans on the
        # forecast itself.
        by_date = lambda table: {pd.Timestamp(d).normalize(): v for d, v in (table or {}).items()}
        self.guard_low_by_date, self.guard_high_by_date = by_date(guard_low_by_date), by_date(guard_high_by_date)
        self.rows = {}    # (service day, EFA) -> schedule row
        self.plans = {}   # service day -> [(SoE at block's first period, at its last)]

    def _lead_in(self, service_date: pd.Timestamp, now: pd.Timestamp) -> list:
        """The previous day's commitments still to run between `now` and the new day."""
        P, E = self.battery.power_mw, self.battery.energy_mwh
        prev = service_date - pd.Timedelta(days=1)
        segments = []
        for efa in range(1, 7):
            start = _block_start(prev, efa)
            end = start + pd.Timedelta(hours=EFA_HOURS)
            if end <= now:
                continue
            n_sp = int((end - max(start, now)) / pd.Timedelta(minutes=30))
            row = self.rows.get((prev, efa))
            if row is None:
                segments.append({"lo": 0.0, "hi": E, "dis": P, "chg": P, "n_sp": n_sp})
            else:
                segments.append({"lo": row["soc_min_mwh"], "hi": row["soc_max_mwh"],
                                 "dis": row["discharge_max_mw"], "chg": row["charge_max_mw"],
                                 "n_sp": n_sp})
        return segments

    def allocate(self, service_date: pd.Timestamp, now: pd.Timestamp, soc_now_mwh: float) -> None:
        b = self.battery
        day = service_date.date()
        # Reserved Capacity has bound the Procurement Rules only since 15 Nov 2024, but
        # the running energy requirement applied throughout cannot be met by a stack
        # that leaves no power to recover (SOE guidance §3), so it is held from EAC
        # go-live, when stacking began. Pre-EAC blocks carry a single service, which
        # in sample fortnights stayed within the requirement without it (2026-09-15).
        apply_reserve = splitting_allowed(day)

        lead_in = self._lead_in(service_date, now)
        if self.plan_trading:
            out, labels = self._allocate_with_plan(service_date, now, soc_now_mwh, apply_reserve, lead_in)
        else:
            out, labels = self._allocate_with_formula(service_date, soc_now_mwh, apply_reserve, lead_in)

        for efa, (alloc, label) in enumerate(zip(out["blocks"], labels), start=1):
            q = alloc["q"]
            soc_lo, soc_hi = soc_bounds_mwh(q, b.energy_mwh)
            dis_max, chg_max = arbitrage_power_limits_mw(q, b.power_mw, apply_reserve)
            row = {
                "family": label, "apply_reserve": apply_reserve,
                "soc_min_mwh": soc_lo, "soc_max_mwh": soc_hi,
                "discharge_max_mw": dis_max, "charge_max_mw": chg_max,
                "arb_mw": alloc["arb_mw"],
            }
            row.update({f"q_{p}": q.get(p, 0.0) for p in PRODUCTS})
            self.rows[(service_date, efa)] = row
        self.plans[service_date] = out["soc_plan_mwh"]

    def _offers(self, key: tuple) -> dict:
        """What the battery could offer into one block: prices, caps and delivery costs."""
        return {
            "prices": self.table.get(key, {}),
            "caps": _caps(self.volumes.get(key, {})),
            "costs": _delivery_offer_costs(self.expected_delivery.get(key), self.expected_prices.get(key),
                                           self.battery),
        }

    def _offer_view(self, service_date: pd.Timestamp) -> dict:
        """The forecasts an offer can see: D-1 as forecast the day before, D as at the deadline."""
        previous = service_date - pd.Timedelta(days=1)
        return {day: series for day, series in ((previous, self.forecast.get(previous)),
                                                (service_date, self.offer_forecast.get(service_date)))
                if series is not None}

    def _allocate_with_formula(self, service_date, soc_now_mwh, apply_reserve, lead_in):
        b, day = self.battery, service_date.date()
        view = self._offer_view(service_date)
        blocks, labels = [], []
        for efa in range(1, 7):
            offers = self._offers((service_date, efa))
            arb_value = (
                _shadow_arb_value_per_mw(_efa_prices(view, service_date, efa), b)
                if self.include_arbitrage else 0.0
            )
            if splitting_allowed(day):
                families, label = FAMILIES, "ALL"
            else:
                reference = (service_date - pd.Timedelta(days=1), efa)
                chosen = choose_family(self.table.get(reference, {}), arb_value, b.power_mw, b.energy_mwh,
                                       b.duration_h, apply_reserve=apply_reserve,
                                       caps=_caps(self.volumes.get(reference, {})),
                                       offer_costs=offers["costs"])
                families, label = ((chosen,), chosen) if chosen else ((), "none")
            blocks.append({**offers, "arb_value": arb_value, "families": families})
            labels.append(label)
        out = allocate_day(blocks, b.power_mw, b.energy_mwh, b.duration_h, b.efficiency_rt,
                           soc_now_mwh, apply_reserve=apply_reserve, lead_in=lead_in)
        return out, labels

    @staticmethod
    def _forecast_path(view: dict, start: pd.Timestamp, n_periods: int) -> np.ndarray:
        """Forecast price of each of n_periods settlement periods from `start`, NaN where unknown."""
        path = np.full(n_periods, np.nan)
        for k in range(n_periods):
            ts = start + pd.Timedelta(minutes=30 * k)
            series = view.get(ts.normalize())
            if series is not None:
                path[k] = series.get(_settlement_period(ts), np.nan)
        return path

    def _allocate_with_plan(self, service_date, now, soc_now_mwh, apply_reserve, lead_in):
        """
        Holdings chosen with the day's trading plan (day_ahead.plan_day). Before EAC
        each block's one service is picked first, on the previous day's clearing
        prices as choose_family does, but jointly across the day and priced by the
        same plan.
        """
        b = self.battery
        n_lead = sum(seg["n_sp"] for seg in lead_in)
        view = self._offer_view(service_date)

        def path(series_by_date):
            """The plan's periods in order: the lead-in, then the service day's six blocks."""
            return np.concatenate([
                self._forecast_path(series_by_date, now, n_lead),
                self._forecast_path(series_by_date, _block_start(service_date, 1),
                                    6 * _SETTLEMENT_PERIODS_PER_BLOCK)])

        prices = path(view)
        guard_low = path(self.guard_low_by_date) if self.guard_low_by_date else None
        guard_high = path(self.guard_high_by_date) if self.guard_high_by_date else None

        def plan(blocks, one_service=False):
            return plan_day(blocks, b.power_mw, b.energy_mwh, b.efficiency_rt, b.cycling_cost_per_mwh,
                            soc_now_mwh, prices, apply_reserve=apply_reserve, lead_in=lead_in,
                            one_service=one_service, guard_low=guard_low, guard_high=guard_high,
                            price_shrink=self.price_shrink_by_date.get(service_date, self.price_shrink))

        offers = [self._offers((service_date, efa)) for efa in range(1, 7)]
        if splitting_allowed(service_date.date()):
            families, labels = [FAMILIES] * 6, ["ALL"] * 6
        else:
            reference = []
            for efa, offer in zip(range(1, 7), offers):
                key = (service_date - pd.Timedelta(days=1), efa)
                reference.append({"prices": self.table.get(key, {}), "caps": _caps(self.volumes.get(key, {})),
                                  "costs": offer["costs"], "families": FAMILIES})
            families = plan(reference, one_service=True)["families"]
            labels = [chosen[0] if chosen else "none" for chosen in families]
        return plan([{**offer, "families": fams} for offer, fams in zip(offers, families)]), labels

    def planned_soc(self, date: pd.Timestamp, sp: int) -> float | None:
        """State of energy the plans pass through at the start of a settlement period."""
        service_day, efa = _period_block(date, sp)
        plan = self.plans.get(service_day)
        if plan is None:
            return None
        first, last = plan[efa - 1]
        order = EFA_PERIODS[efa]["prev"] + EFA_PERIODS[efa]["curr"]
        return first + (last - first) * order.index(sp) / (len(order) - 1)

    def schedule(self) -> pd.DataFrame:
        rows = [{"date": d, "efa": e, **row} for (d, e), row in sorted(self.rows.items())]
        return pd.DataFrame(rows, columns=_SCHEDULE_COLUMNS).set_index(["date", "efa"])


def compute_fr_schedule(
    auctions: pd.DataFrame,
    battery: BatterySpec,
    dates: list,
    forecast_prices_by_date: dict | None = None,
    services: list | None = None,
    *,
    include_arbitrage: bool = True,
    pre_eac_rule: str = "d1",
    initial_soc_frac: float = 0.5,
) -> pd.DataFrame:
    """
    Per-EFA-block holdings in each frequency response product, without dispatch.

    State of energy follows each day's own plan, so the battery is assumed to
    reposition within its free power at no energy cost. This is the FR-only
    scenario; with dispatch, run_dispatch allocates each day from the state of
    energy trading has actually left.

    Parameters
    ----------
    dates : list of pd.Timestamp
        Service days to schedule, in order.
    forecast_prices_by_date : {date: Series(index=settlementPeriod)}, optional
        Price signal behind the arbitrage opportunity cost.
    services : list of product codes, optional
        Products the battery may offer. Default all six; [] offers none.
    include_arbitrage : bool
        False values arbitrage at zero.
    pre_eac_rule : "d1"
        How a pre-EAC unit picks its one service: on the previous service day's
        clearing prices for the same block, the latest a bidder could know.
    initial_soc_frac : float
        State of energy before the first day, as a fraction of energy_mwh.

    Returns
    -------
    DataFrame indexed by (date, efa) with q_<product> (MW) for all six products,
    family (service offered: "ALL" after EAC, "none" if nothing), apply_reserve,
    soc_min_mwh, soc_max_mwh, discharge_max_mw, charge_max_mw and arb_mw.
    """
    scheduler = _Scheduler(auctions, battery, forecast_prices_by_date, services,
                           include_arbitrage=include_arbitrage, pre_eac_rule=pre_eac_rule)
    soc = initial_soc_frac * battery.energy_mwh
    for date in dates:
        date = pd.Timestamp(date).normalize()
        close = _bid_close(date)
        planned = scheduler.planned_soc(close.normalize(), _settlement_period(close))
        if planned is None:
            scheduler.allocate(date, date, soc)
        else:
            soc = planned
            scheduler.allocate(date, close, soc)
    return scheduler.schedule()


# ---------------------------------------------------------------------------
# Stream 1 — Ancillary service availability revenue
# ---------------------------------------------------------------------------

def calc_ancillary_revenue(
    auctions: pd.DataFrame,
    schedule: pd.DataFrame,
    availability: dict | None = None,
) -> pd.DataFrame:
    """
    Monthly availability revenue from the products held in each block.

    Revenue = clearing price x MW held x 4 h, scaled by the share of the block's
    settlement periods the unit was available for. A unit that starts a period
    outside its required state of energy is unavailable for it (Service Terms
    6.12), so dispatch passes those shares in.

    Parameters
    ----------
    schedule : DataFrame from compute_fr_schedule()
    availability : {(date, efa): share of the block available}, optional

    Returns
    -------
    DataFrame with columns: [month (Period), service (str), revenue_gbp (float)]
    """
    empty = pd.DataFrame(columns=["month", "service", "revenue_gbp"])
    if schedule is None or schedule.empty:
        return empty

    held = schedule[[f"q_{p}" for p in PRODUCTS]].rename(columns=lambda c: c[2:])
    long = held.stack().reset_index()
    long.columns = ["date", "efa", "service", "mw"]
    long = long[long["mw"] > 1e-9]
    if long.empty:
        return empty

    prices = (
        auctions.assign(date=pd.to_datetime(auctions["EFA Date"]).dt.normalize())
        .rename(columns={"EFA": "efa", "Service": "service", "Clearing Price": "price"})
        [["date", "efa", "service", "price"]]
    )
    long = long.merge(prices, on=["date", "efa", "service"], how="left")

    if availability:
        share = pd.Series(availability, dtype=float)
        keys = pd.MultiIndex.from_arrays([long["date"], long["efa"]])
        long["available"] = share.reindex(keys).fillna(1.0).to_numpy()
    else:
        long["available"] = 1.0

    long["revenue_gbp"] = long["price"] * long["mw"] * EFA_HOURS * long["available"]
    long["month"] = long["date"].dt.to_period("M")
    return long.groupby(["month", "service"])["revenue_gbp"].sum().reset_index()


# ---------------------------------------------------------------------------
# Stage 2 — dispatch around the commitments
# ---------------------------------------------------------------------------

def run_dispatch(
    apx_by_date: dict,
    battery: BatterySpec,
    dates: list,
    forecast_prices_by_date: dict,
    *,
    scheduler: _Scheduler | None = None,
    schedule: pd.DataFrame | None = None,
    initial_soc_frac: float = 0.5,
    horizon: int = 96,
    delivery: pd.DataFrame | None = None,
    price_seeking: bool = True,
    forecast_vintages: bool = False,
    early_forecast_prices_by_date: dict | None = None,
) -> tuple[list, list, dict]:
    """
    Rolling MPC dispatch over every settlement period, around FR commitments.

    Each service day's commitments enter at its bid deadline: decided there and
    then by `scheduler`, from the state of energy dispatch has left, or read
    from a fixed `schedule`. Until then the battery plans as if uncommitted, as
    an operator would have to. Each period's LP trades only the power the
    commitments leave on each side and holds state of energy where they need it.

    What they need is NESO's Minimum State of Energy Requirement, per direction
    (Service Terms 6.11): at the start of each block, the Contracted Response
    Energy Volume in store for the Low products and as headroom for the High
    products. With a `delivery` table (MWh per MW contracted per settlement
    period, from response_delivery) the contracts are called on. Low delivery
    takes energy out of the store and High delivery puts it in, less the
    round-trip loss, and each lowers its own direction's requirement by the
    energy delivered. The requirement then climbs back by the Energy Recovery
    Adjustment Volume, and the battery recovers energy to keep up, drawing on
    its Reserved Capacity when the power left for trading is not enough.
    Delivered energy is neither paid nor charged (Service Terms 16) but wears
    the battery. Within a block delivery lowers state of energy and requirement
    alike, so the LP plans on no further delivery; where a later block restores
    the full requirement, it plans on delivery continuing at its average over the
    previous day, which the operator has already seen. A settlement period that
    starts outside the requirement counts as unavailable (Service Terms 6.12).

    price_seeking=False is a site with no interest in wholesale arbitrage: its LP
    ignores prices and trades only to keep state of energy where its contracts
    need it, at a small cost per MWh so it moves no more energy than it must.

    Trades execute at actual prices. Per-period revenue can be negative when a
    forecast misleads, or when recovering delivered energy costs money; both are
    realistic outcomes.

    forecast_vintages=True plans only on forecasts that existed at the time. A
    forecast of day X from forecast_prices_by_date uses all of X-1, so it is known
    from midnight on X and drives today's periods. Tomorrow's come from
    early_forecast_prices_by_date, each day's forecast as it stood before the
    previous day ended (data to X-2); with none given, as for perfect foresight,
    forecast_prices_by_date is used for tomorrow too. No honest forecast exists
    for the day after tomorrow, so the plan ends at midnight after tomorrow: 49 to
    96 periods rather than `horizon`. Without vintages every period of the horizon
    reads forecast_prices_by_date, which for naive means tomorrow's slot holds
    today's actual prices.

    Returns
    -------
    energy_rows : list of {date, imbalance_revenue_gbp, cycling_cost_gbp, mwh_cycled,
        delivery_mwh, delivery_cycling_cost_gbp}, one per period in which energy moved
    soc_trajectory : list of (date, sp, soc_frac, soc_min_frac, soc_max_frac) after
        each period, with the requirement state of energy must meet at the start of the next
    breaches : {(date, efa): settlement periods started outside the requirement}
    """
    from src.optimisation.mpc import DT, solve_mpc

    if (scheduler is None) == (schedule is None):
        raise ValueError("pass exactly one of scheduler or schedule")

    P, E = battery.power_mw, battery.energy_mwh
    eta, wear = battery.efficiency_rt, battery.cycling_cost_per_mwh
    dates = [pd.Timestamp(d).normalize() for d in dates]
    periods = [(d, sp) for d in dates for sp in range(1, 49)]
    n = len(periods)
    position = {p: i for i, p in enumerate(periods)}
    forecast = {pd.Timestamp(d).normalize(): s for d, s in forecast_prices_by_date.items()}
    early = (forecast if early_forecast_prices_by_date is None else
             {pd.Timestamp(d).normalize(): s for d, s in early_forecast_prices_by_date.items()})
    fixed = schedule.to_dict("index") if schedule is not None else {}

    actual  = np.full(n, np.nan)
    predict = np.full(n, np.nan)
    predict_early = np.full(n, np.nan)
    soc_lo  = np.zeros(n)      # full Low requirement for the period's block: MWh in store
    soc_hi  = np.full(n, E)    # E less the full High requirement: MWh of headroom
    rev_hi  = np.zeros(n)
    dis_max = np.full(n, P)
    chg_max = np.full(n, P)
    reserve_out = np.zeros(n)  # Reserved Capacity MW available to recover High delivery
    reserve_in  = np.zeros(n)  # ... and Low delivery
    held    = np.zeros(n, dtype=bool)
    blocks  = []
    block_periods = defaultdict(list)

    # Delivery per MW contracted in each period; the MWh it moves depends on what is held
    per_mw = np.zeros((n, len(DELIVERY_COLUMNS)))
    if delivery is not None and not delivery.empty:
        table = delivery.assign(settlementDate=pd.to_datetime(delivery["settlementDate"]).dt.normalize())
        per_mw = (table.set_index(["settlementDate", "settlementPeriod"])[DELIVERY_COLUMNS]
                  .reindex(pd.MultiIndex.from_tuples(periods)).fillna(0.0).to_numpy())
    delivered_out = np.zeros(n)
    delivered_in  = np.zeros(n)
    q_low_by_period = np.zeros((n, len(_LOW_PRODUCTS)))
    q_high_by_period = np.zeros((n, len(_HIGH_PRODUCTS)))

    # Average delivery per MW over the preceding day, from past periods only
    cumulative = np.vstack([np.zeros(len(DELIVERY_COLUMNS)), np.cumsum(per_mw, axis=0)])
    lookback = np.minimum(np.arange(n), DELIVERY_PLANNING_WINDOW)
    recent_per_mw = (cumulative[np.arange(n)] - cumulative[np.arange(n) - lookback]) / np.maximum(lookback, 1)[:, None]

    # Running Minimum State of Energy Requirement per direction, and its adjustments
    need_lo, need_hi = np.zeros(n), np.zeros(n)
    adjust_lo, adjust_hi = np.zeros(n), np.zeros(n)

    for i, (d, sp) in enumerate(periods):
        a = apx_by_date.get(d)
        if a is not None and sp in a.index:
            actual[i] = float(a.loc[sp])
        f = forecast.get(d)
        if f is not None and sp in f.index:
            predict[i] = float(f.loc[sp])
        f = early.get(d)
        if f is not None and sp in f.index:
            predict_early[i] = float(f.loc[sp])
        key = _period_block(d, sp)
        blocks.append(key)
        block_periods[key].append(i)

    block_start = np.zeros(n, dtype=int)
    for i in range(1, n):
        block_start[i] = block_start[i - 1] if blocks[i] == blocks[i - 1] else i

    # A day's commitments are known from its bid deadline, or from its own first
    # period when the deadline falls before the backtest or on a day with no prices.
    known_at = defaultdict(list)
    for day in dates:
        close = _bid_close(day)
        known_at[position.get((close.normalize(), _settlement_period(close)), position[(day, 1)])].append(day)

    def settle_requirement(j):
        """Minimum State of Energy Requirement at the start of period j (Service Terms 6.11 ii)."""
        s = block_start[j]
        if j == s:
            need_lo[j], need_hi[j] = soc_lo[j], rev_hi[j]
            return
        k = j - 1
        adjust_lo[k] = energy_recovery_adjustment(k, s, soc_lo, need_lo, adjust_lo)
        adjust_hi[k] = energy_recovery_adjustment(k, s, rev_hi, need_hi, adjust_hi)
        need_lo[j] = min(soc_lo[j], need_lo[k] - delivered_out[k] + adjust_lo[k])
        need_hi[j] = min(rev_hi[j], need_hi[k] - delivered_in[k] + adjust_hi[k])

    def plan_prices(i, idx):
        """Today's periods on the day-ahead forecast; with vintages, tomorrow's on the early one."""
        if not forecast_vintages:
            return predict[idx]
        return np.where(day_of[idx] == day_of[i], predict[idx], predict_early[idx])

    def planning_range(i, last):
        """
        State-of-energy range the LP plans against at the start of periods i..last:
        the current block's requirement carried forward with no further delivery,
        then each later block's full requirement.
        """
        lo, hi = soc_lo[i:last + 1].copy(), soc_hi[i:last + 1].copy()
        s = block_start[i]
        end = i + 1
        while end <= last and block_start[end] == s:
            end += 1
        for need, adjust, rev, is_low_side in ((need_lo, adjust_lo, soc_lo[i], True),
                                               (need_hi, adjust_hi, rev_hi[i], False)):
            m, g, revs = list(need[s:i + 1]), list(adjust[s:i]), [rev] * (end - s)
            for t in range(i + 1, end):
                k = t - 1 - s
                g.append(energy_recovery_adjustment(k, 0, revs, m, g))
                m.append(min(rev, m[k] + g[k]))
            projected = np.array(m[i - s:])
            if is_low_side:
                lo[: end - i] = projected
            else:
                hi[: end - i] = E - projected
        return lo, hi

    # With vintages, where each period's plan must end: after tomorrow, the last day
    # with a forecast that already exists (or after today if tomorrow is missing)
    day_of = np.repeat(np.arange(len(dates)), 48)
    consecutive = np.array([dates[k + 1] - dates[k] == pd.Timedelta(days=1) for k in range(len(dates) - 1)]
                           + [False])
    plan_limit = np.where(consecutive[day_of], (day_of + 2) * 48, (day_of + 1) * 48)

    tol_mwh = 1e-3
    soc = initial_soc_frac * E
    energy_rows, soc_traj, breaches = [], [], {}

    for i, (d, sp) in enumerate(periods):
        for day in known_at.get(i, ()):
            if scheduler is not None:
                scheduler.allocate(day, d + pd.Timedelta(minutes=30 * (sp - 1)), soc)
            rows = scheduler.rows if scheduler is not None else fixed
            for efa in range(1, 7):
                row = rows.get((day, efa))
                idx = block_periods.get((day, efa))
                if row is None or not idx:
                    continue
                soc_lo[idx], soc_hi[idx] = row["soc_min_mwh"], row["soc_max_mwh"]
                rev_hi[idx] = E - row["soc_max_mwh"]
                dis_max[idx], chg_max[idx] = row["discharge_max_mw"], row["charge_max_mw"]
                held[idx] = sum(row[f"q_{p}"] for p in PRODUCTS) > 1e-9
                share = {fam: RESERVED_CAPACITY_SHARE[fam] if row["apply_reserve"] else 0.0 for fam in FAMILIES}
                reserve_out[idx] = sum(share[family(p)] * row[f"q_{p}"] for p in _HIGH_PRODUCTS)
                reserve_in[idx] = sum(share[family(p)] * row[f"q_{p}"] for p in _LOW_PRODUCTS)
                q_low = np.array([row[f"q_{p}"] for p in _LOW_PRODUCTS])
                q_high = np.array([row[f"q_{p}"] for p in _HIGH_PRODUCTS])
                q_low_by_period[idx], q_high_by_period[idx] = q_low, q_high
                delivered_out[idx] = per_mw[idx][:, _LOW_COLUMNS] @ q_low
                delivered_in[idx] = per_mw[idx][:, _HIGH_COLUMNS] @ q_high
        settle_requirement(i)

        # Availability is judged at the start of each settlement period
        if held[i] and (soc < need_lo[i] - tol_mwh or soc > E - need_hi[i] + tol_mwh):
            breaches[blocks[i]] = breaches.get(blocks[i], 0) + 1

        e_dis = e_chg = 0.0
        if not np.isnan(actual[i]):
            h_end = min(i + horizon, n, plan_limit[i]) if forecast_vintages else min(i + horizon, n)
            idx = np.arange(i, h_end)
            last = min(h_end, n - 1)
            lo, hi = planning_range(i, last)
            points = np.arange(i, last + 1)
            if h_end == n:
                lo, hi, points = np.append(lo, lo[-1]), np.append(hi, hi[-1]), np.append(points, last)
            # A later block restores the full requirement, so plan on delivery before it
            expected_out = q_low_by_period[idx] @ recent_per_mw[i, _LOW_COLUMNS]
            expected_in = q_high_by_period[idx] @ recent_per_mw[i, _HIGH_COLUMNS]
            later = block_start[points] != block_start[i]
            lo = np.where(later, lo + np.concatenate([[0.0], np.cumsum(expected_out)]), lo)
            hi = np.where(later, hi - eta * np.concatenate([[0.0], np.cumsum(expected_in)]), hi)
            # Reserved Capacity is held for energy recovery (SOE guidance), so every
            # period of the plan may use it to keep a requirement reachable
            e_dis, e_chg = solve_mpc(
                soc_current=soc,
                price_forecast=(np.nan_to_num(plan_prices(i, idx), nan=0.0) if price_seeking
                                else np.zeros(len(idx))),
                arb_mw_schedule=dis_max[idx],
                soc_min=lo,
                soc_max=hi,
                energy_mwh=E,
                efficiency_rt=eta,
                cycling_cost_per_mwh=wear,
                horizon=len(idx),
                charge_mw_schedule=chg_max[idx],
                trade_cost_per_mwh=0.0 if price_seeking else RECOVERY_TRADE_COST_GBP_PER_MWH,
                reserve_dis_mw=reserve_out[idx],
                reserve_chg_mw=reserve_in[idx],
            )

        out, into = delivered_out[i], delivered_in[i]
        if e_dis > 0 or e_chg > 0 or out > 0 or into > 0:
            energy_rows.append({
                "date":                      d,
                "imbalance_revenue_gbp":     actual[i] * (e_dis - e_chg) if (e_dis > 0 or e_chg > 0) else 0.0,
                "cycling_cost_gbp":          wear * e_dis,
                "mwh_cycled":                e_dis,
                "delivery_mwh":              out,
                "delivery_cycling_cost_gbp": wear * out,
            })
        soc = float(np.clip(soc - e_dis + eta * e_chg - out + eta * into, 0.0, E))

        nxt = min(i + 1, n - 1)
        if nxt != i:
            settle_requirement(nxt)
        soc_traj.append((d, sp, soc / E, need_lo[nxt] / E, (E - need_hi[nxt]) / E))

    return energy_rows, soc_traj, breaches


# ---------------------------------------------------------------------------
# Full backtest runner
# ---------------------------------------------------------------------------

def run_strategy(
    auctions: pd.DataFrame,
    market_index: pd.DataFrame,
    battery: BatterySpec,
    forecast_prices_by_date: dict,
    services: list | None = None,
    start_date=None,
    end_date=None,
    *,
    initial_soc_frac: float = 0.5,
    horizon: int = 96,
    include_arbitrage: bool = True,
    pre_eac_rule: str = "d1",
    delivery: pd.DataFrame | None = None,
    offer_valuation: str = "formula",
    price_shrink: float = 1.0,
    early_forecast_prices_by_date: dict | None = None,
    offers_at_bid_time: bool = False,
    forecast_vintages: bool = False,
    price_shrink_by_date: dict | None = None,
    guard_low_by_date: dict | None = None,
    guard_high_by_date: dict | None = None,
) -> dict:
    """
    The shared engine behind every strategy: schedule, dispatch, settle.

    Only the price signal differs between perfect foresight, naive and ML. It
    sets the arbitrage opportunity cost in the schedule and drives dispatch;
    revenue is always realised against actual prices.

    include_arbitrage=False is a site with no interest in wholesale arbitrage:
    its offers give arbitrage no value and its dispatch trades only to keep its
    contracts deliverable. It uses no price signal, so every strategy gives the
    same result.

    With a `delivery` table the contracts are called on as GB frequency actually
    moved, and the expected cost of that energy is priced into each offer.
    Without one they are never called on.

    offer_valuation and price_shrink choose how offers price the trading they
    give up; see OFFER_VALUATIONS and _Scheduler.

    early_forecast_prices_by_date is each day's forecast as it stood before the
    previous day ended, built from data to D-2. offers_at_bid_time gives it to
    the offers for D, which close at 14:00 on D-1; forecast_vintages gives it to
    dispatch for tomorrow's periods and ends each plan after tomorrow (see
    run_dispatch). Perfect foresight passes none: its prices are known either way,
    and vintages then only shorten the horizon, keeping the engine identical
    across strategies.
    """
    services = ALL_SERVICES if services is None else list(services)
    apx_by_date = _apx_by_date(market_index)
    dates = [d for d in sorted(apx_by_date) if _in_range(d, start_date, end_date)]
    if not include_arbitrage:
        forecast_prices_by_date, early_forecast_prices_by_date = {}, None

    has_delivery = delivery is not None and not delivery.empty
    scheduler = _Scheduler(
        auctions, battery, forecast_prices_by_date, services,
        include_arbitrage=include_arbitrage, pre_eac_rule=pre_eac_rule,
        expected_delivery=_expected_delivery(delivery) if has_delivery else None,
        expected_prices=_expected_block_prices(market_index) if has_delivery else None,
        offer_valuation=offer_valuation, price_shrink=price_shrink,
        offer_forecast_prices_by_date=early_forecast_prices_by_date if offers_at_bid_time else None,
        price_shrink_by_date=price_shrink_by_date if include_arbitrage else None,
        guard_low_by_date=guard_low_by_date if include_arbitrage else None,
        guard_high_by_date=guard_high_by_date if include_arbitrage else None,
    )
    if dates:
        energy_rows, soc_traj, breaches = run_dispatch(
            apx_by_date, battery, dates, forecast_prices_by_date, scheduler=scheduler,
            initial_soc_frac=initial_soc_frac, horizon=horizon,
            delivery=delivery if has_delivery else None,
            price_seeking=include_arbitrage,
            forecast_vintages=forecast_vintages,
            early_forecast_prices_by_date=early_forecast_prices_by_date,
        )
    else:
        energy_rows, soc_traj, breaches = [], [], {}
    schedule = scheduler.schedule()

    availability = {k: 1.0 - v / _SETTLEMENT_PERIODS_PER_BLOCK for k, v in breaches.items()}
    anc = calc_ancillary_revenue(auctions, schedule, availability)
    if not anc.empty:
        anc_wide = anc.pivot_table(index="month", columns="service", values="revenue_gbp", fill_value=0)
        anc_wide.columns = [f"{c}_rev" for c in anc_wide.columns]
    else:
        anc_wide = pd.DataFrame()

    if energy_rows:
        daily = pd.DataFrame(energy_rows)
        daily["month"] = pd.to_datetime(daily["date"]).dt.to_period("M")
        imb_wide = daily.groupby("month")[["imbalance_revenue_gbp", "cycling_cost_gbp", "mwh_cycled",
                                          "delivery_mwh", "delivery_cycling_cost_gbp"]].sum()
    else:
        imb_wide = pd.DataFrame()

    if not schedule.empty:
        low = schedule[[f"q_{p}" for p in PRODUCTS if is_low(p)]].sum(axis=1)
        high = schedule[[f"q_{p}" for p in PRODUCTS if not is_low(p)]].sum(axis=1)
        committed = np.maximum(low, high)
        avg_fr_mw = float(committed.mean())
        avg_arb_mw = float(np.minimum(schedule["discharge_max_mw"], schedule["charge_max_mw"]).mean())
        blocks_committed = int((committed > 1e-9).sum())
    else:
        avg_fr_mw, avg_arb_mw, blocks_committed = 0.0, battery.power_mw, 0

    extras = {
        "pre_eac_rule":        pre_eac_rule,
        "include_arbitrage":   include_arbitrage,
        "fr_blocks_committed": blocks_committed,
        "soe_breach_periods":  int(sum(breaches.values())),
        "soe_breach_blocks":   len(breaches),
        "auction_share_cap":   AUCTION_SHARE_CAP,
        "delivery_modelled":   has_delivery,
        "offer_valuation":     offer_valuation,
        "price_shrink":        price_shrink,
        "offers_at_bid_time":  offers_at_bid_time,
        "forecast_vintages":   forecast_vintages,
        "dynamic_shrink":      bool(price_shrink_by_date) and include_arbitrage,
        "guard_bands":         bool(guard_low_by_date) and include_arbitrage,
    }
    result = _build_result(anc_wide, imb_wide, battery, avg_fr_mw, avg_arb_mw, soc_traj, extras)
    result["schedule"] = schedule
    return result


def run_backtest(
    auctions: pd.DataFrame,
    market_index: pd.DataFrame,
    battery: BatterySpec,
    services: list = None,
    start_date=None,
    end_date=None,
    initial_soc_frac: float = 0.5,
    horizon: int = 96,
    *,
    include_arbitrage: bool = True,
    pre_eac_rule: str = "d1",
    delivery: pd.DataFrame | None = None,
    offer_valuation: str = "formula",
    price_shrink: float = 1.0,
    forecast_vintages: bool = False,
) -> dict:
    """
    Perfect-foresight revenue backtest: actual day-D prices are the signal.

    Parameters
    ----------
    auctions      : DataFrame from load_auctions()
    market_index  : DataFrame from load_market_index() — APXMIDP prices for trading.
    battery       : BatterySpec instance
    services      : products the battery may offer (default: all six; [] for arbitrage only)
    start_date    : inclusive start date (str or datetime)
    end_date      : inclusive end date (str or datetime)
    initial_soc_frac : starting state of energy as a fraction of battery.energy_mwh
    horizon       : MPC planning horizon in settlement periods (default 96 = 48h)
    include_arbitrage : False is a site with no interest in arbitrage, trading only to
                    keep its contracts deliverable
    pre_eac_rule  : "d1" — see compute_fr_schedule
    delivery      : response delivery table (response_delivery.parquet), or None to
                    leave contracts uncalled
    offer_valuation, price_shrink : how offers price trading; see run_strategy
    forecast_vintages : end each dispatch plan after tomorrow, as the forecast
                    strategies must; see run_strategy

    Returns
    -------
    dict with keys:
      'monthly'        : wide-format DataFrame, one row per month
      'summary'        : dict of aggregate stats
      'soc_trajectory' : DataFrame of TRAJECTORY_COLUMNS
      'schedule'       : DataFrame of per-block holdings (see compute_fr_schedule)
    """
    apx_by_date = _apx_by_date(market_index)
    forecast = {d: s for d, s in apx_by_date.items() if _in_range(d, start_date, end_date)}
    return run_strategy(
        auctions, market_index, battery, forecast, services, start_date, end_date,
        initial_soc_frac=initial_soc_frac, horizon=horizon,
        include_arbitrage=include_arbitrage, pre_eac_rule=pre_eac_rule, delivery=delivery,
        offer_valuation=offer_valuation, price_shrink=price_shrink,
        forecast_vintages=forecast_vintages,
    )


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity_table(
    auctions: pd.DataFrame,
    market_index: pd.DataFrame,
    base_spec: BatterySpec,
    power_range: list = None,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """
    Run the backtest across a range of battery sizes, holding other parameters fixed.
    Returns a DataFrame suitable for display as a summary table.

    Each battery size is allocated and dispatched independently. Sizes above
    100 MW run into the per-product Maximum Sell Size, and stop being plausible
    as price-takers well before that.
    """
    if power_range is None:
        power_range = [10, 25, 50, 100]

    rows = []
    for mw in power_range:
        spec = BatterySpec(
            power_mw=mw,
            duration_h=base_spec.duration_h,
            efficiency_rt=base_spec.efficiency_rt,
            cycling_cost_per_mwh=base_spec.cycling_cost_per_mwh,
        )
        result = run_backtest(
            auctions, market_index, spec,
            start_date=start_date, end_date=end_date,
        )
        s = result["summary"]
        rows.append({
            "Power (MW)":              mw,
            "Energy (MWh)":            round(mw * base_spec.duration_h, 0),
            "Total Net Revenue (£k)":  round(s.get("total_net", 0) / 1_000, 1),
            "Ann. Net Revenue (£k/yr)": round(s.get("annualised_net", 0) / 1_000, 1),
            "Revenue / MW (£k/MW/yr)": round(s.get("annualised_per_mw", 0) / 1_000, 1),
        })

    return pd.DataFrame(rows)
