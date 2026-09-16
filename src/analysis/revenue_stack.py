"""
BESS Revenue Stack Backtester
==============================
Models the revenue a grid-scale battery could have earned from:

  1. Dynamic Response availability payments (DC, DM and DR, High and Low).
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
  - the Reserved Capacity rule applies from its codification (15 Nov 2024).

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

Key assumptions:
  - The battery is a price-taker: its bids do not move clearing prices.
  - It offers each product at its opportunity cost, so it holds a product only
    when the clearing price beats the arbitrage value the capacity would forgo.
  - Frequency response activations are not simulated, so the state-of-energy
    requirement is held at its start-of-block value for the whole block.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analysis.fr_allocation import allocate_day, choose_family
from src.analysis.neso_rules import (
    EAC_BID_CLOSE,
    FAMILIES,
    PRE_EAC_BID_CLOSE,
    PRODUCTS,
    arbitrage_power_limits_mw,
    is_low,
    reserve_rule_applies,
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

# How a pre-EAC unit picks its one service per block. "d1" is the default;
# "always_dc" mirrors what most of the 2022-23 fleet did and is reported as a
# sensitivity (see methodology).
PRE_EAC_RULES = ("d1", "always_dc")

_SETTLEMENT_PERIODS_PER_BLOCK = 8

TRAJECTORY_COLUMNS = ["date", "sp", "soc_frac", "soc_min_frac", "soc_max_frac"]

# Every cached table carries the same columns, whichever streams a scenario has
REVENUE_COLUMNS = [f"{p}_rev" for p in PRODUCTS] + ["imbalance_revenue_gbp", "cycling_cost_gbp", "mwh_cycled"]


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
    cost_cols = [c for c in monthly.columns if c == "cycling_cost_gbp"]

    # Apply availability factor to every revenue stream and cycling cost proportionally.
    # This models the fraction of committed periods/days where the asset is actually
    # available — accounting for planned maintenance, unplanned faults, and curtailment.
    for col in rev_cols + cost_cols:
        monthly[col] = monthly[col] * battery.availability_factor
    if "mwh_cycled" in monthly.columns:
        monthly["mwh_cycled"] = monthly["mwh_cycled"] * battery.availability_factor

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
    services it may offer, and whether the Reserved Capacity rule binds, follow
    the service day's date. Decisions are kept, because the previous day's
    commitments limit how state of energy can move before the next day starts.
    """

    def __init__(self, auctions, battery, forecast_prices_by_date=None, services=None, *,
                 include_arbitrage=True, pre_eac_rule="d1"):
        if pre_eac_rule not in PRE_EAC_RULES:
            raise ValueError(f"pre_eac_rule must be one of {PRE_EAC_RULES}, got {pre_eac_rule!r}")
        services = ALL_SERVICES if services is None else list(services)
        self.table = _clearing_prices(auctions, services)
        self.forecast = {pd.Timestamp(d).normalize(): s for d, s in (forecast_prices_by_date or {}).items()}
        self.battery = battery
        self.include_arbitrage = include_arbitrage
        self.pre_eac_rule = pre_eac_rule
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
        apply_reserve = reserve_rule_applies(day)

        blocks, labels = [], []
        for efa in range(1, 7):
            prices = self.table.get((service_date, efa), {})
            arb_value = (
                _shadow_arb_value_per_mw(_efa_prices(self.forecast, service_date, efa), b)
                if self.include_arbitrage else 0.0
            )
            if splitting_allowed(day):
                families, label = FAMILIES, "ALL"
            elif self.pre_eac_rule == "always_dc":
                families, label = ("DC",), "DC"
            else:
                reference = self.table.get((service_date - pd.Timedelta(days=1), efa), {})
                chosen = choose_family(reference, arb_value, b.power_mw, b.energy_mwh, b.duration_h,
                                       apply_reserve=apply_reserve)
                families, label = ((chosen,), chosen) if chosen else ((), "none")
            blocks.append({"prices": prices, "arb_value": arb_value, "families": families})
            labels.append(label)

        out = allocate_day(blocks, b.power_mw, b.energy_mwh, b.duration_h, b.efficiency_rt,
                           soc_now_mwh, apply_reserve=apply_reserve,
                           lead_in=self._lead_in(service_date, now))

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
    Per-EFA-block holdings in each Dynamic Response product, without dispatch.

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
    pre_eac_rule : "d1" or "always_dc"
        How a pre-EAC unit picks its one service. "d1" chooses on the previous
        service day's clearing prices for the same block, the latest a bidder
        could know; "always_dc" mirrors the 2022-23 fleet's habit.
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
) -> tuple[list, list, dict]:
    """
    Rolling MPC dispatch over every settlement period, around FR commitments.

    Each service day's commitments enter at its bid deadline: decided there and
    then by `scheduler`, from the state of energy dispatch has left, or read
    from a fixed `schedule`. Until then the battery plans as if uncommitted, as
    an operator would have to. Each period's LP trades only the power the
    commitments leave on each side and holds state of energy inside the range
    they require.

    Dispatch decisions are executed at actual prices. Per-period revenue can be
    negative when forecast error causes an unfavourable trade — that is the
    realistic operational outcome and is intentional.

    Returns
    -------
    arb_rows : list of {date, imbalance_revenue_gbp, cycling_cost_gbp, mwh_cycled}
    soc_trajectory : list of (date, sp, soc_frac, soc_min_frac, soc_max_frac) after
        each period, with the range state of energy must be inside at the start of the next
    breaches : {(date, efa): settlement periods started outside the required range}
    """
    from src.optimisation.mpc import solve_mpc

    if (scheduler is None) == (schedule is None):
        raise ValueError("pass exactly one of scheduler or schedule")

    P, E = battery.power_mw, battery.energy_mwh
    dates = [pd.Timestamp(d).normalize() for d in dates]
    periods = [(d, sp) for d in dates for sp in range(1, 49)]
    n = len(periods)
    position = {p: i for i, p in enumerate(periods)}
    forecast = {pd.Timestamp(d).normalize(): s for d, s in forecast_prices_by_date.items()}
    fixed = schedule.to_dict("index") if schedule is not None else {}

    actual  = np.full(n, np.nan)
    predict = np.full(n, np.nan)
    soc_lo  = np.zeros(n)
    soc_hi  = np.full(n, E)
    dis_max = np.full(n, P)
    chg_max = np.full(n, P)
    held    = np.zeros(n, dtype=bool)
    blocks  = []
    block_periods = defaultdict(list)

    for i, (d, sp) in enumerate(periods):
        a = apx_by_date.get(d)
        if a is not None and sp in a.index:
            actual[i] = float(a.loc[sp])
        f = forecast.get(d)
        if f is not None and sp in f.index:
            predict[i] = float(f.loc[sp])
        key = _period_block(d, sp)
        blocks.append(key)
        block_periods[key].append(i)

    # A day's commitments are known from its bid deadline, or from its own first
    # period when the deadline falls before the backtest or on a day with no prices.
    known_at = defaultdict(list)
    for day in dates:
        close = _bid_close(day)
        known_at[position.get((close.normalize(), _settlement_period(close)), position[(day, 1)])].append(day)

    tol_mwh = 1e-3
    soc = initial_soc_frac * E
    arb_rows, soc_traj, breaches = [], [], {}

    def record(i, d, sp):
        nxt = min(i + 1, n - 1)
        soc_traj.append((d, sp, soc / E, soc_lo[nxt] / E, soc_hi[nxt] / E))

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
                dis_max[idx], chg_max[idx] = row["discharge_max_mw"], row["charge_max_mw"]
                held[idx] = sum(row[f"q_{p}"] for p in PRODUCTS) > 1e-9

        # Availability is judged at the start of each settlement period
        if held[i] and (soc < soc_lo[i] - tol_mwh or soc > soc_hi[i] + tol_mwh):
            breaches[blocks[i]] = breaches.get(blocks[i], 0) + 1

        if np.isnan(actual[i]):
            record(i, d, sp)
            continue

        h_end = min(i + horizon, n)
        idx = np.arange(i, h_end)
        end = min(h_end, n - 1)

        e_dis, e_chg = solve_mpc(
            soc_current=soc,
            price_forecast=np.nan_to_num(predict[idx], nan=0.0),
            arb_mw_schedule=dis_max[idx],
            soc_min=np.append(soc_lo[idx], soc_lo[end]),
            soc_max=np.append(soc_hi[idx], soc_hi[end]),
            energy_mwh=E,
            efficiency_rt=battery.efficiency_rt,
            cycling_cost_per_mwh=battery.cycling_cost_per_mwh,
            horizon=len(idx),
            charge_mw_schedule=chg_max[idx],
        )

        if e_dis > 0 or e_chg > 0:
            price = actual[i]
            arb_rows.append({
                "date":                  d,
                "imbalance_revenue_gbp": price * e_dis - price * e_chg,
                "cycling_cost_gbp":      battery.cycling_cost_per_mwh * e_dis,
                "mwh_cycled":            e_dis,
            })
            soc = float(np.clip(soc - e_dis + e_chg * battery.efficiency_rt, 0.0, E))

        record(i, d, sp)

    return arb_rows, soc_traj, breaches


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
) -> dict:
    """
    The shared engine behind every strategy: schedule, dispatch, settle.

    Only the price signal differs between perfect foresight, naive and ML. It
    sets the arbitrage opportunity cost in the schedule and drives dispatch;
    revenue is always realised against actual prices.
    """
    services = ALL_SERVICES if services is None else list(services)

    if include_arbitrage:
        apx_by_date = _apx_by_date(market_index)
        dates = [d for d in sorted(apx_by_date) if _in_range(d, start_date, end_date)]
    else:
        days = pd.to_datetime(auctions["EFA Date"]).dt.normalize().unique()
        dates = [pd.Timestamp(d) for d in sorted(days) if _in_range(pd.Timestamp(d), start_date, end_date)]

    if include_arbitrage and dates:
        scheduler = _Scheduler(auctions, battery, forecast_prices_by_date, services,
                               include_arbitrage=True, pre_eac_rule=pre_eac_rule)
        arb_rows, soc_traj, breaches = run_dispatch(
            apx_by_date, battery, dates, forecast_prices_by_date, scheduler=scheduler,
            initial_soc_frac=initial_soc_frac, horizon=horizon,
        )
        schedule = scheduler.schedule()
    else:
        # FR only: state of energy follows the allocation's own plan, so there
        # are no breaches and no trading stream. Documented in the methodology.
        schedule = compute_fr_schedule(
            auctions, battery, dates, None, services, include_arbitrage=False,
            pre_eac_rule=pre_eac_rule, initial_soc_frac=initial_soc_frac,
        )
        arb_rows, soc_traj, breaches = [], [], {}

    availability = {k: 1.0 - v / _SETTLEMENT_PERIODS_PER_BLOCK for k, v in breaches.items()}
    anc = calc_ancillary_revenue(auctions, schedule, availability)
    if not anc.empty:
        anc_wide = anc.pivot_table(index="month", columns="service", values="revenue_gbp", fill_value=0)
        anc_wide.columns = [f"{c}_rev" for c in anc_wide.columns]
    else:
        anc_wide = pd.DataFrame()

    if arb_rows:
        daily = pd.DataFrame(arb_rows)
        daily["month"] = pd.to_datetime(daily["date"]).dt.to_period("M")
        imb_wide = daily.groupby("month").agg(
            imbalance_revenue_gbp=("imbalance_revenue_gbp", "sum"),
            cycling_cost_gbp=("cycling_cost_gbp", "sum"),
            mwh_cycled=("mwh_cycled", "sum"),
        )
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
) -> dict:
    """
    Perfect-foresight revenue backtest: actual day-D prices are the signal.

    Parameters
    ----------
    auctions      : DataFrame from load_auctions()
    market_index  : DataFrame from load_market_index() — used for APXMIDP arbitrage.
    battery       : BatterySpec instance
    services      : products the battery may offer (default: all six; [] for arbitrage only)
    start_date    : inclusive start date (str or datetime)
    end_date      : inclusive end date (str or datetime)
    initial_soc_frac : starting state of energy as a fraction of battery.energy_mwh
    horizon       : MPC planning horizon in settlement periods (default 96 = 48h)
    include_arbitrage : False runs the FR-only scenario, with no dispatch
    pre_eac_rule  : "d1" (default) or "always_dc" — see compute_fr_schedule

    Returns
    -------
    dict with keys:
      'monthly'        : wide-format DataFrame, one row per month
      'summary'        : dict of aggregate stats
      'soc_trajectory' : DataFrame of TRAJECTORY_COLUMNS, or None for FR only
      'schedule'       : DataFrame of per-block holdings (see compute_fr_schedule)
    """
    apx_by_date = _apx_by_date(market_index)
    forecast = {d: s for d, s in apx_by_date.items() if _in_range(d, start_date, end_date)}
    return run_strategy(
        auctions, market_index, battery, forecast, services, start_date, end_date,
        initial_soc_frac=initial_soc_frac, horizon=horizon,
        include_arbitrage=include_arbitrage, pre_eac_rule=pre_eac_rule,
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
