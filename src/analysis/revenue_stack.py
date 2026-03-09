"""
BESS Revenue Stack Backtester
==============================
Models the historical revenue a BESS asset could have earned from:

  1. Ancillary service availability payments (DC High/Low, DR High/Low, DM High/Low)
     Revenue = clearing_price (£/MW/h) × committed_MW × hours_per_EFA_block
     Services are treated as stackable: DCH + DRH + DMH bid the same physical MW
     against different frequency response speeds — this is valid for the GB market.

  2. Wholesale energy arbitrage
     A daily intrinsic model: charge in the N cheapest settlement periods,
     discharge in the N most expensive, constrained by battery duration.
     Uses the APXMIDP market index price as the spot price reference — this
     is the actual GB day-ahead/intraday settlement reference price and a
     significantly more realistic proxy than the imbalance settlement price (SSP),
     which can reach extreme negative values during high-renewable periods.
     Only executes a cycle on days where the net profit exceeds zero after
     accounting for round-trip efficiency losses and cycling wear cost.

Key assumptions:
  - Battery maintains sufficient SoC headroom to provide both H (discharge)
    and L (charge) ancillary services simultaneously.
  - Ancillary cycling cost (energy delivery during frequency events) is minor
    relative to availability payments and is not separately modelled.
  - Cycling wear cost applies to imbalance arbitrage trades only.
  - One arbitrage cycle per day maximum (conservative).
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np

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

# FR SoC band: battery must stay within [lower, upper] × energy_mwh
# to maintain headroom for both DC High (discharge) and DC Low (charge) delivery.
FR_SOC_LOWER = 0.10
FR_SOC_UPPER = 0.90


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
    return pd.concat(slices)

def _build_arb_mw_by_period(arb_sched_map: dict) -> dict:
    """
    Expand an EFA-block arb allocation map to individual settlement periods.

    Parameters
    ----------
    arb_sched_map : dict
        {(date, efa): arb_mw} — output of the arb schedule computation.

    Returns
    -------
    dict mapping (date, settlement_period) → arb_mw (float).
    EFA 1's D-1 periods (47, 48) are correctly attributed to their calendar date.
    """
    result = {}
    one_day = pd.Timedelta(days=1)
    for (date, efa), mw in arb_sched_map.items():
        for sp in EFA_PERIODS[efa]["curr"]:
            result[(date, sp)] = mw
        for sp in EFA_PERIODS[efa]["prev"]:
            result[(date - one_day, sp)] = mw
    return result


def _run_mpc_dispatch(
    all_periods: list,
    actual_by_period: dict,
    forecast_by_period: dict,
    arb_mw_by_period: dict,
    battery: "BatterySpec",
    initial_soc_frac: float = 0.5,
    horizon: int = 96,
) -> tuple[list, list]:
    """
    Rolling MPC dispatch loop — re-solves the LP at every 30-min settlement period.

    The LP plans over the next `horizon` periods and returns only the first
    period's decision. The FR SoC band [FR_SOC_LOWER, FR_SOC_UPPER] is enforced
    as a hard constraint on all H+1 SoC variables, forcing the battery to
    pre-condition its SoC for future FR delivery obligations.

    Dispatch decisions are executed unconditionally at actual prices. Per-period
    revenue can be negative when forecast error causes an unfavourable trade —
    this is the realistic operational outcome and is intentional.

    Approximation: rolling horizon is not globally optimal. A single LP over the
    full backtest period would be, but rolling MPC reflects real operational
    constraints (decision made before future prices are known).

    Parameters
    ----------
    all_periods : list of (date, sp) tuples
        Chronologically ordered settlement periods to iterate over.
    actual_by_period : dict
        {(date, sp): actual_price £/MWh} — used for revenue settlement.
    forecast_by_period : dict
        {(date, sp): forecast_price £/MWh} — used for LP planning.
        Equal to actual_by_period for the perfect-foresight strategy.
        Missing periods default to 0.0 (LP treats them as zero-revenue).
    arb_mw_by_period : dict
        {(date, sp): arb_mw} — residual MW available after FR commitment.
    battery : BatterySpec
    initial_soc_frac : float
        Starting SoC fraction (default 0.5 = 50%).
    horizon : int
        LP planning horizon in settlement periods (default 96 = 48 hours).

    Returns
    -------
    arb_rows : list of dicts [{date, imbalance_revenue_gbp, cycling_cost_gbp}]
    soc_trajectory : list of (date, sp, soc_frac) tuples
    """
    from src.optimisation.mpc import solve_mpc

    n = len(all_periods)
    actual_arr   = np.array([actual_by_period.get(p, np.nan)   for p in all_periods])
    forecast_arr = np.array([forecast_by_period.get(p, np.nan) for p in all_periods])
    arb_mw_arr   = np.array([arb_mw_by_period.get(p, 0.0)      for p in all_periods])

    soc     = initial_soc_frac * battery.energy_mwh
    soc_min = FR_SOC_LOWER * battery.energy_mwh
    soc_max = FR_SOC_UPPER * battery.energy_mwh

    arb_rows      = []
    soc_trajectory = []

    for i, (date, sp) in enumerate(all_periods):
        soc_frac = soc / battery.energy_mwh

        if np.isnan(actual_arr[i]):
            soc_trajectory.append((date, sp, soc_frac))
            continue

        h_end = min(i + horizon, n)
        fc  = np.where(np.isnan(forecast_arr[i:h_end]), 0.0, forecast_arr[i:h_end])
        mw  = arb_mw_arr[i:h_end]

        e_dis, e_chg = solve_mpc(
            soc_current=soc,
            price_forecast=fc,
            arb_mw_schedule=mw,
            soc_min=soc_min,
            soc_max=soc_max,
            energy_mwh=battery.energy_mwh,
            efficiency_rt=battery.efficiency_rt,
            cycling_cost_per_mwh=battery.cycling_cost_per_mwh,
            horizon=len(fc),
        )

        if e_dis > 0 or e_chg > 0:
            actual_p = actual_arr[i]
            gross    = actual_p * e_dis - actual_p * e_chg
            wear     = battery.cycling_cost_per_mwh * e_dis
            soc      = float(np.clip(
                soc - e_dis + e_chg * battery.efficiency_rt,
                0, battery.energy_mwh,
            ))
            arb_rows.append({
                "date":                  date,
                "imbalance_revenue_gbp": gross,
                "cycling_cost_gbp":      wear,
                "mwh_cycled":            e_dis,
            })
            soc_frac = soc / battery.energy_mwh

        soc_trajectory.append((date, sp, soc_frac))

    return arb_rows, soc_trajectory


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
    "Imbalance": "#C9400A",  # warm orange-red
    "Cycling cost": "#8B2020",  # dark red
}


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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _filter_dates(df: pd.DataFrame, date_col: str, start_date, end_date) -> pd.DataFrame:
    if start_date is not None:
        df = df[df[date_col] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df[date_col] <= pd.Timestamp(end_date)]
    return df


def _build_result(
    anc_wide: pd.DataFrame,
    imb_wide: pd.DataFrame,
    battery: "BatterySpec",
    avg_fr_mw: float,
    avg_arb_mw: float,
    soc_traj: list,
) -> dict:
    """
    Merge ancillary and arbitrage monthly streams, apply availability factor,
    compute net revenue columns, and return the standard result dict.

    Shared by run_backtest() and run_forecast_backtest() to eliminate duplicate
    logic. Handles the optional mwh_cycled column produced by MPC dispatch.

    Returns {"monthly": DataFrame, "summary": dict, "soc_trajectory": DataFrame | None}.
    """
    frames = [f for f in [anc_wide, imb_wide] if not f.empty]
    if not frames:
        return {"monthly": pd.DataFrame(), "summary": {}, "soc_trajectory": None}

    monthly = frames[0].join(frames[1:], how="outer").fillna(0).reset_index()
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
        label = col.replace("_rev", "") if col.endswith("_rev") else "Imbalance"
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

    soc_trajectory_df = (
        pd.DataFrame(soc_traj, columns=["date", "sp", "soc_frac"])
        if soc_traj else None
    )
    return {"monthly": monthly, "summary": summary, "soc_trajectory": soc_trajectory_df}


# ---------------------------------------------------------------------------
# Stream 1 — Ancillary service availability revenue
# ---------------------------------------------------------------------------

def calc_ancillary_revenue(
    auctions: pd.DataFrame,
    battery: BatterySpec,
    services: list,
    start_date=None,
    end_date=None,
    min_price: float = 0.0,
    fr_mw: float = None,
    fr_schedule: "pd.Series | None" = None,
) -> pd.DataFrame:
    """
    Monthly availability revenue from frequency response auctions.

    Negative clearing prices occur in oversupplied GB markets (notably DR High).
    The min_price floor (default 0.0) models an operator who sets a bid price
    floor and opts out of any EFA block clearing below that threshold — a
    rational strategy any real participant would adopt.

    Parameters
    ----------
    fr_mw : float, optional
        Fixed MW committed to FR for all days. Ignored when fr_schedule is
        provided. Defaults to battery.power_mw when both are None.
    fr_schedule : pd.Series, optional
        Per-day FR commitment (index = date, values = fr_mw). When provided,
        each auction record is scaled by the corresponding day's fr_mw value
        rather than a single fixed figure.

    Returns
    -------
    DataFrame with columns: [month (Period), service (str), revenue_gbp (float)]
    """
    df = _filter_dates(auctions.copy(), "EFA Date", start_date, end_date)
    df = df[df["Service"].isin(services)]
    df = df[df["Clearing Price"] >= min_price]

    if df.empty:
        return pd.DataFrame(columns=["month", "service", "revenue_gbp"])

    df = df.copy()

    if fr_schedule is not None:
        if isinstance(fr_schedule.index, pd.MultiIndex):
            # Per-(date, EFA block) schedule — join on both keys
            sched_map = {
                (pd.Timestamp(d).normalize(), int(e)): float(v)
                for (d, e), v in fr_schedule.items()
            }
            df["fr_mw_d"] = df.apply(
                lambda r: sched_map.get(
                    (pd.Timestamp(r["EFA Date"]).normalize(), int(r["EFA"])),
                    battery.power_mw,
                ),
                axis=1,
            )
        else:
            # Legacy daily schedule (date-only index)
            sched_map = {pd.Timestamp(k).normalize(): float(v) for k, v in fr_schedule.items()}
            df["fr_mw_d"] = (
                df["EFA Date"].dt.normalize()
                .map(sched_map)
                .fillna(battery.power_mw)
            )
        df["revenue_gbp"] = df["Clearing Price"] * df["fr_mw_d"] * EFA_HOURS
    else:
        if fr_mw is None:
            fr_mw = battery.power_mw
        if fr_mw == 0:
            return pd.DataFrame(columns=["month", "service", "revenue_gbp"])
        df["revenue_gbp"] = df["Clearing Price"] * fr_mw * EFA_HOURS

    df["month"] = df["EFA Date"].dt.to_period("M")

    return (
        df.groupby(["month", "Service"])["revenue_gbp"]
        .sum()
        .reset_index()
        .rename(columns={"Service": "service"})
    )


# ---------------------------------------------------------------------------
# Stream 2 — Imbalance / spot arbitrage revenue
# ---------------------------------------------------------------------------

def calc_imbalance_revenue(
    market_index: pd.DataFrame,
    battery: BatterySpec,
    start_date=None,
    end_date=None,
    arb_mw: float = None,
    arb_energy_mwh: float = None,
    arb_schedule: "pd.Series | None" = None,
    initial_soc_frac: float = 0.5,
    dispatch_method: str = "greedy",
    horizon: int = 96,
) -> tuple[pd.DataFrame, list]:
    """
    Monthly wholesale energy arbitrage revenue using an EFA-block dispatch model.

    Price reference: APXMIDP — the GB spot market settlement reference.
    N2EXMIDP is excluded (near-zero prices, unreliable signal).

    Each day is divided into 6 EFA blocks (4h each, 8 settlement periods each).
    For each block independently:
      - Identify the N cheapest periods → charge here
      - Identify the N most expensive periods → discharge here
      - Gross profit = avg_discharge × energy_out − avg_charge × energy_in
      - Deduct cycling wear cost = cycling_cost_per_mwh × energy_out
      - Only execute if gross profit > cycling wear cost AND SoC headroom allows

    SoC is tracked across all 6 blocks within a day and carried into the next day.
    The battery must stay within [FR_SOC_LOWER, FR_SOC_UPPER] × energy_mwh to
    maintain headroom for simultaneous DC High and DC Low delivery.

    Parameters
    ----------
    arb_mw : float, optional
        Fixed MW available for arbitrage. Ignored when arb_schedule is provided.
        Defaults to battery.power_mw when both are None.
    arb_energy_mwh : float, optional
        Fixed usable energy (MWh). Ignored when arb_schedule is provided.
    arb_schedule : pd.Series, optional
        Per-(date, efa) arb MW as a MultiIndex Series, or per-date Series (legacy).
        When provided, energy capacity is computed per block as arb_mw_d × duration_h.
    initial_soc_frac : float
        Starting SoC as a fraction of energy_mwh (default 0.5 = 50%).
    dispatch_method : str
        "greedy" (default) — EFA-block greedy N-period selection.
        "mpc" — rolling LP re-solved every settlement period; uses actual prices
        as the LP forecast (perfect-foresight case).
    horizon : int
        MPC planning horizon in settlement periods (default 96 = 48 hours).
        Ignored when dispatch_method="greedy".

    Returns
    -------
    (monthly_df, soc_trajectory) : tuple
        monthly_df : DataFrame [month, imbalance_revenue_gbp, cycling_cost_gbp]
            imbalance_revenue_gbp is GROSS profit before deducting cycling wear.
            With MPC dispatch, per-period revenue can be negative (forecast error).
        soc_trajectory : list of (date, sp, soc_frac) tuples — populated only
            when dispatch_method="mpc", empty list otherwise.
    """
    _empty = pd.DataFrame(columns=["month", "imbalance_revenue_gbp", "cycling_cost_gbp"])

    if arb_schedule is None:
        if arb_mw is None:
            arb_mw = battery.power_mw
        if arb_energy_mwh is None:
            arb_energy_mwh = battery.energy_mwh
        if arb_mw == 0 or arb_energy_mwh == 0:
            return _empty, []

    if market_index.empty:
        return _empty, []

    # Build a full apx_by_date lookup from the entire market_index (no date filter)
    # so that EFA 1's cross-day D-1 lookup succeeds on the first day of the period.
    apx_full = (
        market_index[market_index["dataProvider"] == "APXMIDP"]
        .copy()
        .assign(settlementDate=lambda d: pd.to_datetime(d["settlementDate"]).dt.normalize())
    )
    apx_full = apx_full[apx_full["settlementPeriod"] <= 48]  # drop DST extra periods
    apx_by_date = {
        date: grp.set_index("settlementPeriod")["price"]
        for date, grp in apx_full.groupby("settlementDate")
    }

    # Determine the sorted set of dates to iterate over (user-selected range only)
    apx_filt = _filter_dates(apx_full.copy(), "settlementDate", start_date, end_date)
    if apx_filt.empty:
        return _empty, []
    sorted_dates = sorted(apx_filt["settlementDate"].unique())

    n_periods = max(1, int(battery.duration_h * 2))

    # Build arb schedule map: {(date, efa): arb_mw} or {date: arb_mw} (legacy)
    use_efa_sched = False
    arb_sched_map = {}
    if arb_schedule is not None:
        if isinstance(arb_schedule.index, pd.MultiIndex):
            arb_sched_map = {
                (pd.Timestamp(d).normalize(), int(e)): float(v)
                for (d, e), v in arb_schedule.items()
            }
            use_efa_sched = True
        else:
            arb_sched_map = {
                pd.Timestamp(k).normalize(): float(v)
                for k, v in arb_schedule.items()
            }

    # ------------------------------------------------------------------
    # MPC dispatch path
    # ------------------------------------------------------------------
    if dispatch_method == "mpc":
        if not use_efa_sched:
            # MPC requires per-(date, efa) schedule; build a uniform one if needed
            flat_mw = arb_mw if arb_mw is not None else battery.power_mw
            arb_sched_map = {
                (date, efa): flat_mw
                for date in sorted_dates
                for efa in range(1, 7)
            }

        arb_mw_by_period = _build_arb_mw_by_period(arb_sched_map)
        all_periods = [(date, sp) for date in sorted_dates for sp in range(1, 49)]
        actual_by_period = {
            (d, int(sp)): float(price)
            for d, series in apx_by_date.items()
            for sp, price in series.items()
        }

        rows, soc_traj = _run_mpc_dispatch(
            all_periods=all_periods,
            actual_by_period=actual_by_period,
            forecast_by_period=actual_by_period,   # PF: actual prices as forecast
            arb_mw_by_period=arb_mw_by_period,
            battery=battery,
            initial_soc_frac=initial_soc_frac,
            horizon=horizon,
        )

        if not rows:
            return _empty, soc_traj

        daily = pd.DataFrame(rows)
        daily["month"] = pd.to_datetime(daily["date"]).dt.to_period("M")
        monthly_df = (
            daily.groupby("month")
            .agg(
                imbalance_revenue_gbp=("imbalance_revenue_gbp", "sum"),
                cycling_cost_gbp=("cycling_cost_gbp", "sum"),
            )
            .reset_index()
        )
        return monthly_df, soc_traj

    # ------------------------------------------------------------------
    # Greedy dispatch path (original)
    # ------------------------------------------------------------------
    soc     = initial_soc_frac * battery.energy_mwh
    soc_min = FR_SOC_LOWER * battery.energy_mwh
    soc_max = FR_SOC_UPPER * battery.energy_mwh
    rows    = []

    for date in sorted_dates:
        for efa in range(1, 7):
            efa_prices = _efa_prices(apx_by_date, date, efa)
            if len(efa_prices) < n_periods * 2:
                continue

            # Resolve arb MW for this block
            if arb_schedule is not None:
                if use_efa_sched:
                    arb_mw_d = arb_sched_map.get((date, efa), 0.0)
                else:
                    arb_mw_d = arb_sched_map.get(date, 0.0)
                if arb_mw_d <= 0:
                    continue
                nominal_energy_out = arb_mw_d * battery.duration_h
            else:
                nominal_energy_out = arb_energy_mwh

            # SoC-constrained energy bounds
            energy_out = min(nominal_energy_out, max(0.0, soc - soc_min))
            energy_in  = min(
                energy_out / battery.efficiency_rt,
                max(0.0, soc_max - soc),
            )
            if energy_out <= 0 or energy_in <= 0:
                continue

            avg_charge    = efa_prices.nsmallest(n_periods).mean()
            avg_discharge = efa_prices.nlargest(n_periods).mean()
            gross_profit  = avg_discharge * energy_out - avg_charge * energy_in
            cycling_wear  = battery.cycling_cost_per_mwh * energy_out

            if gross_profit > cycling_wear:
                soc = soc - energy_out + energy_in * battery.efficiency_rt
                soc = float(np.clip(soc, 0, battery.energy_mwh))
                rows.append({
                    "date":                  date,
                    "imbalance_revenue_gbp": gross_profit,
                    "cycling_cost_gbp":      cycling_wear,
                })

    if not rows:
        return _empty, []

    daily = pd.DataFrame(rows)
    daily["month"] = pd.to_datetime(daily["date"]).dt.to_period("M")

    return (
        daily.groupby("month")
        .agg(
            imbalance_revenue_gbp=("imbalance_revenue_gbp", "sum"),
            cycling_cost_gbp=("cycling_cost_gbp", "sum"),
        )
        .reset_index()
    ), []


# ---------------------------------------------------------------------------
# Full backtest runner
# ---------------------------------------------------------------------------

def run_backtest(
    auctions: pd.DataFrame,
    market_index: pd.DataFrame,
    battery: BatterySpec,
    services: list = None,
    start_date=None,
    end_date=None,
    fr_mw: float = None,
    initial_soc_frac: float = 0.5,
    dispatch_method: str = "greedy",
    horizon: int = 96,
) -> dict:
    """
    Run the full revenue stack backtest.

    Parameters
    ----------
    auctions      : DataFrame from load_auctions()
    market_index  : DataFrame from load_market_index() — used for APXMIDP spot arbitrage.
                    Pass an empty DataFrame to exclude the arbitrage stream.
    battery       : BatterySpec instance
    services      : list of service codes to include (default: all six)
    start_date    : inclusive start date (str or datetime)
    end_date      : inclusive end date (str or datetime)
    fr_mw         : float, optional
        MW committed to FR availability services for the entire period. When
        provided, a fixed split is applied — used by sensitivity_table and similar callers.
        When omitted (default), a per-EFA-block dynamic allocation is computed using
        the confirmed FR clearing price vs a perfect-foresight shadow arb estimate.
        Falls back to full FR commitment if market_index is empty.
    initial_soc_frac : float
        Starting SoC as a fraction of battery.energy_mwh (default 0.5 = 50%).
    dispatch_method : str
        "greedy" (default) or "mpc". See calc_imbalance_revenue for details.
    horizon : int
        MPC planning horizon in settlement periods (default 96 = 48h).

    Returns
    -------
    dict with keys:
      'monthly'        : wide-format DataFrame, one row per month
      'summary'        : dict of aggregate stats
      'soc_trajectory' : DataFrame [date, sp, soc_frac] when dispatch_method="mpc", else None
    """
    if services is None:
        services = ALL_SERVICES

    # --- Resolve FR/arbitrage split ---
    if fr_mw is not None:
        # Fixed allocation — used by sensitivity_table and similar callers
        fr_mw     = float(np.clip(fr_mw, 0, battery.power_mw))
        fr_sch    = None
        arb_sch   = None
        avg_fr_mw = fr_mw
    else:
        # Dynamic perfect-foresight allocation: use actual prices as the
        # allocation signal (confirmed FR clearing price vs shadow arb value)
        if not market_index.empty:
            apx = market_index[market_index["dataProvider"] == "APXMIDP"].copy()
            apx["settlementDate"] = pd.to_datetime(apx["settlementDate"]).dt.normalize()
            apx_filt = _filter_dates(apx, "settlementDate", start_date, end_date)
            pf_prices = {
                date: grp.set_index("settlementPeriod")["price"]
                for date, grp in apx_filt.groupby("settlementDate")
            }
            fr_sch    = compute_daily_fr_schedule(
                auctions, pf_prices, battery, services, start_date, end_date
            )
            # arb_sch is the complement of fr_sch, same MultiIndex
            arb_sch   = (battery.power_mw - fr_sch).clip(lower=0)
            avg_fr_mw = float(fr_sch.mean()) if len(fr_sch) > 0 else battery.power_mw
        else:
            # No market data: full FR commitment, no arbitrage
            fr_mw     = battery.power_mw
            fr_sch    = None
            arb_sch   = None
            avg_fr_mw = fr_mw
    avg_arb_mw = battery.power_mw - avg_fr_mw

    # --- Ancillary ---
    anc = calc_ancillary_revenue(
        auctions, battery, services, start_date, end_date,
        fr_schedule=fr_sch, fr_mw=fr_mw if fr_sch is None else None,
    )
    if not anc.empty:
        anc_wide = anc.pivot_table(
            index="month", columns="service", values="revenue_gbp", fill_value=0
        )
        anc_wide.columns = [f"{c}_rev" for c in anc_wide.columns]
    else:
        anc_wide = pd.DataFrame()

    # --- Arbitrage ---
    if arb_sch is not None:
        imb, soc_traj = calc_imbalance_revenue(
            market_index, battery, start_date, end_date,
            arb_schedule=arb_sch, initial_soc_frac=initial_soc_frac,
            dispatch_method=dispatch_method, horizon=horizon,
        )
    else:
        arb_mw_fixed         = battery.power_mw - (fr_mw if fr_mw is not None else battery.power_mw)
        arb_energy_mwh_fixed = arb_mw_fixed * battery.duration_h
        imb, soc_traj = calc_imbalance_revenue(
            market_index, battery, start_date, end_date,
            arb_mw=arb_mw_fixed, arb_energy_mwh=arb_energy_mwh_fixed,
            initial_soc_frac=initial_soc_frac,
            dispatch_method=dispatch_method, horizon=horizon,
        )
    if not imb.empty:
        imb_wide = imb.set_index("month")
    else:
        imb_wide = pd.DataFrame()

    # --- Merge streams, apply availability factor, compute summary ---
    return _build_result(anc_wide, imb_wide, battery, avg_fr_mw, avg_arb_mw, soc_traj)


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

    Each battery size uses the dynamic day-ahead allocation model, so the
    FR/arbitrage split is independently optimised for each size.
    """
    if power_range is None:
        power_range = [10, 25, 50, 100, 200]

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


# ---------------------------------------------------------------------------
# Day-ahead FR/arbitrage capacity allocator
# ---------------------------------------------------------------------------

def compute_daily_fr_schedule(
    auctions: pd.DataFrame,
    forecast_prices_by_date: dict,
    battery: BatterySpec,
    services: list = None,
    start_date=None,
    end_date=None,
) -> pd.Series:
    """
    Day-ahead FR/arbitrage capacity allocation optimiser — EFA block granularity.

    For each day D and each EFA block (1–6) with confirmed auction data, compares:

    FR value per MW (£/MW/block):
        Clearing price × EFA_HOURS for that specific EFA block and service set.
        EAC auctions clear day-ahead, so the clearing price for day D is known
        by end of D-1 — no look-ahead bias.

    Shadow arb value per MW (£/MW/block):
        Estimated net profit from one arbitrage cycle within the block's
        8 settlement periods using forecast prices:
        (avg_discharge − avg_charge / η − cycling_cost) × duration_h.

    Allocation rule (proportional, per block):
        fr_fraction = fr_value / (fr_value + arb_value)

    Capacity flows toward the better-paying stream each block without
    all-or-nothing switching. Blocks where arb is unprofitable receive
    full FR commitment.

    Parameters
    ----------
    forecast_prices_by_date : dict
        {date: pd.Series(index=settlementPeriod, values=price)} — the price
        signal for the shadow arb estimate. Pass actual day-D prices for
        perfect-foresight allocation; D-1 prices or ML predictions for
        realistic day-ahead allocation.
    battery : BatterySpec
    services : list of FR service codes to include (default: ALL_SERVICES)
    start_date, end_date : applied to auctions filter only

    Returns
    -------
    pd.Series with pd.MultiIndex of (date: pd.Timestamp, efa: int), values = fr_mw.
    """
    if services is None:
        services = ALL_SERVICES

    # Per-EFA-block FR value per MW from confirmed auction clearing prices.
    # EAC auctions clear day-ahead for each EFA block independently.
    df_a = _filter_dates(auctions.copy(), "EFA Date", start_date, end_date)
    df_a = df_a[df_a["Service"].isin(services)]
    df_a = df_a[df_a["Clearing Price"] >= 0.0]

    # Sum clearing prices (×EFA_HOURS) across services for each (EFA Date, EFA block)
    if not df_a.empty:
        df_a["EFA Date"] = pd.to_datetime(df_a["EFA Date"]).dt.normalize()
        efa_fr_value = (
            df_a.groupby(["EFA Date", "EFA"])["Clearing Price"]
            .apply(lambda x: (x * EFA_HOURS).sum())
        )  # MultiIndex (EFA Date, EFA) → £/MW/block
    else:
        efa_fr_value = pd.Series(dtype=float)

    # Build a full apx_by_date dict from forecast_prices_by_date so _efa_prices
    # can handle EFA 1's cross-day (D-1 periods 47–48) lookup.
    apx_by_date = {
        pd.Timestamp(d).normalize(): prices
        for d, prices in forecast_prices_by_date.items()
    }

    n_periods = max(1, int(battery.duration_h * 2))
    schedule  = {}

    all_dates = sorted({pd.Timestamp(d).normalize() for d in forecast_prices_by_date})

    for date_ts in all_dates:
        for efa in range(1, 7):
            # FR value per MW: confirmed clearing price for this (date, block)
            fr_value = float(efa_fr_value.get((date_ts, efa), 0.0))

            # Shadow arb value per MW: estimated from the block's forecast prices
            fp_block = _efa_prices(apx_by_date, date_ts, efa)
            if len(fp_block) >= n_periods * 2:
                avg_discharge = fp_block.nlargest(n_periods).mean()
                avg_charge    = fp_block.nsmallest(n_periods).mean()
                net_per_mw = (
                    avg_discharge
                    - avg_charge / battery.efficiency_rt
                    - battery.cycling_cost_per_mwh
                ) * battery.duration_h
                arb_value = max(0.0, net_per_mw)
            else:
                arb_value = 0.0

            total       = fr_value + arb_value
            fr_fraction = (fr_value / total) if total > 0 else 1.0
            schedule[(date_ts, efa)] = fr_fraction * battery.power_mw

    if not schedule:
        return pd.Series(dtype=float, name="fr_mw")

    idx = pd.MultiIndex.from_tuples(list(schedule.keys()), names=["date", "efa"])
    return pd.Series(list(schedule.values()), index=idx, name="fr_mw")
