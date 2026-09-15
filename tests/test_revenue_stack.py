"""Revenue engine: EFA structure, the NESO-compliant schedule, availability and dispatch."""
import pandas as pd
import pytest

from src.analysis.neso_rules import PRODUCTS, is_feasible, splitting_allowed
from src.analysis.revenue_stack import (
    ALL_SERVICES,
    EFA_HOURS,
    EFA_PERIODS,
    REFERENCE_BATTERY,
    REVENUE_COLUMNS,
    TRAJECTORY_COLUMNS,
    BatterySpec,
    _bid_close,
    _block_start,
    _period_block,
    calc_ancillary_revenue,
    compute_fr_schedule,
    run_backtest,
    run_dispatch,
)

P, D = 50.0, 2.0
BATTERY = BatterySpec(power_mw=P, duration_h=D, availability_factor=1.0)


# --- EFA block structure -------------------------------------------------

def test_six_efa_blocks():
    assert sorted(EFA_PERIODS) == [1, 2, 3, 4, 5, 6]


def test_efa_blocks_tile_a_day_exactly_once():
    """Blocks partition all 48 periods: SP1-46 today, SP47-48 rolled into
    the next day's block 1 (which spans midnight)."""
    curr = [sp for b in EFA_PERIODS.values() for sp in b["curr"]]
    prev = [sp for b in EFA_PERIODS.values() for sp in b["prev"]]
    assert sorted(curr + prev) == list(range(1, 49))
    assert len(curr + prev) == len(set(curr + prev)), "a period is claimed by two blocks"


def test_each_block_is_four_hours():
    for block, spec in EFA_PERIODS.items():
        assert len(spec["curr"]) + len(spec["prev"]) == 8, f"EFA {block} is not 8 half-hours"


def test_only_block_one_spans_midnight():
    assert EFA_PERIODS[1]["prev"] == [47, 48]
    for block in (2, 3, 4, 5, 6):
        assert EFA_PERIODS[block]["prev"] == []


def test_late_periods_belong_to_the_next_service_days_first_block():
    day = pd.Timestamp("2026-01-01")
    assert _period_block(day, 1) == (day, 1)
    assert _period_block(day, 7) == (day, 2)
    assert _period_block(day, 46) == (day, 6)
    assert _period_block(day, 47) == (day + pd.Timedelta(days=1), 1)
    assert _block_start(day, 1) == pd.Timestamp("2025-12-31 23:00")


def test_offers_close_the_afternoon_before():
    """14:00 D-1 under EAC; 14:30 D-1 before it."""
    assert _bid_close(pd.Timestamp("2026-01-05")) == pd.Timestamp("2026-01-04 14:00")
    assert _bid_close(pd.Timestamp("2023-06-05")) == pd.Timestamp("2023-06-04 14:30")


# --- Service and battery definitions ------------------------------------

def test_services_are_high_low_pairs():
    assert len(ALL_SERVICES) == 6
    for fam in ("DC", "DR", "DM"):
        assert f"{fam}H" in ALL_SERVICES and f"{fam}L" in ALL_SERVICES


def test_reference_battery_energy_is_power_times_duration():
    assert REFERENCE_BATTERY.energy_mwh == pytest.approx(
        REFERENCE_BATTERY.power_mw * REFERENCE_BATTERY.duration_h
    )


def test_round_trip_efficiency_is_a_fraction_not_a_percentage():
    assert 0 < REFERENCE_BATTERY.efficiency_rt <= 1


# --- Synthetic inputs ----------------------------------------------------------

def _auctions(prices_by_day: dict) -> pd.DataFrame:
    """prices_by_day: {date str: {product: price}} applied to all six blocks."""
    return pd.DataFrame([
        {"EFA Date": pd.Timestamp(day), "Service": svc, "EFA": efa,
         "Clearing Price": price, "Cleared Volume": 100.0}
        for day, prices in prices_by_day.items()
        for svc, price in prices.items()
        for efa in range(1, 7)
    ])


def _market_index(days: list, low=40.0, high=160.0) -> pd.DataFrame:
    """Cheap first half of each day, expensive second half, so arbitrage is worth doing."""
    return pd.DataFrame([
        {"settlementDate": pd.Timestamp(day), "settlementPeriod": sp, "dataProvider": "APXMIDP",
         "price": low if sp <= 24 else high}
        for day in days for sp in range(1, 49)
    ])


def _schedule(auctions, days, **kwargs):
    return compute_fr_schedule(auctions, BATTERY, [pd.Timestamp(d) for d in days], **kwargs)


def _held(row) -> dict:
    return {p: row[f"q_{p}"] for p in PRODUCTS if row[f"q_{p}"] > 1e-9}


EVERYTHING_PAYS = {"DCH": 3.0, "DCL": 12.0, "DMH": 4.0, "DML": 8.0, "DRH": 6.0, "DRL": 15.0}


# --- The schedule obeys NESO's rules ---------------------------------------------

def test_every_block_is_neso_feasible_across_all_three_rule_eras():
    days = ["2023-06-01", "2023-06-02", "2024-11-14", "2024-11-15", "2026-01-01"]
    sched = _schedule(_auctions({d: EVERYTHING_PAYS for d in days}), days, include_arbitrage=False)
    for (date, _efa), row in sched.iterrows():
        assert is_feasible(_held(row), P, P * D, apply_reserve=row["apply_reserve"],
                           single_family=not splitting_allowed(date.date()), tol=1e-5), (date, row)


def test_negative_prices_commit_nothing():
    sched = _schedule(_auctions({"2026-01-01": {p: -5.0 for p in PRODUCTS}}), ["2026-01-01"],
                      include_arbitrage=False)
    assert (sched[[f"q_{p}" for p in PRODUCTS]] == 0).all().all()


def test_pre_eac_service_is_chosen_on_the_previous_days_prices():
    """DR was best yesterday; DC is best today. A bidder could only know yesterday's."""
    prices = {
        "2023-06-01": {"DCH": 1.0, "DCL": 1.0, "DMH": 1.0, "DML": 1.0, "DRH": 10.0, "DRL": 30.0},
        "2023-06-02": {"DCH": 1.0, "DCL": 40.0, "DMH": 1.0, "DML": 1.0, "DRH": 1.0, "DRL": 2.0},
    }
    row = _schedule(_auctions(prices), ["2023-06-02"], include_arbitrage=False).iloc[0]
    assert row["family"] == "DR"
    assert row["q_DCL"] == 0.0 and row["q_DRL"] > 0.0


def test_always_dc_rule_holds_dc_before_eac():
    prices = {"2023-06-01": EVERYTHING_PAYS, "2023-06-02": EVERYTHING_PAYS}
    row = _schedule(_auctions(prices), ["2023-06-02"], include_arbitrage=False,
                    pre_eac_rule="always_dc").iloc[0]
    assert row["family"] == "DC"
    assert row["q_DRL"] == 0.0 and row["q_DCL"] > 0.0


def test_unknown_pre_eac_rule_is_rejected():
    with pytest.raises(ValueError):
        _schedule(_auctions({"2023-06-01": EVERYTHING_PAYS}), ["2023-06-01"], pre_eac_rule="hindsight")


def test_reserve_rule_switches_on_at_codification():
    """Symmetric DM takes full power the day before 15 Nov 2024 and 41.67 MW from it."""
    dm = {"DMH": 10.0, "DML": 10.0}
    sched = _schedule(_auctions({"2024-11-14": dm, "2024-11-15": dm}), ["2024-11-14", "2024-11-15"],
                      include_arbitrage=False, services=["DMH", "DML"])
    before = sched.loc[pd.Timestamp("2024-11-14")].iloc[0]
    after = sched.loc[pd.Timestamp("2024-11-15")].iloc[0]
    assert before["q_DMH"] == pytest.approx(P) and not before["apply_reserve"]
    assert after["q_DMH"] == pytest.approx(P / 1.2, rel=1e-4) and after["apply_reserve"]


def test_arbitrage_only_scenario_holds_nothing_and_frees_the_whole_battery():
    sched = _schedule(_auctions({"2026-01-01": EVERYTHING_PAYS}), ["2026-01-01"],
                      include_arbitrage=False, services=[])
    assert (sched[[f"q_{p}" for p in PRODUCTS]] == 0).all().all()
    assert (sched["soc_min_mwh"] == 0).all() and (sched["soc_max_mwh"] == P * D).all()
    assert (sched["discharge_max_mw"] == P).all() and (sched["charge_max_mw"] == P).all()


# --- Availability revenue --------------------------------------------------------

def test_availability_revenue_is_price_times_mw_held_times_four_hours():
    auctions = _auctions({"2026-01-01": {"DCL": 10.0}})
    sched = _schedule(auctions, ["2026-01-01"], include_arbitrage=False, services=["DCL"])
    held = sched["q_DCL"].sum()
    total = calc_ancillary_revenue(auctions, sched)["revenue_gbp"].sum()
    assert total == pytest.approx(10.0 * held * EFA_HOURS)


def test_unavailable_periods_lose_their_share_of_the_payment():
    auctions = _auctions({"2026-01-01": {"DCL": 10.0}})
    sched = _schedule(auctions, ["2026-01-01"], include_arbitrage=False, services=["DCL"])
    full = calc_ancillary_revenue(auctions, sched)["revenue_gbp"].sum()
    day = pd.Timestamp("2026-01-01")
    half_of_one_block = calc_ancillary_revenue(auctions, sched, {(day, 3): 0.5})["revenue_gbp"].sum()
    assert half_of_one_block == pytest.approx(full - full / 6 / 2)


def test_output_is_long_format_by_month_and_service():
    auctions = _auctions({"2026-01-01": EVERYTHING_PAYS})
    sched = _schedule(auctions, ["2026-01-01"], include_arbitrage=False)
    df = calc_ancillary_revenue(auctions, sched)
    assert list(df.columns) == ["month", "service", "revenue_gbp"]


# --- Dispatch ----------------------------------------------------------------------

def _manual_schedule(day, *, block_limits, default_charge=50.0):
    rows = []
    for efa in range(1, 7):
        lo, held_mw = block_limits.get(efa, (0.0, 0.0))
        row = {"date": day, "efa": efa, "family": "ALL", "apply_reserve": True,
               "soc_min_mwh": lo, "soc_max_mwh": P * D, "discharge_max_mw": P,
               "charge_max_mw": default_charge, "arb_mw": 0.0}
        row.update({f"q_{p}": 0.0 for p in PRODUCTS})
        row["q_DRL"] = held_mw
        rows.append(row)
    return pd.DataFrame(rows).set_index(["date", "efa"])


def test_an_unreachable_requirement_is_recorded_as_unavailability():
    """Need 95 MWh in store from 11:00 but can only charge at 1 MW from empty."""
    day = pd.Timestamp("2026-03-02")
    sched = _manual_schedule(day, block_limits={4: (95.0, 40.0)}, default_charge=1.0)
    flat = {day: pd.Series(50.0, index=range(1, 49))}
    _, _, breaches = run_dispatch(flat, BATTERY, [day], flat, schedule=sched, initial_soc_frac=0.0)
    assert breaches.get((day, 4)) == 8


def test_tomorrows_commitments_are_not_anticipated_before_the_bid_deadline():
    """
    Tomorrow's 07:00 block needs 90 MWh in store. Offers close at 14:00 today,
    so the battery should not start charging for it before then - and should
    still arrive compliant.
    """
    day1, day2 = pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-03")
    sched = pd.concat([_manual_schedule(day1, block_limits={}),
                       _manual_schedule(day2, block_limits={3: (90.0, 40.0)})])
    flat = {d: pd.Series(50.0, index=range(1, 49)) for d in (day1, day2)}
    _, traj, breaches = run_dispatch(flat, BATTERY, [day1, day2], flat, schedule=sched,
                                     initial_soc_frac=0.0)
    soc = pd.DataFrame(traj, columns=TRAJECTORY_COLUMNS)
    before_deadline = soc[(soc.date == day1) & (soc.sp <= 28)]          # decisions up to 13:30
    assert before_deadline.soc_frac.max() == pytest.approx(0.0, abs=1e-6)
    assert (day2, 3) not in breaches
    # The trajectory carries the range the next period must start inside
    at_0630 = soc[(soc.date == day2) & (soc.sp == 14)].iloc[0]
    assert at_0630.soc_min_frac == pytest.approx(0.9) and at_0630.soc_frac >= 0.9 - 1e-5


def test_dispatch_needs_exactly_one_source_of_commitments():
    day = pd.Timestamp("2026-03-02")
    flat = {day: pd.Series(50.0, index=range(1, 49))}
    with pytest.raises(ValueError):
        run_dispatch(flat, BATTERY, [day], flat)


# --- End to end --------------------------------------------------------------------

DAYS = [f"2026-01-0{i}" for i in range(1, 6)]


def test_full_backtest_is_neso_feasible_and_never_out_of_position():
    result = run_backtest(_auctions({d: EVERYTHING_PAYS for d in DAYS}), _market_index(DAYS), BATTERY)
    s = result["summary"]
    assert not result["monthly"].empty
    assert {"soe_breach_periods", "pre_eac_rule", "fr_blocks_committed"} <= set(s)
    assert s["soe_breach_periods"] == 0
    for (date, _efa), row in result["schedule"].iterrows():
        assert is_feasible(_held(row), P, P * D, apply_reserve=row["apply_reserve"], tol=1e-5)


def test_holdings_that_freeze_state_of_energy_do_not_strand_the_battery():
    """
    31.25 MW DC High with 46.9 MW DR Low leaves no free power either way. Before
    the allocation planned state of energy across blocks, trading could leave the
    store outside that combination's range with no way back, losing whole blocks.
    """
    auctions = _auctions({d: {"DCH": 20.0, "DRL": 30.0} for d in DAYS})
    result = run_backtest(auctions, _market_index(DAYS, low=-20.0, high=300.0), BATTERY)
    assert result["summary"]["soe_breach_periods"] == 0
    assert result["schedule"]["q_DCH"].max() > 0


def test_fr_revenue_never_exceeds_selling_the_same_mw_into_every_product():
    auctions = _auctions({d: EVERYTHING_PAYS for d in DAYS})
    result = run_backtest(auctions, _market_index(DAYS), BATTERY, include_arbitrage=False)
    fr = sum(v for k, v in result["summary"]["breakdown"].items() if k != "Arbitrage")
    old = sum(EVERYTHING_PAYS.values()) * EFA_HOURS * 6 * len(DAYS) * P
    assert 0 < fr < old


def test_fr_only_scenario_has_no_trading_stream():
    days = DAYS[:3]
    result = run_backtest(_auctions({d: EVERYTHING_PAYS for d in days}), _market_index(days),
                          BATTERY, include_arbitrage=False)
    assert result["summary"]["breakdown"].get("Arbitrage", 0) == 0
    assert result["soc_trajectory"] is None


def test_every_scenario_writes_the_same_columns():
    auctions = _auctions({d: EVERYTHING_PAYS for d in DAYS[:2]})
    market = _market_index(DAYS[:2])
    for kwargs in ({}, {"include_arbitrage": False}, {"services": []}):
        monthly = run_backtest(auctions, market, BATTERY, **kwargs)["monthly"]
        assert set(REVENUE_COLUMNS) <= set(monthly.columns), kwargs
