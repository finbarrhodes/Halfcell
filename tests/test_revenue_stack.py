"""Revenue stack structure and availability revenue arithmetic."""
import pandas as pd
import pytest

from src.analysis.revenue_stack import (
    ALL_SERVICES, EFA_PERIODS, FR_SOC_LOWER, FR_SOC_UPPER,
    BatterySpec, REFERENCE_BATTERY, calc_ancillary_revenue,
)


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


def test_blocks_are_contiguous_and_ascending():
    ordered = [sp for b in sorted(EFA_PERIODS) for sp in EFA_PERIODS[b]["curr"]]
    assert ordered == list(range(1, 47))


# --- Service and battery definitions ------------------------------------

def test_services_are_high_low_pairs():
    assert len(ALL_SERVICES) == 6
    for family in ("DC", "DR", "DM"):
        assert f"{family}H" in ALL_SERVICES
        assert f"{family}L" in ALL_SERVICES


def test_fr_soc_band_is_a_valid_interval():
    assert 0 < FR_SOC_LOWER < FR_SOC_UPPER < 1


def test_reference_battery_energy_is_power_times_duration():
    assert REFERENCE_BATTERY.energy_mwh == pytest.approx(
        REFERENCE_BATTERY.power_mw * REFERENCE_BATTERY.duration_h
    )


def test_round_trip_efficiency_is_a_fraction_not_a_percentage():
    assert 0 < REFERENCE_BATTERY.efficiency_rt <= 1


# --- Availability revenue arithmetic ------------------------------------

def _auctions(price=10.0, days=31, services=("DCH",)):
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    return pd.DataFrame([
        {"EFA Date": d, "Service": s, "EFA": efa,
         "Clearing Price": price, "Cleared Volume": 100.0}
        for d in dates for s in services for efa in range(1, 7)
    ])


def _total(df):
    return df["revenue_gbp"].sum() if not df.empty else 0.0


def test_availability_revenue_matches_price_times_mw_times_hours():
    """revenue = price (£/MW/h) x MW x 4h per block x 6 blocks x days."""
    battery = BatterySpec(power_mw=10.0, duration_h=2.0, availability_factor=1.0)
    df = calc_ancillary_revenue(_auctions(price=10.0, days=31), battery, ["DCH"], fr_mw=10.0)
    assert _total(df) == pytest.approx(10.0 * 10.0 * 4 * 6 * 31, rel=1e-6)


def test_revenue_scales_linearly_with_committed_mw():
    battery = BatterySpec(power_mw=50.0, availability_factor=1.0)
    a = _total(calc_ancillary_revenue(_auctions(), battery, ["DCH"], fr_mw=10.0))
    b = _total(calc_ancillary_revenue(_auctions(), battery, ["DCH"], fr_mw=20.0))
    assert b == pytest.approx(a * 2, rel=1e-6)


def test_zero_fr_commitment_earns_nothing():
    battery = BatterySpec(power_mw=50.0)
    assert _total(calc_ancillary_revenue(_auctions(), battery, ["DCH"], fr_mw=0.0)) == 0.0


def test_output_is_long_format_by_month_and_service():
    battery = BatterySpec(power_mw=10.0)
    df = calc_ancillary_revenue(
        _auctions(services=("DCH", "DRL")), battery, ["DCH", "DRL"], fr_mw=10.0
    )
    assert list(df.columns) == ["month", "service", "revenue_gbp"]
    assert set(df["service"]) == {"DCH", "DRL"}


def test_unselected_services_contribute_nothing():
    battery = BatterySpec(power_mw=10.0)
    df = calc_ancillary_revenue(_auctions(services=("DCH", "DRL")), battery, ["DCH"], fr_mw=10.0)
    assert set(df["service"]) == {"DCH"}


def test_negative_clearing_prices_are_included_by_default():
    """Negative clearing is a real feature of an oversupplied flexibility market.

    ~13.5% of GB auction records in the dataset clear below zero, concentrated in
    DRH and DMH. Flooring them at zero overstated FR income by roughly 12% of
    total modelled revenue, so signed prices are now the default.
    """
    battery = BatterySpec(power_mw=10.0, availability_factor=1.0)
    total = _total(calc_ancillary_revenue(_auctions(price=-5.0), battery, ["DCH"], fr_mw=10.0))
    assert total == pytest.approx(-5.0 * 10.0 * 4 * 6 * 31, rel=1e-6)


def test_price_floor_is_opt_in():
    """min_price still available for an operator who bids a floor and opts out."""
    battery = BatterySpec(power_mw=10.0, availability_factor=1.0)
    df = calc_ancillary_revenue(
        _auctions(price=-5.0), battery, ["DCH"], fr_mw=10.0, min_price=0.0
    )
    assert _total(df) == 0.0


def test_date_filtering_restricts_the_period():
    battery = BatterySpec(power_mw=10.0)
    full = _total(calc_ancillary_revenue(_auctions(days=31), battery, ["DCH"], fr_mw=10.0))
    part = _total(calc_ancillary_revenue(
        _auctions(days=31), battery, ["DCH"], fr_mw=10.0,
        start_date="2026-01-01", end_date="2026-01-15",
    ))
    assert 0 < part < full


# --- FR/arbitrage allocation guard --------------------------------------

def _forecast(days=3, flat=50.0):
    """Flat forecast prices — no spread, so shadow arbitrage value is zero."""
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    return {d: pd.Series({sp: flat for sp in range(1, 49)}) for d in dates}


def _auctions_priced(price, days=3):
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    return pd.DataFrame([
        {"EFA Date": d, "Service": "DRH", "EFA": efa,
         "Clearing Price": price, "Cleared Volume": 100.0}
        for d in dates for efa in range(1, 7)
    ])


def test_allocation_never_commits_negative_mw():
    """A block whose services net out negative must not yield negative MW.

    The proportional split divides by (fr_value + arb_value); an unclamped
    negative fr_value gives a negative fraction, and the committed MW flows
    into the dispatch LP's power bounds.
    """
    from src.analysis.revenue_stack import compute_daily_fr_schedule

    battery = BatterySpec(power_mw=50.0)
    sched = compute_daily_fr_schedule(
        _auctions_priced(-5.0), _forecast(), battery, services=["DRH"]
    )
    assert (sched >= 0).all(), f"negative MW committed: min={sched.min()}"


def test_allocation_commits_nothing_to_fr_at_negative_prices():
    """Blocks clearing negative overall should be released to arbitrage."""
    from src.analysis.revenue_stack import compute_daily_fr_schedule

    battery = BatterySpec(power_mw=50.0)
    sched = compute_daily_fr_schedule(
        _auctions_priced(-5.0), _forecast(), battery, services=["DRH"]
    )
    assert (sched == 0).all(), (
        "a negative-clearing block was committed to FR — the degenerate branch "
        "used to commit 100% here"
    )


def test_allocation_is_bounded_by_installed_power():
    from src.analysis.revenue_stack import compute_daily_fr_schedule

    battery = BatterySpec(power_mw=50.0)
    for price in (-20.0, -0.01, 0.0, 0.01, 25.0):
        sched = compute_daily_fr_schedule(
            _auctions_priced(price), _forecast(), battery, services=["DRH"]
        )
        assert (sched >= 0).all() and (sched <= battery.power_mw + 1e-9).all(), (
            f"allocation out of bounds at clearing price {price}"
        )


def test_positive_prices_still_commit_to_fr():
    from src.analysis.revenue_stack import compute_daily_fr_schedule

    battery = BatterySpec(power_mw=50.0)
    sched = compute_daily_fr_schedule(
        _auctions_priced(20.0), _forecast(), battery, services=["DRH"]
    )
    assert (sched > 0).all()
