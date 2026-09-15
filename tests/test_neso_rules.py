"""
NESO Dynamic Response rules, checked against NESO's own published examples.

Where a test reproduces a figure or worked example from a NESO document, the
docstring names it, so a failure points straight at the source to re-read.
"""
import pytest

from src.analysis.neso_rules import (
    DELIVERY_DURATION_H,
    ENERGY_RECOVERY_SHARE,
    FAMILIES,
    MAX_SELL_SIZE_MW,
    RESERVED_CAPACITY_SHARE,
    arbitrage_power_limits_mw,
    capacity_use_mw,
    is_feasible,
    response_energy_mwh,
    soc_bounds_mwh,
)


# --- Published parameters -------------------------------------------------

def test_delivery_durations_are_15_30_60_minutes():
    assert DELIVERY_DURATION_H == {"DC": 0.25, "DM": 0.5, "DR": 1.0}


def test_reserved_shares_match_nesos_derivation():
    """SoE guidance §3: share = duration_h x 20% recovery x 2."""
    for fam in FAMILIES:
        derived = DELIVERY_DURATION_H[fam] * ENERGY_RECOVERY_SHARE * 2
        assert RESERVED_CAPACITY_SHARE[fam] == pytest.approx(derived), fam


def test_release_3_worked_example():
    """Aug 2024 submission: a 20 MW DC contract has a 5 MWh REV and needs 2 MW reserved."""
    low, high = response_energy_mwh({"DCL": 20.0})
    assert (low, high) == (pytest.approx(5.0), 0.0)
    _, imp = capacity_use_mw({"DCL": 20.0})
    assert imp == pytest.approx(2.0)


# --- NESO SoE guidance §3, Figures 1 and 2 ---------------------------------

def test_figure_1_single_direction_needs_no_bid_direction_reserve():
    """50 MW of DM one way on a 50 MW / 50 MWh unit is allowed."""
    assert is_feasible({"DML": 50.0}, power_mw=50.0, energy_mwh=50.0)


def test_figure_2_bidirectional_dm_fits_at_40_mw():
    """40 MW of DM each way plus the 20% reserves fits a 50 MW / 50 MWh unit."""
    assert is_feasible({"DMH": 40.0, "DML": 40.0}, power_mw=50.0, energy_mwh=50.0)


def test_figure_2_limit_is_just_under_42_mw_each_way():
    """x + 0.2x <= 50 gives 41.67 MW, so 41 fits and 42 does not."""
    assert is_feasible({"DMH": 41.0, "DML": 41.0}, power_mw=50.0, energy_mwh=50.0)
    assert not is_feasible({"DMH": 42.0, "DML": 42.0}, power_mw=50.0, energy_mwh=50.0)


# --- The rule the old model broke ---------------------------------------------

def test_the_same_mw_cannot_be_sold_into_three_high_products():
    """The previous model booked 50 MW into DCH, DMH and DRH at once."""
    assert not is_feasible(
        {"DCH": 50.0, "DMH": 50.0, "DRH": 50.0}, power_mw=50.0, energy_mwh=100.0
    )


def test_splitting_one_direction_across_services_is_allowed():
    """EAC splitting: 20 + 15 + 5 MW across DC, DM and DR on a 50 MW unit."""
    assert is_feasible(
        {"DCH": 20.0, "DMH": 15.0, "DRH": 5.0}, power_mw=50.0, energy_mwh=100.0
    )


def test_capacity_is_checked_per_direction_not_pooled():
    """
    50 MW DCL plus 20 MW DCH needs 52 MW on the export side (50 + 10% of 20).

    A pooled reading (everything against 2x the rating) would allow it; a unit
    physically cannot export 52 MW from a 50 MW rating, so it must be refused.
    """
    export, _ = capacity_use_mw({"DCL": 50.0, "DCH": 20.0})
    assert export == pytest.approx(52.0)
    assert not is_feasible({"DCL": 50.0, "DCH": 20.0}, power_mw=50.0, energy_mwh=100.0)


def test_pre_eac_units_offer_one_service_per_window():
    q = {"DCL": 20.0, "DML": 20.0}
    assert is_feasible(q, power_mw=50.0, energy_mwh=100.0)
    assert not is_feasible(q, power_mw=50.0, energy_mwh=100.0, single_family=True)


def test_reserve_can_be_switched_off_for_eras_before_it_applied():
    q = {"DMH": 45.0, "DML": 45.0}
    assert not is_feasible(q, power_mw=50.0, energy_mwh=100.0)
    assert is_feasible(q, power_mw=50.0, energy_mwh=100.0, apply_reserve=False)


# --- Energy -----------------------------------------------------------------

def test_energy_binds_for_bidirectional_dr_on_a_one_hour_unit():
    """60-minute DR both ways needs REV in store and as headroom together."""
    assert is_feasible({"DRH": 25.0, "DRL": 25.0}, power_mw=50.0, energy_mwh=50.0)
    assert not is_feasible({"DRH": 26.0, "DRL": 26.0}, power_mw=50.0, energy_mwh=50.0)


def test_stacked_rev_sums_across_services_in_a_direction():
    low, high = response_energy_mwh({"DCL": 20.0, "DML": 10.0, "DRL": 5.0})
    assert low == pytest.approx(20 * 0.25 + 10 * 0.5 + 5 * 1.0)
    assert high == 0.0


def test_max_sell_size_caps_each_product():
    assert is_feasible({"DCL": MAX_SELL_SIZE_MW}, power_mw=500.0, energy_mwh=1000.0)
    assert not is_feasible({"DCL": MAX_SELL_SIZE_MW + 1}, power_mw=500.0, energy_mwh=1000.0)


# --- What dispatch inherits ---------------------------------------------------

def test_soc_bounds_are_asymmetric_for_one_direction():
    """Low only: energy must stay in store; headroom is unconstrained."""
    lo, hi = soc_bounds_mwh({"DCL": 40.0}, energy_mwh=100.0)
    assert (lo, hi) == (pytest.approx(10.0), pytest.approx(100.0))


def test_soc_bounds_narrow_to_the_middle_for_bidirectional_dr():
    lo, hi = soc_bounds_mwh({"DRH": 40.0, "DRL": 40.0}, energy_mwh=100.0)
    assert (lo, hi) == (pytest.approx(40.0), pytest.approx(60.0))


def test_arbitrage_limits_exclude_commitments_and_reserve():
    dis, chg = arbitrage_power_limits_mw({"DCL": 30.0}, power_mw=50.0)
    assert dis == pytest.approx(20.0)          # 50 - 30 exported for DCL
    assert chg == pytest.approx(47.0)          # 50 - 3 reserved to recover


# --- Rule changes over the backtest window ------------------------------------

from datetime import date

from src.analysis.neso_rules import (
    EAC_GO_LIVE,
    RESERVE_RULE_EFFECTIVE,
    reserve_rule_applies,
    splitting_allowed,
)


def test_splitting_starts_at_eac_go_live():
    assert not splitting_allowed(date(2023, 11, 1))
    assert splitting_allowed(EAC_GO_LIVE)


def test_reserve_rule_starts_when_codified():
    """Procurement Rules v2.0, effective 15 November 2024."""
    assert RESERVE_RULE_EFFECTIVE == date(2024, 11, 15)
    assert not reserve_rule_applies(date(2024, 11, 14))
    assert reserve_rule_applies(RESERVE_RULE_EFFECTIVE)


def test_rules_tighten_in_order():
    """Splitting arrived with EAC a year before the reserve became binding."""
    assert EAC_GO_LIVE < RESERVE_RULE_EFFECTIVE
