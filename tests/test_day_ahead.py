"""
The bid-deadline LP that chooses holdings together with a trading plan.

The economic tests use one hand-checkable day on a 50 MW / 100 MWh battery with
no Reserved Capacity: power is cheap in block 1 (£20), dear in block 5 (£200)
and middling otherwise (£80), so the day's trade is to charge overnight and sell
into the evening. What the LP will hold then follows from which side of the
battery that trade needs, and when - the effects a per-block formula cannot see.
"""
import numpy as np
import pytest

from src.analysis.fr_allocation import EFA_HOURS, allocate_day
from src.analysis.neso_rules import PRODUCTS, capacity_use_mw, is_feasible, soc_bounds_mwh
from src.optimisation.day_ahead import planning_prices, plan_day
from src.optimisation.mpc import DT

P, E, ETA, WEAR = 50.0, 100.0, 0.90, 3.0
EVENING_PEAK = [20.0] * 8 + [80.0] * 24 + [200.0] * 8 + [80.0] * 8


def day(block_prices, prices=EVENING_PEAK, soc_now=50.0, *, reserve=False, families=("DC", "DM", "DR"),
        lead_in=(), **kw):
    blocks = [{"prices": bp, "families": families} for bp in block_prices]
    return plan_day(blocks, P, E, ETA, WEAR, soc_now, prices, apply_reserve=reserve, lead_in=lead_in, **kw)


def only(block, offer, n=6):
    """Offer `offer` in one block (1-based) and nothing elsewhere."""
    return [offer if b == block else {} for b in range(1, n + 1)]


def trading_value(blocked, holding, prices, soc_now=20.0, **kw):
    """The plan's trading value with `holding` forced into each block in `blocked`."""
    fixed = [holding if b in blocked else {} for b in range(1, 7)]
    return day([{}] * 6, prices, soc_now, fixed=fixed, **kw)["trading_value_gbp"]


# --- Mechanics ------------------------------------------------------------------

def test_the_plan_follows_the_state_equation_and_stays_in_the_store():
    out = day(only(1, {"DCL": 10.0}))
    soc, dis, chg = out["soc_path_mwh"], out["discharge_mw"], out["charge_mw"]
    np.testing.assert_allclose(soc[1:], soc[:-1] - dis * DT + ETA * chg * DT, atol=1e-6)
    assert soc.min() >= -1e-6 and soc.max() <= E + 1e-6


def test_prices_must_cover_every_period_from_bid_time_to_the_end():
    with pytest.raises(ValueError, match="expected 66 prices"):
        day([{}] * 6, prices=[50.0] * 48, lead_in=[{"lo": 0, "hi": E, "dis": P, "chg": P, "n_sp": 18}])


@pytest.mark.parametrize("seed", range(4))
def test_holdings_are_neso_feasible_and_the_plan_keeps_them_deliverable(seed):
    rng = np.random.default_rng(300 + seed)
    for _ in range(6):
        reserve = bool(rng.integers(0, 2))
        block_prices = [{p: float(rng.uniform(-10, 40)) for p in PRODUCTS} for _ in range(6)]
        prices = rng.normal(80, 60, 48)
        out = day(block_prices, prices, float(rng.uniform(0, E)), reserve=reserve)
        soc, dis, chg = out["soc_path_mwh"], out["discharge_mw"], out["charge_mw"]
        for b, block in enumerate(out["blocks"]):
            assert is_feasible(block["q"], P, E, apply_reserve=reserve, tol=1e-5)
            lo, hi = soc_bounds_mwh(block["q"], E)
            periods = slice(8 * b, 8 * b + 8)
            assert soc[periods].min() >= lo - 1e-5 and soc[periods].max() <= hi + 1e-5
            export, imports = capacity_use_mw(block["q"], reserve)
            assert (dis[periods] + export).max() <= P + 1e-5
            assert (chg[periods] + imports).max() <= P + 1e-5


def test_with_no_price_signal_it_holds_what_the_formula_allocator_holds_at_zero_value():
    """With nothing to trade for, both allocators only need a deliverable path."""
    block_prices = [{"DCH": 12.0, "DCL": 9.0, "DRL": 14.0}, {"DMH": 6.0, "DML": 6.0},
                    {"DRH": 11.0}, {"DCL": 7.0}, {"DCH": 3.0, "DRH": 4.0}, {"DML": 8.0}]
    lp = day(block_prices, prices=[np.nan] * 48, reserve=True)
    formula = allocate_day([{"prices": bp, "arb_value": 0.0, "families": ("DC", "DM", "DR")}
                            for bp in block_prices], P, E, 2.0, ETA, 50.0, apply_reserve=True)
    for ours, theirs in zip(lp["blocks"], formula["blocks"]):
        assert ours["fr_revenue_gbp"] == pytest.approx(theirs["fr_revenue_gbp"], rel=1e-6)


# --- What the day view changes ----------------------------------------------------

def test_with_equal_prices_it_sells_low_overnight_and_high_into_the_evening_peak():
    """
    Overnight the trade needs the charge side, which DC High takes: holding all of it
    would mean charging at £80 instead of £20, so it holds only as much as leaves
    enough power to fill the store. Low takes the discharge side, which nothing
    needs overnight. At the evening peak the sides swap.
    """
    dc = {"DCH": 15.0, "DCL": 15.0}
    out = day([dc, {}, {}, {}, dc, {}], soc_now=20.0, terminal_value_per_mwh=0.0)
    night, peak = out["blocks"][0]["q"], out["blocks"][4]["q"]
    assert night["DCL"] == pytest.approx(P) and night["DCH"] < P - 10
    assert peak["DCH"] == pytest.approx(P) and peak["DCL"] < P - 10


def test_high_overnight_is_held_only_to_the_point_where_it_starts_to_cost():
    """80 MWh to charge from 20 at 90% needs 88.9 MWh in, which 22.2 MW does in 4 h."""
    out = day(only(1, {"DCH": 15.0}), soc_now=20.0, terminal_value_per_mwh=0.0)
    assert out["blocks"][0]["q"]["DCH"] == pytest.approx(P - 80.0 / ETA / EFA_HOURS, rel=1e-4)


def test_what_it_will_hold_depends_on_state_of_energy_at_bid_time():
    """Holding DC High overnight costs nothing if the store is already nearly full."""
    full = day(only(1, {"DCH": 15.0}), soc_now=87.5, terminal_value_per_mwh=0.0)
    empty = day(only(1, {"DCH": 15.0}), soc_now=12.5, terminal_value_per_mwh=0.0)
    assert full["blocks"][0]["q"]["DCH"] == pytest.approx(P)
    assert empty["blocks"][0]["q"]["DCH"] < P - 20


def test_two_evening_peaks_are_substitutes_so_their_values_do_not_add():
    """Keeping both peaks free is worth one sale of the store, not one for each."""
    two_peaks = [20.0] * 8 + [80.0] * 24 + [200.0] * 16
    blocks_both = {"DCH": P, "DCL": P}
    base = trading_value({5, 6}, blocks_both, two_peaks, terminal_value_per_mwh=0.0)
    free_5 = trading_value({6}, blocks_both, two_peaks, terminal_value_per_mwh=0.0) - base
    free_6 = trading_value({5}, blocks_both, two_peaks, terminal_value_per_mwh=0.0) - base
    free_both = trading_value(set(), blocks_both, two_peaks, terminal_value_per_mwh=0.0) - base
    assert free_both < 0.7 * (free_5 + free_6)
    assert free_both >= max(free_5, free_6) - 1e-6


def test_offered_low_in_both_peaks_it_leaves_just_enough_free_to_sell_the_store_once():
    """
    A per-block formula prices two flat £200 blocks at no arbitrage value and holds
    both in full. The plan keeps exactly the store's worth of discharge free between them.
    """
    two_peaks = [20.0] * 8 + [80.0] * 24 + [200.0] * 16
    out = day([{}, {}, {}, {}, {"DCL": 15.0}, {"DCL": 15.0}], two_peaks, soc_now=20.0, families=("DC",))
    free_mwh = sum((P - out["blocks"][b]["q"].get("DCL", 0.0)) * EFA_HOURS for b in (4, 5))
    assert free_mwh == pytest.approx(E, rel=1e-4)


def test_a_holding_costs_what_the_plan_loses_by_carrying_it():
    """fixed= prices a holding: it earns no FR, and the plan can only lose from it."""
    free = trading_value(set(), {}, EVENING_PEAK)
    held = trading_value({1}, {"DCH": P}, EVENING_PEAK)
    assert 0 < free - held


# --- Prices and the end of the plan ---------------------------------------------------

def test_missing_prices_take_the_mean_and_shrink_pulls_towards_it():
    path, terminal = planning_prices([10.0, np.nan, 30.0, 40.0], 4, WEAR, price_shrink=0.5)
    mean = 80.0 / 3
    np.testing.assert_allclose(path, mean + 0.5 * (np.array([10.0, mean, 30.0, 40.0]) - mean))
    assert terminal == pytest.approx(mean - WEAR)


def test_no_prices_at_all_give_trading_no_value():
    path, terminal = planning_prices([np.nan] * 48, 48, WEAR)
    assert not path.any() and terminal == 0.0


def test_a_fully_shrunk_forecast_holds_as_if_there_were_nothing_to_trade():
    dc = {"DCH": 1.0, "DCL": 1.0}
    shrunk = day([dc] * 6, soc_now=50.0, price_shrink=0.0)
    flat = day([dc] * 6, prices=[np.nan] * 48, soc_now=50.0)
    for a, b in zip(shrunk["blocks"], flat["blocks"]):
        assert a["fr_revenue_gbp"] == pytest.approx(b["fr_revenue_gbp"], rel=1e-6)


def test_energy_left_at_the_end_is_worth_the_terminal_value():
    flat = [50.0] * 48
    emptied = day([{}] * 6, flat, soc_now=50.0, terminal_value_per_mwh=0.0)
    kept = day([{}] * 6, flat, soc_now=50.0, terminal_value_per_mwh=60.0)
    assert emptied["soc_path_mwh"][-1] == pytest.approx(0.0, abs=1e-6)
    assert kept["soc_path_mwh"][-1] == pytest.approx(E, abs=1e-6)


# --- Pre-EAC and commitments already held ------------------------------------------------

def test_before_eac_each_block_holds_at_most_one_service():
    offers = [{"DCL": 15.0, "DRL": 14.0, "DMH": 13.0, "DCH": 12.0}] * 6
    out = day(offers, soc_now=50.0, one_service=True, reserve=True)
    for block, services in zip(out["blocks"], out["families"]):
        assert len(services) <= 1
        assert is_feasible(block["q"], P, E, apply_reserve=True, single_family=True, tol=1e-5)
    assert any(out["families"])


def test_an_out_of_position_lead_in_still_plans():
    """Starting below a range the unit already holds must not make the LP fail."""
    held_low = {"lo": 80.0, "hi": E, "dis": 0.0, "chg": 0.0, "n_sp": 18}
    out = day([{"DCL": 10.0}] * 6, [60.0] * 18 + EVENING_PEAK, soc_now=20.0, lead_in=[held_low])
    assert len(out["blocks"]) == 6 and out["blocks"][1]["q"]["DCL"] > 0
    # Frozen through the lead-in: no power either way
    assert out["soc_path_mwh"][18] == pytest.approx(20.0)
