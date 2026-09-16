"""
Per-block FR / arbitrage allocation under NESO's participation rules.

The central property: whatever prices arrive, the holding the allocator picks
must be one NESO would accept (checked with neso_rules.is_feasible), and it must
never earn more than the old model, which sold the same MW into every product.
"""
import numpy as np
import pytest

from src.analysis.fr_allocation import EFA_HOURS, allocate_block, allocate_day, choose_family
from src.analysis.neso_rules import (
    PRODUCTS,
    arbitrage_power_limits_mw,
    family,
    is_feasible,
    soc_bounds_mwh,
)

P, E, D = 50.0, 100.0, 2.0     # reference battery: 50 MW / 100 MWh


def alloc(prices, arb=0.0, *, reserve=True, families=("DC", "DM", "DR"), power=P, energy=E, duration=D):
    return allocate_block(prices, arb, power, energy, duration, apply_reserve=reserve, families=families)


# --- Basics ------------------------------------------------------------------

def test_negative_prices_commit_nothing():
    out = alloc({p: -5.0 for p in PRODUCTS})
    assert out["q"] == {} and out["fr_revenue_gbp"] == 0.0


def test_zero_prices_commit_nothing():
    """A zero price earns nothing but would still bind state of charge."""
    assert alloc({p: 0.0 for p in PRODUCTS})["q"] == {}


def test_missing_and_nan_products_are_not_offered():
    out = alloc({"DCL": 10.0, "DRL": float("nan")})
    assert set(out["q"]) == {"DCL"}


def test_high_arbitrage_value_keeps_capacity_for_trading():
    """£1,000/MW/block of arbitrage beats £10/MW/h x 4h = £40/MW/block of FR."""
    out = alloc({p: 10.0 for p in PRODUCTS}, arb=1_000.0)
    assert out["q"] == {} and out["arb_mw"] == pytest.approx(P)


# --- The bug being fixed -------------------------------------------------------

def test_three_high_products_share_the_power_rather_than_each_taking_it():
    out = alloc({"DCH": 10.0, "DMH": 10.0, "DRH": 10.0})
    high = sum(out["q"].values())
    assert high <= P + 1e-6, f"committed {high} MW of High on a {P} MW unit"


def test_never_earns_more_than_the_old_same_mw_everywhere_model():
    prices = {"DCH": 3.0, "DCL": 12.0, "DMH": 4.0, "DML": 8.0, "DRH": 6.0, "DRL": 15.0}
    old = sum(prices.values()) * EFA_HOURS * P
    assert alloc(prices)["fr_revenue_gbp"] < old


# --- Rules carried through -----------------------------------------------------

def test_symmetric_dm_stops_at_the_reserve_limit():
    """NESO SoE guidance Figure 2 logic: x + 0.2x <= 50 gives 41.67 MW each way."""
    out = alloc({"DMH": 10.0, "DML": 10.0})
    assert out["q"]["DMH"] == pytest.approx(50 / 1.2, rel=1e-4)
    assert out["q"]["DML"] == pytest.approx(50 / 1.2, rel=1e-4)


def test_without_the_reserve_rule_symmetric_dm_takes_full_power():
    out = alloc({"DMH": 10.0, "DML": 10.0}, reserve=False)
    assert out["q"]["DMH"] == pytest.approx(P) and out["q"]["DML"] == pytest.approx(P)


def test_energy_limits_bidirectional_dr_on_a_one_hour_battery():
    """
    60-minute DR on 50 MW / 50 MWh: response energy caps DRH + DRL at 50 MW in total.

    At equal prices any split of that 50 MW is optimal, so only the total is
    pinned; power alone would have allowed 50 MW each way.
    """
    out = alloc({"DRH": 10.0, "DRL": 10.0}, reserve=False, energy=50.0, duration=1.0)
    total = out["q"].get("DRH", 0.0) + out["q"].get("DRL", 0.0)
    assert total == pytest.approx(50.0, rel=1e-4)


def test_single_family_restriction_is_respected():
    out = alloc({p: 10.0 for p in PRODUCTS}, families=["DM"])
    assert {family(p) for p in out["q"]} == {"DM"}


def test_choose_family_picks_the_better_service():
    prices = {"DCH": 1.0, "DCL": 2.0, "DMH": 1.0, "DML": 1.0, "DRH": 9.0, "DRL": 20.0}
    assert choose_family(prices, 0.0, P, E, D, apply_reserve=True) == "DR"


def test_choose_family_returns_none_when_trading_is_worth_more():
    prices = {p: 1.0 for p in PRODUCTS}
    assert choose_family(prices, 1_000.0, P, E, D, apply_reserve=True) is None


# --- Property: every allocation is one NESO would accept -----------------------

@pytest.mark.parametrize("seed", range(5))
def test_allocations_are_always_neso_feasible(seed):
    rng = np.random.default_rng(seed)
    for _ in range(60):
        power = float(rng.choice([10.0, 50.0, 100.0]))
        duration = float(rng.choice([1.0, 2.0]))
        reserve = bool(rng.integers(0, 2))
        fams = ["DC", "DM", "DR"] if rng.integers(0, 2) else [str(rng.choice(["DC", "DM", "DR"]))]
        prices = {p: float(rng.uniform(-20, 40)) for p in PRODUCTS}
        out = allocate_block(prices, float(rng.uniform(0, 400)), power, power * duration, duration,
                             apply_reserve=reserve, families=fams)
        assert is_feasible(out["q"], power, power * duration, apply_reserve=reserve,
                           single_family=len(fams) == 1, tol=1e-5), (prices, out)


# --- Whole days: state of energy must be able to reach every block ----------------

ETA = 0.90


def day(prices_by_block, soc_now, *, lead_in=(), arb=0.0, reserve=True, families=("DC", "DM", "DR")):
    blocks = [{"prices": pr, "arb_value": arb, "families": families} for pr in prices_by_block]
    return allocate_day(blocks, P, E, D, ETA, soc_now, apply_reserve=reserve, lead_in=lead_in)


def free(n_sp):
    return {"lo": 0.0, "hi": E, "dis": P, "chg": P, "n_sp": n_sp}


def plan_is_deliverable(out, *, reserve=True, tol=1e-5):
    """Each block's planned SoE is in its range, and every move fits the power left free."""
    plan = out["soc_plan_mwh"]
    for b, alloc in enumerate(out["blocks"]):
        lo, hi = soc_bounds_mwh(alloc["q"], E)
        dis, chg = arbitrage_power_limits_mw(alloc["q"], P, reserve)
        first, last = plan[b]
        moves = [(first, last, 3.5)] + ([(last, plan[b + 1][0], 0.5)] if b + 1 < len(plan) else [])
        for soe in (first, last):
            if not lo - tol <= soe <= hi + tol:
                return False
        for start, end, hours in moves:
            if end - start > hours * ETA * chg + tol or start - end > hours * dis + tol:
                return False
    return True


def test_a_day_of_self_contained_blocks_matches_block_by_block():
    """Symmetric DM keeps SoE mid-range, so linking the blocks changes nothing."""
    prices = [{"DMH": 10.0, "DML": 10.0}] * 6
    out = day(prices, soc_now=50.0)
    single = alloc(prices[0])
    for block in out["blocks"]:
        assert block["q"] == pytest.approx(single["q"], rel=1e-6)


def test_a_full_battery_cannot_take_high_response_it_has_no_headroom_for():
    """
    DC High needs headroom to absorb power. Bidding at the block start with the store
    full, block 1 must decline it; by block 2 there has been time to make room.
    """
    out = day([{"DCH": 10.0}] * 6, soc_now=E)
    assert out["blocks"][0]["q"] == {}
    assert out["blocks"][1]["q"]["DCH"] == pytest.approx(P)
    assert plan_is_deliverable(out)


def test_half_an_hour_of_free_power_before_the_block_makes_enough_room():
    out = day([{"DCH": 10.0}] * 6, soc_now=E, lead_in=[free(1)])
    assert out["blocks"][0]["q"]["DCH"] == pytest.approx(P)


def test_a_holding_that_freezes_state_of_energy_needs_it_already_in_range():
    """
    31.25 MW DC High with 46.9 MW DR Low uses all 50 MW on both sides, so SoE cannot
    move during the block. The previous evening held 50 MW DR Low - no discharge room
    - and the store is full, so SoE is stuck above the combination's 92.2 MWh ceiling.
    Seen in the 2025 backtest as whole blocks lost to unavailability.
    """
    prices = [{"DCH": 20.0, "DRL": 30.0}] * 6
    evening = {"lo": 50.0, "hi": E, "dis": 0.0, "chg": 30.0, "n_sp": 18}

    stuck = day(prices, soc_now=E, lead_in=[evening])
    assert stuck["blocks"][0]["q"].get("DCH", 0.0) == pytest.approx(0.0, abs=1e-6)
    assert plan_is_deliverable(stuck)

    in_range = day(prices, soc_now=70.0, lead_in=[evening])
    assert in_range["blocks"][0]["q"]["DCH"] == pytest.approx(31.25, rel=1e-4)
    assert in_range["blocks"][0]["q"]["DRL"] == pytest.approx(46.875, rel=1e-4)


def test_an_unavoidable_breach_already_under_way_still_allocates():
    """Starting below a range the unit already holds must not make the LP fail."""
    held_low = {"lo": 80.0, "hi": E, "dis": 0.0, "chg": 0.0, "n_sp": 18}
    out = day([{"DCL": 10.0}] * 6, soc_now=20.0, lead_in=[held_low])
    assert len(out["blocks"]) == 6 and out["blocks"][0]["q"]["DCL"] > 0
    assert plan_is_deliverable(out)


@pytest.mark.parametrize("seed", range(5))
def test_days_are_neso_feasible_deliverable_and_never_beat_independent_blocks(seed):
    rng = np.random.default_rng(100 + seed)
    for _ in range(12):
        reserve = bool(rng.integers(0, 2))
        prices = [{p: float(rng.uniform(-20, 40)) for p in PRODUCTS} for _ in range(6)]
        arb = float(rng.uniform(0, 300))
        lead_in = [{"lo": lo, "hi": lo + float(rng.uniform(0, E - lo)),
                    "dis": float(rng.uniform(0, P)), "chg": float(rng.uniform(0, P)),
                    "n_sp": int(rng.integers(1, 9))}
                   for lo in rng.uniform(0, E, size=int(rng.integers(0, 3)))]
        out = day(prices, float(rng.uniform(0, E)), lead_in=lead_in, arb=arb, reserve=reserve)

        assert plan_is_deliverable(out, reserve=reserve)
        for pr, block in zip(prices, out["blocks"]):
            assert is_feasible(block["q"], P, E, apply_reserve=reserve, tol=1e-5)
            independent = alloc(pr, arb, reserve=reserve)["fr_revenue_gbp"]
            assert block["fr_revenue_gbp"] <= independent + 1e-3 or block["arb_mw"] < P
