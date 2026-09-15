"""
MPC dispatch LP: mechanics, economics, and the response-driven limits it inherits.

Each test is a single small solve, so the suite stays in the fast default run.
"""
import numpy as np
import pytest

from src.optimisation.mpc import DT, solve_mpc

E, ETA, WEAR = 100.0, 0.90, 3.0


def plan(prices, soc0=50.0, dis=50.0, chg=None, lo=0.0, hi=E, wear=WEAR, eta=ETA):
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    return solve_mpc(
        soc_current=soc0,
        price_forecast=prices,
        arb_mw_schedule=np.full(n, dis) if np.ndim(dis) == 0 else np.asarray(dis, dtype=float),
        soc_min=lo, soc_max=hi, energy_mwh=E,
        efficiency_rt=eta, cycling_cost_per_mwh=wear, horizon=n,
        charge_mw_schedule=None if chg is None else (np.full(n, chg) if np.ndim(chg) == 0 else np.asarray(chg, dtype=float)),
        return_plan=True,
    )


# --- Mechanics -----------------------------------------------------------------

def test_state_of_energy_follows_the_state_equation():
    out = plan([20, 20, 150, 150, 20, 150, 20, 150])
    soc, dis, chg = out["soc"], out["p_dis"], out["p_chg"]
    expected = soc[:-1] - dis * DT + chg * ETA * DT
    np.testing.assert_allclose(soc[1:], expected, atol=1e-5)


def test_state_of_energy_stays_within_the_physical_store():
    out = plan([0] * 8 + [300] * 8, soc0=10.0)
    assert out["soc"].min() >= -1e-6 and out["soc"].max() <= E + 1e-6


def test_power_limits_apply_separately_to_each_side():
    out = plan([10, 10, 10, 10, 200, 200, 200, 200], soc0=50.0, dis=30.0, chg=12.0)
    assert out["p_dis"].max() <= 30.0 + 1e-6
    assert out["p_chg"].max() <= 12.0 + 1e-6


def test_scalar_bounds_still_work():
    """Backwards compatible with the fixed-band callers."""
    e_dis, e_chg = solve_mpc(50.0, np.array([10.0, 200.0]), np.array([50.0, 50.0]),
                             10.0, 90.0, E, ETA, WEAR, horizon=2)
    assert e_dis >= 0.0 and e_chg >= 0.0


# --- Economics -------------------------------------------------------------------

def test_flat_prices_do_not_trade_from_empty():
    """Nothing to sell, and buying at a flat price only loses to efficiency and wear."""
    out = plan([80.0] * 16, soc0=0.0)
    assert out["p_dis"].sum() == pytest.approx(0.0, abs=1e-5)
    assert out["p_chg"].sum() == pytest.approx(0.0, abs=1e-5)


def test_stored_energy_has_no_value_at_the_horizon_end():
    """
    Known limitation, pinned so it stays visible. With no terminal value on state
    of energy, charge left in store at the end of the horizon is worth nothing to
    the LP, so it sells it down even at a flat price. The 48-hour rolling horizon
    keeps this from dominating the backtest; a learned terminal value is the fix.
    """
    out = plan([80.0] * 16, soc0=50.0)
    assert out["p_dis"].sum() * DT == pytest.approx(50.0, rel=1e-3)


def test_a_spread_below_break_even_does_not_trade():
    """Buying at 100 and selling at 110 loses money at 90% efficiency and £3 wear."""
    out = plan([100.0] * 4 + [110.0] * 4, soc0=0.0)
    assert out["p_dis"].sum() == pytest.approx(0.0, abs=1e-5)


def test_a_wide_spread_charges_low_and_discharges_high():
    out = plan([20.0] * 4 + [250.0] * 4, soc0=0.0)
    assert out["p_chg"][:4].sum() > 0 and out["p_dis"][4:].sum() > 0
    assert out["p_dis"][:4].sum() == pytest.approx(0.0, abs=1e-5)


def test_no_simultaneous_charge_and_discharge_at_a_positive_spread():
    out = plan([20.0, 250.0] * 8, soc0=50.0)
    assert np.minimum(out["p_dis"], out["p_chg"]).max() == pytest.approx(0.0, abs=1e-4)


# --- Response requirements ----------------------------------------------------------

def test_prepositions_for_a_later_stored_energy_requirement():
    """A Low commitment from period 4 needs 60 MWh in store; start empty and flat-priced."""
    lo = np.array([0.0] * 4 + [60.0] * 5)
    out = plan([80.0] * 8, soc0=0.0, lo=lo)
    assert out["soc"][4:].min() >= 60.0 - 1e-4
    assert out["shortfall_mwh"].sum() == pytest.approx(0.0, abs=1e-4)


def test_holds_headroom_for_a_high_commitment():
    hi = np.full(9, 30.0)
    out = plan([10.0] * 8, soc0=30.0, hi=hi)        # cheap power, but no room to take it
    assert out["soc"].max() <= 30.0 + 1e-4


def test_never_breaches_to_capture_a_price_spike():
    """£3,000/MWh at the end is worth less than the breach it would take."""
    lo = np.full(9, 50.0)
    out = plan([80.0] * 7 + [3000.0], soc0=50.0, lo=lo)
    assert out["shortfall_mwh"].sum() == pytest.approx(0.0, abs=1e-4)


def test_an_unreachable_requirement_still_solves_and_reports_the_shortfall():
    """Start empty, need 80 MWh next period, charge limited to 20 MW: cannot comply."""
    lo = np.array([0.0] + [80.0] * 4)
    out = plan([80.0] * 4, soc0=0.0, chg=20.0, lo=lo)
    assert out is not None
    assert out["shortfall_mwh"][1] > 0
    assert out["p_chg"][0] == pytest.approx(20.0, rel=1e-3)   # moving towards compliance
