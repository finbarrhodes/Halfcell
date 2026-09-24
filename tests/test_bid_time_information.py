"""
What an offer may know. Offers for day D close at 14:00 on D-1, so a forecast of D
built from all of D-1 uses ten hours that had not happened yet. These tests pin the
bid-time variants: features and the naive forecast one day further back, and the
scheduler giving offers that forecast while D-1's own periods keep the one made
the day before.
"""
import numpy as np
import pandas as pd
import pytest

from src.analysis.features import build_feature_matrix
from src.analysis.price_forecast import naive_day_prices, run_forecast_backtest
from src.analysis.revenue_stack import REFERENCE_BATTERY, _Scheduler


def _prices(start="2025-01-01", days=40):
    """Each day's price is its day number, so a lag reads off which day it came from."""
    dates = pd.date_range(start, periods=days, freq="D")
    return pd.DataFrame([
        {"settlementDate": d, "settlementPeriod": sp, "dataProvider": "APXMIDP", "price": float(i)}
        for i, d in enumerate(dates) for sp in range(1, 49)
    ])


def _generation(start="2025-01-01", days=40):
    dates = pd.date_range(start, periods=days, freq="D")
    return pd.DataFrame([{"settlementDate": d, "fuelGroup": "Wind", "generation": 1000.0 + i}
                         for i, d in enumerate(dates)])


def test_bid_time_features_come_from_the_day_before_yesterday():
    day_ahead = build_feature_matrix(_prices(), _generation())
    bid_time = build_feature_matrix(_prices(), _generation(), information_lag_days=2)
    target = pd.Timestamp("2025-01-30")                     # day number 29
    da = day_ahead[day_ahead["settlementDate"] == target].iloc[0]
    bt = bid_time[bid_time["settlementDate"] == target].iloc[0]
    assert (da["apx_lag_1d"], da["prev_day_mean"], da["gen_wind"]) == (28.0, 28.0, 1028.0)
    assert (bt["apx_lag_1d"], bt["prev_day_mean"], bt["gen_wind"]) == (27.0, 27.0, 1027.0)
    assert bt["apx_lag_14d"] == 29.0 - 15


def test_features_cannot_include_the_day_being_forecast():
    with pytest.raises(ValueError, match="information_lag_days"):
        build_feature_matrix(_prices(), _generation(), information_lag_days=0)


def test_the_bid_time_naive_forecast_is_the_last_complete_day():
    prices = _prices()
    assert naive_day_prices(prices, pd.Timestamp("2025-01-30")).iloc[0] == 28.0
    assert naive_day_prices(prices, pd.Timestamp("2025-01-30"), days_back=2).iloc[0] == 27.0


def test_ml_offers_at_bid_time_need_their_own_forecast_table():
    with pytest.raises(ValueError, match="early_predictions"):
        run_forecast_backtest(strategy="ml", market_index=_prices(), auctions=pd.DataFrame(),
                              battery=REFERENCE_BATTERY, services=[], start_date=None, end_date=None,
                              predictions=pd.DataFrame({"settlementDate": [], "settlementPeriod": [],
                                                        "prediction": []}),
                              offer_information="bid_time")


def test_an_unknown_information_set_is_rejected():
    with pytest.raises(ValueError, match="offer_information"):
        run_forecast_backtest(strategy="naive", market_index=_prices(), auctions=pd.DataFrame(),
                              battery=REFERENCE_BATTERY, services=[], start_date=None, end_date=None,
                              offer_information="hindsight")


def test_offers_see_the_bid_time_forecast_of_d_and_yesterdays_forecast_of_d_minus_1():
    day = pd.Timestamp("2025-03-10")
    series = lambda value: pd.Series(value, index=range(1, 49), dtype=float)
    scheduler = _Scheduler(
        pd.DataFrame(columns=["EFA Date", "Service", "EFA", "Clearing Price", "Cleared Volume"]),
        REFERENCE_BATTERY,
        forecast_prices_by_date={day - pd.Timedelta(days=1): series(1.0), day: series(2.0)},
        offer_forecast_prices_by_date={day - pd.Timedelta(days=1): series(8.0), day: series(9.0)},
    )
    view = scheduler._offer_view(day)
    assert view[day - pd.Timedelta(days=1)].iloc[0] == 1.0      # made on D-2: known by the deadline
    assert view[day].iloc[0] == 9.0                              # as it stood at the deadline
    # 14:00 to midnight is still calendar D-1, including EFA 1's first hour from 23:00
    path = scheduler._forecast_path(view, pd.Timestamp("2025-03-09 14:00"), 22)
    assert np.all(path[:20] == 1.0) and np.all(path[20:] == 9.0)


# --- Dispatch plans only on forecasts that already exist ---------------------------------

def _record_plans(monkeypatch):
    """Capture the price path and length of every dispatch plan, and trade nothing."""
    import src.optimisation.mpc as mpc
    plans = []

    def fake_solve(**kwargs):
        plans.append(np.asarray(kwargs["price_forecast"], dtype=float))
        # As solve_mpc does: the reserve's MWh too when the caller keeps an account of them
        reporting = kwargs.get("return_reserve") or kwargs.get("recovery_allowance") is not None
        return (0.0, 0.0, 0.0, 0.0) if reporting else (0.0, 0.0)

    monkeypatch.setattr(mpc, "solve_mpc", fake_solve)
    return plans


def _dispatch(days, *, vintages, early=True, horizon=96):
    """Three-day run where each forecast is tagged by where it came from."""
    from src.analysis.revenue_stack import run_dispatch
    flat = lambda value: pd.Series(float(value), index=range(1, 49))
    actual = {d: flat(50) for d in days}
    # Day-ahead forecast of day k reads 100 + k, the early one 200 + k
    day_ahead = {d: flat(100 + k) for k, d in enumerate(days)}
    early_fc = {d: flat(200 + k) for k, d in enumerate(days)} if early else None
    schedule = pd.DataFrame([
        {"date": d, "efa": efa, "family": "none", "apply_reserve": False, "soc_min_mwh": 0.0,
         "soc_max_mwh": 100.0, "discharge_max_mw": 50.0, "charge_max_mw": 50.0, "arb_mw": 50.0,
         **{f"q_{p}": 0.0 for p in ("DCH", "DCL", "DMH", "DML", "DRH", "DRL")}}
        for d in days for efa in range(1, 7)
    ]).set_index(["date", "efa"])
    run_dispatch(actual, REFERENCE_BATTERY, days, day_ahead, schedule=schedule, horizon=horizon,
                 forecast_vintages=vintages, early_forecast_prices_by_date=early_fc)


DAYS3 = [pd.Timestamp("2025-03-10") + pd.Timedelta(days=k) for k in range(3)]


def test_without_vintages_tomorrow_is_planned_on_a_forecast_that_needs_today(monkeypatch):
    """The leak being fixed: at 14:00 on day 0 the plan reads day 1's day-ahead forecast."""
    plans = _record_plans(monkeypatch)
    _dispatch(DAYS3, vintages=False)
    at_1400 = plans[28]
    assert len(at_1400) == 96
    assert set(at_1400[20:68]) == {101.0}          # tomorrow, from a forecast built on all of today


def test_with_vintages_today_uses_the_day_ahead_forecast_and_tomorrow_the_early_one(monkeypatch):
    plans = _record_plans(monkeypatch)
    _dispatch(DAYS3, vintages=True)
    at_1400 = plans[28]                            # day 0, period 29
    assert len(at_1400) == 20 + 48                 # the rest of today, then all of tomorrow
    assert set(at_1400[:20]) == {100.0}
    assert set(at_1400[20:]) == {201.0}


def test_with_vintages_the_plan_ends_after_tomorrow(monkeypatch):
    plans = _record_plans(monkeypatch)
    _dispatch(DAYS3, vintages=True)
    lengths = [len(p) for p in plans]
    assert lengths[0] == 96 and lengths[47] == 49  # 00:00 and 23:30 on day 0
    assert lengths[48] == 96                       # a new day moves the limit on
    assert lengths[-1] == 1                        # the last day has no tomorrow in the data


def test_perfect_foresight_keeps_its_prices_and_only_loses_horizon(monkeypatch):
    plans = _record_plans(monkeypatch)
    _dispatch(DAYS3, vintages=True, early=False)
    at_1400 = plans[28]
    assert len(at_1400) == 68 and set(at_1400[20:]) == {101.0}


def test_a_missing_tomorrow_ends_the_plan_tonight(monkeypatch):
    plans = _record_plans(monkeypatch)
    _dispatch([DAYS3[0], DAYS3[2]], vintages=True)
    assert len(plans[28]) == 20
