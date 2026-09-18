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
    with pytest.raises(ValueError, match="offer_predictions"):
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
