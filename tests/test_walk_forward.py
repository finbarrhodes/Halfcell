"""
Walk-forward retraining: the property that matters is that no day is ever predicted
by a model that was trained on it.

Fits run on a small synthetic feature matrix, so the whole module stays in the fast
suite. The real matrix is 130k rows and an RF fit on it takes ~30 s.
"""
import numpy as np
import pandas as pd
import pytest

from src.analysis.features import FEATURE_COLS
from src.analysis.price_forecast import (
    forecast_series_by_date,
    walk_forward_origins,
    walk_forward_predictions,
)

PERIODS = 4          # settlement periods per day, enough to exercise the shape
FEATURES = FEATURE_COLS[:3]


def _feature_frame(start="2024-01-01", days=560, seed=0):
    """A synthetic matrix with a trend, so a model fitted on old data misprices new days."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    rows = []
    for i, date in enumerate(dates):
        for sp in range(1, PERIODS + 1):
            level = 50.0 + 0.05 * i + 8.0 * sp
            rows.append({
                "settlementDate": date,
                "settlementPeriod": sp,
                "apx_price": level + rng.normal(0, 3),
                **{col: level + rng.normal(0, 1) for col in FEATURES},
            })
    return pd.DataFrame(rows)


# --- Origins ---------------------------------------------------------------------

def test_origins_start_on_the_first_of_a_month_and_span_the_window():
    origins = walk_forward_origins("2024-02-14", "2024-12-31", cadence_months=3)
    assert origins[0] == pd.Timestamp("2024-02-01")
    assert origins[-1] <= pd.Timestamp("2024-12-31")
    assert all((b - a).days >= 89 for a, b in zip(origins, origins[1:]))


def test_cadence_controls_how_often_the_model_is_refitted():
    monthly = walk_forward_origins("2024-01-01", "2024-12-31", cadence_months=1)
    yearly = walk_forward_origins("2024-01-01", "2024-12-31", cadence_months=12)
    assert len(monthly) == 12 and len(yearly) == 1


# --- The no-leakage property ------------------------------------------------------

@pytest.fixture(scope="module")
def walked():
    frame = _feature_frame()
    predictions, folds = walk_forward_predictions(
        frame, "2025-01-01", "2025-06-30", model_type="rf", cadence_months=3
    )
    return frame, predictions, folds


def test_no_day_is_predicted_by_a_model_that_trained_on_it(walked):
    _frame, predictions, _folds = walked
    assert not predictions.empty
    assert (predictions["origin"] <= predictions["settlementDate"]).all()


def test_each_fold_trains_only_on_history_before_its_origin(walked):
    _frame, _predictions, folds = walked
    for fold in folds:
        assert fold["train_start"] < fold["origin"]
        assert fold["train_rows"] > 0
        assert fold["origin"] <= fold["predicts_until"]


def test_folds_do_not_overlap_and_run_in_order(walked):
    _frame, _predictions, folds = walked
    for earlier, later in zip(folds, folds[1:]):
        assert earlier["predicts_until"] < later["origin"]


def test_every_day_in_the_window_gets_a_forecast(walked):
    _frame, predictions, _folds = walked
    days = pd.to_datetime(predictions["settlementDate"]).dt.normalize().unique()
    expected = pd.date_range("2025-01-01", "2025-06-30", freq="D")
    assert len(days) == len(expected)
    assert predictions.groupby("settlementDate").size().eq(PERIODS).all()


def test_each_fold_reports_the_metrics_of_the_days_it_predicted(walked):
    _frame, _predictions, folds = walked
    for fold in folds:
        assert fold["metrics"]["n_samples"] > 0
        assert fold["metrics"]["rmse"] > 0


# --- Rolling windows ----------------------------------------------------------------

def test_a_rolling_window_forgets_older_history():
    frame = _feature_frame()
    _predictions, folds = walk_forward_predictions(
        frame, "2025-04-01", "2025-06-30", model_type="rf", cadence_months=3, train_years=0.5
    )
    fold = folds[0]
    assert pd.Timestamp(fold["train_start"]) == pd.Timestamp("2024-10-01")
    expanding = walk_forward_predictions(
        frame, "2025-04-01", "2025-06-30", model_type="rf", cadence_months=3
    )[1][0]
    assert fold["train_rows"] < expanding["train_rows"]


# --- Handing predictions to dispatch -------------------------------------------------

def test_predictions_convert_to_the_series_dispatch_expects(walked):
    _frame, predictions, _folds = walked
    by_date = forecast_series_by_date(predictions)
    day = pd.Timestamp("2025-02-03")
    assert list(by_date[day].index) == list(range(1, PERIODS + 1))
    assert by_date[day].notna().all()


def test_a_date_range_filters_the_prediction_table(walked):
    _frame, predictions, _folds = walked
    by_date = forecast_series_by_date(predictions, "2025-03-01", "2025-03-31")
    assert min(by_date) == pd.Timestamp("2025-03-01")
    assert max(by_date) == pd.Timestamp("2025-03-31")


def test_skipping_origins_leaves_them_unfitted_so_a_cache_can_be_extended():
    frame = _feature_frame()
    full = walk_forward_predictions(frame, "2025-01-01", "2025-06-30", model_type="rf", cadence_months=3)[1]
    origins = [f["origin"] for f in full]
    _predictions, folds = walk_forward_predictions(
        frame, "2025-01-01", "2025-06-30", model_type="rf", cadence_months=3,
        skip_origins=origins[:-1],
    )
    assert [f["origin"] for f in folds] == origins[-1:]


# --- Spread calibration -----------------------------------------------------------

def test_spread_calibration_is_signed_so_over_and_under_prediction_differ():
    from src.analysis.price_forecast import spread_calibration

    dates = ["2026-01-01"] * 3 + ["2026-01-02"] * 3
    actual = [10.0, 20.0, 30.0, 10.0, 20.0, 30.0]           # spread 20 on both days
    wide = [0.0, 25.0, 50.0, 0.0, 25.0, 50.0]               # spread 50: over by 30
    narrow = [18.0, 20.0, 22.0, 18.0, 20.0, 22.0]           # spread 4: under by 16

    assert spread_calibration(dates, actual, wide) == {"spread_bias": 30.0, "spread_mae": 30.0}
    assert spread_calibration(dates, actual, narrow) == {"spread_bias": -16.0, "spread_mae": 16.0}


def test_opposite_errors_cancel_in_the_bias_but_not_in_the_mae():
    """The pair a single RMSE would hide: one day too wide, one too narrow."""
    from src.analysis.price_forecast import spread_calibration

    dates = ["2026-01-01"] * 2 + ["2026-01-02"] * 2
    actual = [10.0, 30.0, 10.0, 30.0]
    predicted = [0.0, 40.0, 15.0, 25.0]                     # +20 then -10
    assert spread_calibration(dates, actual, predicted) == {"spread_bias": 5.0, "spread_mae": 15.0}


def test_a_perfect_spread_forecast_scores_zero_on_both():
    from src.analysis.price_forecast import spread_calibration

    dates = ["2026-01-01"] * 2
    assert spread_calibration(dates, [10.0, 30.0], [12.0, 32.0]) == {"spread_bias": 0.0, "spread_mae": 0.0}


def test_fold_metrics_carry_the_spread_statistics(walked):
    _frame, _predictions, folds = walked
    for fold in folds:
        assert "spread_bias" in fold["metrics"] and "spread_mae" in fold["metrics"]
