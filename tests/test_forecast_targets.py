"""
Model specs: what an estimator is fitted to, and under which loss.

Two experiments live here. The residual target fits the deviation from the naive
forecast rather than the price, so the estimator stops rediscovering the shape
yesterday already carries. The asymmetric loss charges a forecast for moving
further from that baseline than the day actually did — the spread it invents,
which RMSE and Spearman do not charge for.

Fits run on a small synthetic matrix so the module stays in the fast suite.
"""
import numpy as np
import pandas as pd
import pytest

from src.analysis.features import FEATURE_COLS
from src.analysis.forecasting_models import (
    NAIVE_COL,
    ModelSpec,
    _asymmetric_grad_hess,
    _build_model,
    _ResidualModel,
    parse_model_spec,
)
from src.analysis.price_forecast import predict_day_prices, train_forecast_model

PERIODS = 4
OTHER_FEATURES = [c for c in FEATURE_COLS[:3] if c != NAIVE_COL]


def _feature_frame(days=400, seed=0):
    """
    A matrix where the price is yesterday's price plus a settlement-period effect,
    so the naive forecast is most of the answer and the residual is the rest.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    rows = []
    for i, date in enumerate(dates):
        for sp in range(1, PERIODS + 1):
            naive = 50.0 + 0.05 * i + 8.0 * sp
            rows.append({
                "settlementDate": date,
                "settlementPeriod": sp,
                NAIVE_COL: naive,
                "apx_price": naive + 4.0 * sp + rng.normal(0, 1),
                **{col: naive + rng.normal(0, 1) for col in OTHER_FEATURES},
            })
    return pd.DataFrame(rows)


# --- Spec parsing -----------------------------------------------------------------

def test_a_bare_base_is_the_shipped_configuration():
    assert parse_model_spec("rf") == ModelSpec(base="rf", target="price", loss="squared")


def test_modifiers_name_the_target_and_the_loss():
    assert parse_model_spec("rf-residual").target == "residual"
    assert parse_model_spec("hgb-pinball").loss == "pinball"

    both = parse_model_spec("lgb-residual-asym")
    assert (both.base, both.target, both.loss) == ("lgb", "residual", "asymmetric")


def test_a_modifier_carries_its_own_value():
    assert parse_model_spec("hgb-pinball_0.4").alpha == pytest.approx(0.4)
    assert parse_model_spec("xgb-residual-asym_3").penalty == pytest.approx(3.0)


def test_spec_strings_stay_usable_as_filenames():
    # The walk-forward harness caches one table per spec, named after it
    for spec in ("rf-residual", "hgb-pinball_0.4", "xgb-residual-asym_3"):
        assert not set(spec) & set('/\\:*?"<>|')


def test_an_unknown_modifier_is_rejected_rather_than_ignored():
    with pytest.raises(ValueError, match="Unknown modifier"):
        parse_model_spec("rf-resdiual")


def test_a_loss_is_refused_on_a_base_that_cannot_fit_it():
    with pytest.raises(ValueError, match="no quantile objective"):
        parse_model_spec("rf-pinball")
    with pytest.raises(ValueError, match="no custom objective"):
        parse_model_spec("hgb-residual-asym")


def test_the_asymmetric_loss_requires_the_residual_target():
    # Without it there is no origin to call a move 'further than the day went'
    with pytest.raises(ValueError, match="only defined on the residual target"):
        parse_model_spec("lgb-asym")


def test_an_out_of_range_quantile_is_rejected():
    with pytest.raises(ValueError, match="must lie in"):
        parse_model_spec("hgb-pinball_1.5")


# --- The residual target ----------------------------------------------------------

def test_the_residual_model_fits_the_deviation_and_adds_the_baseline_back():
    frame = _feature_frame(days=40)
    X = frame[[NAIVE_COL] + OTHER_FEATURES]
    y = frame["apx_price"]

    class Recorder:
        def fit(self, X, y):
            self.target = np.asarray(y)
            return self

        def predict(self, X):
            return np.zeros(len(X))

    model = _ResidualModel(Recorder()).fit(X, y)

    # What the estimator saw is the deviation, not the price
    assert model._model.target == pytest.approx(y.to_numpy() - X[NAIVE_COL].to_numpy())
    # A predicted deviation of zero is the naive forecast itself
    assert model.predict(X) == pytest.approx(X[NAIVE_COL].to_numpy())


def test_the_residual_model_says_so_when_the_baseline_column_is_missing():
    frame = _feature_frame(days=10)
    with pytest.raises(KeyError, match=NAIVE_COL):
        _ResidualModel(_build_model("rf")).fit(frame[OTHER_FEATURES], frame["apx_price"])


def test_the_residual_target_trains_and_predicts_in_price_space():
    frame = _feature_frame()
    model, cols, _train, test = train_forecast_model(
        frame, model_type="rf-residual", test_start="2024-10-01"
    )
    assert NAIVE_COL in cols

    day = pd.Timestamp("2024-11-05")
    predicted = predict_day_prices(model, cols, frame, day)
    actual = frame[frame["settlementDate"] == day].set_index("settlementPeriod")["apx_price"]
    assert len(predicted) == PERIODS
    # Price space, not deviation space: within a few £ of the day it forecasts
    assert predicted.to_numpy() == pytest.approx(actual.to_numpy(), abs=5.0)
    assert test["rmse"] < 5.0


def test_the_residual_target_beats_the_price_target_when_the_baseline_carries_the_level():
    """
    On a series that is naive plus a period effect, fitting the deviation should not
    be worse than fitting the level — the baseline is handed to it for free. This is
    the property the experiment rests on, not a claim about real prices.
    """
    frame = _feature_frame()
    _, _, _, price = train_forecast_model(frame, model_type="rf", test_start="2024-10-01")
    _, _, _, residual = train_forecast_model(frame, model_type="rf-residual", test_start="2024-10-01")
    assert residual["rmse"] <= price["rmse"]


# --- The asymmetric loss ----------------------------------------------------------

def test_overshooting_the_day_is_charged_more_than_undershooting_it():
    # Same 5 £/MWh error either side of an actual move of +10
    _, over = _asymmetric_grad_hess(y_true=[10.0], y_pred=[15.0], penalty=3.0)
    _, under = _asymmetric_grad_hess(y_true=[10.0], y_pred=[5.0], penalty=3.0)
    assert over == pytest.approx([3.0])
    assert under == pytest.approx([1.0])


def test_a_move_in_the_wrong_direction_is_charged_on_its_size_not_its_sign():
    # Predicting a big move the wrong way invents spread; a small one does not
    _, big = _asymmetric_grad_hess(y_true=[1.0], y_pred=[-20.0], penalty=2.0)
    _, small = _asymmetric_grad_hess(y_true=[10.0], y_pred=[-2.0], penalty=2.0)
    assert big == pytest.approx([2.0])
    assert small == pytest.approx([1.0])


def test_the_gradient_points_away_from_the_error():
    grad, _ = _asymmetric_grad_hess(y_true=[10.0, 10.0], y_pred=[15.0, 5.0], penalty=2.0)
    # Overshoot pushes down and carries the penalty; undershoot pushes up at unit weight
    assert grad == pytest.approx([10.0, -5.0])


def test_a_symmetric_penalty_is_plain_squared_error():
    grad, hess = _asymmetric_grad_hess(y_true=[10.0, 10.0], y_pred=[15.0, 5.0], penalty=1.0)
    assert grad == pytest.approx([5.0, -5.0])
    assert hess == pytest.approx([1.0, 1.0])
