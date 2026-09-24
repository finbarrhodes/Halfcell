"""
Electricity Price Forecasting — Model Definitions
===================================================
Self-contained model classes and factory function for price forecasting.
All models expose a sklearn-compatible fit(X, y) / predict(X) interface and
are wrapped by _LogTransformModel in price_forecast.train_forecast_model().

Models
------
_LogTransformModel  : Target-transform wrapper (signed-log1p) applied to any
                      base estimator; handles negative prices gracefully.
_ResidualModel      : Target-transform wrapper fitting the deviation from the
                      naive forecast rather than the price itself.
_LEARModel          : LEAR (Lasso Estimated AutoRegressive) — 48 per-period
                      LassoLarsIC regressors following Lago et al. (2021).
_DNNModel           : Fully-connected DNN — 4 hidden layers (512→256→128→64),
                      ReLU, Dropout(0.15), early stopping; single global model
                      across all 48 settlement periods.
_build_model()      : Factory — returns the appropriate base estimator for a
                      ModelSpec (or a spec string).

Specs
-----
parse_model_spec()  : "rf", "rf-residual", "hgb-pinball_0.4",
                      "lgb-residual-asym_3" — base estimator, what it is fitted
                      to, and under which loss. See parse_model_spec.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Model specs — base estimator, target parameterisation, training loss
# ---------------------------------------------------------------------------

# The naive forecast, as a feature column. build_feature_matrix lags every price
# feature by information_lag_days, so this column holds the last complete day's
# price for the same settlement period — exactly what naive_day_prices() returns
# at the matching lag. The residual target is measured against it for that reason:
# the two stay aligned when the offer stage moves the cutoff back to D-2, where
# naive is D-2's prices and this column is too.
NAIVE_COL = "apx_lag_1d"

# Which backends can be fitted under something other than squared error. Pinball is
# a built-in objective in all three; the asymmetric loss is a custom gradient, which
# only the two boosting libraries with a callable-objective interface accept.
_PINBALL_BASES = ("xgb", "lgb", "hgb")
_ASYMMETRIC_BASES = ("xgb", "lgb")


@dataclass(frozen=True)
class ModelSpec:
    """What to fit, to what, and under which loss."""

    base: str                 # "rf", "xgb", "lgb", "hgb", "lear", "dnn"
    target: str = "price"     # "price" or "residual"
    loss: str = "squared"     # "squared", "pinball" or "asymmetric"
    alpha: float = 0.5        # pinball quantile
    penalty: float = 2.0      # asymmetric: cost of overshoot relative to undershoot


def parse_model_spec(spec: str) -> ModelSpec:
    """
    Parse a model spec string.

        "rf"                    Random Forest on the price, as shipped
        "rf-residual"           ... on the price's deviation from the naive forecast
        "hgb-pinball"           sklearn boosting at the median rather than the mean
        "lgb-pinball_0.4"       LightGBM at the 0.4 quantile
        "xgb-residual-asym_3"   ... on the residual, overshoot charged 3x undershoot

    Modifiers are '-' separated and carry an optional '_' value. A spec string is
    also a cache key and a filename in the walk-forward harness, which is why the
    value separator is '_' rather than ':'.
    """
    base, *modifiers = spec.split("-")
    fields: dict = {}
    for modifier in modifiers:
        name, _, value = modifier.partition("_")
        if name == "residual":
            fields["target"] = "residual"
        elif name == "pinball":
            fields["loss"] = "pinball"
            if value:
                fields["alpha"] = float(value)
        elif name == "asym":
            fields["loss"] = "asymmetric"
            if value:
                fields["penalty"] = float(value)
        else:
            raise ValueError(
                f"Unknown modifier '{modifier}' in model spec '{spec}'. Expected "
                "'residual', 'pinball[_alpha]' or 'asym[_penalty]'."
            )

    parsed = ModelSpec(base=base, **fields)
    if parsed.loss == "pinball":
        if parsed.base not in _PINBALL_BASES:
            raise ValueError(
                f"'{parsed.base}' has no quantile objective; pinball needs one of "
                f"{', '.join(_PINBALL_BASES)}."
            )
        if not 0.0 < parsed.alpha < 1.0:
            raise ValueError(f"pinball alpha must lie in (0, 1), got {parsed.alpha}")
    if parsed.loss == "asymmetric":
        if parsed.base not in _ASYMMETRIC_BASES:
            raise ValueError(
                f"'{parsed.base}' takes no custom objective; the asymmetric loss needs "
                f"one of {', '.join(_ASYMMETRIC_BASES)}."
            )
        if parsed.target != "residual":
            raise ValueError(
                "The asymmetric loss is only defined on the residual target: it charges a "
                "forecast for moving further from the naive baseline than the day did, "
                f"which needs that baseline as the origin. Use '{parsed.base}-residual-asym'."
            )
        if parsed.penalty <= 0:
            raise ValueError(f"asymmetric penalty must be positive, got {parsed.penalty}")
    return parsed


# ---------------------------------------------------------------------------
# Asymmetric loss — charges a forecast for spread it invents
# ---------------------------------------------------------------------------

def _asymmetric_grad_hess(y_true, y_pred, penalty: float) -> tuple:
    """
    Gradient and Hessian of a squared error that charges overshoot `penalty` times
    undershoot, on the residual target.

        L = ½ w (pred − actual)²,   w = penalty where |pred| > |actual|, else 1

    On the residual target both arguments are deviations from the naive forecast, so
    |pred| > |actual| means the forecast moved further from yesterday's shape than
    the day actually did — it invented spread. That is the error the 2026-09-17
    benchmark showed RMSE and Spearman do not charge for, and the one that costs
    money twice: a bad trade, and an inflated shadow arbitrage value that declines
    frequency response contracts worth having. Undershooting only forgoes upside, so
    it keeps unit weight.

    The rule catches a move in either direction, which is what spread inflation is:
    predicting −20 where the day moved +1 is charged, predicting −2 where it moved
    +10 is not.
    """
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    weight = np.where(np.abs(predicted) > np.abs(actual), float(penalty), 1.0)
    return weight * (predicted - actual), weight


def _asymmetric_objective(penalty: float):
    """The custom objective callable XGBRegressor and LGBMRegressor both accept."""
    def objective(y_true, y_pred):
        return _asymmetric_grad_hess(y_true, y_pred, penalty)
    return objective


# ---------------------------------------------------------------------------
# Log-transform model wrapper
# ---------------------------------------------------------------------------

class _LogTransformModel:
    """
    Wraps a sklearn/xgboost/lightgbm estimator with a signed-log1p target
    transform so that fit() and predict() both operate in price space.

    Signed-log1p handles negative prices gracefully:
        transform  : sign(y) * log1p(|y|)
        inverse    : sign(p) * expm1(|p|)
    """

    def __init__(self, base_model):
        self._model = base_model
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X, y):
        y_log = np.sign(y) * np.log1p(np.abs(y))
        self._model.fit(X, y_log)
        # None for a base estimator that reports no importances, such as
        # HistGradientBoostingRegressor
        self.feature_importances_ = getattr(self._model, "feature_importances_", None)
        return self

    def predict(self, X):
        pred_log = self._model.predict(X)
        return np.sign(pred_log) * np.expm1(np.abs(pred_log))


# ---------------------------------------------------------------------------
# Residual model wrapper — fits the deviation from the naive forecast
# ---------------------------------------------------------------------------

class _ResidualModel:
    """
    Fits the base estimator on the price's deviation from the naive forecast, and
    adds that forecast back at predict time.

    The naive forecast is already a feature (NAIVE_COL), so it is read out of X
    rather than passed beside it, and predict(X) keeps the single-argument interface
    every caller uses.

    What changes is where the estimator spends its capacity. Fitted on the price, a
    tree spends most of its splits rediscovering the daily shape that yesterday
    already carries, and the walk-forward benchmark could not separate the result
    from persistence at all (Diebold-Mariano p = 0.26 on squared error). Fitted on
    the deviation, that shape is free and the splits go to the part persistence gets
    wrong. It also puts the origin at the naive forecast, which is what makes the
    asymmetric loss well posed.

    No signed-log transform here, unlike _LogTransformModel: deviations are already
    signed and centred near zero, and that transform exists to compress a
    heavy-tailed positive price level.
    """

    def __init__(self, base_model, naive_col: str = NAIVE_COL):
        self._model = base_model
        self._naive_col = naive_col
        self.feature_importances_: np.ndarray | None = None

    def _naive(self, X) -> np.ndarray:
        if self._naive_col not in X:
            raise KeyError(
                f"the residual target needs the naive forecast column "
                f"'{self._naive_col}' in X, which build_feature_matrix supplies"
            )
        return np.asarray(X[self._naive_col], dtype=float)

    def fit(self, X, y):
        self._model.fit(X, np.asarray(y, dtype=float) - self._naive(X))
        self.feature_importances_ = getattr(self._model, "feature_importances_", None)
        return self

    def predict(self, X):
        return self._model.predict(X) + self._naive(X)


# ---------------------------------------------------------------------------
# LEAR model — 48 per-settlement-period Lasso regressors
# ---------------------------------------------------------------------------

class _LEARModel:
    """
    Lasso Estimated AutoRegressive (LEAR) model for electricity price forecasting.

    Trains one LassoLarsIC (AIC criterion) per settlement period (48 total),
    each operating on a StandardScaler-normalised feature space so that Lasso's
    uniform L1 penalty is not biased by feature scale differences.

    Expects X to contain a 'settlementPeriod' column (int, 1–48) used for
    period routing; this column is popped internally before Lasso fitting and
    prediction — it must NOT appear in the caller's feature_cols list.

    feature_importances_ is exposed as the mean of |coef| across all 48
    period models, aligned to the feature columns (excluding settlementPeriod).

    _lear_extra_df is set by train_forecast_model after fitting so that
    predict_day_prices can retrieve the pre-built wide lag features without
    rebuilding them on every call.
    """

    def __init__(self) -> None:
        self._models:  dict = {}   # sp -> fitted LassoLarsIC
        self._scalers: dict = {}   # sp -> fitted StandardScaler
        self._feat_names: list | None = None
        self.feature_importances_: np.ndarray | None = None
        self._lear_extra_df: pd.DataFrame | None = None  # cached post-fit

    def fit(self, X: pd.DataFrame, y) -> "_LEARModel":
        from sklearn.linear_model import LassoLarsIC
        from sklearn.preprocessing import StandardScaler

        sp_vals  = X["settlementPeriod"].values
        X_feat   = X.drop(columns=["settlementPeriod"])
        self._feat_names = list(X_feat.columns)
        feat_arr = X_feat.values.astype(float)
        y_arr    = np.asarray(y, dtype=float)

        all_coefs: list = []
        for sp in range(1, 49):
            mask = sp_vals == sp
            if mask.sum() < 30:
                continue
            scaler = StandardScaler()
            X_sc   = scaler.fit_transform(feat_arr[mask])
            m      = LassoLarsIC(criterion="aic", fit_intercept=True, max_iter=500)
            m.fit(X_sc, y_arr[mask])
            self._models[sp]  = m
            self._scalers[sp] = scaler
            all_coefs.append(np.abs(m.coef_))

        n_feat = len(self._feat_names)
        self.feature_importances_ = (
            np.mean(all_coefs, axis=0) if all_coefs else np.zeros(n_feat)
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        sp_vals = X["settlementPeriod"].values
        X_feat  = X.drop(columns=["settlementPeriod"]).values.astype(float)
        out     = np.zeros(len(X))
        for sp, m in self._models.items():
            mask = sp_vals == sp
            if not mask.any():
                continue
            X_sc     = self._scalers[sp].transform(X_feat[mask])
            out[mask] = m.predict(X_sc)
        return out


# ---------------------------------------------------------------------------
# DNN model — single global fully-connected network (Lago et al., 2021)
# ---------------------------------------------------------------------------

class _DNNModel:
    """
    Fully-connected deep neural network for electricity price forecasting,
    following Lago et al. (2021): a single global model across all 48
    settlement periods, with SP encoded as a plain numeric input feature.

    Architecture: 4 hidden layers (512 → 256 → 128 → 64), ReLU, Dropout(0.15).
    Trained with Adam + ReduceLROnPlateau scheduler; early stopping on a
    chronological 10 % validation hold-out from the end of the training window.

    sklearn-compatible interface: fit(X, y) / predict(X).
    X may be a DataFrame or ndarray; column order must be consistent.
    Internal StandardScaler normalises inputs before the network sees them.

    _lear_extra_df is set externally by train_forecast_model after fitting so
    that predict_day_prices can retrieve the wide lag features without
    rebuilding them on every call.
    """

    def __init__(
        self,
        hidden_dims: tuple = (512, 256, 128, 64),
        dropout: float = 0.15,
        lr: float = 1e-3,
        batch_size: int = 512,
        max_epochs: int = 200,
        patience: int = 15,
        val_frac: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.hidden_dims  = hidden_dims
        self.dropout      = dropout
        self.lr           = lr
        self.batch_size   = batch_size
        self.max_epochs   = max_epochs
        self.patience     = patience
        self.val_frac     = val_frac
        self.random_state = random_state

        self._net:    object | None = None
        self._scaler: object | None = None
        self.feature_importances_: np.ndarray | None = None
        self._lear_extra_df: pd.DataFrame | None = None

    def _build_net(self, n_in: int):
        import torch.nn as nn
        layers: list = []
        in_dim = n_in
        for h in self.hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(self.dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y) -> "_DNNModel":
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler

        torch.manual_seed(self.random_state)
        rng = np.random.RandomState(self.random_state)

        X_np = X.values if hasattr(X, "values") else np.asarray(X, dtype=float)
        y_np = np.asarray(y, dtype=float)

        # Chronological train/val split — val is the last val_frac rows
        n_val   = max(1, int(len(X_np) * self.val_frac))
        n_train = len(X_np) - n_val
        X_tr, X_val = X_np[:n_train], X_np[n_train:]
        y_tr, y_val = y_np[:n_train], y_np[n_train:]

        self._scaler = StandardScaler()
        X_tr_sc  = self._scaler.fit_transform(X_tr).astype(np.float32)
        X_val_sc = self._scaler.transform(X_val).astype(np.float32)

        X_tr_t  = torch.from_numpy(X_tr_sc)
        X_val_t = torch.from_numpy(X_val_sc)
        y_tr_t  = torch.from_numpy(y_tr.astype(np.float32))
        y_val_t = torch.from_numpy(y_val.astype(np.float32))

        self._net = self._build_net(X_tr_sc.shape[1])
        optimiser = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, patience=5, factor=0.5, min_lr=1e-5
        )
        loss_fn = nn.MSELoss()

        best_val   = float("inf")
        best_state = None
        no_improve = 0

        for _ in range(self.max_epochs):
            self._net.train()
            idx = rng.permutation(n_train)
            for start in range(0, n_train, self.batch_size):
                batch_idx = idx[start : start + self.batch_size]
                xb = X_tr_t[batch_idx]
                yb = y_tr_t[batch_idx]
                optimiser.zero_grad()
                loss = loss_fn(self._net(xb).squeeze(1), yb)
                loss.backward()
                optimiser.step()

            self._net.eval()
            with torch.no_grad():
                val_loss = loss_fn(self._net(X_val_t).squeeze(1), y_val_t).item()
            scheduler.step(val_loss)

            if val_loss < best_val - 1e-6:
                best_val   = val_loss
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)

        self.feature_importances_ = np.zeros(X_tr_sc.shape[1])
        return self

    def predict(self, X) -> np.ndarray:
        import torch
        self._net.eval()
        X_np = X.values if hasattr(X, "values") else np.asarray(X, dtype=float)
        X_sc = self._scaler.transform(X_np).astype(np.float32)
        with torch.no_grad():
            return self._net(torch.from_numpy(X_sc)).squeeze(1).numpy()


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _build_model(spec):
    """Instantiate and return the base estimator for a ModelSpec or a spec string."""
    if isinstance(spec, str):
        spec = parse_model_spec(spec)

    if spec.base == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=300,
            max_features=0.5,       # outperforms "sqrt" with correlated lag features
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        )
    elif spec.base == "xgb":
        from xgboost import XGBRegressor
        objective: dict = {}
        if spec.loss == "pinball":
            objective = {"objective": "reg:quantileerror", "quantile_alpha": spec.alpha}
        elif spec.loss == "asymmetric":
            # base_score is where boosting starts before the first tree. Its default
            # suits a price level, not a residual, and a custom objective gets no
            # automatic estimate — so start at no deviation from naive.
            objective = {"objective": _asymmetric_objective(spec.penalty), "base_score": 0.0}
        return XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,     # reduces overfitting on ~20-month training set
            reg_alpha=0.05,         # L1 regularisation
            n_jobs=-1,
            random_state=42,
            verbosity=0,
            **objective,
        )
    elif spec.base == "lgb":
        from lightgbm import LGBMRegressor
        objective = {}
        if spec.loss == "pinball":
            objective = {"objective": "quantile", "alpha": spec.alpha}
        elif spec.loss == "asymmetric":
            objective = {"objective": _asymmetric_objective(spec.penalty)}
        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
            **objective,
        )
    elif spec.base == "hgb":
        # The sklearn-native boosting backend. It earns its place by having a
        # built-in quantile objective, so the pinball experiment runs on a bare
        # checkout without xgboost or lightgbm installed.
        from sklearn.ensemble import HistGradientBoostingRegressor
        quantile = {"loss": "quantile", "quantile": spec.alpha} if spec.loss == "pinball" else {}
        return HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.03,
            max_leaf_nodes=63,
            min_samples_leaf=20,
            l2_regularization=0.05,
            random_state=42,
            **quantile,
        )
    elif spec.base == "lear":
        return _LEARModel()
    elif spec.base == "dnn":
        return _DNNModel()
    else:
        raise ValueError(
            f"Unknown model base '{spec.base}'. Use 'rf', 'xgb', 'lgb', 'hgb', 'lear', or 'dnn'."
        )
