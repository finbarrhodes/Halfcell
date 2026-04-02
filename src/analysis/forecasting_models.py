"""
Electricity Price Forecasting — Model Definitions
===================================================
Self-contained model classes and factory function for price forecasting.
All models expose a sklearn-compatible fit(X, y) / predict(X) interface and
are wrapped by _AsinhTransformModel in price_forecast.train_forecast_model().

Models
------
_AsinhTransformModel    : Target-transform wrapper (arcsinh) applied to any
                          base estimator; handles negative prices gracefully.
_LEARModel              : LEAR (Lasso Estimated AutoRegressive) — 48 per-period
                          LassoLarsIC regressors following Lago et al. (2021).
_LEARCalibratedEnsemble : Calibration-window ensemble for LEAR: N instances
                          trained on different window lengths, predictions
                          simple-averaged (Lago et al., 2021).
_DNNModel               : Fully-connected DNN — 4 hidden layers (512→256→128→64),
                          ReLU, Dropout(0.15), early stopping; single global model
                          across all 48 settlement periods.
_build_model()          : Factory — returns the appropriate base estimator for a
                          model_type string ("rf", "xgb", "lgb", "lear", "dnn").
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Asinh-transform model wrapper
# ---------------------------------------------------------------------------

class _AsinhTransformModel:
    """
    Wraps a sklearn/xgboost/lightgbm estimator with an arcsinh target
    transform so that fit() and predict() both operate in price space.

    arcsinh is variance-stabilising, symmetric around zero, and handles
    negative prices natively — unlike log transforms. Recommended by
    Lago et al. (2021) for electricity price forecasting as renewable
    penetration increases the frequency of zero and negative price periods.

        transform  : arcsinh(y)
        inverse    : sinh(p)
    """

    def __init__(self, base_model):
        self._model = base_model
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X, y):
        y_tr = np.arcsinh(y)
        self._model.fit(X, y_tr)
        self.feature_importances_ = self._model.feature_importances_
        return self

    def predict(self, X):
        pred_tr = self._model.predict(X)
        return np.sinh(pred_tr)


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

        n_features = feat_arr.shape[1]
        all_coefs: list = []
        for sp in range(1, 49):
            mask = sp_vals == sp
            # LassoLarsIC requires n_samples > n_features to estimate noise
            # variance; also enforce a 30-row floor for numerical stability.
            if mask.sum() <= max(n_features, 30):
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
# LEAR calibration-window ensemble (Lago et al., 2021)
# ---------------------------------------------------------------------------

class _LEARCalibratedEnsemble:
    """
    Calibration-window ensemble for LEAR, following Lago et al. (2021).

    Wraps N _AsinhTransformModel(_LEARModel()) instances trained on different
    window lengths. Predictions are simple-averaged across all constituents.
    Combining short windows (adapts to recent regime shifts) with long windows
    (stable baselines) consistently outperforms any single calibration window.

    Exposes the same interface as a single _AsinhTransformModel so all
    downstream consumers (predict_day_prices, get_feature_importances) work
    without modification:
        predict(X)           -> np.ndarray  (averaged across all windows)
        feature_importances_ -> np.ndarray  (averaged across all windows)
        ._model              -> first constituent's _LEARModel, satisfying
                                isinstance(model._model, _LEARModel) checks
    """

    def __init__(self, constituents: list) -> None:
        if not constituents:
            raise ValueError("_LEARCalibratedEnsemble requires at least one constituent.")
        self._constituents = constituents
        # Point ._model at the first constituent's inner _LEARModel so that
        # isinstance checks and _lear_extra_df / _feat_names access in
        # predict_day_prices and get_feature_importances work transparently.
        self._model = constituents[0]._model

        fi_arrays = [
            m.feature_importances_ for m in constituents
            if m.feature_importances_ is not None
        ]
        self.feature_importances_ = (
            np.mean(fi_arrays, axis=0) if fi_arrays else None
        )

    def predict(self, X) -> np.ndarray:
        # Average in arcsinh-transformed space, then apply sinh() once.
        # Averaging in price space (after sinh) allows the exponential tail of
        # sinh to amplify magnitude errors from any constituent — particularly
        # for spike periods where arcsinh-space errors are small but price-space
        # errors are enormous. Spearman survives this (ranks are scale-invariant)
        # but RMSE and Spike-RMSE do not.
        raw = np.stack([m._model.predict(X) for m in self._constituents], axis=0)
        return np.sinh(raw.mean(axis=0))


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

def _build_model(model_type: str):
    """Instantiate and return the base estimator for the given model type."""
    if model_type == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=300,
            max_features=0.5,       # outperforms "sqrt" with correlated lag features
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        )
    elif model_type == "xgb":
        from xgboost import XGBRegressor
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
        )
    elif model_type == "lgb":
        from lightgbm import LGBMRegressor
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
        )
    elif model_type == "lear":
        return _LEARModel()
    elif model_type == "dnn":
        return _DNNModel()
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'rf', 'xgb', 'lgb', 'lear', or 'dnn'.")
