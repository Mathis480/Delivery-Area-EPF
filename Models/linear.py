"""
Linear models for per-QH EPF forecasting:
  1. Standard OLS LinearRegression (LR) — trains on X_trainval
  2. LASSO with cross-validation on the validation fold — trains final on X_trainval
"""
from __future__ import annotations

import os
import pickle
import warnings
from typing import Optional

import numpy as np
from sklearn.linear_model import Lasso, LassoLarsCV, LinearRegression

from config import LASSO_CV_DAYS, LASSO_MAX_ITER, LASSO_TOL, SCALERS_DIR, TRAINED_MODELS_DIR


# ---------------------------------------------------------------------------
# Ordinary Least Squares (LR)
# ---------------------------------------------------------------------------

def train_lr(X_trainval: np.ndarray, y_trainval: np.ndarray) -> LinearRegression:
    """
    Fit unregularized OLS on the combined train+val window.
    """
    model = LinearRegression(fit_intercept=True, n_jobs=1)
    model.fit(X_trainval, y_trainval)
    return model


def predict_lr(model: LinearRegression, X: np.ndarray) -> np.ndarray:
    """Predict scaled price difference."""
    return model.predict(X)


def get_lr_weights(model: LinearRegression) -> tuple[np.ndarray, float]:
    """
    Return (coef_, intercept_) from a fitted LinearRegression.
    Used to initialize the MAML-NN linear bypass projection layer.
    """
    return model.coef_.copy(), float(model.intercept_)


# ---------------------------------------------------------------------------
# LASSO (LassoLarsCV — matches replication_cSVR methodology)
# ---------------------------------------------------------------------------

def _build_temporal_cv_splits(
    n_samples: int, cv_days: int = LASSO_CV_DAYS,
) -> list[tuple[list[int], list[int]]]:
    """
    Build an expanding-window temporal CV split for LASSO alpha selection.

    Each fold uses all data before the test index as training.
    The last `cv_days` observations serve as individual holdout folds.
    This mirrors the approach in replication_cSVR/forecasting_simulation.py.
    """
    splits = []
    for cv_test_idx in range(cv_days):
        test_pos = n_samples - cv_days + cv_test_idx
        train_indices = list(range(0, test_pos))
        test_indices = [test_pos]
        splits.append((train_indices, test_indices))
    return splits


def train_lasso(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_trainval: np.ndarray,
    y_trainval: np.ndarray,
    alpha: float | None = None,
) -> tuple[Lasso, float]:
    """
    LASSO with LassoLarsCV alpha selection (recalibrated every call).

    Alpha selection uses an expanding-window temporal CV over the last
    LASSO_CV_DAYS days of the trainval window, following Marcjasz et al.
    The final model is fitted on the full trainval set with the selected alpha.
    """
    if alpha is not None:
        model = Lasso(
            alpha=alpha,
            max_iter=LASSO_MAX_ITER,
            tol=LASSO_TOL,
            fit_intercept=True,
            random_state=42,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model.fit(X_trainval, y_trainval)
        return model, alpha

    # Build temporal expanding-window CV splits on trainval
    n_trainval = len(X_trainval)
    cv_splits = _build_temporal_cv_splits(n_trainval, cv_days=LASSO_CV_DAYS)

    cv_model = LassoLarsCV(
        cv=cv_splits,
        max_iter=LASSO_MAX_ITER,
        fit_intercept=True,
        n_jobs=-1,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        cv_model.fit(X_trainval, y_trainval)

    best_alpha = float(cv_model.alpha_)

    # Refit final model on full trainval set with the selected alpha
    final_model = Lasso(
        alpha=best_alpha,
        max_iter=LASSO_MAX_ITER,
        tol=LASSO_TOL,
        fit_intercept=True,
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        final_model.fit(X_trainval, y_trainval)

    return final_model, best_alpha


def predict_lasso(model: Lasso, X: np.ndarray) -> np.ndarray:
    """Predict scaled price difference."""
    return model.predict(X)


def get_lasso_weights(model: Lasso) -> tuple[np.ndarray, float]:
    """Return (coef_, intercept_) from a fitted Lasso."""
    return model.coef_.copy(), float(model.intercept_)


# ---------------------------------------------------------------------------
# Model & Scaler persistence utilities
# ---------------------------------------------------------------------------

def save_scaler(scaler, zone: str, qh_idx: int, date_str: str, folder: str = SCALERS_DIR) -> str:
    """Save fitted StandardScaler to Models/scalers/."""
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"scaler_{zone}_QH{qh_idx:02d}_{date_str}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(scaler, f)
    return filename


def save_trained_model(model, name: str, zone: str, qh_idx: int, date_str: str, folder: str = TRAINED_MODELS_DIR) -> str:
    """Save fitted model object to Models/trained_models/."""
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{name}_{zone}_QH{qh_idx:02d}_{date_str}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    return filename
