"""
LASSO with penalty selection by LARS and expanding-window time-series cross-validation (Puć & Janczura, 2024).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LassoLarsCV

from config import LASSO_CV_DAYS, LASSO_MAX_ITER


def temporal_cv_splits(n_samples: int, cv_days: int = LASSO_CV_DAYS) -> list[tuple[list[int], list[int]]]:
    """Each of the last `cv_days` observations is held out once; training uses all earlier observations."""
    return [(list(range(t)), [t]) for t in range(n_samples - cv_days, n_samples)]


def train_lasso(X_train: np.ndarray, y_train: np.ndarray) -> LassoLarsCV:
    """LARS path per fold, lambda with the lowest CV MSE; the LARS model is refitted on the full window."""
    return LassoLarsCV(cv=temporal_cv_splits(len(X_train)), max_iter=LASSO_MAX_ITER, n_jobs=1).fit(X_train, y_train)
