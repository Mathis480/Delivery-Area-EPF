"""
cSVR (kernel Support Vector Regression) for EPF — faithful port of Marcjasz et al. / Puc et al.

Key idea:
  1. Feature distances are computed via Euclidean distance (cdist).
  2. The kernel is a Laplacian kernel: K = exp(width * dist)
     where width = log(2 - 2 * q_kernel) / quantile(dist, q_data).
  3. The target difference (y) is standardized (zero-mean, unit-variance) during training
     so that standard SVR hyperparameters (epsilon=0.1, C=1.0) operate on the standardized scale,
     and inverse-transformed at prediction time.
  4. Near-zero-variance features are removed consistently between train and test.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.svm import SVR

from config import (
    CSVR_C,
    CSVR_EPSILON,
    CSVR_NORM,
    CSVR_Q_DATA,
    CSVR_Q_KERNEL,
)


def _remove_zerovar(
    X: np.ndarray, threshold: float = 1e-10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove near-zero-variance columns from X.

    Returns
    -------
    X_filtered : np.ndarray  — columns with sufficient variance
    mask       : np.ndarray  — boolean mask of kept columns (for reuse at test time)
    """
    var = np.var(X, axis=0)
    mask = var > threshold
    if mask.sum() == 0:
        mask = np.ones(X.shape[1], dtype=bool)
    return X[:, mask], mask


@dataclass
class CSVRModel:
    """Fitted cSVR — stores everything needed for test-time prediction."""
    svr: SVR
    X_train: np.ndarray          # filtered training features (for computing test distance)
    feature_mask: np.ndarray     # boolean mask from variance filter
    width: float                 # kernel exponent scaling parameter
    y_mean: float                # mean of target difference for denormalization
    y_std: float                 # std of target difference for denormalization
    norm: int


def train_csvr(
    X_trainval: np.ndarray,
    y_trainval: np.ndarray,
    epsilon: float = CSVR_EPSILON,
    C: float = CSVR_C,
    q_kernel: float = CSVR_Q_KERNEL,
    q_data: float = CSVR_Q_DATA,
    norm: int = CSVR_NORM,
) -> CSVRModel:
    """
    Fit cSVR on the combined train+val set.

    Steps:
      1. Filter zero-variance columns.
      2. Standardize target y_trainval (zero mean, unit variance).
      3. Compute pairwise distance matrix on training features.
      4. Compute Laplace kernel width and K_train = exp(width * dist_train).
      5. Fit SVR(kernel='precomputed', epsilon=epsilon, C=C).
    """
    # 1. Feature variance filter
    X_filt, mask = _remove_zerovar(X_trainval)

    # 2. Target standardization
    y_mean = float(np.mean(y_trainval))
    y_std = float(np.std(y_trainval))
    if y_std <= 1e-8:
        y_std = 1.0
    y_standarized = (y_trainval - y_mean) / y_std

    # 3. Intermediate distance kernel
    metric = "euclidean" if norm == 2 else "cityblock"
    dist_train = cdist(X_filt, X_filt, metric=metric)

    # 4. Laplace kernel parameters (Marcjasz et al.)
    q_dist = float(np.quantile(dist_train, q_data))
    if q_dist <= 1e-8:
        q_dist = 1.0
    width = float(np.log(2.0 - 2.0 * q_kernel) / q_dist)

    K_train = np.exp(width * dist_train)

    # 5. Fit SVR on precomputed kernel
    svr = SVR(kernel="precomputed", epsilon=epsilon, C=C)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        svr.fit(K_train, y_standarized)

    return CSVRModel(
        svr=svr,
        X_train=X_filt,
        feature_mask=mask,
        width=width,
        y_mean=y_mean,
        y_std=y_std,
        norm=norm,
    )


def predict_csvr(model: CSVRModel, X_test: np.ndarray) -> np.ndarray:
    """
    Predict for test sample(s) using stored kernel and target parameters.

    Returns
    -------
    y_pred_diff : np.ndarray
        Predicted price difference (unstandardized, same unit as y_trainval).
    """
    # Apply same feature mask
    X_filt = X_test[:, model.feature_mask]

    # Compute distance to training points
    metric = "euclidean" if model.norm == 2 else "cityblock"
    dist_test = cdist(X_filt, model.X_train, metric=metric)

    # Precomputed test kernel: K_test ∈ R^{n_test × n_train}
    K_test = np.exp(model.width * dist_test)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        pred_standarized = model.svr.predict(K_test)

    # Invert target standardization
    y_pred_diff = pred_standarized * model.y_std + model.y_mean
    return y_pred_diff
