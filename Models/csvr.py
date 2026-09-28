"""
cSVR (Corrected Support Vector Regression) for EPF — faithful port of Puć & Janczura (2024).

Theoretical Foundation:
-----------------------
Following Puć & Janczura (2024, Eq. 12; arXiv:2411.16237v1) and their official replication
repository (https://github.com/pucandrzej/replication_cSVR/tree/main/Forecasting):

1. **Feature Distance (Laplace Kernel):**
   Pairwise Euclidean distances d_X(x_i, x_j) = ||x_i - x_j||_2 are computed on standardized
   and zero-variance-filtered features. The Laplace kernel width is set via quantile heuristics:
       l = log(2 - 2 * q_kernel) / quantile(d_X, q_data)
   with q_kernel = 0.75 and q_data = 0.50.

2. **Naive Benchmark Gaussian Correction Kernel (Eq. 12):**
   To incorporate local elasticity and suppress leverage points (training samples with close
   features but distant prices), the Laplace kernel is multiplied by a Gaussian kernel
   on the standardized naive benchmark price:
       K_cSVR(x_i, x_j) = exp( l * ||x_i - x_j||_2 - 1 / (2 * sigma^2) * (P_i^naive - P_j^naive)^2 )
   where sigma is determined via data quantiles:
       sigma = quantile((P_i^naive - P_j^naive)^2, q_data_naive) / Phi^-1(q_kernel_naive)
   with q_data_naive = 0.75 and q_kernel_naive = 0.75.

3. **Target Standardisation & Inversion:**
   Target difference (y = VWAP_0to30 - VWAP_90to105) is standardized to zero mean and unit
   variance on the training set, allowing standard SVR parameters (C=1.0, epsilon=0.1)
   to operate on the normalized scale before inverse-transforming predictions.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.stats
from scipy.spatial.distance import cdist
from sklearn.svm import SVR

from config import (
    CSVR_C,
    CSVR_EPSILON,
    CSVR_NORM,
    CSVR_Q_DATA,
    CSVR_Q_KERNEL,
    CSVR_USE_GAUSSIAN_CORRECTION,
    CSVR_Q_KERNEL_NAIVE,
    CSVR_Q_DATA_NAIVE,
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
    X_train: np.ndarray                  # filtered training features
    feature_mask: np.ndarray             # boolean mask from variance filter
    width: float                         # Laplace kernel exponent scaling parameter
    norm: int                            # L1 or L2 norm
    bm_std_train: Optional[np.ndarray] = None  # standardized naive benchmark training series
    bm_mean: float = 0.0                 # training mean of naive benchmark
    bm_std: float = 1.0                  # training std of naive benchmark
    sigma: Optional[float] = None        # Gaussian naive correction scale parameter


def train_csvr(
    X_trainval: np.ndarray,
    y_trainval: np.ndarray,
    bm_trainval: Optional[np.ndarray] = None,
    epsilon: float = CSVR_EPSILON,
    C: float = CSVR_C,
    q_kernel: float = CSVR_Q_KERNEL,
    q_data: float = CSVR_Q_DATA,
    norm: int = CSVR_NORM,
    use_gaussian_correction: bool = CSVR_USE_GAUSSIAN_CORRECTION,
    q_kernel_naive: float = CSVR_Q_KERNEL_NAIVE,
    q_data_naive: float = CSVR_Q_DATA_NAIVE,
) -> CSVRModel:
    """
    Fit cSVR on the combined train+val set following Puć & Janczura (2024, Eq. 12).

    Steps:
      1. Filter zero-variance columns.
      2. Compute pairwise Euclidean distance matrix on training features.
      3. Compute Laplace kernel width: width = log(2 - 2*q_kernel) / quantile(dist, q_data).
      4. If use_gaussian_correction and bm_trainval is provided:
         Multiply Laplace kernel by Gaussian naive kernel (Puć & Janczura 2024, Eq. 12).
      5. Fit SVR(kernel='precomputed', epsilon=epsilon, C=C).
    """
    # 1. Feature variance filter
    X_filt, mask = _remove_zerovar(X_trainval)

    # 2. Intermediate distance kernel
    metric = "euclidean" if norm == 2 else "cityblock"
    dist_train = cdist(X_filt, X_filt, metric=metric)

    # 3. Laplace kernel parameters (Marcjasz et al. / Puć & Janczura)
    q_dist = float(np.quantile(dist_train, q_data))
    if q_dist <= 1e-8:
        q_dist = 1.0
    width = float(np.log(2.0 - 2.0 * q_kernel) / q_dist)

    # Base Laplace exponent
    kernel_exponent = width * dist_train

    # 4. Optional Gaussian Naive Benchmark Correction (Puć & Janczura 2024, Eq. 12)
    bm_std_train = None
    bm_mean = 0.0
    bm_std = 1.0
    sigma = None

    if use_gaussian_correction and bm_trainval is not None and len(bm_trainval) == len(X_filt):
        bm_mean = float(np.mean(bm_trainval))
        bm_std = float(np.std(bm_trainval))
        if bm_std <= 1e-8:
            bm_std = 1.0
        bm_std_train = (bm_trainval - bm_mean) / bm_std
        d_naive_train = np.abs(bm_std_train[:, None] - bm_std_train[None, :]) ** 2

        q_naive = float(np.quantile(d_naive_train, q_data_naive))
        z_crit = float(scipy.stats.norm.ppf(q_kernel_naive, loc=0, scale=1))
        sigma = q_naive / (z_crit if abs(z_crit) > 1e-8 else 1.0)
        if sigma <= 1e-8:
            sigma = 1.0

        gaussian_penalty = (1.0 / (2.0 * (sigma ** 2))) * d_naive_train
        kernel_exponent = kernel_exponent - gaussian_penalty

    K_train = np.exp(kernel_exponent)

    # 5. Fit SVR on precomputed kernel (y is pre-standardized by the data loader)
    svr = SVR(kernel="precomputed", epsilon=epsilon, C=C)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        svr.fit(K_train, y_trainval)

    return CSVRModel(
        svr=svr,
        X_train=X_filt,
        feature_mask=mask,
        width=width,
        norm=norm,
        bm_std_train=bm_std_train,
        bm_mean=bm_mean,
        bm_std=bm_std,
        sigma=sigma,
    )


def predict_csvr(
    model: CSVRModel,
    X_test: np.ndarray,
    bm_test: Optional[float] = None,
) -> np.ndarray:
    """
    Predict for test sample(s) using stored kernel and target parameters.
    Applies Gaussian correction if bm_test is provided and model was trained with correction.

    Returns
    -------
    y_pred_scaled : np.ndarray
        Predicted standardized price difference.
    """
    # Apply same feature mask
    X_filt = X_test[:, model.feature_mask]

    # Compute distance to training points
    metric = "euclidean" if model.norm == 2 else "cityblock"
    dist_test = cdist(X_filt, model.X_train, metric=metric)

    kernel_exponent = model.width * dist_test

    # Apply Gaussian correction kernel if active
    if model.sigma is not None and bm_test is not None and model.bm_std_train is not None:
        bm_test_scaled = (bm_test - model.bm_mean) / model.bm_std
        d_naive_test = np.abs(bm_test_scaled - model.bm_std_train[None, :]) ** 2
        gaussian_penalty = (1.0 / (2.0 * (model.sigma ** 2))) * d_naive_test
        kernel_exponent = kernel_exponent - gaussian_penalty

    # Precomputed test kernel: K_test ∈ R^{n_test × n_train}
    K_test = np.exp(kernel_exponent)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        y_pred = model.svr.predict(K_test)

    return y_pred
