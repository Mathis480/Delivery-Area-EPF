"""
Corrected Support Vector Regression (cSVR) with Laplace and Gaussian correction kernels.
Based on Puć & Janczura (2024).
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


def _remove_zerovar(X: np.ndarray, threshold: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Remove near-zero-variance columns from X."""
    var = np.var(X, axis=0)
    mask = var > threshold
    if mask.sum() == 0:
        mask = np.ones(X.shape[1], dtype=bool)
    return X[:, mask], mask


@dataclass
class CSVRModel:
    """Fitted cSVR model parameters and state."""
    svr: SVR
    X_train: np.ndarray
    feature_mask: np.ndarray
    width: float
    norm: int
    bm_std_train: Optional[np.ndarray] = None
    bm_mean: float = 0.0
    bm_std: float = 1.0
    sigma: Optional[float] = None


def train_csvr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    bm_train: Optional[np.ndarray] = None,
    epsilon: float = CSVR_EPSILON,
    C: float = CSVR_C,
    q_kernel: float = CSVR_Q_KERNEL,
    q_data: float = CSVR_Q_DATA,
    norm: int = CSVR_NORM,
    use_gaussian_correction: bool = CSVR_USE_GAUSSIAN_CORRECTION,
    q_kernel_naive: float = CSVR_Q_KERNEL_NAIVE,
    q_data_naive: float = CSVR_Q_DATA_NAIVE,
    **kwargs,
) -> CSVRModel:
    """Fit cSVR with Laplace kernel and optional Gaussian naive benchmark correction."""
    if bm_train is None and "bm_trainval" in kwargs:
        bm_train = kwargs["bm_trainval"]

    X_filt, mask = _remove_zerovar(X_train)
    metric = "euclidean" if norm == 2 else "cityblock"
    dist_train = cdist(X_filt, X_filt, metric=metric)

    q_dist = float(np.quantile(dist_train, q_data))
    if q_dist <= 1e-8:
        q_dist = 1.0
    width = float(np.log(2.0 - 2.0 * q_kernel) / q_dist)
    kernel_exponent = width * dist_train

    bm_std_train, bm_mean, bm_std, sigma = None, 0.0, 1.0, None
    if use_gaussian_correction and bm_train is not None and len(bm_train) == len(X_filt):
        bm_mean = float(np.mean(bm_train))
        bm_std = float(np.std(bm_train))
        if bm_std <= 1e-8:
            bm_std = 1.0
        bm_std_train = (bm_train - bm_mean) / bm_std
        d_naive_train = np.abs(bm_std_train[:, None] - bm_std_train[None, :]) ** 2

        q_naive = float(np.quantile(d_naive_train, q_data_naive))
        z_crit = float(scipy.stats.norm.ppf(q_kernel_naive, loc=0, scale=1))
        sigma = q_naive / (z_crit if abs(z_crit) > 1e-8 else 1.0)
        if sigma <= 1e-8:
            sigma = 1.0

        gaussian_penalty = (1.0 / (2.0 * (sigma ** 2))) * d_naive_train
        kernel_exponent = kernel_exponent - gaussian_penalty

    K_train = np.exp(kernel_exponent)
    svr = SVR(kernel="precomputed", epsilon=epsilon, C=C)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        svr.fit(K_train, y_train)

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
    """Predict using fitted cSVR kernel and model."""
    X_filt = X_test[:, model.feature_mask]
    metric = "euclidean" if model.norm == 2 else "cityblock"
    dist_test = cdist(X_filt, model.X_train, metric=metric)
    kernel_exponent = model.width * dist_test

    if model.sigma is not None and bm_test is not None and model.bm_std_train is not None:
        bm_test_scaled = (bm_test - model.bm_mean) / model.bm_std
        d_naive_test = np.abs(bm_test_scaled - model.bm_std_train[None, :]) ** 2
        gaussian_penalty = (1.0 / (2.0 * (model.sigma ** 2))) * d_naive_test
        kernel_exponent = kernel_exponent - gaussian_penalty

    K_test = np.exp(kernel_exponent)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        return model.svr.predict(K_test)