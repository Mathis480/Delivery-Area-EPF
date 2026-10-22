"""
Corrected Support Vector Regression (cSVR) of Puć & Janczura (2024): Laplace kernel on the features
multiplied by a Gaussian correction kernel on the standardized naive price (Eq. 12), with quantile-based widths.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.stats
from scipy.spatial.distance import cdist
from sklearn.svm import SVR

from config import CSVR_C, CSVR_EPSILON, CSVR_NORM, CSVR_Q_DATA, CSVR_Q_DATA_NAIVE, CSVR_Q_KERNEL, CSVR_Q_KERNEL_NAIVE

METRIC = "euclidean" if CSVR_NORM == 2 else "cityblock"


@dataclass
class CSVRModel:
    svr: SVR
    X_train: np.ndarray
    bm_train_std: np.ndarray  # standardized naive price of the calibration days
    bm_mean: float
    bm_std: float
    width: float              # Laplace width (negative sign included)
    sigma: float              # Gaussian correction scale


def _kernel(model: CSVRModel, X: np.ndarray, bm_std: np.ndarray, dist: np.ndarray | None = None) -> np.ndarray:
    """Kernel against the calibration days; `dist` reuses an already computed feature distance matrix."""
    if dist is None:
        dist = cdist(X, model.X_train, metric=METRIC)
    correction = (1.0 / (2.0 * model.sigma ** 2)) * np.abs(bm_std[:, None] - model.bm_train_std[None, :]) ** 2
    return np.exp(model.width * dist - correction)


def train_csvr(X_train: np.ndarray, y_train: np.ndarray, bm_train: np.ndarray) -> CSVRModel:
    dist = cdist(X_train, X_train, metric=METRIC)
    width = float(np.log(2.0 - 2.0 * CSVR_Q_KERNEL) / np.quantile(dist, CSVR_Q_DATA))

    bm_mean, bm_std = float(np.mean(bm_train)), float(np.std(bm_train))
    bm_train_std = (bm_train - bm_mean) / bm_std
    d_naive = (bm_train_std[:, None] - bm_train_std[None, :]) ** 2
    sigma = float(np.quantile(d_naive, CSVR_Q_DATA_NAIVE) / scipy.stats.norm.ppf(CSVR_Q_KERNEL_NAIVE))

    model = CSVRModel(SVR(kernel="precomputed", epsilon=CSVR_EPSILON, C=CSVR_C), X_train, bm_train_std,
                      bm_mean, bm_std, width, sigma)
    model.svr.fit(_kernel(model, X_train, bm_train_std, dist=dist), y_train)
    return model


def predict_csvr(model: CSVRModel, X_test: np.ndarray, bm_test: float) -> np.ndarray:
    bm_test_std = np.atleast_1d((bm_test - model.bm_mean) / model.bm_std)
    return model.svr.predict(_kernel(model, X_test, bm_test_std))
