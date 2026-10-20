"""
Rolling inverse-MAE forecast averaging (Marcjasz et al., 2018; Puć & Janczura, 2024) and
statistical tests on daily loss vectors (Diebold & Mariano, 1995; Giacomini & White, 2006).
"""
from __future__ import annotations

import numpy as np
import scipy.stats

from config import ENSEMBLE_CALIB_DAYS, ENSEMBLE_POWER


def inverse_mae_average(
    predictions: dict[str, np.ndarray], y_true_2d: np.ndarray,
    calib_window: int = ENSEMBLE_CALIB_DAYS, power: float = ENSEMBLE_POWER,
) -> tuple[np.ndarray, np.ndarray]:
    """Average (days x 96) forecasts with per-product weights from the MAE of the preceding `calib_window` days.

    Equal weights during the first `calib_window` days. Returns the averaged forecast and the
    daily mean weight of every member (days x members).
    """
    preds = np.stack(list(predictions.values()))  # members x days x 96
    n_days = y_true_2d.shape[0]
    ens = np.empty_like(y_true_2d, dtype=np.float64)
    weights = np.empty((n_days, len(predictions)))
    for d in range(n_days):
        if d < calib_window:
            w = np.full(preds.shape[::2], 1.0 / len(predictions))
        else:
            mae = np.nanmean(np.abs(preds[:, d - calib_window: d] - y_true_2d[None, d - calib_window: d]), axis=1)
            inv = mae ** -power
            w = inv / inv.sum(axis=0, keepdims=True)
        ens[d] = np.sum(w * preds[:, d], axis=0)
        weights[d] = w.mean(axis=1)
    return ens, weights


def daily_loss(pred_2d: np.ndarray, y_true_2d: np.ndarray) -> np.ndarray:
    """Mean absolute error per day (L1 norm of the daily error vector / 96)."""
    return np.nanmean(np.abs(pred_2d - y_true_2d), axis=1)


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 7) -> tuple[float, float]:
    """Daily-vector Diebold-Mariano test with Newey-West variance (lag h).

    One-sided p-value for H1: forecast B is more accurate than forecast A.
    """
    d = loss_a - loss_b
    n, d_bar = len(d), d.mean()
    gamma = [np.mean((d[k:] - d_bar) * (d[: n - k] - d_bar)) for k in range(h + 1)]
    var = gamma[0] + 2 * sum((1 - k / (h + 1)) * gamma[k] for k in range(1, h + 1))
    stat = float(d_bar / np.sqrt(var / n))
    return stat, float(1 - scipy.stats.norm.cdf(stat))


def gw_test(loss_a: np.ndarray, loss_b: np.ndarray) -> tuple[float, float]:
    """Conditional Giacomini-White test (one-step horizon, instruments: constant and lagged loss differential).

    One-sided p-value for H1: forecast B is more accurate than forecast A (as in Lago et al., 2021).
    """
    d = loss_a - loss_b
    instruments = np.column_stack([np.ones(len(d) - 1), d[:-1]])
    z = instruments * d[1:, None]
    n = len(z)
    beta, *_ = np.linalg.lstsq(z, np.ones(n), rcond=None)
    r2 = 1 - np.mean((np.ones(n) - z @ beta) ** 2)
    stat = float(n * r2 * np.sign(d.mean()))
    return stat, float(1 - scipy.stats.chi2.cdf(stat, z.shape[1]))
