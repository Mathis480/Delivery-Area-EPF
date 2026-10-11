"""
Rolling Inverse-MAE Forecast Averaging and Statistical Evaluation.
Based on Marcjasz et al. (2018) and Puć & Janczura (2024).
"""
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import scipy.stats


def compute_rolling_weighted_average(
    predictions: Dict[str, np.ndarray],
    y_true_2d: np.ndarray,
    benchmark_2d: np.ndarray,
    calib_window: int = 14,
    power: float = 1.0,
    warmup_weights: Optional[Dict[str, float]] = None,
    qh_adaptive: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, float]]:
    """
    Computes rolling out-of-sample forecast ensemble using ex-ante inverse-MAE weights.
    When qh_adaptive=True, computes separate calibration weights for each 15-minute QH.
    """
    model_names = list(predictions.keys())
    n_models = len(model_names)
    n_days, n_qh = y_true_2d.shape

    for name, pred in predictions.items():
        if pred.shape != (n_days, n_qh):
            raise ValueError(f"Model '{name}' shape {pred.shape} does not match y_true_2d {(n_days, n_qh)}")

    preds_all = np.array([predictions[m] for m in model_names])
    pred_ens_2d = np.zeros((n_days, n_qh), dtype=np.float64)
    weights_history = np.zeros((n_days, n_models), dtype=np.float64)

    if warmup_weights is None:
        init_w = np.ones(n_models) / n_models
    else:
        init_w = np.array([warmup_weights.get(m, 1.0 / n_models) for m in model_names])
        init_w = init_w / np.sum(init_w)

    for d in range(n_days):
        p_today = preds_all[:, d, :]
        if d < calib_window:
            weights_history[d] = init_w
            pred_ens_2d[d] = init_w @ p_today
        else:
            cal_p = preds_all[:, d - calib_window : d, :]
            cal_y = y_true_2d[d - calib_window : d, :]
            axis = 1 if qh_adaptive else (1, 2)
            maes = np.nan_to_num(np.nanmean(np.abs(cal_p - cal_y[None, ...]), axis=axis), nan=1e6)
            maes = np.maximum(maes, 1e-4)
            inv = 1.0 / (maes ** power)
            if qh_adaptive:
                w_q = inv / np.sum(inv, axis=0, keepdims=True)
                pred_ens_2d[d] = np.sum(w_q * p_today, axis=0)
                weights_history[d] = np.mean(w_q, axis=1)
            else:
                w = inv / np.sum(inv)
                pred_ens_2d[d] = w @ p_today
                weights_history[d] = w

    weights_df = pd.DataFrame(weights_history, columns=model_names)

    mask = ~np.isnan(y_true_2d) & ~np.isnan(pred_ens_2d)
    y_eval, b_eval, e_eval = y_true_2d[mask], benchmark_2d[mask], pred_ens_2d[mask]
    mae_naive = float(np.mean(np.abs(y_eval - b_eval)))
    mae_ens = float(np.mean(np.abs(y_eval - e_eval)))
    rmse_ens = float(np.sqrt(np.mean((y_eval - e_eval) ** 2)))
    rmae = mae_ens / mae_naive if mae_naive > 0 else 1.0

    metrics = {
        "mae_naive": mae_naive,
        "mae_ensemble": mae_ens,
        "rmae_ensemble": rmae,
        "rmse_ensemble": rmse_ens,
    }
    return pred_ens_2d, weights_df, metrics


def compute_rolling_qh_adaptive_ensemble(
    predictions: Dict[str, np.ndarray],
    y_true_2d: np.ndarray,
    benchmark_2d: np.ndarray,
    calib_window: int = 28,
    power: float = 2.0,
    warmup_weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, float]]:
    """Time-of-day adaptive ensemble optimizing ex-ante weights per 15-minute QH."""
    return compute_rolling_weighted_average(
        predictions, y_true_2d, benchmark_2d, calib_window, power, warmup_weights, qh_adaptive=True
    )


def dm_test(actual: np.ndarray, p1: np.ndarray, p2: np.ndarray, h: int = 7) -> Tuple[float, float]:
    """Multivariate daily-vector Diebold-Mariano test with Newey-West HAC variance."""
    n = (min(len(actual), len(p1), len(p2)) // 96) * 96
    d = np.mean(np.abs(actual[:n].reshape(-1, 96) - p1[:n].reshape(-1, 96)), axis=1) - \
        np.mean(np.abs(actual[:n].reshape(-1, 96) - p2[:n].reshape(-1, 96)), axis=1)
    d_bar, D = np.mean(d), len(d)
    gamma = [np.mean((d[k:] - d_bar) * (d[:D - k] - d_bar)) for k in range(h + 1)]
    var_hac = gamma[0] + 2 * sum((1 - k / (h + 1)) * gamma[k] for k in range(1, h + 1))
    stat = float(d_bar / np.sqrt(max(var_hac, 1e-12) / D))
    return stat, float(2 * (1 - scipy.stats.norm.cdf(abs(stat))))
