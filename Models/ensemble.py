"""
Rolling Weighted Forecast Averaging Ensemble Generator.

Direct Methodological Foundation:
1. Marcjasz, Serafin & Weron (2018, Energies 11(9), 2364):
   Introduced the rolling inverse-MAE (1/MAE) window-weighting scheme for electricity prices:
   w_j^W = (1 / MAE_j^W) / sum_k (1 / MAE_k^W).
2. Puć & Janczura (2024, Eq. 25; arXiv:2411.16237v1 & 'intel_avg_generator.py'):
   Adopted the inverse-MAE rolling scheme over calibration windows W in {7, 14, 21, 28} days
   to average cSVR feature sets.
3. Our implementation:
   Vectorized 2D ex-ante rolling execution of Puć & Janczura's 'intel_avg_generator.py'
   with calibration window W = 14 days and strictly linear inverse-MAE weighting (p = 1.0).
"""
from typing import Dict, List, Optional, Tuple
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
    Computes rolling out-of-sample forecast ensemble using strictly ex-ante inverse-MAE weights.
    When qh_adaptive=True (default), computes separate ex-ante calibration weights for each 15-minute
    quarter-hour position across past days (Puć & Janczura, 2024, intel_avg_generator.py), strictly
    preventing any D-1 late-night information spillover.
    """
    model_names = list(predictions.keys())
    n_models = len(model_names)
    n_days, n_qh = y_true_2d.shape

    # Validate shapes
    for name, pred in predictions.items():
        if pred.shape != (n_days, n_qh):
            raise ValueError(f"Model '{name}' shape {pred.shape} does not match y_true_2d shape {(n_days, n_qh)}")

    pred_ens_2d = np.zeros((n_days, n_qh), dtype=np.float64)
    weights_history = np.zeros((n_days, n_models), dtype=np.float64)

    # Default warmup weights
    if warmup_weights is None:
        init_w = np.ones(n_models) / n_models
    else:
        init_w = np.array([warmup_weights.get(m, 1.0 / n_models) for m in model_names])
        init_w = init_w / np.sum(init_w)

    for d in range(n_days):
        if d < calib_window:
            # Warmup period: use initial fixed weights
            w = init_w.copy()
            weights_history[d] = w
            for q in range(n_qh):
                for i_m, m in enumerate(model_names):
                    pred_ens_2d[d, q] += w[i_m] * predictions[m][d, q]
        else:
            if not qh_adaptive:
                # Daily aggregated calibration window [d - calib_window : d]
                cal_true = y_true_2d[d - calib_window : d]
                maes = []
                for m in model_names:
                    cal_pred = predictions[m][d - calib_window : d]
                    mask = ~np.isnan(cal_true) & ~np.isnan(cal_pred)
                    mae_m = float(np.mean(np.abs(cal_true[mask] - cal_pred[mask]))) if np.sum(mask) > 0 else 1e6
                    maes.append(max(mae_m, 1e-4))

                inv_mae = 1.0 / (np.array(maes) ** power)
                w = inv_mae / np.sum(inv_mae)
                weights_history[d] = w

                for q in range(n_qh):
                    for i_m, m in enumerate(model_names):
                        pred_ens_2d[d, q] += w[i_m] * predictions[m][d, q]
            else:
                # Quarter-Hour Specific Adaptive Calibration
                day_w_avg = np.zeros(n_models)
                for q in range(n_qh):
                    cal_true_q = y_true_2d[d - calib_window : d, q]
                    maes_q = []
                    for m in model_names:
                        cal_pred_q = predictions[m][d - calib_window : d, q]
                        mask = ~np.isnan(cal_true_q) & ~np.isnan(cal_pred_q)
                        mae_m = float(np.mean(np.abs(cal_true_q[mask] - cal_pred_q[mask]))) if np.sum(mask) > 0 else 1e6
                        maes_q.append(max(mae_m, 1e-4))

                    inv_mae_q = 1.0 / (np.array(maes_q) ** power)
                    w_q = inv_mae_q / np.sum(inv_mae_q)
                    day_w_avg += w_q

                    for i_m, m in enumerate(model_names):
                        pred_ens_2d[d, q] += w_q[i_m] * predictions[m][d, q]

                weights_history[d] = day_w_avg / n_qh

    weights_df = pd.DataFrame(weights_history, columns=model_names)

    # Compute overall test metrics (excluding NaNs)
    mask_eval = ~np.isnan(y_true_2d.ravel()) & ~np.isnan(pred_ens_2d.ravel())
    y_eval = y_true_2d.ravel()[mask_eval]
    b_eval = benchmark_2d.ravel()[mask_eval]
    e_eval = pred_ens_2d.ravel()[mask_eval]

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
    """Time-of-Day Adaptive Ensemble: dynamically optimizes ex-ante weights per 15-minute QH."""
    return compute_rolling_weighted_average(
        predictions=predictions,
        y_true_2d=y_true_2d,
        benchmark_2d=benchmark_2d,
        calib_window=calib_window,
        power=power,
        warmup_weights=warmup_weights,
        qh_adaptive=True,
    )


# Backward-compatible alias
compute_rolling_intelligent_ensemble = compute_rolling_weighted_average


def dm_test(actual: np.ndarray, p1: np.ndarray, p2: np.ndarray, h: int = 7) -> Tuple[float, float]:
    """Multivariate daily-vector Diebold-Mariano test with Newey-West HAC variance (Ziel & Weron, 2018)."""
    n = (min(len(actual), len(p1), len(p2)) // 96) * 96
    d = np.mean(np.abs(actual[:n].reshape(-1, 96) - p1[:n].reshape(-1, 96)), axis=1) - \
        np.mean(np.abs(actual[:n].reshape(-1, 96) - p2[:n].reshape(-1, 96)), axis=1)
    d_bar, D = np.mean(d), len(d)
    gamma = [np.mean((d[k:] - d_bar) * (d[:D - k] - d_bar)) for k in range(h + 1)]
    var_hac = gamma[0] + 2 * sum((1 - k / (h + 1)) * gamma[k] for k in range(1, h + 1))
    stat = float(d_bar / np.sqrt(max(var_hac, 1e-12) / D))
    return stat, float(2 * (1 - scipy.stats.norm.cdf(abs(stat))))
