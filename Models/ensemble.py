"""
Rolling Weighted Forecast Averaging Ensemble Generator.
Adapts the weighted forecast averaging methodology of Puć & Janczura (2024, Eq. 25;
implemented as 'intel_avg_generator.py' in their replication code) and Bates & Granger (1969)
to combine multiple model forecasts and feature-set predictions with the naive benchmark
using rolling ex-ante inverse-MAE weighting over a historical calibration window.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def compute_rolling_weighted_average(
    predictions: Dict[str, np.ndarray],
    y_true_2d: np.ndarray,
    benchmark_2d: np.ndarray,
    calib_window: int = 28,
    power: float = 2.0,
    warmup_weights: Optional[Dict[str, float]] = None,
    qh_adaptive: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, float]]:
    """
    Computes rolling out-of-sample forecast ensemble using strictly ex-ante inverse-MAE weights.
    When qh_adaptive=True, computes separate ex-ante calibration weights for each quarter-hour position
    to dynamically adapt to solar intraday duck curve volatility.
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
