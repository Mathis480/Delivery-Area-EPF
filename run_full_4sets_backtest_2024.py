"""
Full 4-Feature-Set Backtest Runner (2024) across German Delivery Areas (DE1..DE4).

Evaluates all 4 Feature Sets:
  - Set 1 (Macro): Own VWAP history, auction prices, AR lags, fundamentals, balancing
  - Set 2 (Neighbor): Adjacent QH contract price trajectories, spreads, volumes
  - Set 3 (Fundamental): Pure fundamentals — solar, wind, load, reserves, calendar
  - Set 4 (Balance): Zone-level bilateral intraday commercial flow balances

Models evaluated across all 4 sets:
  - LASSO (LassoLarsCV with expanding-window temporal CV)
  - cSVR (Corrected SVR with Laplacian kernel, Puć & Janczura 2024)
  - MAML-NN (Meta-learning NN with NCL backbone diversification, Finn et al. 2017)

Ensembles computed (rolling inverse-MAE weighting, Puć & Janczura 2024; Bates & Granger 1969):
  - Pure cSVR Ensemble across all 4 sets
  - Pure MAML-NN Ensemble across all 4 sets (with NCL-diversified backbones)
  - Consolidated Hybrid Intelligent Ensemble across all 12 models + naive
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats

from config import (
    RESULTS_DIR,
    LOOKBACK_DAYS,
    VAL_DAYS,
    N_QH,
    ZONES,
    FEATURE_SET_MACRO,
    FEATURE_SET_NEIGHBOR,
    FEATURE_SET_FUNDAMENTAL,
    FEATURE_SET_BALANCE,
    MAML_INNER_STEPS,
    MAML_INNER_LR,
    MAML_CLIP_RESIDUAL,
    MAML_PROX_SHRINK,
    MAML_USE_LINEAR_BYPASS,
    MAML_BYPASS_SATURATION_M,
    MAML_SUPPORT_SELECTION,
    MAML_SUPPORT_K,
    MAML_SUPPORT_POOL_DAYS,
    MAML_SUPPORT_WEIGHTING,
    MAML_KERNEL_TAU,
    CSVR_USE_GAUSSIAN_CORRECTION,
    ENSEMBLE_CALIB_DAYS,
    ENSEMBLE_POWER,
)
from data_loader import QHDataLoader
from Models.linear import train_lasso, predict_lasso
from Models.csvr import train_csvr, predict_csvr
from Models.maml_nn import MAMLManager
from Models.ensemble import compute_rolling_intelligent_ensemble


def _backtransform(y_pred_scaled: float, y_mean: float, y_std: float, benchmark_eur: float) -> float:
    return float(y_pred_scaled * y_std + y_mean + benchmark_eur)


def dm_test(actual: np.ndarray, pred1: np.ndarray, pred2: np.ndarray):
    e1 = np.abs(actual - pred1)
    e2 = np.abs(actual - pred2)
    d = e1 - e2
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    stat = mean_d / np.sqrt(var_d / len(d))
    p_val = 2 * (1 - stats.norm.cdf(np.abs(stat)))
    return stat, p_val


def run_zone_full4sets(zone: str, force_recompute_maml: bool = True):
    print(f"\n=======================================================")
    print(f"       STARTING FULL 4-SETS BACKTEST FOR {zone} (2024)")
    print(f"=======================================================")

    in_npz = os.path.join(RESULTS_DIR, "annual_run_2024", f"results_{zone}_multiset_2024.npz")
    if not os.path.exists(in_npz):
        in_npz = os.path.join(RESULTS_DIR, "annual_run_2024", f"results_{zone}_2024.npz")
    if not os.path.exists(in_npz):
        raise FileNotFoundError(f"Base 2024 results not found for {zone} at {in_npz}")

    base_data = np.load(in_npz)
    dates_arr = base_data["dates"]
    qh_arr = base_data["qh_idx"]
    y_true_arr = base_data["y_true"]
    bm_arr = base_data["benchmark"]
    n_total = len(dates_arr)

    # Initialize data dict with base values
    data = {k: base_data[k] for k in base_data.files}

    loader = QHDataLoader()

    # Pre-train / load MAML backbones for all 4 sets
    maml_s1 = MAMLManager(zone, loader.n_features(zone, FEATURE_SET_MACRO), feature_set=FEATURE_SET_MACRO)
    maml_s1.pretrain_backbone(loader, verbose=False)

    maml_s2 = MAMLManager(zone, loader.n_features(zone, FEATURE_SET_NEIGHBOR), feature_set=FEATURE_SET_NEIGHBOR)
    maml_s2.pretrain_backbone(loader, verbose=False)

    maml_s3 = MAMLManager(zone, loader.n_features(zone, FEATURE_SET_FUNDAMENTAL), feature_set=FEATURE_SET_FUNDAMENTAL)
    maml_s3.pretrain_backbone(loader, verbose=False)

    maml_s4 = MAMLManager(zone, loader.n_features(zone, FEATURE_SET_BALANCE), feature_set=FEATURE_SET_BALANCE)
    maml_s4.pretrain_backbone(loader, verbose=False)

    out_npz = os.path.join(RESULTS_DIR, "annual_run_2024", f"results_{zone}_full4sets_2024.npz")

    has_canonical_csvr = "pred_csvr_s1" in base_data and len(base_data["pred_csvr_s1"]) == n_total
    if has_canonical_csvr:
        print(f"[{zone}] Fast-loading canonical cSVR and LASSO predictions ({n_total} QH) - skipping redundant model training.")
        pred_csvr_s1 = list(base_data["pred_csvr_s1"])
        pred_csvr_s2 = list(base_data["pred_csvr_s2"])
        pred_csvr_s3 = list(base_data["pred_csvr_s3"])
        pred_csvr_s4 = list(base_data["pred_csvr_s4"])
        pred_lasso_s1 = list(base_data["pred_lasso_s1"])
        pred_lasso_s2 = list(base_data["pred_lasso_s2"])
        pred_lasso_s3 = list(base_data["pred_lasso_s3"])
        pred_lasso_s4 = list(base_data["pred_lasso_s4"])
    else:
        pred_lasso_s1, pred_lasso_s2, pred_lasso_s3, pred_lasso_s4 = [], [], [], []
        pred_csvr_s1, pred_csvr_s2, pred_csvr_s3, pred_csvr_s4 = [], [], [], []

    checkpoint_path = os.path.join(RESULTS_DIR, "annual_run_2024", f"checkpoint_{zone}_full4sets.npz")
    if force_recompute_maml or not os.path.exists(checkpoint_path):
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
            except Exception:
                pass
        start_idx = 0
        pred_maml_s1 = []
        pred_maml_s2 = []
        pred_maml_s3 = []
        pred_maml_s4 = []
    else:
        print(f"[{zone}] Found existing checkpoint at {checkpoint_path}, loading...")
        ckpt = np.load(checkpoint_path)
        pred_maml_s1 = list(ckpt["pred_maml_s1"])
        pred_maml_s2 = list(ckpt["pred_maml_s2"])
        pred_maml_s3 = list(ckpt["pred_maml_s3"])
        pred_maml_s4 = list(ckpt["pred_maml_s4"])
        start_idx = len(pred_maml_s1)
        print(f"[{zone}] Resumed MAML from checkpoint at index {start_idx}/{n_total} ({start_idx/n_total*100:.1f}%).")

    t0 = time.time()
    pool_days = MAML_SUPPORT_POOL_DAYS if MAML_SUPPORT_SELECTION == "regime_l1" else VAL_DAYS
    print(f"[{zone}] Running full 4-sets inference across {n_total} quarter-hours (starting at {start_idx})...")
    print(f"       Configuration: LinearBypass={MAML_USE_LINEAR_BYPASS}, Support={MAML_SUPPORT_SELECTION} (K={MAML_SUPPORT_K}, Pool={pool_days}d), InnerSteps={MAML_INNER_STEPS}, Weighting={MAML_SUPPORT_WEIGHTING} (tau={MAML_KERNEL_TAU})")

    for i in tqdm(range(start_idx, n_total), desc=f"{zone} 4-Sets"):
        d_str = str(dates_arr[i])
        q = int(qh_arr[i])
        bench_eur = float(bm_arr[i])

        # ---------------- SET 1 (Macro) ----------------
        w1 = loader.get_window(zone, q, d_str, feature_set=FEATURE_SET_MACRO, lookback_days=LOOKBACK_DAYS, val_days=VAL_DAYS, filter_zero_var=False)
        if w1 is not None and w1.X_trainval is not None:
            try:
                m_l1, _ = train_lasso(w1.X_trainval, w1.y_trainval)
                lin_c1 = m_l1.coef_ if MAML_USE_LINEAR_BYPASS else None
                lin_i1 = m_l1.intercept_ if MAML_USE_LINEAR_BYPASS else 0.0
                p_maml_scaled1 = maml_s1.adapt_and_predict(
                    qh_idx=q,
                    X_support=w1.X_trainval,
                    y_support=w1.y_trainval,
                    X_test=w1.X_test_full,
                    linear_coef=lin_c1,
                    linear_intercept=lin_i1,
                    inner_steps=MAML_INNER_STEPS,
                    inner_lr=MAML_INNER_LR,
                    prox_shrink=MAML_PROX_SHRINK,
                    clip_residual=MAML_CLIP_RESIDUAL,
                    bypass_saturation_m=MAML_BYPASS_SATURATION_M,
                    selection_mode=MAML_SUPPORT_SELECTION,
                    support_k=MAML_SUPPORT_K,
                    return_residual=False,
                )
                pred_maml_s1.append(_backtransform(p_maml_scaled1, w1.y_mean, w1.y_std, bench_eur))
            except Exception:
                pred_maml_s1.append(bench_eur)
        else:
            pred_maml_s1.append(bench_eur)

        # ---------------- SET 2 (Neighbor) ----------------
        w2 = loader.get_window(zone, q, d_str, feature_set=FEATURE_SET_NEIGHBOR, lookback_days=LOOKBACK_DAYS, val_days=VAL_DAYS, filter_zero_var=False)
        if w2 is not None and w2.X_trainval is not None:
            try:
                m_l2, _ = train_lasso(w2.X_trainval, w2.y_trainval)
                lin_c2 = m_l2.coef_ if MAML_USE_LINEAR_BYPASS else None
                lin_i2 = m_l2.intercept_ if MAML_USE_LINEAR_BYPASS else 0.0
                p_maml_scaled2 = maml_s2.adapt_and_predict(
                    qh_idx=q,
                    X_support=w2.X_trainval,
                    y_support=w2.y_trainval,
                    X_test=w2.X_test_full,
                    linear_coef=lin_c2,
                    linear_intercept=lin_i2,
                    inner_steps=MAML_INNER_STEPS,
                    inner_lr=MAML_INNER_LR,
                    prox_shrink=MAML_PROX_SHRINK,
                    clip_residual=MAML_CLIP_RESIDUAL,
                    bypass_saturation_m=MAML_BYPASS_SATURATION_M,
                    selection_mode=MAML_SUPPORT_SELECTION,
                    support_k=MAML_SUPPORT_K,
                    return_residual=False,
                )
                pred_maml_s2.append(_backtransform(p_maml_scaled2, w2.y_mean, w2.y_std, bench_eur))
            except Exception:
                pred_maml_s2.append(bench_eur)
        else:
            pred_maml_s2.append(bench_eur)

        # ---------------- SET 3 (Fundamental) ----------------
        w3 = loader.get_window(zone, q, d_str, feature_set=FEATURE_SET_FUNDAMENTAL, lookback_days=LOOKBACK_DAYS, val_days=VAL_DAYS, filter_zero_var=False)
        if w3 is not None and w3.X_trainval is not None:
            try:
                m_l3, _ = train_lasso(w3.X_trainval, w3.y_trainval)
                lin_c3 = m_l3.coef_ if MAML_USE_LINEAR_BYPASS else None
                lin_i3 = m_l3.intercept_ if MAML_USE_LINEAR_BYPASS else 0.0
                p_maml_scaled3 = maml_s3.adapt_and_predict(
                    qh_idx=q,
                    X_support=w3.X_trainval,
                    y_support=w3.y_trainval,
                    X_test=w3.X_test_full,
                    linear_coef=lin_c3,
                    linear_intercept=lin_i3,
                    inner_steps=MAML_INNER_STEPS,
                    inner_lr=MAML_INNER_LR,
                    prox_shrink=MAML_PROX_SHRINK,
                    clip_residual=MAML_CLIP_RESIDUAL,
                    bypass_saturation_m=MAML_BYPASS_SATURATION_M,
                    selection_mode=MAML_SUPPORT_SELECTION,
                    support_k=MAML_SUPPORT_K,
                    return_residual=False,
                )
                pred_maml_s3.append(_backtransform(p_maml_scaled3, w3.y_mean, w3.y_std, bench_eur))
            except Exception:
                pred_maml_s3.append(bench_eur)
        else:
            pred_maml_s3.append(bench_eur)

        # ---------------- SET 4 (Balance) ----------------
        w4 = loader.get_window(zone, q, d_str, feature_set=FEATURE_SET_BALANCE, lookback_days=LOOKBACK_DAYS, val_days=VAL_DAYS, filter_zero_var=False)
        if w4 is not None and w4.X_trainval is not None:
            try:
                m_l4, _ = train_lasso(w4.X_trainval, w4.y_trainval)
                lin_c4 = m_l4.coef_ if MAML_USE_LINEAR_BYPASS else None
                lin_i4 = m_l4.intercept_ if MAML_USE_LINEAR_BYPASS else 0.0
                p_maml_scaled4 = maml_s4.adapt_and_predict(
                    qh_idx=q,
                    X_support=w4.X_trainval,
                    y_support=w4.y_trainval,
                    X_test=w4.X_test_full,
                    linear_coef=lin_c4,
                    linear_intercept=lin_i4,
                    inner_steps=MAML_INNER_STEPS,
                    inner_lr=MAML_INNER_LR,
                    prox_shrink=MAML_PROX_SHRINK,
                    clip_residual=MAML_CLIP_RESIDUAL,
                    bypass_saturation_m=MAML_BYPASS_SATURATION_M,
                    selection_mode=MAML_SUPPORT_SELECTION,
                    support_k=MAML_SUPPORT_K,
                    return_residual=False,
                )
                pred_maml_s4.append(_backtransform(p_maml_scaled4, w4.y_mean, w4.y_std, bench_eur))
            except Exception:
                pred_maml_s4.append(bench_eur)
        else:
            pred_maml_s4.append(bench_eur)

        if (i + 1) % 1000 == 0 or (i + 1) == n_total:
            np.savez(
                checkpoint_path,
                pred_lasso_s1=np.array(pred_lasso_s1, dtype=np.float32),
                pred_lasso_s2=np.array(pred_lasso_s2, dtype=np.float32),
                pred_lasso_s3=np.array(pred_lasso_s3, dtype=np.float32),
                pred_lasso_s4=np.array(pred_lasso_s4, dtype=np.float32),
                pred_csvr_s1=np.array(pred_csvr_s1, dtype=np.float32),
                pred_csvr_s2=np.array(pred_csvr_s2, dtype=np.float32),
                pred_csvr_s3=np.array(pred_csvr_s3, dtype=np.float32),
                pred_csvr_s4=np.array(pred_csvr_s4, dtype=np.float32),
                pred_maml_s1=np.array(pred_maml_s1, dtype=np.float32),
                pred_maml_s2=np.array(pred_maml_s2, dtype=np.float32),
                pred_maml_s3=np.array(pred_maml_s3, dtype=np.float32),
                pred_maml_s4=np.array(pred_maml_s4, dtype=np.float32),
            )


    print(f"[{zone}] Inference loop finished in {time.time() - t0:.1f}s.")

    # Store 1D arrays in data dict
    data["pred_lasso_s1"] = np.array(pred_lasso_s1, dtype=np.float32)
    data["pred_lasso_s2"] = np.array(pred_lasso_s2, dtype=np.float32)
    data["pred_lasso_s3"] = np.array(pred_lasso_s3, dtype=np.float32)
    data["pred_lasso_s4"] = np.array(pred_lasso_s4, dtype=np.float32)
    data["pred_lasso"] = data["pred_lasso_s1"]

    data["pred_csvr_s1"] = np.array(pred_csvr_s1, dtype=np.float32)
    data["pred_csvr_s2"] = np.array(pred_csvr_s2, dtype=np.float32)
    data["pred_csvr_s3"] = np.array(pred_csvr_s3, dtype=np.float32)
    data["pred_csvr_s4"] = np.array(pred_csvr_s4, dtype=np.float32)
    data["pred_csvr"] = data["pred_csvr_s1"]

    data["pred_maml_s1"] = np.array(pred_maml_s1, dtype=np.float32)
    data["pred_maml_s2"] = np.array(pred_maml_s2, dtype=np.float32)
    data["pred_maml_s3"] = np.array(pred_maml_s3, dtype=np.float32)
    data["pred_maml_s4"] = np.array(pred_maml_s4, dtype=np.float32)
    data["pred_maml"] = data["pred_maml_s1"]

    # Map to 2D daily matrices
    dates_daily = data["dates_daily"]
    n_days = len(dates_daily)
    date_to_idx = {d: idx for idx, d in enumerate(dates_daily)}

    for key in [
        "pred_lasso_s1", "pred_lasso_s2", "pred_lasso_s3", "pred_lasso_s4",
        "pred_csvr_s1", "pred_csvr_s2", "pred_csvr_s3", "pred_csvr_s4",
        "pred_maml_s1", "pred_maml_s2", "pred_maml_s3", "pred_maml_s4",
    ]:
        mat_2d = np.full((n_days, N_QH), np.nan, dtype=np.float64)
        arr_1d = data[key]
        for idx in range(n_total):
            d_i = date_to_idx[dates_arr[idx]]
            q = int(qh_arr[idx])
            mat_2d[d_i, q] = float(arr_1d[idx])
        data[f"{key}_2d"] = mat_2d

    data["pred_lasso_2d"] = data["pred_lasso_s1_2d"]
    data["pred_csvr_2d"] = data["pred_csvr_s1_2d"]
    data["pred_maml_2d"] = data["pred_maml_s1_2d"]

    y_true_2d = data["y_true_2d"]
    bm_2d = data["benchmark_2d"]

    # -------------------------------------------------------------
    # 1. PURE cSVR MULTI-SET ENSEMBLE (S1 + S2 + S3 + S4)
    # -------------------------------------------------------------
    csvr_4sets = {
        "csvr_s1": data["pred_csvr_s1_2d"],
        "csvr_s2": data["pred_csvr_s2_2d"],
        "csvr_s3": data["pred_csvr_s3_2d"],
        "csvr_s4": data["pred_csvr_s4_2d"],
    }
    p_csvr_ens_pure_2d, _, m_csvr_pure = compute_rolling_intelligent_ensemble(
        csvr_4sets, y_true_2d, bm_2d, calib_window=ENSEMBLE_CALIB_DAYS, power=ENSEMBLE_POWER
    )
    p_csvr_ens_wn_2d, _, m_csvr_wn = compute_rolling_intelligent_ensemble(
        {**csvr_4sets, "naive": bm_2d}, y_true_2d, bm_2d, calib_window=ENSEMBLE_CALIB_DAYS, power=ENSEMBLE_POWER
    )

    # -------------------------------------------------------------
    # 2. PURE MAML-NN MULTI-SET ENSEMBLE (S1 + S2 + S3 + S4)
    # -------------------------------------------------------------
    maml_4sets = {
        "maml_s1": data["pred_maml_s1_2d"],
        "maml_s2": data["pred_maml_s2_2d"],
        "maml_s3": data["pred_maml_s3_2d"],
        "maml_s4": data["pred_maml_s4_2d"],
    }
    p_maml_ens_pure_2d, w_maml_pure, m_maml_pure = compute_rolling_intelligent_ensemble(
        maml_4sets, y_true_2d, bm_2d, calib_window=ENSEMBLE_CALIB_DAYS, power=ENSEMBLE_POWER
    )
    p_maml_ens_wn_2d, w_maml_wn, m_maml_wn = compute_rolling_intelligent_ensemble(
        {**maml_4sets, "naive": bm_2d}, y_true_2d, bm_2d, calib_window=ENSEMBLE_CALIB_DAYS, power=ENSEMBLE_POWER
    )

    # Map ensemble arrays to 1D
    def _to_1d(mat_2d):
        res = np.zeros(n_total, dtype=np.float32)
        for idx in range(n_total):
            d_i = date_to_idx[dates_arr[idx]]
            q = int(qh_arr[idx])
            res[idx] = mat_2d[d_i, q]
        return res

    data["pred_csvr_ens_pure_2d"] = p_csvr_ens_pure_2d
    data["pred_csvr_ens_pure"] = _to_1d(p_csvr_ens_pure_2d)
    data["pred_csvr_ens_wn_2d"] = p_csvr_ens_wn_2d
    data["pred_csvr_ens_wn"] = _to_1d(p_csvr_ens_wn_2d)

    data["pred_maml_ens_pure_2d"] = p_maml_ens_pure_2d
    data["pred_maml_ens_pure"] = _to_1d(p_maml_ens_pure_2d)
    data["pred_maml_ens_wn_2d"] = p_maml_ens_wn_2d
    data["pred_maml_ens_wn"] = _to_1d(p_maml_ens_wn_2d)

    # Save to both NPZ files for full backward compatibility
    np.savez_compressed(out_npz, **data)
    multiset_npz = os.path.join(RESULTS_DIR, "annual_run_2024", f"results_{zone}_multiset_2024.npz")
    np.savez_compressed(multiset_npz, **data)

    weights_csv_maml = os.path.join(RESULTS_DIR, "annual_run_2024", f"ensemble_weights_{zone}_maml_2024.csv")
    w_maml_pure.to_csv(weights_csv_maml, index=False)

    # Print Full Summary Table
    mae_naive = float(np.mean(np.abs(y_true_arr - bm_arr)))
    print(f"\n==================== {zone} FULL 4-SETS ANNUAL RESULTS (2024) ====================")
    print(f"Naive Benchmark:             {mae_naive:.4f} EUR/MWh (1.0000)")
    print("--------------------------------------------------------------------------------")
    print(f"Set 1 (Macro):")
    print(f"  LASSO S1:                  {np.mean(np.abs(y_true_arr - data['pred_lasso_s1'])):.4f} EUR/MWh ({np.mean(np.abs(y_true_arr - data['pred_lasso_s1']))/mae_naive:.4f})")
    print(f"  cSVR S1:                   {np.mean(np.abs(y_true_arr - data['pred_csvr_s1'])):.4f} EUR/MWh ({np.mean(np.abs(y_true_arr - data['pred_csvr_s1']))/mae_naive:.4f})")
    print(f"  MAML S1 (Tuned):           {np.mean(np.abs(y_true_arr - data['pred_maml_s1'])):.4f} EUR/MWh ({np.mean(np.abs(y_true_arr - data['pred_maml_s1']))/mae_naive:.4f})")
    print("--------------------------------------------------------------------------------")
    print(f"Set 2 (Neighbor Contracts):")
    print(f"  LASSO S2:                  {np.mean(np.abs(y_true_arr - data['pred_lasso_s2'])):.4f} EUR/MWh ({np.mean(np.abs(y_true_arr - data['pred_lasso_s2']))/mae_naive:.4f})")
    print(f"  cSVR S2:                   {np.mean(np.abs(y_true_arr - data['pred_csvr_s2'])):.4f} EUR/MWh ({np.mean(np.abs(y_true_arr - data['pred_csvr_s2']))/mae_naive:.4f})")
    print(f"  MAML S2 (Tuned):           {np.mean(np.abs(y_true_arr - data['pred_maml_s2'])):.4f} EUR/MWh ({np.mean(np.abs(y_true_arr - data['pred_maml_s2']))/mae_naive:.4f})")
    print("--------------------------------------------------------------------------------")
    print(f"Set 3 (Pure Fundamentals):")
    print(f"  LASSO S3:                  {np.mean(np.abs(y_true_arr - data['pred_lasso_s3'])):.4f} EUR/MWh ({np.mean(np.abs(y_true_arr - data['pred_lasso_s3']))/mae_naive:.4f})")
    print(f"  cSVR S3:                   {np.mean(np.abs(y_true_arr - data['pred_csvr_s3'])):.4f} EUR/MWh ({np.mean(np.abs(y_true_arr - data['pred_csvr_s3']))/mae_naive:.4f})")
    print(f"  MAML S3 (Tuned):           {np.mean(np.abs(y_true_arr - data['pred_maml_s3'])):.4f} EUR/MWh ({np.mean(np.abs(y_true_arr - data['pred_maml_s3']))/mae_naive:.4f})")
    print("--------------------------------------------------------------------------------")
    print(f"Set 4 (Zone Balances):")
    print(f"  LASSO S4:                  {np.mean(np.abs(y_true_arr - data['pred_lasso_s4'])):.4f} EUR/MWh ({np.mean(np.abs(y_true_arr - data['pred_lasso_s4']))/mae_naive:.4f})")
    print(f"  cSVR S4:                   {np.mean(np.abs(y_true_arr - data['pred_csvr_s4'])):.4f} EUR/MWh ({np.mean(np.abs(y_true_arr - data['pred_csvr_s4']))/mae_naive:.4f})")
    print(f"  MAML S4 (Tuned):           {np.mean(np.abs(y_true_arr - data['pred_maml_s4'])):.4f} EUR/MWh ({np.mean(np.abs(y_true_arr - data['pred_maml_s4']))/mae_naive:.4f})")
    print("--------------------------------------------------------------------------------")
    print(f"Multi-Set Ensembles (Strict Versus):")
    print(f"  Pure cSVR Ensemble (S1..S4):    {m_csvr_pure['mae_ensemble']:.4f} EUR/MWh ({m_csvr_pure['rmae_ensemble']:.4f})")
    print(f"  cSVR Ensemble (+Naive):         {m_csvr_wn['mae_ensemble']:.4f} EUR/MWh ({m_csvr_wn['rmae_ensemble']:.4f})")
    print(f"  Pure MAML-NN Ensemble (S1..S4): {m_maml_pure['mae_ensemble']:.4f} EUR/MWh ({m_maml_pure['rmae_ensemble']:.4f})")
    print(f"  MAML-NN Ensemble (+Naive):      {m_maml_wn['mae_ensemble']:.4f} EUR/MWh ({m_maml_wn['rmae_ensemble']:.4f})")

    # DM Tests
    dm_stat_c, dm_p_c = dm_test(y_true_arr, data["pred_csvr_ens_wn"], bm_arr)
    dm_stat_m, dm_p_m = dm_test(y_true_arr, data["pred_maml_ens_wn"], bm_arr)
    dm_pure_vs, dm_pure_p = dm_test(y_true_arr, data["pred_maml_ens_pure"], data["pred_csvr_ens_pure"])
    dm_wn_vs, dm_wn_p = dm_test(y_true_arr, data["pred_maml_ens_wn"], data["pred_csvr_ens_wn"])

    print(f"\nDiebold-Mariano Tests:")
    print(f"  cSVR Ens (+Naive) vs Naive:   DM = {dm_stat_c:.2f} (p = {dm_p_c:.4e})")
    print(f"  MAML Ens (+Naive) vs Naive:   DM = {dm_stat_m:.2f} (p = {dm_p_m:.4e})")
    print(f"  MAML vs cSVR (Pure Ensemble): DM = {dm_pure_vs:.2f} (p = {dm_pure_p:.4e})")
    print(f"  MAML vs cSVR (+Naive Ens):    DM = {dm_wn_vs:.2f} (p = {dm_wn_p:.4e})")

    return {
        "zone": zone,
        "mae_naive": mae_naive,
        "mae_csvr_ens_pure": m_csvr_pure["mae_ensemble"],
        "rmae_csvr_ens_pure": m_csvr_pure["rmae_ensemble"],
        "mae_csvr_ens_wn": m_csvr_wn["mae_ensemble"],
        "rmae_csvr_ens_wn": m_csvr_wn["rmae_ensemble"],
        "mae_maml_ens_pure": m_maml_pure["mae_ensemble"],
        "rmae_maml_ens_pure": m_maml_pure["rmae_ensemble"],
        "mae_maml_ens_wn": m_maml_wn["mae_ensemble"],
        "rmae_maml_ens_wn": m_maml_wn["rmae_ensemble"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", type=str, default="all", choices=["all", "DE1", "DE2", "DE3", "DE4"])
    parser.add_argument("--force", action="store_true", help="Force recomputing MAML from scratch")
    args = parser.parse_args()

    target_zones = ZONES if args.zone == "all" else [args.zone]
    all_summaries = []

    for z in target_zones:
        summary = run_zone_full4sets(z, force_recompute_maml=args.force)
        all_summaries.append(summary)

    df_summary = pd.DataFrame(all_summaries)
    summary_csv = os.path.join(RESULTS_DIR, "annual_run_2024", "full_4sets_summary_metrics_2024.csv")
    df_summary.to_csv(summary_csv, index=False)
    print("\n=======================================================")
    print("      ALL 4 ZONES COMPLETED - FINAL SUMMARY TABLE      ")
    print("=======================================================")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
