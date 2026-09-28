#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast Vector Exploration on June 2024 (DE2 Amprion) using Pre-cached SummerContext:
Tests:
  1. Baseline: Pool=30d (X_val_full), Lasso m=0.15, Feature distance only.
  2. Full Lookback Pool: Pool=822d (X_trainval), Lasso m=0.15.
  3. Full Lookback Pool + Benchmark-Conditioned Support Selection (cSVR-style).
  4. Consistent Saturation: Support residual computed against saturated linear bypass.
  5. Smooth Linear Bypass: Ridge (alpha=100) instead of Lasso.
  6. Pure Neural Net: No Linear Bypass (direct residual to naive benchmark).
"""
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge

from config import (
    FEATURE_SET_MACRO,
    FEATURE_SET_NEIGHBOR,
    FEATURE_SET_FUNDAMENTAL,
    FEATURE_SET_BALANCE,
    MAML_BACKBONE_BATCH,
    MAML_META_LR,
)
from Models.ensemble import compute_rolling_intelligent_ensemble
from optimize_maml_large_scale_june2024 import SummerEvaluationContext, build_maml_net, pretrain_gncl_backbones

FEATURE_SETS = [
    FEATURE_SET_MACRO,
    FEATURE_SET_NEIGHBOR,
    FEATURE_SET_FUNDAMENTAL,
    FEATURE_SET_BALANCE,
]

def run_vector(
    ctx: SummerEvaluationContext,
    backbones: dict,
    desc: str,
    pool_mode: str = "val_30",          # "val_30" vs "full_822"
    use_bm_dist: bool = False,          # add |bm_i - bm_test| to distance
    bm_weight: float = 1.0,
    bypass_mode: str = "lasso_m015",    # "lasso_m015", "lasso_consistent", "ridge_m015", "none"
    bypass_m: float = 0.15,
    inner_steps: int = 8,
    inner_lr: float = 0.0025,
    support_k: int = 28,
    tau: float = 4.0,
):
    print(f"\n=======================================================")
    print(f"Testing Vector: {desc}")
    print(f"=======================================================")
    t0 = time.time()
    n_days, n_qh = 30, 96
    preds_eur = {}
    preds_2d = {}

    for s in FEATURE_SETS:
        model = build_maml_net(ctx.X_arrays[s].shape[1], hidden_sizes=[128, 64], dropout=0.0, l1_reg=1e-4, activation="tanh")
        b_weights = backbones[s]
        model.set_weights(b_weights)
        variables = model.trainable_variables
        lr_t = tf.constant(inner_lr, tf.float32)

        @tf.function(reduce_retracing=True)
        def _adapt_step(xs, rs, xte, w_k):
            for _ in range(inner_steps):
                with tf.GradientTape() as tape:
                    p = model(xs, training=True)
                    loss = tf.reduce_sum(w_k * tf.abs(rs - p))
                grads = tape.gradient(loss, variables)
                grads, _ = tf.clip_by_global_norm(grads, 1.0)
                for v, g in zip(variables, grads):
                    if g is not None:
                        v.assign_sub(lr_t * g)
            return model(xte, training=False)

        pred_list = []
        for i in range(ctx.n_eval):
            w = ctx.window_cache[s][i]
            coef_vec, intercept = ctx.lasso_cache[s][i]
            bench_eur = float(ctx.bm[i])

            if w is None or w.X_val_full is None:
                pred_list.append(bench_eur)
                continue

            model.set_weights(b_weights)

            # 1. Pool selection
            if pool_mode == "full_822" and w.X_trainval is not None:
                X_pool = w.X_trainval
                y_pool = w.y_trainval
                bm_pool = w.bm_trainval
            else:
                X_pool = w.X_val_full
                y_pool = w.y_val
                bm_pool = w.bm_trainval[-len(y_pool):] if w.bm_trainval is not None else None

            # 2. Linear bypass handling
            if bypass_mode == "none":
                coef_use = None
                p_lin_safe = 0.0
                r_pool = y_pool.ravel().astype(np.float32)[:, None]
            elif bypass_mode == "ridge_m015":
                # Quick Ridge fit on pool
                m_ridge = Ridge(alpha=100.0).fit(X_pool, y_pool)
                coef_use = m_ridge.coef_.astype(np.float32)
                int_use = float(m_ridge.intercept_)
                p_lin_sup = (X_pool @ coef_use + int_use).astype(np.float32)
                r_pool = (y_pool.ravel() - p_lin_sup).astype(np.float32)[:, None]
                p_lin_test = float((w.X_test_full.ravel() @ coef_use) + int_use)
                p_lin_safe = float(bypass_m * np.tanh(p_lin_test / bypass_m)) if bypass_m > 0 else p_lin_test
            elif bypass_mode == "lasso_consistent":
                coef_use = coef_vec
                if coef_use is not None:
                    p_lin_sup = (X_pool @ coef_use + intercept).astype(np.float32)
                    p_lin_sup_safe = (bypass_m * np.tanh(p_lin_sup / bypass_m)).astype(np.float32) if bypass_m > 0 else p_lin_sup
                    r_pool = (y_pool.ravel() - p_lin_sup_safe).astype(np.float32)[:, None]
                    p_lin_test = float((w.X_test_full.ravel() @ coef_use) + intercept)
                    p_lin_safe = float(bypass_m * np.tanh(p_lin_test / bypass_m)) if bypass_m > 0 else p_lin_test
                else:
                    r_pool = y_pool.ravel().astype(np.float32)[:, None]
                    p_lin_safe = 0.0
            else: # "lasso_m015" (current baseline)
                coef_use = coef_vec
                if coef_use is not None:
                    p_lin_sup = (X_pool @ coef_use + intercept).astype(np.float32)
                    r_pool = (y_pool.ravel() - p_lin_sup).astype(np.float32)[:, None]
                    p_lin_test = float((w.X_test_full.ravel() @ coef_use) + intercept)
                    p_lin_safe = float(bypass_m * np.tanh(p_lin_test / bypass_m)) if bypass_m > 0 else p_lin_test
                else:
                    r_pool = y_pool.ravel().astype(np.float32)[:, None]
                    p_lin_safe = 0.0

            # 3. Distance calculation
            dists = cdist(X_pool, w.X_test_full.reshape(1, -1), metric="cityblock").ravel()
            if use_bm_dist and bm_pool is not None:
                bm_std = float(np.std(bm_pool)) if np.std(bm_pool) > 1e-4 else 1.0
                dists = dists + bm_weight * (np.abs(bm_pool - bench_eur) / bm_std)

            k = min(support_k, len(X_pool))
            sel_idx = np.argsort(dists)[:k]
            d_sel = dists[sel_idx]
            scale = max(float(np.std(d_sel)), 1e-4) * tau
            w_k = np.exp(-(d_sel - d_sel[0]) / scale)
            w_k = (w_k / np.sum(w_k)).astype(np.float32)[:, None]

            xs = tf.cast(X_pool[sel_idx], tf.float32)
            rs = tf.cast(r_pool[sel_idx], tf.float32)
            xte = tf.cast(w.X_test_full.reshape(1, -1), tf.float32)
            wt = tf.cast(w_k, tf.float32)

            r_p = float(_adapt_step(xs, rs, xte, wt).numpy().ravel()[0])
            r_p = float(np.clip(r_p, -0.5, 0.5))

            p_scaled = p_lin_safe + r_p
            p_final_eur = p_scaled * w.y_std + w.y_mean + bench_eur
            pred_list.append(p_final_eur)

        preds_eur[s] = np.array(pred_list, dtype=np.float64)
        preds_2d[s] = preds_eur[s].reshape(n_days, n_qh)
        mae_s = float(np.mean(np.abs(ctx.y_true - preds_eur[s])))
        print(f"  {s}: MAE = {mae_s:.4f} EUR/MWh")
        tf.keras.backend.clear_session()

    # Ensembles
    y_true_2d = ctx.y_true.reshape(n_days, n_qh)
    bm_2d = ctx.bm.reshape(n_days, n_qh)

    p_ens_p1, _, m_p1 = compute_rolling_intelligent_ensemble({f"m_{s}": preds_2d[s] for s in FEATURE_SETS}, y_true_2d, bm_2d, calib_window=14, power=1.0)
    p_ens_p4, _, m_p4 = compute_rolling_intelligent_ensemble({f"m_{s}": preds_2d[s] for s in FEATURE_SETS}, y_true_2d, bm_2d, calib_window=14, power=4.0)
    p_ens_wn_p1, _, m_wn_p1 = compute_rolling_intelligent_ensemble({**{f"m_{s}": preds_2d[s] for s in FEATURE_SETS}, "naive": bm_2d}, y_true_2d, bm_2d, calib_window=14, power=1.0)
    p_ens_wn_p4, _, m_wn_p4 = compute_rolling_intelligent_ensemble({**{f"m_{s}": preds_2d[s] for s in FEATURE_SETS}, "naive": bm_2d}, y_true_2d, bm_2d, calib_window=14, power=4.0)

    elapsed = time.time() - t0
    print(f"  => Pure Ens (p=1): {m_p1['mae_ensemble']:.4f} (rMAE: {m_p1['rmae_ensemble']:.4f})")
    print(f"  => Pure Ens (p=4): {m_p4['mae_ensemble']:.4f} (rMAE: {m_p4['rmae_ensemble']:.4f})")
    print(f"  => +Naive Ens(p=1): {m_wn_p1['mae_ensemble']:.4f} (rMAE: {m_wn_p1['rmae_ensemble']:.4f})")
    print(f"  => +Naive Ens(p=4): {m_wn_p4['mae_ensemble']:.4f} (rMAE: {m_wn_p4['rmae_ensemble']:.4f})")
    print(f"  [Elapsed: {elapsed:.1f}s]")

    return {
        "desc": desc,
        "mae_pure_p1": m_p1["mae_ensemble"],
        "rmae_pure_p1": m_p1["rmae_ensemble"],
        "mae_pure_p4": m_p4["mae_ensemble"],
        "rmae_pure_p4": m_p4["rmae_ensemble"],
        "mae_wn_p1": m_wn_p1["mae_ensemble"],
        "rmae_wn_p1": m_wn_p1["rmae_ensemble"],
        "mae_wn_p4": m_wn_p4["mae_ensemble"],
        "rmae_wn_p4": m_wn_p4["rmae_ensemble"],
        "elapsed": elapsed,
    }

if __name__ == "__main__":
    ctx = SummerEvaluationContext(zone="DE2", month="2024-06")

    print("Pre-training clean [128, 64] backbones (no GNCL)...")
    backbones = pretrain_gncl_backbones(
        ctx=ctx,
        hidden_sizes=[128, 64],
        dropout=0.0,
        l1_reg=1e-4,
        activation="tanh",
        ncl_mode="difference",
        lambda_ncl=0.0,
        ambiguity_type="squared",
        meta_epochs=6,
    )

    results = []

    # Vector 1: Current Baseline (Pool=30d, Lasso m=0.15, Cityblock)
    r1 = run_vector(ctx, backbones, "1. Baseline (Pool=30d, Lasso m=0.15)", pool_mode="val_30", bypass_mode="lasso_m015")
    results.append(r1)

    # Vector 2: Full History Pool (822d like cSVR)
    r2 = run_vector(ctx, backbones, "2. Full History Pool (822d like cSVR)", pool_mode="full_822", bypass_mode="lasso_m015")
    results.append(r2)

    # Vector 3: Full History Pool + Benchmark-Conditioned Selection
    r3 = run_vector(ctx, backbones, "3. Full History Pool + BM-Conditioned Support", pool_mode="full_822", use_bm_dist=True, bm_weight=2.0, bypass_mode="lasso_m015")
    results.append(r3)

    # Vector 4: Consistent Saturation (Saturate Support & Test)
    r4 = run_vector(ctx, backbones, "4. Consistent Saturation (Support & Test)", pool_mode="val_30", bypass_mode="lasso_consistent", bypass_m=0.15)
    results.append(r4)

    # Vector 5: Consistent Saturation + Full Pool
    r5 = run_vector(ctx, backbones, "5. Consistent Saturation + Full Pool (822d)", pool_mode="full_822", bypass_mode="lasso_consistent", bypass_m=0.15)
    results.append(r5)

    # Vector 6: Pure Neural Network (No Bypass)
    r6 = run_vector(ctx, backbones, "6. Pure Neural Net (No Bypass, Pool=822d)", pool_mode="full_822", bypass_mode="none")
    results.append(r6)

    # Vector 7: Pure Neural Net with BM-Conditioned Support
    r7 = run_vector(ctx, backbones, "7. Pure Neural Net + BM-Conditioned Support (822d)", pool_mode="full_822", use_bm_dist=True, bm_weight=2.0, bypass_mode="none")
    results.append(r7)

    # Summary
    df = pd.DataFrame(results)
    out_csv = "Results/vector_exploration_june2024_de2.csv"
    df.to_csv(out_csv, index=False)

    print("\n" + "="*95)
    print("FINAL COMPARISON OF ARCHITECTURAL & ENSEMBLE VECTORS (JUNE 2024 DE2)")
    print(f"BENCHMARKS: Naive = {ctx.mae_naive:.4f} EUR/MWh | cSVR Ens (+Naive) = {ctx.mae_csvr:.4f} EUR/MWh")
    print("="*95)
    cols = ["desc", "mae_pure_p1", "mae_pure_p4", "mae_wn_p1", "mae_wn_p4"]
    print(df[cols].to_string(index=False))
